"""Read-only fleet merge preflight and delivery prioritization.

The auditor never pushes, creates pull requests, or checks out files in a source
repository. It reads remote refs into a temporary bare repository and asks Git's
three-way merge engine to compute the result without updating any branch.
"""
from __future__ import annotations

import json
import base64
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _git(repo: Path, *args: str, check=True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result


def _diff_stats(repo: Path, base: str, feature: str) -> dict:
    result = _git(repo, "diff", "--numstat", f"{base}...{feature}")
    files = []
    added = deleted = 0
    for line in result.stdout.splitlines():
        fields = line.split("\t", 2)
        if len(fields) != 3:
            continue
        plus, minus, name = fields
        files.append(name)
        if plus.isdecimal():
            added += int(plus)
        if minus.isdecimal():
            deleted += int(minus)
    return {"files": sorted(files), "changed_files": len(files), "lines_added": added,
            "lines_deleted": deleted}


def _priority(conflicted: bool, stats: dict) -> str:
    if conflicted:
        return "P0"
    sensitive = any(any(token in name.lower() for token in
                        ("auth", "permission", "credential", "payment", "wallet", "migration"))
                    for name in stats["files"])
    if sensitive or stats["changed_files"] >= 100 or stats["lines_added"] + stats["lines_deleted"] >= 3000:
        return "P1"
    return "P2"


def audit_local_repository(repo: str | Path, *, default_branch: str | None = None,
                           delivery_branches: Iterable[str] | None = None) -> dict:
    """Audit local Git refs. All merge checks are dry runs and leave refs unchanged."""
    root = Path(repo).resolve()
    if not root.is_dir() or _git(root, "rev-parse", "--git-dir").returncode:
        raise ValueError(f"not a git repository: {root}")
    default_branch = default_branch or _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
    if not default_branch:
        default_branch = "main"
    default_ref = default_branch if default_branch.startswith("refs/") else f"refs/heads/{default_branch}"
    branch_names = list(delivery_branches) if delivery_branches is not None else [
        ref.removeprefix("refs/heads/") for ref in
        _git(root, "for-each-ref", "--format=%(refname)", "refs/heads").stdout.splitlines()
        if ref.startswith("refs/heads/feat/aegis-")]
    default_oid = _git(root, "rev-parse", "--verify", default_ref, check=False)
    if default_oid.returncode:
        raise ValueError(f"default branch does not exist: {default_branch}")
    default_oid = default_oid.stdout.strip()
    results = []
    for branch in sorted(set(branch_names)):
        ref = branch if branch.startswith("refs/") else f"refs/heads/{branch}"
        feature_oid = _git(root, "rev-parse", "--verify", ref, check=False)
        if feature_oid.returncode:
            results.append({"branch": branch, "status": "missing_branch", "priority": "P0"})
            continue
        feature_oid = feature_oid.stdout.strip()
        base_result = _git(root, "merge-base", default_oid, feature_oid, check=False)
        if base_result.returncode:
            results.append({"branch": branch, "head": feature_oid, "status": "unrelated_histories",
                            "priority": "P0", "conflicts": ["no common merge base"]})
            continue
        merge_base = base_result.stdout.strip()
        merge = _git(root, "merge-tree", "--write-tree", default_oid, feature_oid, check=False)
        conflicted = merge.returncode != 0
        stats = _diff_stats(root, merge_base, feature_oid)
        feature_commits = set(_git(root, "rev-list", feature_oid, f"^{merge_base}").stdout.splitlines())
        default_commits = set(_git(root, "rev-list", default_oid).stdout.splitlines())
        already_present = feature_commits & default_commits
        merged_into_default = bool(feature_commits) and len(already_present) == len(feature_commits)
        conflicts = []
        if conflicted:
            conflicts = [line for line in (merge.stdout + "\n" + merge.stderr).splitlines()
                         if "CONFLICT" in line or "Automatic merge failed" in line]
        branch_name = branch.removeprefix("refs/heads/").removeprefix("refs/remotes/origin/")
        results.append({"branch": branch_name, "head": feature_oid,
                        "default_head": default_oid, "merge_base": merge_base,
                        "status": "conflict" if conflicted else "mergeable",
                        "conflicted": conflicted, "conflicts": conflicts,
                        "default_branch_zero_pollution": not already_present,
                        "delivery_commits_already_on_default": len(already_present),
                        "feature_already_merged_to_default": merged_into_default,
                        "changes": stats, "priority": _priority(conflicted, stats)})
    results.sort(key=lambda item: (item.get("priority", "P2"), item["branch"]))
    return {"repository": root.name, "path": str(root), "default_branch": default_branch,
            "default_head": default_oid, "deliveries": results,
            "summary": {"delivery_branches": len(results),
                        "mergeable": sum(item.get("status") == "mergeable" for item in results),
                        "conflicts": sum(item.get("status") == "conflict" for item in results),
                        "default_branch_zero_pollution": all(item.get("default_branch_zero_pollution", False)
                                                              for item in results)}}


def _github_json(url: str, token: str | None) -> dict | list:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "aegis-core-pr-orchestrator"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urlopen(Request(url, headers=headers), timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"GitHub read-only request failed: {error}") from error


def discover_github_repositories(owner: str, *, token: str | None = None) -> list[dict]:
    """Read all owner repositories, including private repos available to the active connection."""
    if not owner or "/" in owner:
        raise ValueError("owner must be a GitHub account or organization name")

    def pages(url_template):
        result = []
        page = 1
        while True:
            payload = _github_json(url_template.format(page=page), token)
            if not isinstance(payload, list):
                raise RuntimeError("GitHub repository response was not a list")
            result.extend(item for item in payload if isinstance(item, dict) and item.get("clone_url"))
            if len(payload) < 100:
                break
            page += 1
        return result

    try:
        authenticated = pages(
            "https://api.github.com/user/repos?per_page=100&page={page}&visibility=all&affiliation=owner")
        owned = [repo for repo in authenticated
                 if str((repo.get("owner") or {}).get("login", "")).lower() == owner.lower()]
        if owned:
            return owned
    except RuntimeError:
        pass
    return pages(f"https://api.github.com/users/{owner}/repos?per_page=100&page={{page}}&type=all")


def audit_github_repository(metadata: dict, *, token: str | None = None) -> dict:
    """Read remote branches, clone matching refs into a temp bare repo, and dry-run merges."""
    name = metadata.get("full_name")
    url = metadata.get("clone_url")
    default_branch = metadata.get("default_branch")
    if not all(isinstance(value, str) and value for value in (name, url, default_branch)):
        raise ValueError("repository metadata requires full_name, clone_url, and default_branch")
    env = os.environ.copy()
    if token:
        env["GITHUB_TOKEN"] = token
        authorization = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update({"GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                    "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + authorization})
    refs = subprocess.run(["git", "ls-remote", "--heads", url], capture_output=True, text=True,
                          check=False, env=env, timeout=60)
    if refs.returncode:
        return {"repository": name, "default_branch": default_branch, "status": "ref_read_failed",
                "error": refs.stderr.strip()[:500], "deliveries": []}
    branches = [line.split("\t", 1)[1].removeprefix("refs/heads/") for line in refs.stdout.splitlines()
                if "\trefs/heads/feat/aegis-" in line]
    if not branches:
        return {"repository": name, "default_branch": default_branch, "status": "no_delivery_branches",
                "deliveries": [], "summary": {"delivery_branches": 0}}
    with tempfile.TemporaryDirectory(prefix="aegis-fleet-") as scratch:
        bare = Path(scratch) / "repo.git"
        initialized = subprocess.run(["git", "init", "--quiet", "--bare", str(bare)],
                                      capture_output=True, text=True, check=False, env=env, timeout=30)
        remote = subprocess.run(["git", "--git-dir", str(bare), "remote", "add", "origin", url],
                                capture_output=True, text=True, check=False, env=env, timeout=30)
        if initialized.returncode or remote.returncode:
            failed = initialized if initialized.returncode else remote
            return {"repository": name, "default_branch": default_branch, "status": "clone_failed",
                    "error": failed.stderr.strip()[:500], "deliveries": []}
        refspecs = [f"+refs/heads/{default_branch}:refs/remotes/origin/{default_branch}"]
        refspecs.extend(f"+refs/heads/{branch}:refs/remotes/origin/{branch}" for branch in branches)
        fetch = subprocess.run(["git", "--git-dir", str(bare), "fetch", "--quiet", "--filter=blob:none",
                                "--no-tags", "origin", *refspecs],
                               capture_output=True, text=True, check=False, env=env, timeout=300)
        if fetch.returncode:
            return {"repository": name, "default_branch": default_branch, "status": "fetch_failed",
                    "error": fetch.stderr.strip()[:500], "deliveries": []}
        report = audit_local_repository(bare, default_branch=f"refs/remotes/origin/{default_branch}",
                                        delivery_branches=[f"refs/remotes/origin/{branch}" for branch in branches])
        report["repository"] = name
        report.pop("path", None)
        return report


def render_dashboard(ledger: dict) -> str:
    rows = ["# Aegis Fleet Delivery Dashboard", "",
            "Read-only merge preflight; no branch or pull request was modified.", "",
            "| Repository | Branch | Priority | Merge | Files | + / − | Default branch |",
            "|---|---|---:|---|---:|---:|---|"]
    for repo in ledger.get("repositories", []):
        deliveries = repo.get("deliveries", [])
        if not deliveries:
            rows.append(f"| {repo.get('repository', '?')} | — | — | {repo.get('status', 'no delivery branches')} | 0 | 0 / 0 | — |")
        for item in deliveries:
            stats = item.get("changes", {})
            clean = "clean" if item.get("default_branch_zero_pollution") else "contains delivery"
            rows.append(f"| {repo.get('repository', '?')} | {item.get('branch', '?')} | {item.get('priority', '?')} | "
                        f"{item.get('status', '?')} | {stats.get('changed_files', 0)} | "
                        f"{stats.get('lines_added', 0)} / {stats.get('lines_deleted', 0)} | {clean} |")
    rows += ["", f"Repositories scanned: {ledger.get('summary', {}).get('repositories_scanned', 0)}",
             f"Repositories excluded: {ledger.get('summary', {}).get('repositories_excluded', 0)}",
             f"Delivery branches: {ledger.get('summary', {}).get('delivery_branches', 0)}",
             f"Merge conflicts: {ledger.get('summary', {}).get('conflicts', 0)}", ""]
    return "\n".join(rows)


def audit_fleet(owner: str, *, token: str | None = None,
                exclude_repositories: Iterable[str] = ()) -> dict:
    """Read-only audit of all owner repositories with feat/aegis-* remote branches."""
    exclusions = {value.strip().lower() for value in exclude_repositories if value.strip()}
    repositories = []
    excluded_count = 0
    excluded_names = []
    for metadata in discover_github_repositories(owner, token=token):
        full_name = metadata.get("full_name", "")
        short_name = full_name.rsplit("/", 1)[-1].lower()
        if full_name.lower() in exclusions or short_name in exclusions:
            excluded_count += 1
            excluded_names.append(full_name)
            continue
        report = audit_github_repository(metadata, token=token)
        if report.get("status") != "no_delivery_branches":
            repositories.append(report)
    deliveries = [item for repo in repositories for item in repo.get("deliveries", [])]
    return {"schema_version": "aegis.fleet_audit.v1", "owner": owner,
            "repositories": repositories,
            "summary": {"repositories_scanned": len(repositories),
                        "delivery_branches": len(deliveries),
                        "mergeable": sum(item.get("status") == "mergeable" for item in deliveries),
                        "conflicts": sum(item.get("status") == "conflict" for item in deliveries),
                        "repositories_excluded": excluded_count,
                        "excluded_repositories": sorted(excluded_names),
                        "p0": sum(item.get("priority") == "P0" for item in deliveries),
                        "p1": sum(item.get("priority") == "P1" for item in deliveries),
                        "p2": sum(item.get("priority") == "P2" for item in deliveries)}}


def write_fleet_artifacts(ledger: dict, output_dir: str | Path) -> dict:
    """Atomically persist the structured audit and Markdown summary."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "fleet_dashboard_v16.json"
    markdown_path = root / "fleet_dashboard_v16.md"
    for path, content in ((json_path, json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"),
                          (markdown_path, render_dashboard(ledger))):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    return {"json": str(json_path.resolve()), "markdown": str(markdown_path.resolve())}
