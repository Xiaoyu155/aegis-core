"""Idempotent, dry-run-first PR delivery planning for Aegis fleet branches."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .pr_orchestrator import (
    _github_json,
    _priority,
    audit_github_repository,
    audit_local_repository,
    discover_github_repositories,
)


PROTECTED_REPOSITORIES = {"nova-agent", "checkpoint", "edu-typing"}
_SUCCESS = {"success", "completed", "passed", "green"}


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _commits(repo: Path, base: str, branch: str) -> list[dict]:
    raw = _git(repo, "log", "--date=iso-strict", "--format=%H%x1f%an%x1f%ad%x1f%s",
               f"{base}..{branch}")
    commits = []
    for row in raw.splitlines():
        fields = row.split("\x1f", 3)
        if len(fields) == 4:
            commits.append({"sha": fields[0], "author": fields[1], "date": fields[2], "subject": fields[3]})
    return commits


def _gate_state(value) -> tuple[str, dict]:
    if isinstance(value, str):
        return value.lower(), {"source": "provided", "state": value.lower()}
    if not isinstance(value, dict):
        return "unknown", {"source": "unavailable", "state": "unknown"}
    state = str(value.get("state", value.get("status", "unknown"))).lower()
    checks = value.get("checks")
    if isinstance(checks, list):
        states = [str(item.get("conclusion") or item.get("status") or "unknown").lower()
                  if isinstance(item, dict) else str(item).lower() for item in checks]
        if states and all(item in _SUCCESS for item in states):
            state = "success"
        elif any(item in {"failure", "failed", "cancelled", "timed_out", "error"} for item in states):
            state = "failure"
        elif states:
            state = "pending"
    return state, dict(value)


def _pr_body(repository: str, delivery: dict, commits: list[dict], gate: dict) -> str:
    subjects = commits[:12]
    summary = "\n".join(f"- {item['subject']} (`{item['sha'][:10]}`)" for item in subjects) or "- No commits ahead of the default branch."
    if len(commits) > len(subjects):
        summary += f"\n- … and {len(commits) - len(subjects)} earlier commit(s)."
    checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
    gate_summary = gate.get("summary") or gate.get("state", "unknown")
    if checks:
        gate_summary += ": " + ", ".join(
            f"{item.get('name', 'check')}={item.get('conclusion') or item.get('status', 'unknown')}"
            for item in checks if isinstance(item, dict))
    merge_status = "conflict-free" if delivery.get("status") == "mergeable" else "conflicts found"
    stats = delivery.get("changes", {})
    return (
        "## Defect and delivery summary\n"
        f"Repository: `{repository}`\n\n"
        f"{summary}\n\n"
        "## Red/green probe and CI results\n"
        f"- Gate state: **{gate_summary}**\n"
        f"- Changed files: {stats.get('changed_files', 0)}; lines added/deleted: "
        f"{stats.get('lines_added', 0)}/{stats.get('lines_deleted', 0)}\n\n"
        "## Conflict preflight\n"
        f"- Three-way merge: **{merge_status}**\n"
        f"- Default branch: `{delivery.get('default_branch', '?')}`; head: `{delivery.get('head', '?')}`\n\n"
        "## Manual review checklist\n"
        "- [ ] Confirm defect summary against the changed code and tests.\n"
        "- [ ] Review security, data migration, and compatibility impact.\n"
        "- [ ] Confirm CI probes and required approvals are complete.\n"
        "- [ ] Verify release notes and rollback plan where applicable.\n"
    )


def _build_delivery(repository: str, delivery: dict, commits: list[dict], gate_value,
                    existing_pr: dict | None = None) -> dict:
    gate_state, gate = _gate_state(gate_value)
    title_seed = next((item["subject"] for item in commits if item.get("subject")), "Aegis delivery update")
    title = f"Aegis: {title_seed}"[:120]
    merged = bool(delivery.get("feature_already_merged_to_default"))
    if existing_pr:
        action = "reuse_existing_pr"
    elif merged:
        action = "already_merged"
    else:
        action = "create_pr"
    conflicts = delivery.get("status") == "conflict" or delivery.get("conflicted") is True
    ready = action == "create_pr" and not conflicts and gate_state in _SUCCESS
    return {
        "repository": repository,
        "branch": delivery.get("branch"),
        "head": delivery.get("head"),
        "default_branch": delivery.get("default_branch"),
        "commits": commits,
        "commit_count": len(commits),
        "changes": dict(delivery.get("changes", {})),
        "conflict_preflight": delivery.get("status", "unknown"),
        "risk_priority": delivery.get("priority") or _priority(conflicts, delivery.get("changes", {"files": [], "changed_files": 0, "lines_added": 0, "lines_deleted": 0})),
        "gate_status": gate,
        "existing_pr": existing_pr,
        "action": action,
        "ready_to_execute": ready,
        "title": title,
        "body": _pr_body(repository, delivery, commits, gate),
        "idempotency_key": f"{repository.lower()}:{delivery.get('branch')}:{delivery.get('head')}",
    }


def plan_local_repository(repo: str | Path, *, default_branch: str | None = None,
                          gate_statuses: dict | None = None, existing_prs: dict | None = None,
                          delivery_branches=None) -> list[dict]:
    root = Path(repo).resolve()
    audited = audit_local_repository(root, default_branch=default_branch,
                                     delivery_branches=delivery_branches)
    plans = []
    for delivery in audited["deliveries"]:
        branch = delivery.get("branch", "")
        commits = _commits(root, audited["default_head"], branch) if delivery.get("head") else []
        gate = (gate_statuses or {}).get(branch, (gate_statuses or {}).get(delivery.get("head")))
        existing = (existing_prs or {}).get(branch)
        item = dict(delivery)
        item["default_branch"] = audited["default_branch"]
        plans.append(_build_delivery(f"local/{root.name}", item, commits, gate, existing))
    return plans


def _github_get_pull(repository: str, owner: str, branch: str, base: str, token: str | None):
    params = urlencode({"state": "open", "head": f"{owner}:{branch}", "base": base})
    payload = _github_json(f"https://api.github.com/repos/{repository}/pulls?{params}", token)
    if not isinstance(payload, list):
        return None
    return payload[0] if payload else None


def _github_gate(repository: str, sha: str, token: str | None) -> dict:
    status = _github_json(f"https://api.github.com/repos/{repository}/commits/{sha}/status", token)
    check_data = _github_json(f"https://api.github.com/repos/{repository}/commits/{sha}/check-runs?per_page=100", token)
    checks = check_data.get("check_runs", []) if isinstance(check_data, dict) else []
    states = [str(item.get("conclusion") or item.get("status") or "unknown").lower()
              for item in checks if isinstance(item, dict)]
    combined = str(status.get("state", "unknown")).lower() if isinstance(status, dict) else "unknown"
    if combined in {"failure", "error"} or any(item in {"failure", "cancelled", "timed_out", "action_required"} for item in states):
        state = "failure"
    elif states and all(item == "success" for item in states) and combined in {"success", "none", "unknown"}:
        state = "success"
    elif states or combined == "pending":
        state = "pending"
    else:
        state = "unknown"
    return {"state": state, "summary": f"commit status={combined}; check runs={len(checks)}",
            "checks": checks, "commit_status": status}


def plan_github_fleet(owner: str, *, token: str | None = None,
                      exclude_repositories=()) -> dict:
    """Read remote delivery refs and checks, then create submit-ready PR drafts."""
    exclusions = {str(item).strip().lower() for item in (*PROTECTED_REPOSITORIES, *exclude_repositories) if str(item).strip()}
    repositories = []
    excluded = []
    for metadata in discover_github_repositories(owner, token=token):
        full_name = metadata.get("full_name", "")
        short = full_name.rsplit("/", 1)[-1]
        if full_name.lower() in exclusions or short.lower() in exclusions:
            excluded.append(full_name)
            continue
        audited = audit_github_repository(metadata, token=token)
        if audited.get("status") == "no_delivery_branches":
            continue
        deliveries = []
        for delivery in audited.get("deliveries", []):
            branch = delivery.get("branch", "")
            head = delivery.get("head")
            if not head:
                deliveries.append(_build_delivery(full_name, delivery, [], None))
                continue
            compare = _github_json(
                f"https://api.github.com/repos/{full_name}/compare/{quote(metadata['default_branch'], safe='')}...{quote(branch, safe='')}",
                token)
            commits_payload = compare.get("commits", []) if isinstance(compare, dict) else []
            commits = [{"sha": row.get("sha", ""),
                        "author": (row.get("commit", {}).get("author") or {}).get("name", ""),
                        "date": (row.get("commit", {}).get("author") or {}).get("date", ""),
                        "subject": (row.get("commit", {}).get("message", "").splitlines() or [""])[0]}
                       for row in commits_payload if isinstance(row, dict)]
            gate = _github_gate(full_name, head, token)
            pr = _github_get_pull(full_name, owner, branch, metadata["default_branch"], token)
            existing = ({"number": pr.get("number"), "url": pr.get("html_url"), "title": pr.get("title")}
                        if pr else None)
            delivery["default_branch"] = metadata["default_branch"]
            deliveries.append(_build_delivery(full_name, delivery, commits, gate, existing))
        repositories.append({"repository": full_name, "default_branch": metadata.get("default_branch"),
                             "deliveries": deliveries})
    flattened = [item for repo in repositories for item in repo["deliveries"]]
    return {"schema_version": "aegis.fleet_ship_plan.v1", "owner": owner,
            "mode": "dry_run", "executed": False,
            "repositories": repositories, "excluded_repositories": sorted(excluded),
            "summary": {"repositories_scanned": len(repositories), "deliveries": len(flattened),
                        "ready_to_execute": sum(item["ready_to_execute"] for item in flattened),
                        "existing_prs": sum(item["action"] == "reuse_existing_pr" for item in flattened),
                        "conflicts": sum(item["conflict_preflight"] == "conflict" for item in flattened)}}


def plan_local_fleet(repositories, *, default_branch: str | None = None,
                     exclude_repositories=(), gate_statuses=None, existing_prs=None) -> dict:
    exclusions = {str(item).strip().lower() for item in (*PROTECTED_REPOSITORIES, *exclude_repositories) if str(item).strip()}
    reports = []
    excluded = []
    for path in repositories:
        root = Path(path).resolve()
        if root.name.lower() in exclusions or str(root).lower() in exclusions:
            excluded.append(str(root))
            continue
        gates = (gate_statuses or {}).get(str(root), {})
        prs = (existing_prs or {}).get(str(root), {})
        reports.append({"repository": root.name,
                        "deliveries": plan_local_repository(root, default_branch=default_branch,
                                                            gate_statuses=gates, existing_prs=prs)})
    flattened = [item for repo in reports for item in repo["deliveries"]]
    return {"schema_version": "aegis.fleet_ship_plan.v1", "owner": None,
            "mode": "dry_run", "executed": False, "repositories": reports,
            "excluded_repositories": sorted(excluded),
            "summary": {"repositories_scanned": len(reports), "deliveries": len(flattened),
                        "ready_to_execute": sum(item["ready_to_execute"] for item in flattened),
                        "existing_prs": sum(item["action"] == "reuse_existing_pr" for item in flattened),
                        "conflicts": sum(item["conflict_preflight"] == "conflict" for item in flattened)}}


def execute_ship_plan(plan: dict, *, token: str | None) -> dict:
    """Create only ready PRs; existing PRs are reused and never duplicated."""
    if not token:
        raise ValueError("--execute requires a GitHub token in the configured token environment variable")
    results = []
    for repository in plan.get("repositories", []):
        for delivery in repository.get("deliveries", []):
            if delivery.get("action") == "reuse_existing_pr":
                results.append({"repository": delivery["repository"], "branch": delivery["branch"],
                                "status": "reused", "pull_request": delivery.get("existing_pr")})
                continue
            if delivery.get("action") != "create_pr" or not delivery.get("ready_to_execute"):
                results.append({"repository": delivery.get("repository"), "branch": delivery.get("branch"),
                                "status": "held", "reason": "gate, conflict, or idempotency precondition not satisfied"})
                continue
            repository_name = delivery["repository"]
            if repository_name.startswith("local/") or "/" not in repository_name:
                raise ValueError("--execute only accepts GitHub owner/repository plans")
            owner, repo_name = repository_name.split("/", 1)
            base = delivery["default_branch"]
            branch = delivery["branch"]
            existing = _github_get_pull(repository_name, owner, branch, base, token)
            if existing:
                results.append({"repository": repository_name, "branch": branch, "status": "reused",
                                "pull_request": {"number": existing.get("number"), "url": existing.get("html_url"),
                                                 "title": existing.get("title")}})
                continue
            latest_branch = _github_json(
                f"https://api.github.com/repos/{repository_name}/branches/{quote(branch, safe='')}", token)
            latest_head = ((latest_branch.get("commit") or {}).get("sha")
                           if isinstance(latest_branch, dict) else None)
            if latest_head != delivery.get("head"):
                results.append({"repository": repository_name, "branch": branch, "status": "held",
                                "reason": "delivery branch head changed after dry-run"})
                continue
            latest_gate = _github_gate(repository_name, latest_head, token)
            if latest_gate.get("state") != "success":
                results.append({"repository": repository_name, "branch": branch, "status": "held",
                                "reason": "latest GitHub checks are not green", "gate_status": latest_gate})
                continue
            fresh_audit = audit_github_repository(
                {"full_name": repository_name, "clone_url": f"https://github.com/{repository_name}.git",
                 "default_branch": base}, token=token)
            fresh_delivery = next((item for item in fresh_audit.get("deliveries", [])
                                   if item.get("branch") == branch), None)
            if not fresh_delivery or fresh_delivery.get("status") != "mergeable" or fresh_delivery.get("head") != latest_head:
                results.append({"repository": repository_name, "branch": branch, "status": "held",
                                "reason": "fresh three-way conflict preflight is not clean"})
                continue
            payload = json.dumps({"title": delivery["title"], "body": delivery["body"],
                                  "head": branch, "base": base}).encode()
            request = Request(f"https://api.github.com/repos/{owner}/{repo_name}/pulls", data=payload,
                              headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}",
                                       "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json",
                                       "User-Agent": "aegis-core-fleet-shipper"}, method="POST")
            with urlopen(request, timeout=30) as response:
                created = json.loads(response.read().decode("utf-8"))
            results.append({"repository": delivery["repository"], "branch": delivery["branch"],
                            "status": "created", "pull_request": {"number": created.get("number"),
                                                                       "url": created.get("html_url")}})
    return {**plan, "mode": "execute", "executed": True, "execution_results": results,
            "executed_at": datetime.now(timezone.utc).isoformat()}
