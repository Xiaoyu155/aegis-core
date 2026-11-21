"""Read open GitHub pull requests and produce a head-pinned review checklist."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from .combat import _remote_repository


_DEFAULT_RECEIPTS = frozenset(
    f"v16-gates / {os} / Python {version}"
    for os in ("ubuntu-latest", "windows-latest") for version in ("3.11", "3.12")
)


def _pages(root, endpoint):
    result = subprocess.run(
        ["gh", "api", "--method", "GET", "--paginate", "--slurp", endpoint],
        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(f"GitHub read failed for {endpoint}; check gh authentication and access")
    return json.loads(result.stdout)


def _review(root, repo, pull, receipt_names):
    number = pull["number"]
    sha = pull["head"]["sha"]
    if type(number) is not int or number <= 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("invalid GitHub pull request number or head SHA")
    detail = _pages(root, f"repos/{repo}/pulls/{number}")[0]
    if detail.get("state") != "open" or detail.get("head", {}).get("sha") != sha:
        return {"number": number, "title": pull["title"], "head": sha, "risk": "high",
                "issues": ["pull_changed_during_read"], "receipts": [], "merge_command": None}
    checks = {}
    for page in _pages(root, f"repos/{repo}/commits/{sha}/check-runs?filter=latest&per_page=100"):
        for check in page["check_runs"]:
            key = (check["name"], check.get("app", {}).get("slug"))
            if key not in checks or check["id"] > checks[key]["id"]:
                checks[key] = check
    statuses = _pages(root, f"repos/{repo}/commits/{sha}/status?per_page=100")[0]
    receipts = [{"name": check["name"], "status": check["status"], "conclusion": check["conclusion"],
                 "url": check.get("html_url"), "head": sha}
                for (name, app), check in checks.items()
                if name in receipt_names and app == "github-actions" and check.get("head_sha") == sha]
    issues = []
    if detail.get("mergeable") is not True:
        issues.append("mergeability_unconfirmed" if detail.get("mergeable") is None else "merge_conflict")
    if detail.get("mergeable_state") != "clean":
        issues.append("github_branch_gate_not_clean")
    if detail.get("draft"):
        issues.append("draft")
    if {item["name"] for item in receipts} != receipt_names:
        issues.append("aegis_receipt_missing")
    elif any(item["status"] != "completed" or item["conclusion"] != "success" for item in receipts):
        issues.append("aegis_receipt_not_green")
    if any(check.get("status") != "completed" or check.get("conclusion") not in
           {"success", "neutral", "skipped"} for check in checks.values()) or (
        statuses.get("total_count", 0) and statuses.get("state") != "success"
    ):
        issues.append("checks_not_green")
    changed = detail.get("changed_files", 0)
    churn = detail.get("additions", 0) + detail.get("deletions", 0)
    risk = "high" if issues else "medium" if changed >= 100 or churn >= 3000 else "low"
    return {"number": number, "title": pull["title"], "url": pull["html_url"], "head": sha,
            "risk": risk, "issues": issues, "receipts": receipts, "changed_files": changed,
            "merge_command": (f"gh pr merge {number} --repo {repo} --match-head-commit {sha} --squash"
                              if not issues else None)}


def review_fleet(repo_root=".", *, receipt_name=None) -> dict:
    root = Path(repo_root).expanduser().resolve(strict=True)
    repo = _remote_repository(root)
    if not repo or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*", repo):
        raise ValueError("review-fleet requires a github.com origin")
    pulls = {}
    for page in _pages(root, f"repos/{repo}/pulls?state=open&sort=created&direction=asc&per_page=100"):
        for pull in page:
            pulls[pull["number"]] = pull
    receipt_names = {receipt_name} if receipt_name else _DEFAULT_RECEIPTS
    items = [_review(root, repo, pull, receipt_names) for pull in pulls.values()]
    rank = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda item: (rank[item["risk"]], -len(item["issues"]), item["number"]))
    return {"repository": repo, "count": len(items), "pull_requests": items}
