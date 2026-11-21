"""Preflight external requirements for a real Luna run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen

from aegis_core.env_doctor import EnvDoctor
from aegis_core.enforce import inspect_keys, key_directory
from aegis_core.sandbox import run_git

from .model_router import config_path, load_config


def _git(root: Path, *args: str, check=True):
    return run_git(root, *args, check=check).stdout.decode("utf-8", errors="replace").strip()


def _remote_repository(root: Path) -> str | None:
    remote = _git(root, "remote", "get-url", "origin", check=False)
    match = re.search(r"github\.com[:/]([^/]+/[^/.]+)(?:\.git)?$", remote)
    return match.group(1) if match else None


def _ci_state(root: Path) -> dict:
    repo = _remote_repository(root)
    if not repo:
        return {"status": "unknown", "reason": "GitHub origin is not configured"}
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD", check=False)
    branch = branch.split("/", 1)[-1] if branch else "main"
    url = f"https://api.github.com/repos/{repo}/actions/runs?branch={branch}&per_page=1"
    try:
        request = Request(url, headers={"Accept": "application/vnd.github+json",
                                        "User-Agent": "luna-combat-check"})
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("workflow_runs", [])
        if not rows:
            return {"status": "unknown", "reason": "no workflow runs found", "branch": branch}
        run = rows[0]
        return {"status": "ready" if run.get("status") == "completed" and run.get("conclusion") == "success"
                else "blocked", "branch": branch, "conclusion": run.get("conclusion"),
                "run_url": run.get("html_url"), "head_sha": run.get("head_sha")}
    except Exception as error:
        return {"status": "unknown", "reason": type(error).__name__, "branch": branch}


def _model_gateway(config_path: Path, root: Path | None = None) -> dict:
    if not config_path.is_file():
        return {"status": "blocked", "reason": "model routing configuration is missing"}
    try:
        production = load_config(config_path, root or config_path.parent)
        config = json.loads(config_path.read_text(encoding="utf-8-sig")) if not production else None
    except (OSError, ValueError):
        return {"status": "blocked", "reason": "model routing configuration is invalid"}
    if production:
        from urllib.parse import urlsplit
        checked = []
        ready = set()
        for slot, endpoint in production.models.items():
            reachable = False
            if endpoint.configured:
                parsed = urlsplit(endpoint.base_url)
                try:
                    with socket.create_connection((parsed.hostname, parsed.port or
                                                   (443 if parsed.scheme == "https" else 80)),
                                                  timeout=0.35):
                        reachable = True
                except (OSError, ValueError):
                    pass
            if reachable:
                ready.add(slot)
            checked.append({"slot_id": slot, "configured": endpoint.configured,
                            "reachable": reachable, "status": "ready" if reachable else "blocked"})
        missing = [role for role, route in production.routes.items() if ready.isdisjoint(route)]
        return {"status": "blocked" if missing else "ready", "ready_slots": sorted(ready),
                "checked_slots": checked, "unavailable_roles": missing}
    checked, ready = [], []
    for tier, nodes in config.get("tiers", {}).items():
        for node in nodes:
            if not node.get("enabled"):
                continue
            model = os.environ.get(node.get("model_env", ""), "") if node.get("model_env") else node.get("model", "")
            base = os.environ.get(node.get("base_url_env", ""), "") if node.get("base_url_env") else node.get("base_url", "")
            key_name = node.get("api_key_env", "")
            credential = not key_name or bool(os.environ.get(key_name))
            item = {"slot_id": node.get("id"), "tier": tier,
                    "model_configured": bool(model), "credential_configured": credential,
                    "endpoint_configured": bool(base), "reachable": False}
            try:
                from urllib.parse import urlsplit
                parsed = urlsplit(base)
                if parsed.hostname:
                    port = parsed.port or (443 if parsed.scheme == "https" else 80)
                    with socket.create_connection((parsed.hostname, port), timeout=0.35):
                        item["reachable"] = True
            except (OSError, ValueError):
                pass
            item_ready = bool(model and base and credential and item["reachable"])
            item["status"] = "ready" if item_ready else "blocked"
            if item_ready:
                ready.append(item["slot_id"])
            checked.append(item)
    return {"status": "ready" if ready else "blocked", "ready_slots": ready,
            "checked_slots": checked}


def _sandbox_state(root: Path) -> dict:
    base = _git(root, "rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="luna-combat-worktree-") as parent:
        path = Path(parent) / "checkout"
        added = run_git(root, "worktree", "add", "--detach", str(path), base, check=False)
        if added.returncode:
            return {"status": "blocked", "reason": added.stderr.decode("utf-8", "replace")[-300:]}
        try:
            return {"status": "ready", "isolated": True, "base_commit": base}
        finally:
            run_git(root, "worktree", "remove", "--force", str(path), check=False)


def _git_permissions(root: Path) -> dict:
    local_write = os.access(root, os.W_OK) and _remote_repository(root) is not None
    gh = shutil.which("gh")
    authenticated = False
    remote_write = False
    if gh:
        auth = subprocess.run([gh, "auth", "status", "--hostname", "github.com"],
                              capture_output=True, text=True, timeout=5, check=False)
        authenticated = auth.returncode == 0
        repo = _remote_repository(root)
        if authenticated and repo:
            view = subprocess.run([gh, "api", f"repos/{repo}"], capture_output=True, text=True,
                                  timeout=5, check=False)
            if view.returncode == 0:
                try:
                    remote_write = bool(json.loads(view.stdout).get("permissions", {}).get("push"))
                except json.JSONDecodeError:
                    pass
    elif os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        authenticated = True
    return {"status": "ready" if local_write and authenticated and remote_write else "blocked",
            "local_branch_writable": bool(local_write), "github_authenticated": authenticated,
            "remote_push_permission": remote_write,
            "reason": None if local_write and authenticated and remote_write else
            "requires GitHub authentication with push permission"}


def combat_check(repo_root: str | Path = ".", *, models: str | Path | None = None,
                 keys_dir: str | Path | None = None) -> dict:
    root = Path(repo_root).expanduser().resolve(strict=True)
    doctor = EnvDoctor(root, cleanup=False)
    try:
        environment = doctor.check(strict=False)
    finally:
        doctor.cleanup_temporary_configs()
    key_state = inspect_keys(keys_dir or key_directory())
    model_path = config_path(root, models)
    python_ok = sys.version_info >= (3, 11)
    if environment.get("python", {}).get("expected"):
        try:
            from packaging.specifiers import SpecifierSet
            python_ok = python_ok and SpecifierSet(environment["python"]["expected"]).contains(
                ".".join(map(str, sys.version_info[:3])), prereleases=True)
        except (ImportError, ValueError):
            pass
    checks = {
        "environment": {"status": "ready" if not environment.get("blocking") and
                         not environment.get("disk", {}).get("low_space") and python_ok else "blocked",
                         "python": environment.get("python"), "disk": environment.get("disk"),
                         "python_supported": python_ok,
                         "missing_dependencies": environment.get("missing_dependencies", [])},
        "ed25519_key": {"status": "ready" if key_state["valid"] else "blocked",
                         "fingerprint": key_state.get("fingerprint"), "errors": key_state.get("errors", [])},
        "git_permissions": _git_permissions(root),
        "sandbox_isolation": _sandbox_state(root),
        "ci": _ci_state(root),
        "model_gateway": _model_gateway(model_path, root),
    }
    status = "READY" if all(item.get("status") == "ready" for item in checks.values()) else "BLOCKED"
    blockers = [name for name, value in checks.items() if value.get("status") != "ready"]
    return {"status": status, "repo": str(root), "checks": checks, "blockers": blockers}
