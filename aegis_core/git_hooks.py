"""Install dependency-free Git hooks that run the local Aegis patch gate."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys


_MARKER = "# aegis-autonomous-gate-v2"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def install_hooks(repo_root: str | Path = ".") -> dict:
    root = Path(repo_root).resolve()
    result = _git(root, "rev-parse", "--git-path", "hooks")
    if result.returncode:
        raise ValueError("repo_root is not inside a Git worktree")
    hooks = Path(result.stdout.strip())
    if not hooks.is_absolute():
        hooks = (root / hooks).resolve()
    hooks.mkdir(parents=True, exist_ok=True)
    backup_dir = root / ".aegis" / "hooks"
    installed = []
    for name, mode in (("pre-commit", "staged"), ("pre-push", "pre-push")):
        target = hooks / name
        chained_backup = None
        if target.is_file():
            current = target.read_text(encoding="utf-8", errors="replace")
            if _MARKER in current:
                installed.append({"hook": name, "status": "already_installed"})
                continue
            backup_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()[:12]
            backup = backup_dir / f"{name}.{digest}.backup"
            if not backup.exists():
                backup.write_bytes(target.read_bytes())
                if os.name != "nt":
                    os.chmod(backup, 0o700)
            chained_backup = backup
        quote = shlex.quote(str(chained_backup)) if chained_backup else None
        script = "\n".join(("#!/bin/sh", _MARKER, "set -u",
                           "if command -v aegis >/dev/null 2>&1; then",
                           f"  aegis hook run --repo {shlex.quote(str(root))} --mode {mode}",
                           "else",
                           f"  {shlex.quote(sys.executable)} -m aegis_core.cli hook run --repo {shlex.quote(str(root))} --mode {mode}",
                           "fi",
                           "status=$?",
                           "if [ \"$status\" -ne 0 ]; then exit \"$status\"; fi",
                           f"exec {quote} \"$@\"" if quote else "exit 0",
                           ""))
        temporary = target.with_suffix(".aegis-tmp")
        temporary.write_text(script, encoding="utf-8", newline="\n")
        os.replace(temporary, target)
        if os.name != "nt":
            os.chmod(target, 0o755)
        installed.append({"hook": name, "status": "installed", "path": str(target),
                          "chained_existing": str(chained_backup) if chained_backup else None})
    marker = root / ".aegis" / "enforce.enabled"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("policy_version=2.0\n", encoding="ascii")
    return {"status": "installed", "repo": str(root), "policy_version": "2.0",
            "enforce_marker": marker.relative_to(root).as_posix(), "hooks": installed}


def remove_hooks(repo_root: str | Path = ".") -> dict:
    root = Path(repo_root).resolve()
    result = _git(root, "rev-parse", "--git-path", "hooks")
    if result.returncode:
        raise ValueError("repo_root is not inside a Git worktree")
    hooks = Path(result.stdout.strip())
    if not hooks.is_absolute():
        hooks = (root / hooks).resolve()
    removed = []
    for name in ("pre-commit", "pre-push"):
        target = hooks / name
        if target.is_file() and _MARKER in target.read_text(encoding="utf-8", errors="replace"):
            target.unlink()
            removed.append(name)
    (root / ".aegis" / "enforce.enabled").unlink(missing_ok=True)
    return {"status": "removed", "removed": removed,
            "existing_hook_backups": sorted(path.name for path in (root / ".aegis" / "hooks").glob("*.backup"))}

