"""Git isolation and acceptance gates; not an OS security sandbox."""
import copy
import ast
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

_EXCLUDED = {".git", ".worktrees", "__pycache__", ".pytest_cache", ".venv"}


def tree_hash(root: Path) -> str:
    """Hash relative names, bytes, executable modes and symlink targets."""
    digest = hashlib.sha256()
    root = Path(root).resolve()
    def walk(directory):
        for path in sorted(directory.iterdir(), key=lambda p: p.name):
            if path.name in _EXCLUDED:
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                yield relative, "link", os.readlink(path).encode()
            elif path.is_dir():
                yield relative, "dir", b""
                yield from walk(path)
            elif path.is_file():
                yield relative, "exec" if path.stat().st_mode & 0o111 else "file", path.read_bytes()
    for name, kind, data in walk(root):
        for part in (name.encode(), kind.encode(), data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def run_exam_gate(worktree_path: Path, test_cmd: str | list[str] = "pytest", *, timeout: float = 60) -> dict:
    path = Path(worktree_path).resolve()
    command = shlex.split(test_cmd, posix=os.name != "nt") if isinstance(test_cmd, str) else list(test_cmd)
    if not command:
        raise ValueError("empty exam command")
    before = tree_hash(path)
    try:
        # A fresh cache prevents same-size/same-second edits from reusing stale pyc.
        with tempfile.TemporaryDirectory(prefix="aegis-pycache-") as cache:
            environment = dict(os.environ, PYTHONPYCACHEPREFIX=cache,
                               PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(command, cwd=path, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout,
                                    env=environment)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        code = 124
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = (stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr) + "\nExam timed out"
    except OSError as error:
        code, stdout, stderr = 127, "", str(error)
    after = tree_hash(path)
    return {"command": command, "exit_code": code, "stdout": stdout, "stderr": stderr,
            "before_hash": before, "after_hash": after, "changed": before != after,
            "reward": 1.0 if code == 0 and before == after else -1.0,
            "passed": code == 0 and before == after}


class Sandbox:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        top = self._git("rev-parse", "--show-toplevel").strip()
        if Path(top).resolve() != self.repo_root:
            raise ValueError("repo_root must be the Git toplevel")
        self._gates = {}
        self._owned = set()
        self._bases = {}
        self._repair_contracts = set()

    def _git(self, *args, cwd=None, input=None):
        # Text-mode stdin rewrites LF to CRLF on Windows, corrupting patches
        # against LF fixtures. Preserve exact UTF-8 patch bytes on every OS.
        result = subprocess.run(["git", *args], cwd=cwd or self.repo_root,
                                input=input.encode("utf-8") if input is not None else None,
                                capture_output=True, check=True)
        return result.stdout.decode("utf-8", errors="replace")

    def _paths(self, ticket_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", ticket_id):
            raise ValueError("invalid ticket_id")
        parent = self.repo_root / ".worktrees"
        if parent.is_symlink():
            raise ValueError("worktree parent must not be a symlink")
        return parent / ticket_id, "aegis/" + ticket_id

    def create(self, ticket_id: str, base: str = "HEAD") -> Path:
        path, branch = self._paths(ticket_id)
        if base.startswith("-"):
            raise ValueError("invalid base")
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        self._git("worktree", "add", "-b", branch, str(path), base)
        self._owned.add(ticket_id)
        self._bases[ticket_id] = self._git("rev-parse", "HEAD", cwd=path).strip()
        return path

    def _owned_path(self, ticket_id):
        path, branch = self._paths(ticket_id)
        if ticket_id not in self._owned:
            raise ValueError("worktree not owned by this lifecycle manager")
        return path, branch

    def apply_patch(self, ticket_id: str, patch: str) -> None:
        path, _ = self._owned_path(ticket_id)
        self._gates.pop(ticket_id, None)
        self._git("apply", "--check", "--whitespace=error", "-", cwd=path, input=patch)
        self._git("apply", "--whitespace=error", "-", cwd=path, input=patch)

    def exam(self, ticket_id: str, test_cmd: str | list[str] = "pytest", *, timeout=60) -> dict:
        path, _ = self._owned_path(ticket_id)
        if ticket_id in self._repair_contracts:
            raise ValueError("repair tickets must use the dual-probe gate")
        result = run_exam_gate(path, test_cmd, timeout=timeout)
        self._gates[ticket_id] = result
        return result.copy()

    def repair(self, ticket_id: str, patch: str, *, fail_before_cmd,
               pass_after_cmd, source_dirs: tuple[str, ...], timeout: float = 60, probe_patch: str = "") -> dict:
        import time
        if not 0 < timeout < float("inf"):
            raise ValueError("positive finite repair timeout required")
        path, _ = self._owned_path(ticket_id)
        self._repair_contracts.add(ticket_id)
        self._gates.pop(ticket_id, None)
        base = self._bases[ticket_id]
        # Include committed edits: HEAD itself must still be the frozen baseline.
        if self._git("rev-parse", "HEAD", cwd=path).strip() != base:
            raise ValueError("repair baseline HEAD changed")
        prior = detect_source_cheat(path, source_dirs, base=base)
        if prior["changed_files"]:
            raise ValueError("repair baseline must be clean")
        if probe_patch:
            validate_probe_patch(probe_patch)
            self.apply_patch(ticket_id, probe_patch)
            self._git("add", "--", "tests/test_aegis_probe.py", cwd=path)
            self._git("-c", "user.name=Aegis", "-c", "user.email=aegis@localhost", "commit", "-m", "Freeze automatic probe", cwd=path)
            base = self._git("rev-parse", "HEAD", cwd=path).strip()
            self._bases[ticket_id] = base
        start = time.monotonic()
        before = run_exam_gate(path, fail_before_cmd, timeout=timeout)
        baseline_failed = before["exit_code"] == 1 and not before["changed"]
        result = {"baseline": before, "baseline_failed": baseline_failed,
                  "final": None, "after_passed": False, "passed": False,
                  "cheat_detected": False, "changed_files": [], "diff_lines": 0,
                  "before_hash": before["before_hash"], "after_hash": before["after_hash"],
                  "reward": -1.0}
        if baseline_failed:
            self.apply_patch(ticket_id, patch)
            scope = detect_source_cheat(path, source_dirs, base=base)
            result.update(scope)
            if not scope["cheat_detected"]:
                remaining = timeout - (time.monotonic() - start)
                if remaining > 0:
                    after = run_exam_gate(path, pass_after_cmd, timeout=remaining)
                    result["final"] = after
                    result["after_passed"] = after["passed"]
                    result["passed"] = after["passed"]
                    result["reward"] = 1.0 if after["passed"] else -0.5
            result["after_hash"] = tree_hash(path)
        result["elapsed_sec"] = time.monotonic() - start
        self._gates[ticket_id] = result
        return copy.deepcopy(result)

    def cleanup(self, ticket_id: str) -> None:
        path, branch = self._owned_path(ticket_id)
        self._git("worktree", "remove", "--force", str(path))
        self._git("branch", "-D", branch)
        self._owned.remove(ticket_id)
        self._gates.pop(ticket_id, None)
        self._bases.pop(ticket_id, None)
        self._repair_contracts.discard(ticket_id)

    def merge(self, ticket_id: str, message: str) -> str:
        path, branch = self._owned_path(ticket_id)
        gate = self._gates.get(ticket_id)
        if not gate or not gate["passed"] or tree_hash(path) != gate["after_hash"]:
            raise ValueError("a passing gate for the current tree is required")
        if self._git("status", "--porcelain", "--", ".", ":(exclude).worktrees").strip():
            raise ValueError("parent working tree must be clean")
        parent = self._git("rev-parse", "HEAD").strip()
        base = self._git("merge-base", "HEAD", branch).strip()
        if parent != base:
            raise ValueError("parent advanced; rebase and re-run the gate")
        self._git("add", "--all", cwd=path)
        staged = self._git("diff", "--cached", "--name-only", cwd=path).strip()
        if staged:
            self._git("commit", "-m", message, cwd=path)
        self._git("merge", "--ff-only", branch)
        return self._git("rev-parse", "HEAD").strip()


def detect_source_cheat(worktree_path: Path, source_dirs: tuple[str, ...], *, base: str = "HEAD") -> dict:
    """Compare tracked/staged/untracked files against the frozen baseline ref."""
    from .mediator import canonical_path, is_business_source
    if not source_dirs:
        raise ValueError("trusted source_dirs required")
    directories = []
    for directory in source_dirs:
        directories.append("" if directory == "." else canonical_path(directory).rstrip("/") + "/")
    if base.startswith("-"):
        raise ValueError("invalid baseline")
    path = Path(worktree_path)
    def git(*args):
        return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, check=True).stdout
    entries = git("diff", "--no-renames", "--numstat", "-z", base, "--").split("\0")
    changes = {}
    for entry in entries:
        if entry:
            added, removed, name = entry.split("\t", 2)
            changes[name] = int(added) + int(removed) if added != "-" and removed != "-" else 0
    for name in git("ls-files", "--others", "--exclude-standard", "-z").split("\0"):
        if name and not any(part in _EXCLUDED for part in Path(name).parts):
            file = path / name
            changes[name] = len(file.read_bytes().splitlines()) if file.is_file() and not file.is_symlink() else 0
    def protected(name):
        p = Path(name)
        return (any(part in {"tests", "test", "config", ".github"} for part in p.parts)
                or p.name.startswith(("test_", "check_", "verify_", "validate_"))
                or p.name in {"conftest.py", "setup.py", "Makefile", ".gitignore", ".gitattributes", ".env"}
                or p.stem.lower() in {"config", "settings", "test", "tests"}
                or p.suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"})
    source_lines = sum(count for name, count in changes.items() if is_business_source(name)
                       and any(not prefix or name.startswith(prefix) for prefix in directories))
    protected_files = sorted(name for name in changes if protected(name))
    return {"changed_files": sorted(changes), "diff_lines": sum(changes.values()),
            "source_changed_lines": source_lines, "protected_changed": protected_files,
            "cheat_detected": source_lines == 0 or bool(protected_files)}


class ProbeNotFailingError(ValueError):
    pass


def validate_probe_patch(patch):
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError('missing probe patch')
    headers = [line[4:] for line in patch.splitlines() if line.startswith('+++ ')]
    old = [line[4:] for line in patch.splitlines() if line.startswith('--- ')]
    if headers != ['b/tests/test_aegis_probe.py'] or old != ['/dev/null']:
        raise ValueError('probe must only create tests/test_aegis_probe.py')
    code = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    tree = ast.parse(code)
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    if not assertions or any(isinstance(node.test, ast.Constant) for node in assertions):
        raise ValueError('probe needs nonconstant assertions')
