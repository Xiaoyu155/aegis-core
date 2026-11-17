"""Git isolation and acceptance gates; not an OS security sandbox."""
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

    def _git(self, *args, cwd=None, input=None):
        return subprocess.run(["git", *args], cwd=cwd or self.repo_root,
                              input=input, text=True, capture_output=True, check=True).stdout

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
        result = run_exam_gate(path, test_cmd, timeout=timeout)
        self._gates[ticket_id] = result
        return result.copy()

    def cleanup(self, ticket_id: str) -> None:
        path, branch = self._owned_path(ticket_id)
        self._git("worktree", "remove", "--force", str(path))
        self._git("branch", "-D", branch)
        self._owned.remove(ticket_id)
        self._gates.pop(ticket_id, None)

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
