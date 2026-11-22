"""Git isolation and acceptance gates; not an OS security sandbox."""
import threading
import copy
import ast
import ctypes
import hashlib
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import tempfile
import time

from .probes import PROBE_INTEGRITY_VIOLATION, classify_probe_failure, verify_probe_files
from .repo_adapter import load_tool_source_globs

_GIT_LOCK = threading.RLock()
_EXCLUDED = {".git", ".worktrees", "__pycache__", ".pytest_cache", ".venv"}


def run_git(root: Path, *args: str, input: bytes | None = None, check: bool = True):
    # Windows requires Git metadata readers to wait while worktree commands update config.
    with _GIT_LOCK:
        return subprocess.run(["git", *args], cwd=root, input=input, capture_output=True, check=check)


def _reap_process_group(process_group: int) -> None:
    """Reap killed grandchildren on Linux after adopting them as a subreaper."""
    if not sys_platform_linux():
        return
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-process_group, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            time.sleep(0.01)


def sys_platform_linux() -> bool:
    # Kept local to avoid changing process semantics on Windows and macOS.
    import sys
    return sys.platform.startswith("linux")


def _adopt_orphaned_children() -> int | None:
    if not sys_platform_linux():
        return None
    # PR_SET_CHILD_SUBREAPER is process-wide. Save and restore its previous value
    # so ordinary subprocesses later in a long-running test/worker process do not
    # accumulate unrelated orphaned helpers here.
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
            return None
        if not previous.value and libc.prctl(36, 1, 0, 0, 0) != 0:
            return None
        return int(previous.value)
    except (AttributeError, OSError):
        return None


def _restore_child_subreaper(previous: int | None) -> None:
    if previous is None or not sys_platform_linux():
        return
    try:
        ctypes.CDLL(None, use_errno=True).prctl(36, previous, 0, 0, 0)
    except (AttributeError, OSError):
        pass


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


def run_exam_gate(worktree_path: Path, test_cmd: str | list[str] = "pytest", *, timeout: float = 60,
                  probe_kind="unit", probe_script_sha256=None, probe_runtime_sha256=None,
                  environment_overrides: dict[str, str] | None = None) -> dict:
    path = Path(worktree_path).resolve()
    command = shlex.split(test_cmd, posix=os.name != "nt") if isinstance(test_cmd, str) else list(test_cmd)
    if not command:
        raise ValueError("empty exam command")
    if environment_overrides is not None:
        if (type(environment_overrides) is not dict
                or any(type(key) is not str or type(value) is not str
                       for key, value in environment_overrides.items())):
            raise ValueError("environment_overrides must map strings to strings")
    before = tree_hash(path)
    timed_out = False
    killed_process_group = False
    probe_hashes = {} if probe_script_sha256 is None else probe_script_sha256
    integrity_error = False
    raw_hashes = {}
    try:
        raw_hashes = verify_probe_files(path, probe_hashes)
        if probe_runtime_sha256 is not None and raw_hashes != probe_runtime_sha256:
            raise ValueError(f"{PROBE_INTEGRITY_VIOLATION}: probe bytes differ from Fail-Before")
        # A fresh cache prevents same-size/same-second edits from reusing stale pyc.
        with tempfile.TemporaryDirectory(prefix="aegis-pycache-") as cache:
            environment = dict(os.environ, PYTHONPYCACHEPREFIX=cache,
                               PYTHONDONTWRITEBYTECODE="1")
            environment.update(environment_overrides or {})
            options = {}
            if os.name == "nt":
                options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                options["start_new_session"] = True
            process = subprocess.Popen(command, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", errors="replace", env=environment,
                                       **options)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                killed_process_group = True
                subreaper_state = None
                if os.name == "nt":
                    # taskkill /T includes descendants that inherited this process group.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=5, check=False)
                    if process.poll() is None:
                        process.kill()
                else:
                    subreaper_state = _adopt_orphaned_children()
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if process.poll() is None:
                        process.kill()
                stdout, stderr = process.communicate()
                try:
                    if os.name != "nt":
                        _reap_process_group(process.pid)
                finally:
                    _restore_child_subreaper(subreaper_state)
                code = 124
                stderr = (stderr or "") + "\nExam timed out; process group killed (TIMEOUT_KILLED)"
    except ValueError as error:
        code, stdout, stderr = 125, "", str(error)
        integrity_error = True
    except OSError as error:
        code, stdout, stderr = 127, "", str(error)
    after = tree_hash(path)
    if not integrity_error:
        try:
            if verify_probe_files(path, probe_hashes) != raw_hashes:
                raise ValueError(f"{PROBE_INTEGRITY_VIOLATION}: probe bytes changed during execution")
        except ValueError as error:
            stderr += "\n" + str(error)
            integrity_error = True
    failure = (PROBE_INTEGRITY_VIOLATION if integrity_error else
               classify_probe_failure(code, stdout + "\n" + stderr, kind=probe_kind, changed=before != after))
    return {"command": command, "exit_code": code, "stdout": stdout, "stderr": stderr,
            "before_hash": before, "after_hash": after, "changed": before != after,
            "reward": 1.0 if failure is None else -1.0, "passed": failure is None,
            "probe_kind": probe_kind, "probe_script_sha256": probe_hashes, "probe_runtime_sha256": raw_hashes,
            "failure_code": failure,
            "timed_out": timed_out, "timeout_killed": timed_out,
            "termination_reason": "TIMEOUT_KILLED" if timed_out else None,
            "killed_process_group": killed_process_group}


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
        result = run_git(cwd or self.repo_root, *args,
                         input=input.encode("utf-8") if input is not None else None)
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

    def apply_patch(self, ticket_id: str, patch: str) -> dict:
        path, _ = self._owned_path(ticket_id)
        self._gates.pop(ticket_id, None)
        from .patch_applier import PatchApplicationError, apply_patch
        try:
            return apply_patch(path, patch)
        except PatchApplicationError as error:
            raise subprocess.CalledProcessError(
                1, ["git", "apply", "--check", "--whitespace=error", "-"],
                output=b"", stderr=str(error).encode("utf-8", errors="replace")) from error

    def exam(self, ticket_id: str, test_cmd: str | list[str] = "pytest", *, timeout=60) -> dict:
        path, _ = self._owned_path(ticket_id)
        if ticket_id in self._repair_contracts:
            raise ValueError("repair tickets must use the dual-probe gate")
        result = run_exam_gate(path, test_cmd, timeout=timeout)
        self._gates[ticket_id] = result
        return result.copy()

    def repair(self, ticket_id: str, patch: str, *, fail_before_cmd,
               pass_after_cmd, source_dirs: tuple[str, ...], timeout: float = 60, probe_patch: str = "",
               interface_lock=None, require_review: bool = False, tool_source_globs=None,
               probe_kind="unit", probe_script_sha256=None, mutation_rule=None,
               require_assertion_red=False, budget=None) -> dict:
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
        globs = load_tool_source_globs(path) if tool_source_globs is None else tool_source_globs
        prior = detect_source_cheat(path, source_dirs, base=base, tool_source_globs=globs)
        if prior["changed_files"]:
            raise ValueError("repair baseline must be clean")
        if interface_lock is not None and interface_lock.baseline != base:
            raise ValueError("interface lock baseline disagrees with the worktree")
        if probe_patch:
            probe_file = validate_probe_patch(probe_patch)
            self.apply_patch(ticket_id, probe_patch)
            self._git("add", "--", probe_file, cwd=path)
            self._git("-c", "user.name=Aegis", "-c", "user.email=aegis@localhost", "commit", "-m", "Freeze automatic probe", cwd=path)
            base = self._git("rev-parse", "HEAD", cwd=path).strip()
            self._bases[ticket_id] = base
        if mutation_rule:
            from .probe_mutator import validate_target
            validate_target(path / mutation_rule["path"], mutation_rule)
        start = time.monotonic()
        probe = {"probe_kind": probe_kind, "probe_script_sha256": probe_script_sha256}
        from functools import partial
        exam = run_exam_gate if budget is None else partial(budget.run_gate, run_exam_gate)
        before = exam(path, fail_before_cmd, timeout=timeout, **probe)
        baseline_failed = (before["exit_code"] == 1 and not before["changed"]
                           and before["failure_code"] not in {"ENVIRONMENT_UNAVAILABLE", PROBE_INTEGRITY_VIOLATION})
        if require_assertion_red:
            from .probe_synth import is_true_red
            baseline_failed = is_true_red(before)
        result = {"baseline": before, "baseline_failed": baseline_failed,
                  "final": None, "after_passed": False, "passed": False,
                  "cheat_detected": False, "changed_files": [], "diff_lines": 0,
                  "timeout_killed": bool(before.get("timeout_killed")),
                  "before_hash": before["before_hash"], "after_hash": before["after_hash"],
                  "reward": -1.0}
        if baseline_failed:
            if budget is not None:
                budget.check()
            self.apply_patch(ticket_id, patch)
            scope = detect_source_cheat(path, source_dirs, base=base, tool_source_globs=globs)
            result.update(scope)
            if not scope["cheat_detected"] and interface_lock is not None:
                differences = interface_lock.violations(path, scope["changed_files"])
                if differences:
                    result.update(gate_failure="INTERFACE_CONTRACT_VIOLATION",
                                  interface_contract_differences=differences)
            if not scope["cheat_detected"] and "gate_failure" not in result:
                remaining = timeout - (time.monotonic() - start)
                if remaining > 0:
                    after = exam(path, pass_after_cmd, timeout=remaining, **probe,
                                          probe_runtime_sha256=before["probe_runtime_sha256"])
                    if require_assertion_red and after["passed"]:
                        from .probe_synth import is_true_green, static_cases
                        if not is_true_green(after, len(static_cases(mutation_rule))):
                            after.update(passed=False, failure_code="PROBE_CASES_INCOMPLETE", reward=-1.0)
                            result["gate_failure"] = "PROBE_CASES_INCOMPLETE"
                    result["final"] = after
                    result["after_passed"] = after["passed"]
                    result["passed"] = after["passed"]
                    result["timeout_killed"] = bool(after.get("timeout_killed"))
                    result["termination_reason"] = after.get("termination_reason")
                    result["reward"] = 1.0 if after["passed"] else (-1.0 if after.get("timeout_killed") else -0.5)
                    if after["passed"] and mutation_rule:
                        from .probe_mutator import MUTATION_PROBE_FAILED, run_mutations
                        remaining = timeout - (time.monotonic() - start)
                        mutation = run_mutations(path, mutation_rule, timeout=remaining, exam=exam) if remaining > 0 else None
                        result["mutation"] = mutation
                        if mutation is None or not mutation["passed"]:
                            result.update(passed=False, gate_failure=MUTATION_PROBE_FAILED, reward=-1.0,
                                          timeout_killed=mutation is None or mutation["timeout_killed"])
                    if result["passed"] and require_review:
                        from .reviewer import audit_candidate
                        review = audit_candidate(path, base, scope["changed_files"])
                        result["review"] = review
                        if not review["passed"]:
                            result.update(passed=False, gate_failure="ADVERSARIAL_AUDIT_REJECTED", reward=-1.0)
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


def detect_source_cheat(worktree_path: Path, source_dirs: tuple[str, ...], *, base: str = "HEAD",
                        tool_source_globs=None) -> dict:
    """Compare tracked/staged/untracked files against the frozen baseline ref."""
    from .mediator import canonical_path, is_business_source
    from .ast_slicer import is_protected_test_path
    if not source_dirs:
        raise ValueError("trusted source_dirs required")
    directories = []
    for directory in source_dirs:
        directories.append("" if directory == "." else canonical_path(directory).rstrip("/") + "/")
    if base.startswith("-"):
        raise ValueError("invalid baseline")
    path = Path(worktree_path)
    globs = load_tool_source_globs(path) if tool_source_globs is None else tool_source_globs
    def git(*args):
        return run_git(path, *args).stdout.decode("utf-8")
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
        return (is_protected_test_path(name)
                or any(part.lower() in {"config", ".github", "__tests__", ".aegis"} for part in p.parts)
                or p.name.startswith(("test_", "check_", "verify_", "validate_"))
                or p.name in {"conftest.py", "setup.py", "Makefile", ".gitignore", ".gitattributes", ".env"}
                or p.stem.lower() in {"config", "settings", "test", "tests"}
                or p.suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"})
    source_lines = sum(count for name, count in changes.items() if is_business_source(name, tool_source_globs=globs)
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
    if headers not in (['b/tests/test_aegis_probe.py'], ['b/tests/test_aegis_probe.mjs']) or old != ['/dev/null']:
        raise ValueError('probe must only create tests/test_aegis_probe.py')
    code = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    if headers[0].endswith(".mjs"):
        if "assert.strictEqual(" not in code:
            raise ValueError("node probe needs a strict equality assertion")
        return headers[0][2:]
    tree = ast.parse(code)
    assertions = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    if not assertions or any(isinstance(node.test, ast.Constant) for node in assertions):
        raise ValueError('probe needs nonconstant assertions')
    return headers[0][2:]
