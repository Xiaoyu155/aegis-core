"""Byte-stable Git snapshot replay and counterfactual patch attribution."""
from __future__ import annotations

from contextlib import contextmanager
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReplayIntegrityError(ValueError):
    """Stored snapshot bytes do not match their frozen integrity digests."""


class TimeTravelReplay:
    """Persist replay evidence beside the WAL ledger and rebuild it in git worktrees."""

    def __init__(self, repo_root: str | Path, database_path: str | Path | None = None):
        self.root = Path(repo_root).resolve()
        self.database_path = Path(database_path).resolve() if database_path else Path(
            self._git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "aegis" / "ledger.sqlite"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS replay_snapshots (
                    ticket_id TEXT PRIMARY KEY,
                    commit_oid TEXT NOT NULL,
                    archive BLOB NOT NULL,
                    archive_sha256 TEXT NOT NULL,
                    probe BLOB NOT NULL,
                    probe_path TEXT,
                    probe_sha256 TEXT NOT NULL,
                    probe_command TEXT NOT NULL,
                    mutation_seeds TEXT NOT NULL,
                    world_context TEXT NOT NULL,
                    candidate_patch BLOB,
                    candidate_patch_sha256 TEXT,
                    observed_result TEXT,
                    created_at REAL NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS replay_snapshots_no_update BEFORE UPDATE ON replay_snapshots
                BEGIN SELECT RAISE(ABORT, 'replay snapshots are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS replay_snapshots_no_delete BEFORE DELETE ON replay_snapshots
                BEGIN SELECT RAISE(ABORT, 'replay snapshots are immutable'); END;
            """)
            columns = {item[1] for item in db.execute("PRAGMA table_info(replay_snapshots)")}
            if "candidate_patch" not in columns:
                db.execute("ALTER TABLE replay_snapshots ADD COLUMN candidate_patch BLOB")
            if "candidate_patch_sha256" not in columns:
                db.execute("ALTER TABLE replay_snapshots ADD COLUMN candidate_patch_sha256 TEXT")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.database_path), timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            yield db
        finally:
            db.close()

    def _git(self, *args, check=True):
        result = subprocess.run(["git", *map(str, args)], cwd=self.root,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if check and result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", "replace").strip() or "git command failed")
        return result

    @staticmethod
    def _command(command):
        if not isinstance(command, (list, tuple)) or not command or any(type(x) is not str or not x for x in command):
            raise ValueError("probe_command must be a nonempty argument array")
        return list(command)

    @staticmethod
    def _relative(path):
        if path is None:
            return None
        value = PurePosixPath(str(path).replace("\\", "/"))
        if value.is_absolute() or not value.parts or ".." in value.parts or "." in value.parts:
            raise ValueError("probe_path must be a safe repository-relative path")
        return value.as_posix()

    def capture(self, ticket_id: str, *, commit: str | None = None, probe: bytes | str = b"",
                probe_path: str | None = None, probe_command, mutation_seeds=(), world_context=None,
                candidate_patch: bytes | str = b"", observed_result: dict | None = None) -> dict:
        """Freeze commit tree, probe bytes, deterministic seeds, model state and command."""
        if type(ticket_id) is not str or not ticket_id.strip() or len(ticket_id) > 200:
            raise ValueError("ticket_id must be nonempty and at most 200 characters")
        command = self._command(probe_command)
        path = self._relative(probe_path)
        if type(probe) is str: probe = probe.encode("utf-8")
        if type(probe) is not bytes:
            raise TypeError("probe must be bytes or text")
        if type(candidate_patch) is str: candidate_patch = candidate_patch.encode("utf-8")
        if type(candidate_patch) is not bytes:
            raise TypeError("candidate_patch must be bytes or text")
        if not isinstance(mutation_seeds, (list, tuple)) or any(type(seed) not in (int, str) for seed in mutation_seeds):
            raise ValueError("mutation_seeds must contain only integers or strings")
        context_text = _canonical(world_context or {})
        seeds_text = _canonical(list(mutation_seeds))
        commit_oid = self._git("rev-parse", "--verify", (commit or "HEAD") + "^{commit}").stdout.decode().strip()
        archive = self._git("archive", "--format=tar", commit_oid).stdout
        record = {"ticket_id": ticket_id, "commit_oid": commit_oid, "archive_sha256": _sha(archive),
                  "probe_sha256": _sha(probe), "probe_path": path, "probe_command": command,
                  "candidate_patch_sha256": _sha(candidate_patch) if candidate_patch else None,
                  "mutation_seeds": list(mutation_seeds), "world_context": json.loads(context_text),
                  "observed_result": observed_result}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                old = db.execute("SELECT * FROM replay_snapshots WHERE ticket_id=?", (ticket_id,)).fetchone()
                if old:
                    same = (old["commit_oid"] == commit_oid and old["archive_sha256"] == record["archive_sha256"]
                            and old["probe_sha256"] == record["probe_sha256"]
                            and old["candidate_patch_sha256"] == record["candidate_patch_sha256"]
                            and bytes(old["candidate_patch"] or b"") == candidate_patch
                            and old["probe_path"] == path and old["probe_command"] == _canonical(command)
                            and old["mutation_seeds"] == seeds_text and old["world_context"] == context_text
                            and old["observed_result"] == (_canonical(observed_result) if observed_result is not None else None))
                    if not same: raise ValueError("ticket already has a different immutable replay snapshot")
                else:
                    db.execute("""INSERT INTO replay_snapshots
                        (ticket_id,commit_oid,archive,archive_sha256,probe,probe_path,probe_sha256,probe_command,
                         mutation_seeds,world_context,candidate_patch,candidate_patch_sha256,observed_result,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                               (ticket_id, commit_oid, sqlite3.Binary(archive), record["archive_sha256"],
                                sqlite3.Binary(probe), path, record["probe_sha256"], _canonical(command),
                                seeds_text, context_text, sqlite3.Binary(candidate_patch), record["candidate_patch_sha256"],
                                _canonical(observed_result) if observed_result is not None else None, time.time()))
                db.commit()
            except BaseException:
                db.rollback(); raise
        return {key: value for key, value in record.items() if key not in {"probe_command", "mutation_seeds", "world_context"}} | {
            "probe_command": command, "mutation_seeds": list(mutation_seeds), "world_context": json.loads(context_text)}

    def snapshot(self, ticket_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM replay_snapshots WHERE ticket_id=?", (ticket_id,)).fetchone()
        if row is None: raise KeyError(ticket_id)
        archive, probe = bytes(row["archive"]), bytes(row["probe"])
        candidate_patch = bytes(row["candidate_patch"] or b"")
        if (_sha(archive) != row["archive_sha256"] or _sha(probe) != row["probe_sha256"]
                or (candidate_patch and _sha(candidate_patch) != row["candidate_patch_sha256"])):
            raise ReplayIntegrityError("replay snapshot digest mismatch")
        return {"ticket_id": row["ticket_id"], "commit_oid": row["commit_oid"], "archive": archive,
                "archive_sha256": row["archive_sha256"], "probe": probe, "probe_path": row["probe_path"],
                "probe_sha256": row["probe_sha256"], "probe_command": json.loads(row["probe_command"]),
                "candidate_patch": candidate_patch, "candidate_patch_sha256": row["candidate_patch_sha256"],
                "mutation_seeds": json.loads(row["mutation_seeds"]),
                "world_context": json.loads(row["world_context"]),
                "observed_result": json.loads(row["observed_result"]) if row["observed_result"] else None}

    def restore(self, ticket_id: str, destination: str | Path | None = None) -> dict:
        """Rebuild the exact Git tree as a detached worktree and materialize the frozen probe."""
        record = self.snapshot(ticket_id)
        owned = destination is None
        target = Path(tempfile.mkdtemp(prefix="aegis-replay-") if owned else destination).resolve()
        if owned: target.rmdir()
        if target.exists(): raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "--detach", str(target), record["commit_oid"])
        if record["candidate_patch"]:
            applied = subprocess.run(["git", "apply", "--check", "-"], cwd=target,
                                     input=record["candidate_patch"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if applied.returncode:
                self.cleanup(target)
                raise ReplayIntegrityError("frozen candidate patch no longer applies to its recorded base")
            subprocess.run(["git", "apply", "-"], cwd=target, input=record["candidate_patch"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        if record["probe_path"]:
            probe = (target / record["probe_path"]).resolve()
            if not probe.is_relative_to(target):
                self._git("worktree", "remove", "--force", str(target)); raise ValueError("probe escaped worktree")
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(record["probe"])
        actual = self._git("archive", "--format=tar", record["commit_oid"]).stdout
        if _sha(actual) != record["archive_sha256"]:
            self.cleanup(target); raise ReplayIntegrityError("Git archive changed during restoration")
        return {"ticket_id": ticket_id, "worktree": str(target), "commit_oid": record["commit_oid"],
                "archive_sha256": record["archive_sha256"], "probe_sha256": record["probe_sha256"],
                "candidate_patch_sha256": record["candidate_patch_sha256"],
                "probe_path": record["probe_path"], "mutation_seeds": record["mutation_seeds"],
                "world_context": record["world_context"], "owned": owned}

    def cleanup(self, worktree: str | Path):
        target = Path(worktree).resolve()
        if not target.is_relative_to(self.root.parent) and not target.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise ValueError("refusing to remove a replay worktree outside the repository's parent or temp directory")
        self._git("worktree", "remove", "--force", str(target), check=False)
        shutil.rmtree(target, ignore_errors=True)
        self._git("worktree", "prune", check=False)

    def replay(self, ticket_id: str, *, timeout: float = 60, destination: str | Path | None = None) -> dict:
        if type(timeout) not in (int, float) or timeout <= 0:
            raise ValueError("timeout must be positive")
        record = self.snapshot(ticket_id)
        restored = self.restore(ticket_id, destination)
        command = record["probe_command"]
        environment = os.environ.copy()
        environment.update({"PYTHONHASHSEED": "0", "AEGIS_MUTATION_SEEDS": _canonical(record["mutation_seeds"])})
        try:
            result = subprocess.run(command, cwd=restored["worktree"], env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=timeout, check=False)
            outcome = {"exit_code": result.returncode, "stdout": result.stdout.decode("utf-8", "replace"),
                       "stderr": result.stderr.decode("utf-8", "replace"),
                       "stdout_sha256": _sha(result.stdout), "stderr_sha256": _sha(result.stderr)}
            outcome["result_sha256"] = _sha(_canonical({"exit_code": outcome["exit_code"],
                                                       "stdout_sha256": outcome["stdout_sha256"],
                                                       "stderr_sha256": outcome["stderr_sha256"]}).encode())
            expected = record["observed_result"]
            outcome["matches_observed"] = None if expected is None else outcome["result_sha256"] == expected.get("result_sha256")
            return {"schema_version": "aegis.replay.v1", "status": "replayed", "ticket_id": ticket_id,
                    "commit_oid": restored["commit_oid"], "archive_sha256": restored["archive_sha256"],
                    "probe_sha256": restored["probe_sha256"],
                    "candidate_patch_sha256": restored["candidate_patch_sha256"],
                    "mutation_seeds": restored["mutation_seeds"],
                    "world_context": restored["world_context"], "worktree": restored["worktree"],
                    "outcome": outcome}
        except subprocess.TimeoutExpired as error:
            return {"schema_version": "aegis.replay.v1", "status": "timeout", "ticket_id": ticket_id,
                    "worktree": restored["worktree"], "timeout_sec": timeout,
                    "stdout_sha256": _sha(error.stdout or b""), "stderr_sha256": _sha(error.stderr or b"")}
        finally:
            if restored["owned"]: self.cleanup(restored["worktree"])

    def counterfactual_blame(self, base_commit: str, candidate_commit: str, probe_command,
                             *, timeout: float = 60) -> dict:
        """Use file-level delta debugging then report changed line ranges that retain failure."""
        command = self._command(probe_command)
        base = self._git("rev-parse", "--verify", base_commit + "^{commit}").stdout.decode().strip()
        candidate = self._git("rev-parse", "--verify", candidate_commit + "^{commit}").stdout.decode().strip()
        changed = self._git("diff", "--name-only", "-z", "--diff-filter=ACDMRTUXB", base, candidate).stdout.decode().split("\0")
        files = sorted(item for item in changed if item)
        with tempfile.TemporaryDirectory(prefix="aegis-blame-") as temp:
            base_dir, candidate_dir = Path(temp) / "base", Path(temp) / "candidate"
            self._git("worktree", "add", "--detach", str(base_dir), base)
            self._git("worktree", "add", "--detach", str(candidate_dir), candidate)
            try:
                base_result = _run(command, base_dir, timeout)
                candidate_result = _run(command, candidate_dir, timeout)
                if base_result["exit_code"] != 0 or candidate_result["exit_code"] == 0:
                    return {"schema_version": "aegis.counterfactual_blame.v1", "status": "not_candidate_regression",
                            "base_commit": base, "candidate_commit": candidate,
                            "base_exit_code": base_result["exit_code"], "candidate_exit_code": candidate_result["exit_code"],
                            "files": files, "culprits": []}
                active = files[:]

                def still_fails(included):
                    include = set(included)
                    for filename in files:
                        _write_commit_file(self.root, candidate_dir, candidate, filename) if filename in include else \
                            _write_commit_file(self.root, candidate_dir, base, filename)
                    return _run(command, candidate_dir, timeout)["exit_code"] != 0

                granularity = 2
                while len(active) > 1:
                    chunks = _chunks(active, granularity)
                    reduced = False
                    for chunk in chunks:
                        complement = [item for item in active if item not in chunk]
                        if complement and still_fails(complement):
                            active, granularity = complement, max(2, granularity - 1)
                            reduced = True; break
                        if still_fails(chunk):
                            active, granularity = chunk, max(2, granularity - 1)
                            reduced = True; break
                    if reduced: continue
                    if granularity >= len(active): break
                    granularity = min(len(active), granularity * 2)
                culprits = []
                for filename in active:
                    before = _show(self.root, base, filename)
                    after = _show(self.root, candidate, filename)
                    line_ranges = _changed_line_ranges(before, after)
                    culprits.append({"file": filename, "line_ranges": line_ranges,
                                     "before_sha256": _sha(before) if before is not None else None,
                                     "after_sha256": _sha(after) if after is not None else None})
                return {"schema_version": "aegis.counterfactual_blame.v1", "status": "attributed",
                        "base_commit": base, "candidate_commit": candidate,
                        "base_exit_code": base_result["exit_code"], "candidate_exit_code": candidate_result["exit_code"],
                        "changed_file_count": len(files), "culprits": culprits,
                        "method": "file-level delta debugging over real detached git worktrees"}
            finally:
                for worktree in (base_dir, candidate_dir):
                    self._git("worktree", "remove", "--force", str(worktree), check=False)
                self._git("worktree", "prune", check=False)


def _run(command, cwd, timeout):
    try:
        with tempfile.TemporaryDirectory(prefix="aegis-probe-pycache-") as cache:
            result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                                         "PYTHONPYCACHEPREFIX": cache}, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"exit_code": 124}
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _show(root, commit, filename):
    result = subprocess.run(["git", "show", f"{commit}:{filename}"], cwd=root,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.stdout if result.returncode == 0 else None


def _write_commit_file(root, worktree, commit, filename):
    content = _show(root, commit, filename)
    path = worktree / filename
    if content is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _chunks(values, count):
    size = max(1, (len(values) + count - 1) // count)
    return [values[index:index + size] for index in range(0, len(values), size)]


def _changed_line_ranges(before: bytes | None, after: bytes | None):
    old = (before or b"").decode("utf-8", "replace").splitlines()
    new = (after or b"").decode("utf-8", "replace").splitlines()
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            result.append({"kind": tag, "base_start": i1 + 1, "base_end": i2,
                           "candidate_start": j1 + 1, "candidate_end": j2,
                           "removed": old[i1:i2], "added": new[j1:j2]})
    return result
