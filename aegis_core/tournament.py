"""Parallel, worktree-isolated candidate evaluation and GRPO pair export."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Iterable

from .ast_slicer import JS_SUFFIXES, is_protected_test_path
from .reviewer import audit_candidate
from .sandbox import run_git

_PAIR_LOCK = threading.Lock()
_BRANCH_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_SOURCE_SUFFIXES = JS_SUFFIXES | {".py", ".html", ".htm", ".css"}
_WEIGHTS = {"static": 0.25, "dynamic": 0.30, "mutation": 0.25,
            "audit": 0.15, "complexity": 0.03, "cost": 0.02}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    strategy: str
    patch: str
    token_cost: int | None = None
    metadata: dict | None = None


def _commands(value, name: str, *, required: bool) -> list[list[str]]:
    if value is None:
        if required:
            raise ValueError(f"{name} must contain at least one real probe command")
        return []
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a sequence of argument arrays")
    result = []
    for command in value:
        if isinstance(command, str) or not isinstance(command, Iterable):
            raise ValueError(f"{name} commands must be argument arrays")
        args = list(command)
        if not args or any(type(arg) is not str or not arg for arg in args):
            raise ValueError(f"{name} commands must contain nonempty string arguments")
        result.append([sys.executable if arg == "{python}" else arg for arg in args])
    if required and not result:
        raise ValueError(f"{name} must contain at least one real probe command")
    return result


class CandidateTournament:
    """Run each patch in a detached Git worktree and score independent gate evidence."""

    def __init__(self, repo_root: str | Path, *, dynamic_commands,
                 mutation_commands, static_commands=None, timeout_sec: float = 60,
                 max_workers: int = 4, dataset_path: str | Path | None = None,
                 allowed_paths: Iterable[str] | None = None, probe_patch: str = ""):
        self.root = Path(repo_root).resolve()
        top = run_git(self.root, "rev-parse", "--show-toplevel").stdout.decode().strip()
        if Path(top).resolve() != self.root:
            raise ValueError("repo_root must be a Git toplevel")
        if type(timeout_sec) not in (int, float) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self.dataset_path = (Path(dataset_path).resolve() if dataset_path is not None
                             else self.root / "datasets" / "grpo_tournament_pairs.jsonl")
        status = run_git(self.root, "status", "--porcelain", "--untracked-files=all").stdout.decode().splitlines()
        dataset_relative = (self.dataset_path.relative_to(self.root).as_posix()
                            if self.dataset_path.is_relative_to(self.root) else None)
        dirty = [line for line in status if line[3:].strip().strip('"') != dataset_relative]
        if dirty:
            raise ValueError("tournament baseline working tree must be clean")
        self.base = run_git(self.root, "rev-parse", "HEAD").stdout.decode().strip()
        self.dynamic_commands = _commands(dynamic_commands, "dynamic_commands", required=True)
        self.mutation_commands = _commands(mutation_commands, "mutation_commands", required=True)
        self.static_commands = _commands(static_commands, "static_commands", required=False)
        self.timeout_sec = float(timeout_sec)
        self.max_workers = max_workers
        self.allowed_paths = None if allowed_paths is None else {Path(item).as_posix() for item in allowed_paths}
        if type(probe_patch) is not str:
            raise ValueError("probe_patch must be text")
        self.probe_patch = probe_patch

    def run(self, candidates: Iterable[Candidate | dict], *, contract: str = "") -> dict:
        candidates = [self._candidate(item) for item in candidates]
        if not candidates:
            raise ValueError("at least one candidate is required")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        output: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(candidates)),
                                thread_name_prefix="aegis-tournament") as pool:
            futures = {pool.submit(self._evaluate, candidate): candidate.candidate_id for candidate in candidates}
            for future in as_completed(futures):
                candidate_id = futures[future]
                try:
                    output[candidate_id] = future.result()
                except Exception as error:  # Each candidate failure is isolated and remains reviewable.
                    output[candidate_id] = {"candidate_id": candidate_id, "accepted": False,
                                            "score": 0.0, "error": type(error).__name__,
                                            "message": str(error)[:500], "static_pass_rate": 0.0,
                                            "dynamic_pass_rate": 0.0,
                                            "mutation_probe_survival_rate": 0.0,
                                            "audit_passed": False, "complexity_delta": 0,
                                            "token_cost": 0, "elapsed_sec": 0.0,
                                            "objectives": {}}
        evaluations = [output[candidate_id] for candidate_id in sorted(output)]
        eligible = [item for item in evaluations if item["accepted"]]
        pareto_ids = _pareto_front(eligible)
        for item in evaluations:
            item["pareto_optimal"] = item["candidate_id"] in pareto_ids
        winner = max((item for item in eligible if item["pareto_optimal"]),
                     key=lambda item: (item["score"], item["candidate_id"]), default=None)
        pairs = self._export_pairs(candidates, output, winner, contract) if winner else []
        return {"schema_version": "aegis.tournament.v1", "base_commit": self.base,
                "candidate_count": len(candidates), "accepted_count": len(eligible),
                "winner": winner["candidate_id"] if winner else None,
                "pareto_front": sorted(pareto_ids), "evaluations": evaluations,
                "grpo_pairs_exported": len(pairs), "grpo_dataset": str(self.dataset_path) if pairs else None,
                "isolation": {"git_worktree_per_candidate": True, "max_parallel": min(self.max_workers, len(candidates)),
                              "main_worktree_changed": False}}

    @staticmethod
    def _candidate(value) -> Candidate:
        if isinstance(value, Candidate):
            candidate = value
        elif isinstance(value, dict):
            candidate = Candidate(candidate_id=value.get("candidate_id", ""), strategy=value.get("strategy", ""),
                                  patch=value.get("patch", ""), token_cost=value.get("token_cost"),
                                  metadata=value.get("metadata"))
        else:
            raise ValueError("candidate must be Candidate or a JSON-like object")
        if (type(candidate.candidate_id) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", candidate.candidate_id)
                or type(candidate.strategy) is not str or not candidate.strategy.strip()
                or type(candidate.patch) is not str or not candidate.patch.strip()):
            raise ValueError("candidate_id, strategy and patch are required")
        if candidate.token_cost is not None and (type(candidate.token_cost) is not int or candidate.token_cost < 0):
            raise ValueError("token_cost must be a nonnegative integer")
        if candidate.metadata is not None:
            json.dumps(candidate.metadata, allow_nan=False)
        return candidate

    def _evaluate(self, candidate: Candidate) -> dict:
        started = time.monotonic()
        parent = self.root / ".worktrees"
        if parent.is_symlink():
            raise ValueError("worktree parent must not be a symlink")
        made_parent = not parent.exists()
        parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="tournament-", dir=parent))
        suffix = hashlib.sha256((candidate.candidate_id + candidate.patch).encode()).hexdigest()[:8]
        safe_id = _BRANCH_SAFE.sub("-", candidate.candidate_id).strip(".-")[:48] or "candidate"
        worktree = scratch / (safe_id + "-" + suffix)
        registered = False
        try:
            run_git(self.root, "worktree", "add", "--detach", str(worktree), self.base)
            registered = True
            candidate_base = self.base
            if self.probe_patch:
                from .sandbox import validate_probe_patch
                probe_file = validate_probe_patch(self.probe_patch)
                staged = run_git(worktree, "apply", "--check", input=self.probe_patch.encode("utf-8"), check=False)
                if staged.returncode:
                    raise ValueError("frozen probe does not apply to the candidate baseline")
                run_git(worktree, "apply", input=self.probe_patch.encode("utf-8"))
                run_git(worktree, "add", "--", probe_file)
                run_git(worktree, "-c", "user.name=Aegis", "-c", "user.email=aegis@localhost",
                        "commit", "-m", "Freeze tournament acceptance probe")
                candidate_base = run_git(worktree, "rev-parse", "HEAD").stdout.decode().strip()
            applied = run_git(worktree, "apply", "--check", input=candidate.patch.encode("utf-8"), check=False)
            if applied.returncode:
                raise ValueError("patch does not apply to the frozen base")
            applied = run_git(worktree, "apply", input=candidate.patch.encode("utf-8"), check=False)
            if applied.returncode:
                raise ValueError("patch application failed")
            changed = [line for line in run_git(worktree, "diff", "--name-only", candidate_base).stdout.decode().splitlines() if line]
            violations = self._scope_violations(changed)
            if not changed or violations:
                raise ValueError("empty or out-of-scope candidate patch: " + ", ".join(violations or ["no changes"]))
            static = self._static_gates(worktree, changed)
            dynamic = self._run_commands(worktree, self.dynamic_commands)
            mutations = self._run_commands(worktree, self.mutation_commands)
            audit = audit_candidate(worktree, self.base, changed)
            complexity_delta = _complexity_delta(self.root, worktree, self.base, changed)
            elapsed = time.monotonic() - started
            token_cost = (candidate.token_cost if candidate.token_cost is not None
                          else len(re.findall(r"\w+|[^\w\s]", candidate.patch, re.UNICODE)))
            static_rate = _rate(static)
            dynamic_rate = _rate(dynamic)
            mutation_rate = _rate(mutations)
            audit_score = 1.0 if audit["passed"] else max(0.0, 1.0 - len(audit["findings"]) / 4)
            complexity_score = 1 / (1 + max(0, complexity_delta) / 10)
            cost_score = 1 / (1 + token_cost / 4000 + elapsed / max(1.0, self.timeout_sec))
            score = sum((_WEIGHTS["static"] * static_rate, _WEIGHTS["dynamic"] * dynamic_rate,
                         _WEIGHTS["mutation"] * mutation_rate, _WEIGHTS["audit"] * audit_score,
                         _WEIGHTS["complexity"] * complexity_score, _WEIGHTS["cost"] * cost_score))
            accepted = (static_rate == 1 and dynamic_rate == 1 and mutation_rate == 1 and audit["passed"])
            objectives = {"static": static_rate, "dynamic": dynamic_rate, "mutation": mutation_rate,
                          "audit": audit_score, "complexity": complexity_score, "cost": cost_score}
            return {"candidate_id": candidate.candidate_id, "strategy": candidate.strategy,
                    "accepted": accepted, "score": round(score, 8), "changed_files": changed,
                    "static": static, "dynamic": dynamic, "mutation": mutations, "audit": audit,
                    "static_pass_rate": static_rate, "dynamic_pass_rate": dynamic_rate,
                    "mutation_probe_survival_rate": mutation_rate, "audit_passed": audit["passed"],
                    "complexity_delta": complexity_delta, "token_cost": token_cost,
                    "elapsed_sec": round(elapsed, 6), "objectives": objectives,
                    "patch_sha256": hashlib.sha256(candidate.patch.encode()).hexdigest(),
                    "metadata": candidate.metadata or {}}
        finally:
            if registered:
                run_git(self.root, "worktree", "remove", "--force", str(worktree), check=False)
            shutil.rmtree(scratch, ignore_errors=True)
            if made_parent:
                try:
                    parent.rmdir()
                except OSError:
                    pass

    def _scope_violations(self, changed: list[str]) -> list[str]:
        violations = []
        for name in changed:
            suffix = Path(name).suffix.lower()
            if (is_protected_test_path(name) or suffix not in _SOURCE_SUFFIXES
                    or (self.allowed_paths is not None and name not in self.allowed_paths)):
                violations.append(name)
        return violations

    def _static_gates(self, worktree: Path, changed: list[str]) -> list[dict]:
        commands = [["git", "diff", "--check"]]
        python_files = [name for name in changed if name.endswith(".py")]
        if python_files:
            commands.append([sys.executable, "-B", "-m", "py_compile", *python_files])
        commands.extend(self.static_commands)
        return self._run_commands(worktree, commands)

    def _run_commands(self, worktree: Path, commands: list[list[str]]) -> list[dict]:
        receipts = []
        for command in commands:
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix="aegis-tournament-cache-") as cache:
                    environment = dict(os.environ, PYTHONPYCACHEPREFIX=cache, PYTHONDONTWRITEBYTECODE="1")
                    result = subprocess.run(command, cwd=worktree, capture_output=True, text=True,
                                            encoding="utf-8", errors="replace", timeout=self.timeout_sec,
                                            env=environment, check=False)
                code, stdout, stderr, timed_out = result.returncode, result.stdout, result.stderr, False
            except subprocess.TimeoutExpired as error:
                code, timed_out = 124, True
                stdout = (error.stdout or b"").decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
                stderr = (error.stderr or b"").decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
                stderr += "\nTOURNAMENT_TIMEOUT"
            except OSError as error:
                code, stdout, stderr, timed_out = 127, "", type(error).__name__, False
            receipts.append({"command": command, "exit_code": code, "passed": code == 0 and not timed_out,
                             "timed_out": timed_out, "stdout": stdout[-4000:], "stderr": stderr[-4000:],
                             "elapsed_sec": round(time.monotonic() - started, 6)})
        return receipts

    def _export_pairs(self, candidates, results, winner, contract: str) -> list[dict]:
        if winner is None:
            return []
        chosen = next(item for item in candidates if item.candidate_id == winner["candidate_id"])
        rejected = [candidate for candidate in candidates if candidate.candidate_id != chosen.candidate_id]
        if not rejected:
            return []
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for candidate in rejected:
            loser_result = results[candidate.candidate_id]
            row = {"schema_version": "aegis.grpo_pair.v1", "source": "candidate_tournament",
                   "prompt": contract[:20000], "chosen_candidate": chosen.candidate_id,
                   "rejected_candidate": candidate.candidate_id, "chosen_patch": chosen.patch,
                   "rejected_patch": candidate.patch, "chosen_score": winner["score"],
                   "rejected_score": loser_result["score"], "chosen_sha256": winner["patch_sha256"],
                   "rejected_sha256": loser_result.get("patch_sha256"),
                   "rejection_evidence": {"accepted": loser_result["accepted"],
                                          "static_pass_rate": loser_result["static_pass_rate"],
                                          "dynamic_pass_rate": loser_result["dynamic_pass_rate"],
                                          "mutation_probe_survival_rate": loser_result["mutation_probe_survival_rate"],
                                          "audit_passed": loser_result["audit_passed"],
                                          "audit_findings": loser_result.get("audit", {}).get("findings", [])}}
            stable_identity = {key: row[key] for key in (
                "schema_version", "source", "prompt", "chosen_candidate", "rejected_candidate",
                "chosen_sha256", "rejected_sha256")}
            row["pair_sha256"] = hashlib.sha256(_canonical_json(stable_identity).encode()).hexdigest()
            rows.append(row)
        if os.name == "nt":
            import msvcrt
            with _PAIR_LOCK, self.dataset_path.open("a+", encoding="utf-8") as stream:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    existing = _read_jsonl(stream)
                    known = {row.get("pair_sha256") for row in existing}
                    stream.seek(0, os.SEEK_END)
                    for row in rows:
                        if row["pair_sha256"] not in known:
                            stream.write(_canonical_json(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            with _PAIR_LOCK, self.dataset_path.open("a+", encoding="utf-8") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                try:
                    existing = _read_jsonl(stream)
                    known = {row.get("pair_sha256") for row in existing}
                    stream.seek(0, os.SEEK_END)
                    for row in rows:
                        if row["pair_sha256"] not in known:
                            stream.write(_canonical_json(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return rows


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_jsonl(stream) -> list[dict]:
    stream.seek(0)
    records = []
    for line in stream:
        if line.strip():
            records.append(json.loads(line))
    return records


def _rate(receipts: list[dict]) -> float:
    return (sum(bool(item["passed"]) for item in receipts) / len(receipts)) if receipts else 0.0


def _complexity_delta(root: Path, worktree: Path, base: str, files: list[str]) -> int:
    def points(source: str, suffix: str) -> int:
        if suffix == ".py":
            try:
                tree = ast.parse(source)
            except SyntaxError:
                return 100000
            nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
                     ast.BoolOp, ast.IfExp, ast.comprehension)
            return sum(isinstance(node, nodes) for node in ast.walk(tree))
        # A bounded lexical estimate for JS/TS branch and boolean operators.
        return len(re.findall(r"\b(?:if|for|while|catch|case)\b|&&|\|\||\?", source))
    delta = 0
    for name in files:
        old = run_git(root, "show", f"{base}:{name}", check=False)
        before = old.stdout.decode("utf-8", errors="replace") if old.returncode == 0 else ""
        target = worktree / name
        after = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        delta += points(after, Path(name).suffix.lower()) - points(before, Path(name).suffix.lower())
    return delta


def _pareto_front(candidates: list[dict]) -> set[str]:
    front = set()
    for item in candidates:
        objectives = item["objectives"]
        dominated = False
        for other in candidates:
            if other is item:
                continue
            theirs = other["objectives"]
            if (all(theirs[key] >= objectives[key] for key in objectives)
                    and any(theirs[key] > objectives[key] for key in objectives)):
                dominated = True
                break
        if not dominated:
            front.add(item["candidate_id"])
    return front
