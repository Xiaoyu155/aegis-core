"""Real Git-worktree candidate races, scoring and replay-pair persistence."""
import difflib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aegis_core.tournament import Candidate, CandidateTournament


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _repo(path):
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Tournament Tests")
    _git(path, "config", "user.email", "tournament@example.invalid")
    (path / "app.py").write_text("def compute(value):\n    return 0\n", encoding="utf-8")
    _git(path, "add", "app.py")
    _git(path, "commit", "-qm", "base")
    return path


def _patch(root, old, new, filename="app.py"):
    source = (root / filename).read_text(encoding="utf-8")
    before = source.splitlines(keepends=True)
    after = source.replace(old, new).splitlines(keepends=True)
    body = difflib.unified_diff(before, after, fromfile="a/" + filename, tofile="b/" + filename)
    return f"diff --git a/{filename} b/{filename}\n" + "".join(body)


def _commands():
    dynamic = [[sys.executable, "-c", "from app import compute; assert compute(2) == 3"]]
    mutation = [[sys.executable, "-c", "from app import compute; assert compute(-1) == 0"]]
    return dynamic, mutation


def _engine(root, tmp_path, **kwargs):
    dynamic, mutation = _commands()
    options = {"timeout_sec": 3, "max_workers": 4, "dataset_path": tmp_path / "pairs.jsonl"}
    options.update(kwargs)
    return CandidateTournament(root, dynamic_commands=dynamic, mutation_commands=mutation, **options)


def _good(root, candidate_id="good", token_cost=None):
    return Candidate(candidate_id, "correct", _patch(root, "return 0", "return max(0, value + 1)"), token_cost)


def test_candidate_is_evaluated_in_a_real_detached_worktree(tmp_path):
    root = _repo(tmp_path / "repo")
    result = _engine(root, tmp_path).run([_good(root)])
    assert result["winner"] == "good"
    assert result["evaluations"][0]["accepted"]
    assert result["isolation"]["git_worktree_per_candidate"]


def test_candidate_worktree_is_removed_after_success(tmp_path):
    root = _repo(tmp_path / "repo")
    _engine(root, tmp_path).run([_good(root)])
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert not (root / ".worktrees").exists()


def test_candidate_worktree_is_removed_after_failed_dynamic_probe(tmp_path):
    root = _repo(tmp_path / "repo")
    bad = Candidate("bad", "wrong result", _patch(root, "return 0", "return 9"))
    result = _engine(root, tmp_path).run([bad])
    assert result["winner"] is None
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_parallel_candidates_do_not_change_the_main_worktree(tmp_path):
    root = _repo(tmp_path / "repo")
    original = (root / "app.py").read_bytes()
    candidates = [
        Candidate("good_a", "safe", _patch(root, "return 0", "return max(0, value + 1)")),
        Candidate("bad_b", "incorrect", _patch(root, "return 0", "return value + 2")),
        Candidate("good_c", "same safe behavior", _patch(root, "return 0", "return max(0, 1 + value)")),
        Candidate("bad_d", "incorrect", _patch(root, "return 0", "return 4")),
    ]
    result = _engine(root, tmp_path, max_workers=4).run(candidates, contract="compute value + 1")
    assert result["candidate_count"] == 4
    assert result["accepted_count"] == 2
    assert (root / "app.py").read_bytes() == original
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_bad_patch_is_rejected_before_probe_execution(tmp_path):
    root = _repo(tmp_path / "repo")
    candidate = Candidate("invalid", "stale patch", "not a git patch")
    result = _engine(root, tmp_path).run([candidate])
    assert result["evaluations"][0]["accepted"] is False
    assert "patch" in result["evaluations"][0]["message"]


def test_test_file_change_is_blocked_as_out_of_scope(tmp_path):
    root = _repo(tmp_path / "repo")
    (root / "tests").mkdir()
    (root / "tests/test_gate.py").write_text("assert True\n")
    _git(root, "add", "tests/test_gate.py")
    _git(root, "commit", "-qm", "test fixture")
    old = (root / "tests/test_gate.py").read_text()
    new = old + "assert False\n"
    body = difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                fromfile="a/tests/test_gate.py", tofile="b/tests/test_gate.py")
    patch = "diff --git a/tests/test_gate.py b/tests/test_gate.py\n" + "".join(body)
    result = _engine(root, tmp_path).run([Candidate("cheat", "test edit", patch)])
    assert result["accepted_count"] == 0


def test_l3_independent_audit_rejects_silent_broad_exception(tmp_path):
    root = _repo(tmp_path / "repo")
    source = "def compute(value):\n    return max(0, value + 1)\n\ndef guard(value):\n    try:\n        return int(value)\n    except Exception:\n        pass\n"
    before = (root / "app.py").read_text().splitlines(keepends=True)
    body = difflib.unified_diff(before, source.splitlines(keepends=True), fromfile="a/app.py", tofile="b/app.py")
    result = _engine(root, tmp_path).run([Candidate("bait", "silently swallow failures", "diff --git a/app.py b/app.py\n" + "".join(body))])
    evaluation = result["evaluations"][0]
    assert not evaluation["accepted"]
    assert not evaluation["audit_passed"]
    assert "EXCEPTION_SWALLOWED" in {item["code"] for item in evaluation["audit"]["findings"]}


def test_syntax_error_fails_static_gate(tmp_path):
    root = _repo(tmp_path / "repo")
    bad_source = "def compute(value):\n    return (\n"
    before = (root / "app.py").read_text().splitlines(keepends=True)
    body = difflib.unified_diff(before, bad_source.splitlines(keepends=True), fromfile="a/app.py", tofile="b/app.py")
    result = _engine(root, tmp_path).run([Candidate("syntax", "invalid syntax", "diff --git a/app.py b/app.py\n" + "".join(body))])
    assert result["evaluations"][0]["static_pass_rate"] < 1


def test_static_dynamic_mutation_audit_and_cost_metrics_are_recorded(tmp_path):
    root = _repo(tmp_path / "repo")
    evaluation = _engine(root, tmp_path).run([_good(root, token_cost=17)])["evaluations"][0]
    assert evaluation["static_pass_rate"] == 1
    assert evaluation["dynamic_pass_rate"] == 1
    assert evaluation["mutation_probe_survival_rate"] == 1
    assert evaluation["audit_passed"]
    assert evaluation["token_cost"] == 17
    assert evaluation["elapsed_sec"] > 0
    assert set(evaluation["objectives"]) == {"static", "dynamic", "mutation", "audit", "complexity", "cost"}


def test_complexity_delta_penalizes_added_branches(tmp_path):
    root = _repo(tmp_path / "repo")
    low = _good(root, "low")
    branch_source = "def compute(value):\n    if value < 0:\n        return 0\n    return value + 1\n"
    before = (root / "app.py").read_text().splitlines(keepends=True)
    body = difflib.unified_diff(before, branch_source.splitlines(keepends=True), fromfile="a/app.py", tofile="b/app.py")
    high = Candidate("branch", "branchy", "diff --git a/app.py b/app.py\n" + "".join(body))
    results = _engine(root, tmp_path).run([low, high])
    by_id = {item["candidate_id"]: item for item in results["evaluations"]}
    assert by_id["branch"]["complexity_delta"] > by_id["low"]["complexity_delta"]


def test_pareto_front_is_reported_separately_from_winner(tmp_path):
    root = _repo(tmp_path / "repo")
    result = _engine(root, tmp_path).run([_good(root, "a", 5), _good(root, "b", 1000)])
    assert result["winner"] in result["pareto_front"]
    assert all(item["pareto_optimal"] == (item["candidate_id"] in result["pareto_front"])
               for item in result["evaluations"])


def test_losing_candidates_export_grpo_comparisons(tmp_path):
    root = _repo(tmp_path / "repo")
    candidates = [_good(root, "winner"), Candidate("loser", "bad", _patch(root, "return 0", "return 9"))]
    result = _engine(root, tmp_path).run(candidates, contract="compute nonnegative value + 1")
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert result["grpo_pairs_exported"] == 1
    assert rows[0]["chosen_candidate"] == "winner"
    assert rows[0]["rejected_candidate"] == "loser"
    assert rows[0]["source"] == "candidate_tournament"
    assert rows[0]["rejection_evidence"]["dynamic_pass_rate"] == 0


def test_single_candidate_does_not_create_a_comparison_pair(tmp_path):
    root = _repo(tmp_path / "repo")
    result = _engine(root, tmp_path).run([_good(root)])
    assert result["grpo_pairs_exported"] == 0
    assert not (tmp_path / "pairs.jsonl").exists()


def test_concurrent_pair_exports_remain_valid_jsonl(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    root = _repo(tmp_path / "repo")
    engine = _engine(root, tmp_path)
    def run(index):
        return engine.run([_good(root, f"winner_{index}"),
                           Candidate(f"loser_{index}", "bad", _patch(root, "return 0", f"return {10+index}"))])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    records = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert sum(item["grpo_pairs_exported"] for item in results) == 4
    assert len(records) == 4
    assert len({item["pair_sha256"] for item in records}) == 4


@pytest.mark.parametrize("commands", [None, [], ["pytest -q"], [[]]])
def test_dynamic_probe_commands_are_required_and_structured(tmp_path, commands):
    root = _repo(tmp_path / "repo")
    with pytest.raises(ValueError):
        CandidateTournament(root, dynamic_commands=commands, mutation_commands=[[sys.executable, "-c", "pass"]])


@pytest.mark.parametrize("timeout", [0, -1, float("nan")])
def test_invalid_timeout_is_rejected(tmp_path, timeout):
    root = _repo(tmp_path / "repo")
    with pytest.raises(ValueError):
        CandidateTournament(root, dynamic_commands=[[sys.executable, "-c", "pass"]],
                            mutation_commands=[[sys.executable, "-c", "pass"]], timeout_sec=timeout)


@pytest.mark.parametrize("workers", [0, -1, 1.2, True])
def test_invalid_worker_count_is_rejected(tmp_path, workers):
    root = _repo(tmp_path / "repo")
    with pytest.raises(ValueError):
        CandidateTournament(root, dynamic_commands=[[sys.executable, "-c", "pass"]],
                            mutation_commands=[[sys.executable, "-c", "pass"]], max_workers=workers)


def test_candidate_id_must_be_unique(tmp_path):
    root = _repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="unique"):
        _engine(root, tmp_path).run([_good(root, "same"), _good(root, "same")])


@pytest.mark.parametrize("candidate", [
    Candidate("bad id!", "strategy", "patch"),
    Candidate("id", "", "patch"),
    Candidate("id", "strategy", ""),
    Candidate("id", "strategy", "patch", -1),
])
def test_candidate_fields_are_validated(tmp_path, candidate):
    root = _repo(tmp_path / "repo")
    with pytest.raises(ValueError):
        _engine(root, tmp_path).run([candidate])


def test_candidate_timeout_is_bounded_and_recorded(tmp_path):
    root = _repo(tmp_path / "repo")
    engine = CandidateTournament(root, dynamic_commands=[[sys.executable, "-c", "import time; time.sleep(1)"]],
                                mutation_commands=[[sys.executable, "-c", "pass"]], timeout_sec=0.05,
                                dataset_path=tmp_path / "pairs.jsonl")
    evaluation = engine.run([_good(root)])["evaluations"][0]
    assert evaluation["dynamic"][0]["timed_out"]
    assert evaluation["dynamic_pass_rate"] == 0


def test_dirty_main_worktree_is_rejected_before_tournament(tmp_path):
    root = _repo(tmp_path / "repo")
    (root / "app.py").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        _engine(root, tmp_path)


def test_allowed_path_scope_is_enforced(tmp_path):
    root = _repo(tmp_path / "repo")
    engine = _engine(root, tmp_path, allowed_paths={"other.py"})
    result = engine.run([_good(root)])
    assert result["accepted_count"] == 0


def test_each_candidate_reports_its_frozen_base(tmp_path):
    root = _repo(tmp_path / "repo")
    base = _git(root, "rev-parse", "HEAD")
    result = _engine(root, tmp_path).run([_good(root)])
    assert result["base_commit"] == base


def test_tournament_rejects_an_uncommitted_baseline(tmp_path):
    root = _repo(tmp_path / "repo")
    (root / "new.py").write_text("x=1\n")
    with pytest.raises(ValueError, match="clean"):
        _engine(root, tmp_path)
