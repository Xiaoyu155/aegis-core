"""Real structured-probe tournament cycles, diagnosis, budget fuse and replay learning."""
import difflib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aegis_core.experience_bank import ExperienceBank
from aegis_core.ledger import Ledger
from aegis_core.self_heal_loop import SelfHealLoop, run_autopilot


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path, source="def clamp(value):\n    return value\n"):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text(".worktrees/\n__pycache__/\n.pytest_cache/\n.cache/\n")
    (root / "app.py").write_text(source, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Self Heal Tests")
    _git(root, "config", "user.email", "self-heal@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    return root


def _patch(root, old, new):
    before = (root / "app.py").read_text(encoding="utf-8").splitlines(keepends=True)
    after = (root / "app.py").read_text(encoding="utf-8").replace(old, new).splitlines(keepends=True)
    body = difflib.unified_diff(before, after, fromfile="a/app.py", tofile="b/app.py")
    return "diff --git a/app.py b/app.py\n" + "".join(body)


def _spec():
    return {"kind": "numeric_bounds", "path": "app.py", "symbol": "clamp",
            "min": 0, "max": 10, "fallback": None, "seed": 42}


def _candidate(root, candidate_id, replacement, strategy="repair"):
    return {"candidate_id": candidate_id, "strategy": strategy,
            "patch": _patch(root, "return value", replacement)}


def _run(root, tmp_path, rounds, **kwargs):
    options = {"contract": "numeric clamp bounds for app.py", "probe_spec": _spec(),
               "candidate_rounds": rounds, "database_path": tmp_path / "ledger.sqlite",
               "dataset_path": tmp_path / "grpo.jsonl", "timeout_sec": 5,
               "max_rounds": len(rounds), "max_context_tokens": 512,
               "max_total_tokens": 20000, "max_wall_clock_sec": 120}
    options.update(kwargs)
    return run_autopilot(root, **options)


def test_contract_expansion_true_red_tournament_and_second_round_converge(tmp_path):
    root = _repo(tmp_path)
    db = tmp_path / "ledger.sqlite"
    ExperienceBank(db).record_arbitration(pattern="numeric DYNAMIC_PROBE_FAILED overflow", rule={})
    rounds = [
        [_candidate(root, "wrong_a", "return 99", "bad constant"),
         _candidate(root, "wrong_b", "return value + 2", "off by two")],
        [_candidate(root, "clamp", "if type(value) is not int:\n        return None\n    return min(10, max(0, value))", "range clamp"),
         _candidate(root, "bait", "return min(10, max(0, value + 2))", "hidden offset")],
    ]
    result = _run(root, tmp_path, rounds)
    assert result["status"] == "succeeded"
    assert result["rounds_attempted"] == 2
    assert result["converged_round"] == 2
    assert result["winner"]["candidate_id"] == "clamp"
    assert result["synthesized_probe"]["baseline_true_red"] == "ASSERTION_FAILED"
    assert result["learning"]["verified_recoveries"] == 1
    assert result["learning"]["experience_ingest"]["added"] == 1
    assert result["learning"]["replayed_hints"]
    assert result["rounds"][1]["experience_hints"]
    assert result["learning"]["grpo_dataset_sha256"]
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1
    ticket = Ledger(db).get_ticket(result["ticket_id"])
    assert ticket["status"] == "succeeded"


def test_autopilot_records_failure_frames_and_assertion_differences(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "wrong", "return 99")]], max_rounds=1)
    assert result["status"] == "failed"
    assert result["diagnosis"]["failure_code"] == "DYNAMIC_PROBE_FAILED"
    assert result["diagnosis"]["frames"]
    assert any(item["code"] == "ASSERTION_MISMATCH" for item in result["diagnosis"]["contract_differences"])
    assert result["rounds"][0]["tournament"]["evaluations"][0]["dynamic_pass_rate"] == 0


def test_provider_receives_structured_failure_and_experience_before_round_two(tmp_path):
    root = _repo(tmp_path)
    db = tmp_path / "ledger.sqlite"
    ExperienceBank(db).record_arbitration(pattern="numeric DYNAMIC_PROBE_FAILED overflow", rule={})
    calls = []
    def provider(context):
        calls.append(context)
        if context["round"] == 1:
            return [_candidate(root, "bad", "return 99")]
        assert context["diagnosis"]["failure_code"] == "DYNAMIC_PROBE_FAILED"
        assert context["experience_hints"]
        return [_candidate(root, "good", "if type(value) is not int:\n        return None\n    return min(10, max(0, value))")]
    result = _run(root, tmp_path, [], candidate_provider=provider, max_rounds=2)
    assert result["status"] == "succeeded"
    assert len(calls) == 2
    assert calls[1]["previous_rounds"][0]["winner"] is None
    assert calls[1]["remaining_tokens"] < calls[0]["remaining_tokens"]


def test_budget_token_fuse_stops_before_candidate_processes(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "too_large", "return 99")]], max_total_tokens=1)
    assert result["status"] == "failed"
    assert result["reason"] == "BUDGET_EXCEEDED_TOKENS"
    assert result["rounds_attempted"] == 0
    assert result["budget"]["tokens_used"] == 0


def test_budget_context_is_bounded_and_reports_recall(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "good", "if type(value) is not int:\n        return None\n    return min(10, max(0, value))")]])
    context = result["impact_context"]
    assert context["token_count"] <= context["max_context_tokens"]
    assert context["critical_symbol_recall"] == 1
    assert "app.py" in context["changed_files"]


def test_fixed_seed_true_red_manifest_has_randomized_cases(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "wrong", "return 99")]], max_rounds=1)
    report = result["synthesized_probe"]
    assert report["case_categories"]["randomized"] == 5
    assert report["case_count"] >= 12


def test_replay_learning_does_not_invent_world_model_predictions(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "bad", "return 99")],
                                   [_candidate(root, "good", "if type(value) is not int:\n        return None\n    return min(10, max(0, value))")]])
    assert result["learning"]["selection_counts"]["no_prediction_count"] == 2
    assert result["learning"]["selected_model_traces"] == 0
    assert result["learning"]["verified_recoveries"] == 1


def test_rounds_exhaustion_closes_ticket_as_failed(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "bad", "return 99")]], max_rounds=1)
    assert Ledger(tmp_path / "ledger.sqlite").get_ticket(result["ticket_id"])["status"] == "failed"
    assert result["reason"] == "self_heal_rounds_exhausted"


def test_autopilot_requires_clean_frozen_baseline(tmp_path):
    root = _repo(tmp_path)
    (root / "app.py").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        _run(root, tmp_path, [[_candidate(root, "bad", "return 99")]])


def test_probe_target_must_be_tracked_in_baseline(tmp_path):
    root = _repo(tmp_path)
    (root / ".cache").mkdir()
    (root / ".cache/new.py").write_text("def f(): pass\n")
    spec = _spec() | {"path": ".cache/new.py", "symbol": "f"}
    with pytest.raises(ValueError, match="tracked"):
        _run(root, tmp_path, [[]], probe_spec=spec, max_rounds=1)


def test_max_round_count_is_bounded(tmp_path):
    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="max_rounds"):
        _run(root, tmp_path, [[_candidate(root, "bad", "return 99")]], max_rounds=9)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_token_budgets_are_rejected(tmp_path, budget):
    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="max_total_tokens"):
        _run(root, tmp_path, [[_candidate(root, "bad", "return 99")]], max_total_tokens=budget)


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_invalid_wall_clock_budgets_are_rejected(tmp_path, budget):
    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="max_wall_clock_sec"):
        _run(root, tmp_path, [[_candidate(root, "bad", "return 99")]], max_wall_clock_sec=budget)


def test_invalid_context_budget_is_rejected_before_synthesis(tmp_path):
    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="max_context_tokens"):
        _run(root, tmp_path, [[_candidate(root, "bad", "return 99")]], max_context_tokens=10)


def test_expansion_can_stop_for_missing_business_constraints(tmp_path):
    root = _repo(tmp_path)
    loop = SelfHealLoop(root, database_path=tmp_path / "db.sqlite")
    result = loop.run(contract="Set max drawdown to {{max_drawdown}}", probe_spec=_spec(),
                      candidate_rounds=[[_candidate(root, "bad", "return 99")]])
    assert result["status"] == "MISSING_CONSTRAINT"
    assert result["phases"] == ["contract", "expand"]
    assert result["questions"][0]["parameter"] == "max_drawdown"


def test_multiple_rounds_preserve_contract_and_probe_hash_for_verified_pair(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "bad", "return 99")],
                                   [_candidate(root, "good", "if type(value) is not int:\n        return None\n    return min(10, max(0, value))")]])
    rows = Ledger(tmp_path / "ledger.sqlite").learning_selection(0)[1]
    assert len(rows) == 1
    failed = rows[0]["failed_trace"]["action_a0"]
    assert failed["contract"] == "numeric clamp bounds for app.py"
    assert failed["probe_script_sha256"]
    assert rows[0]["passed_trace_id"] > rows[0]["failed_trace"]["trace_id"]


def test_failed_run_keeps_main_source_unmodified(tmp_path):
    root = _repo(tmp_path)
    original = (root / "app.py").read_bytes()
    _run(root, tmp_path, [[_candidate(root, "bad", "return 99")]], max_rounds=1)
    assert (root / "app.py").read_bytes() == original
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_grpo_replay_jsonl_contains_chosen_and_rejected_real_patches(tmp_path):
    root = _repo(tmp_path)
    result = _run(root, tmp_path, [[_candidate(root, "bad", "return 99")],
                                   [_candidate(root, "good", "if type(value) is not int:\n        return None\n    return min(10, max(0, value))"),
                                    _candidate(root, "loser", "return value + 2")]])
    records = [json.loads(line) for line in (tmp_path / "grpo.jsonl").read_text().splitlines()]
    assert result["learning"]["grpo_dataset_sha256"]
    assert records[0]["chosen_candidate"] == "good"
    assert records[0]["rejected_candidate"] == "loser"
    assert records[0]["chosen_sha256"] != records[0]["rejected_sha256"]


def test_invalid_candidate_shape_is_reported_as_execution_error(tmp_path):
    root = _repo(tmp_path)
    with pytest.raises(ValueError):
        _run(root, tmp_path, [[{"candidate_id": "broken", "strategy": "missing patch"}]])
