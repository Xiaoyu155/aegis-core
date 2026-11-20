"""Held-out calibration, metric semantics and signed-mode-gate probes."""
import json
import math
import subprocess
import sys

import pytest

from aegis_core.world_model_bench import (
    CalibrationSample,
    evaluate_threshold,
    load_causal_trace_samples,
    load_tournament_samples,
    recommend_threshold,
    run_benchmark,
    split_validation,
    threshold_candidates,
    verify_permit,
)


KEY = "test-only-calibration-key-with-32-bytes"


class FakeModel:
    model_version = "fake-v1"

    def predict(self, state, action, context):
        patch = action["patch"]
        score = 0.1 if "GOOD_PATCH" in patch else 0.9
        return {"risk_probability": score}


def rows(*, positives=10, negatives=10, positive_score=0.1, negative_score=0.9):
    return ([CalibrationSample(f"p-{i}", f"positive-group-{i}", "positive", positive_score, i + 1)
             for i in range(positives)]
            + [CalibrationSample(f"n-{i}", f"negative-group-{i}", "negative", negative_score, i + 1)
               for i in range(negatives)])


@pytest.mark.parametrize("threshold", [0.0, 0.1, 0.2, 0.5, 0.9, 1.0])
def test_threshold_metrics_count_wrong_blocks_misses_recall_and_compute(threshold):
    samples = [CalibrationSample("p", "gp", "positive", 0.1, 2),
               CalibrationSample("n", "gn", "negative", 0.9, 3)]
    result = evaluate_threshold(samples, threshold)
    expected_fp = int(0.1 >= threshold)
    expected_blocked_negative = int(0.9 >= threshold)
    assert result["false_positive_blocks"] == expected_fp
    assert result["false_positive_block_rate"] == expected_fp
    assert result["false_negative_rate"] == 1 - expected_blocked_negative
    assert result["recall"] == expected_blocked_negative
    assert result["compute_savings_rate"] == (2 * expected_fp + 3 * expected_blocked_negative) / 5
    assert result["safe_negative_compute_savings_rate"] == 3 * expected_blocked_negative / 5


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        evaluate_threshold(rows(), threshold)


@pytest.mark.parametrize("sample", [
    {"sample_id": "", "group_id": "g", "label": "positive", "risk_probability": 0.1},
    {"sample_id": "s", "group_id": "", "label": "positive", "risk_probability": 0.1},
    {"sample_id": "s", "group_id": "g", "label": "unknown", "risk_probability": 0.1},
    {"sample_id": "s", "group_id": "g", "label": "positive", "risk_probability": -0.1},
    {"sample_id": "s", "group_id": "g", "label": "positive", "risk_probability": 1.1},
    {"sample_id": "s", "group_id": "g", "label": "positive", "risk_probability": float("nan")},
    {"sample_id": "s", "group_id": "g", "label": "positive", "risk_probability": 0.1, "cost_units": 0},
    {"sample_id": "s", "group_id": "g", "label": "positive", "risk_probability": 0.1, "cost_units": float("inf")},
])
def test_invalid_calibration_rows_fail_closed(sample):
    with pytest.raises((TypeError, ValueError)):
        evaluate_threshold([sample, CalibrationSample("n", "n", "negative", 0.9)], 0.5)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, float("nan")])
def test_validation_fraction_must_leave_independent_training_and_holdout_groups(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        split_validation(rows(), validation_fraction=fraction)


@pytest.mark.parametrize("seed", range(6))
def test_grouped_holdout_split_is_stable_and_never_leaks_groups(seed):
    samples = [CalibrationSample(f"{group}-{index}", f"group-{(group + seed) % 12}",
                                 "positive" if index == 0 else "negative", 0.2 + index / 10)
               for group in range(12) for index in range(2)]
    train, validation = split_validation(samples, validation_fraction=0.25)
    train_groups = {item.group_id for item in train}
    validation_groups = {item.group_id for item in validation}
    assert train and validation
    assert train_groups.isdisjoint(validation_groups)
    assert validation_groups == {item.group_id for item in split_validation(samples, validation_fraction=0.25)[1]}
    assert all(sum(item.group_id == group for item in validation) == 2 for group in validation_groups)


@pytest.mark.parametrize("positive,negative,expected", [
    (0.1, 0.9, True),
    (0.9, 0.1, False),
    (0.5, 0.5, False),
    (0.0, 1.0, True),
])
def test_calibration_gate_recommends_only_when_false_blocks_and_recall_are_safe(positive, negative, expected):
    data = [CalibrationSample(f"p{i}", f"p{i}", "positive", positive) for i in range(8)]
    data += [CalibrationSample(f"n{i}", f"n{i}", "negative", negative) for i in range(8)]
    report = recommend_threshold(data, model_version="m1", signing_key=KEY)
    assert report["gate"]["allowed_mode_transition"] is expected
    assert report["gate"]["status"] == ("permit" if expected else "blocked")


def test_permit_signature_verifies_and_detects_report_tampering():
    report = recommend_threshold(rows(), model_version="model-42", signing_key=KEY)
    assert report["gate"]["allowed_mode_transition"] is True
    assert report["signature_algorithm"] == "HMAC-SHA256"
    assert verify_permit(report, KEY) is True
    changed = json.loads(json.dumps(report))
    changed["validation_metrics"]["recall"] = 0.86
    assert verify_permit(changed, KEY) is False


@pytest.mark.parametrize("key", [None, "short", "123456789012345"])
def test_valid_metrics_without_a_sufficient_signing_key_cannot_authorize_enforce(key):
    report = recommend_threshold(rows(), model_version="model-42", signing_key=key)
    assert report["threshold_recommendation"] is not None
    assert report["gate"]["false_positive_block_rate_below_1_percent"] is True
    assert report["gate"]["recall_at_least_85_percent"] is True
    assert report["gate"]["allowed_mode_transition"] is False
    assert "signature" not in report
    assert verify_permit(report, KEY) is False


def test_false_positive_limit_is_strict_but_allows_less_than_one_percent():
    data = [CalibrationSample(f"p{i}", f"p{i}", "positive", 0.1) for i in range(101)]
    data += [CalibrationSample(f"n{i}", f"n{i}", "negative", 0.9) for i in range(20)]
    data[0] = CalibrationSample("p0", "p0", "positive", 0.9)
    report = recommend_threshold(data, model_version="m", signing_key=KEY)
    assert report["gate"]["allowed_mode_transition"] is True
    assert report["validation_metrics"]["false_positive_block_rate"] < 0.01


def test_gate_does_not_combine_metrics_met_at_different_thresholds():
    data = [CalibrationSample("p1", "p1", "positive", 0.1),
            CalibrationSample("p2", "p2", "positive", 0.9),
            CalibrationSample("n1", "n1", "negative", 0.2)]
    report = recommend_threshold(data, model_version="m", signing_key=KEY)
    assert report["gate"]["false_positive_block_rate_below_1_percent"] is True
    assert report["gate"]["recall_at_least_85_percent"] is True
    assert report["gate"]["joint_threshold_passed"] is False
    assert report["threshold_recommendation"] is None
    assert report["gate"]["allowed_mode_transition"] is False


def test_threshold_candidates_include_exact_and_above_score_boundaries():
    candidates = threshold_candidates(rows(positives=1, negatives=1, positive_score=0.2, negative_score=0.8))
    assert 0.0 in candidates and 1.0 in candidates
    assert 0.2 in candidates and math.nextafter(0.2, 1.0) in candidates
    assert 0.8 in candidates and math.nextafter(0.8, 1.0) in candidates


@pytest.mark.parametrize("row_count", [2, 5, 9])
def test_tournament_loader_uses_winner_and_loser_patches_as_paired_labels(row_count, tmp_path):
    path = tmp_path / "pairs.jsonl"
    records = [{"pair_sha256": f"group-{i}", "prompt": f"task {i}",
                "chosen_patch": "+GOOD_PATCH", "rejected_patch": "+BAD_PATCH"}
               for i in range(row_count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    samples = load_tournament_samples(path, FakeModel())
    assert len(samples) == row_count * 2
    assert sum(item.label == "positive" for item in samples) == row_count
    assert sum(item.label == "negative" for item in samples) == row_count
    assert all(item.risk_probability == (0.1 if item.label == "positive" else 0.9) for item in samples)


def test_causal_trace_loader_uses_only_labeled_repair_outcomes(tmp_path):
    path = tmp_path / "traces.jsonl"
    rows = [
        {"trace_id": 1, "group_id": "g1", "actual_state_s1": {"after_passed": True},
         "action_a0": {"patch": "+GOOD_PATCH"}, "state_s0": {"scenario": "repair"}},
        {"trace_id": 2, "group_id": "g1", "actual_state_s1": {"after_passed": False},
         "action_a0": {"patch": "+BAD_PATCH"}, "state_s0": {"scenario": "repair"}},
        {"trace_id": 3, "group_id": "lifecycle", "actual_state_s1": {"status": "running"},
         "action_a0": {"op": "create_ticket"}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    samples = load_causal_trace_samples(path, FakeModel())
    assert [(item.sample_id, item.label) for item in samples] == [("1", "positive"), ("2", "negative")]


def test_benchmark_report_combines_datasets_and_states_proxy_cost_method(tmp_path):
    tournament = tmp_path / "pairs.jsonl"
    causal = tmp_path / "traces.jsonl"
    records = [{"pair_sha256": f"group-{i}", "prompt": f"task {i}",
                "chosen_patch": "+GOOD_PATCH", "rejected_patch": "+BAD_PATCH"}
               for i in range(12)]
    tournament.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    trace_rows = [{"trace_id": i, "group_id": f"trace-{i}", "actual_state_s1": {"after_passed": i % 2 == 0},
                   "action_a0": {"patch": "+GOOD_PATCH" if i % 2 == 0 else "+BAD_PATCH"},
                   "state_s0": {"scenario": f"trace-{i}"}} for i in range(8)]
    causal.write_text("".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8")
    report = run_benchmark(tmp_path, model=FakeModel(), tournament_path=tournament,
                           causal_trace_path=causal, signing_key=KEY)
    assert report["dataset"]["total_samples"] == 32
    assert report["dataset"]["validation_samples"] > 0
    assert report["dataset"]["training_groups"] > 0
    assert "changed patch lines" in report["dataset"]["cost_estimate_method"]
    assert report["source"] == "held_out_validation"
    assert verify_permit(report, KEY) is True


def test_world_model_bench_cli_exposes_validation_and_signing_controls():
    result = subprocess.run([sys.executable, "-m", "aegis_core.cli", "world-model-bench", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--validation-fraction" in result.stdout
    assert "--threshold" in result.stdout
    assert "--signing-key-env" in result.stdout


def test_recommended_threshold_is_never_issued_for_single_class_holdout():
    positives = [CalibrationSample(f"p{i}", f"p{i}", "positive", 0.1) for i in range(4)]
    with pytest.raises(ValueError, match="positive and negative"):
        recommend_threshold(positives, model_version="m", signing_key=KEY)
