import hashlib
import json
import math
from pathlib import Path

from aegis_core.ledger import Ledger
from aegis_core.runner import causal_prediction_error
from aegis_core.world_model import (
    CausalWorldModel,
    FeatureEncoder,
    STATE_LABELS,
    preblock_reason,
)


ROOT = Path(__file__).resolve().parents[1]


def test_feature_encoder_fits_causal_state_action_and_ast_context():
    records = [{
        "state": {"status": "running"},
        "action": {"patch": "--- a/src/guard.py\n+++ b/src/guard.py\n@@ -1 +1 @@\n-return False\n+return True\n"},
        "context": {"items": [{"source": "def guard(): return False"}]},
        "prior_attempts": [],
    }]
    encoder = FeatureEncoder(max_vocab=32).fit(records)
    features = encoder.encode(records[0]["state"], records[0]["action"], records[0]["context"])
    assert len(features) == encoder.input_size
    assert all(math.isfinite(value) for value in features)
    assert len(encoder.to_dict()["vocabulary"]) <= 32


def test_preblock_detects_test_edits_and_only_repeated_patch_loops():
    cheat = "--- a/tests/test_gate.py\n+++ b/tests/test_gate.py\n@@ -1 +1 @@\n-assert False\n+assert True\n"
    assert preblock_reason({"patch": cheat})[0] == "test_modification"
    patch = "--- a/src/guard.py\n+++ b/src/guard.py\n@@ -1 +1 @@\n-return False\n+return True\n"
    digest = hashlib.sha256(patch.encode()).hexdigest()
    assert preblock_reason({"patch": patch}, [{"patch_sha256": digest}])[0] is None
    assert preblock_reason({"patch": patch}, [{"patch_sha256": digest}, {"patch_sha256": digest}])[0] == "repeated_patch"
    assert preblock_reason({"patch": patch, "repeated_patch": True})[0] == "repeated_patch"
    assert preblock_reason({"patch": patch, "contract": None})[0] is None


def test_trained_artifact_loads_and_predicts_all_causal_heads():
    model = CausalWorldModel.load(ROOT / "artifacts" / "causal_world_model")
    prediction = model.predict(
        {"status": "running", "contract": {"target_files": ["src/guard.py"]}},
        {"op": "repair_attempt", "patch": "--- a/src/guard.py\n+++ b/src/guard.py\n@@ -1 +1 @@\n-return False\n+return True\n"},
        {"items": [{"source": "def guard(): return False"}]},
    )
    assert prediction["predicted_next_state"]["outcome"] in STATE_LABELS
    assert set(prediction["predicted_next_state"]["probabilities"]) == set(STATE_LABELS)
    assert abs(sum(prediction["predicted_next_state"]["probabilities"].values()) - 1.0) < 1e-8
    assert isinstance(prediction["predicted_reward"], float)
    low, high = prediction["predicted_reward_interval"]
    assert low <= prediction["predicted_reward"] <= high
    assert 0 <= prediction["prediction_confidence"] <= 1
    assert prediction["model_version"]


def test_world_model_overrides_cheat_prediction_with_preblock_evidence():
    model = CausalWorldModel.load(ROOT / "artifacts" / "causal_world_model")
    prediction = model.predict({"status": "running"}, {
        "patch": "--- a/tests/test_gate.py\n+++ b/tests/test_gate.py\n@@ -1 +1 @@\n-assert False\n+assert True\n"
    })
    state = prediction["predicted_next_state"]
    assert state["outcome"] == "cheat_detected"
    assert state["decision_source"] == "deterministic_pre_block"
    assert state["high_risk_pre_block"]
    assert state["pre_block_reason"] == "test_modification"
    assert prediction["risk_probability"] == 1.0
    assert state["probabilities"] == {
        "gate_passed": 0.0, "assertion_error": 0.0, "cheat_detected": 1.0,
        "replay_blocked": 0.0, "missing_constraint": 0.0,
    }


def test_causal_prediction_error_compares_state_and_reward():
    error = causal_prediction_error({"outcome": "gate_passed"}, 1.25,
                                    {"passed": True, "reward": 1.0})
    assert error == {
        "predicted_outcome": "gate_passed", "actual_outcome": "gate_passed",
        "state_mismatch": False, "state_error": 0, "reward_absolute_error": 0.25,
        "reward_interval_covered": None, "composite_error": 0.25,
    }
    interval = causal_prediction_error({"outcome": "gate_passed"}, 1.25,
                                       {"passed": True, "reward": 1.0}, [0.9, 1.3])
    assert interval["reward_interval_covered"]
    mismatch = causal_prediction_error({"outcome": "gate_passed"}, 0.0,
                                        {"pre_block_cause": "repeated_patch", "reward": -1.0})
    assert mismatch["actual_outcome"] == "replay_blocked"
    assert mismatch["state_mismatch"]
    assert mismatch["reward_absolute_error"] == 1.0


def test_ledger_exports_immutable_world_model_prediction_and_error(tmp_path):
    ledger = Ledger(tmp_path / "world-model.sqlite")
    ledger.create_ticket("prediction", "test", {"goal": "predict"}, "def run() -> None: ...")
    token = ledger.acquire_lease("prediction", "test")
    state = ledger.get_state("prediction")
    compiled = {"status": "COMPILED"}
    ledger.transition("prediction", token, "queued", "COMPILED", state_s0=state,
                      action_a0={"op": "compile"}, actual_state_s1=compiled, reward=0,
                      compiled_intent={"status": "COMPILED"}, contract_stub="def run() -> None: ...")
    ledger.transition("prediction", token, "COMPILED", "running", state_s0=compiled,
                      action_a0={"op": "start_run"}, actual_state_s1={"status": "running"}, reward=0)
    running = ledger.get_state("prediction")
    prediction = {"outcome": "gate_passed", "probabilities": {"gate_passed": 0.9}}
    ledger.transition("prediction", token, "running", "running", state_s0=running,
                      action_a0={"op": "world_model_prediction"}, actual_state_s1=running, reward=0,
                      predicted_next_state=prediction, predicted_reward=1.2, prediction_confidence=0.9)
    observed = {"status": "running", "world_model_outcome": "gate_passed", "passed": True}
    ledger.transition("prediction", token, "running", "running", state_s0=running,
                      action_a0={"op": "repair_attempt"}, actual_state_s1=observed, reward=1.0,
                      predicted_next_state=prediction, predicted_reward=1.2, prediction_confidence=0.9,
                      prediction_error={"state_error": 0, "reward_absolute_error": 0.2},
                      predicted_reward_interval=[0.8, 1.4])
    output = tmp_path / "traces.jsonl"
    assert ledger.export_jsonl(output) == 5
    traces = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert traces[-1]["predicted_next_state"] == prediction
    assert traces[-1]["predicted_reward"] == 1.2
    assert traces[-1]["predicted_reward_interval"] == [0.8, 1.4]
    assert traces[-1]["prediction_confidence"] == 0.9
    assert traces[-1]["prediction_error"] == {"state_error": 0, "reward_absolute_error": 0.2}
    assert traces[-1]["actual_state_s1"]["world_model_outcome"] == "gate_passed"
