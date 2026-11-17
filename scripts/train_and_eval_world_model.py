"""Train and blindly evaluate the compact Aegis causal world model on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from aegis_core.world_model import (
    CausalWorldNet,
    FeatureEncoder,
    STATE_LABELS,
    _extract_patch,
    preblock_reason,
)


SFT_FILES = ("sft_mediator_train.jsonl", "sft_mediator_eval.jsonl")
SYNTHETIC_PROVENANCE = "executed_template_variant"
CAMPAIGN_GROUP_ALIAS = {
    "depth": "template:cat_depth",
    "leverage": "template:quant_notional_cap",
    "regime": "template:quant_guard_period",
}
TEST_GROUPS = {
    "template:cat_depth",
    "template:quant_guard_period",
    "template:quant_signal_period",
    "ticket:cheat",
}
VALIDATION_GROUPS = {
    "template:cat_blank",
    "template:quant_notional_cap",
    "ticket:self_heal",
}


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _template_family(value: str) -> str:
    return value.rsplit("-", 1)[0] if value.rsplit("-", 1)[-1] in {"positive", "negative"} else value


def build_corpus(root: Path = ROOT):
    """Select 1,050 synthetic action outcomes and labeled physical transitions."""
    examples = []
    synthetic_count = 0
    for filename in SFT_FILES:
        rows = _read_jsonl(root / "datasets" / filename)
        for index, row in enumerate(rows):
            if row.get("provenance") != SYNTHETIC_PROVENANCE:
                continue
            messages = row.get("messages", [])
            if len(messages) < 2 or not isinstance(row.get("reward"), (int, float)):
                raise ValueError(f"invalid executed SFT row: {filename}:{index}")
            user_text = messages[-2].get("content", "")
            try:
                prompt = json.loads(user_text)
            except (json.JSONDecodeError, TypeError):
                prompt = {"prompt": user_text}
            # Case numbering and template IDs are split keys, never model inputs.
            state = {key: value for key, value in prompt.items() if key not in {"case", "template_id"}}
            action_text = messages[-1].get("content", "")
            action = {"completion": action_text}
            reward = float(row["reward"])
            label = "gate_passed" if reward > 0 else "assertion_error"
            family = _template_family(row["template_id"])
            examples.append({
                "sample_id": f"synthetic:{filename}:{index}",
                "group_id": f"template:{family}",
                "state": state,
                "action": action,
                "context": prompt.get("context", prompt.get("source", {})),
                "prior_attempts": [],
                "label": label,
                "reward": reward,
                "high_risk": preblock_reason(action, [])[0] is not None,
                "provenance": "synthetic_template_execution",
            })
            synthetic_count += 1

    trace_path = root / "datasets" / "campaign" / "campaign_traces.jsonl"
    physical_rows = _read_jsonl(trace_path)
    history = defaultdict(list)
    real_actionable = 0
    for row in physical_rows:
        ticket = row["ticket_id"]
        action = row.get("action_a0", {})
        state = row.get("state_s0", {})
        actual = row.get("actual_state_s1", {})
        operation = action.get("op")
        prior_attempts = [{"patch_sha256": digest} for digest in history[ticket]]
        if operation == "repair_attempt":
            if actual.get("cheat_detected"):
                label = "cheat_detected"
            elif actual.get("repeated_patch"):
                label = "replay_blocked"
            elif actual.get("passed") or actual.get("after_passed"):
                label = "gate_passed"
            else:
                label = "assertion_error"
            sample_action = dict(action)
            sample_action["target_files"] = row.get("action_a0", {}).get("contract", {}).get("target_files", [])
            group_id = CAMPAIGN_GROUP_ALIAS.get(ticket, f"ticket:{ticket}")
            reward = float(row["reward"])
            examples.append({
                "sample_id": f"campaign:{ticket}:{row['step_idx']}",
                "group_id": group_id,
                "state": state,
                "action": sample_action,
                "context": action.get("context", {}),
                "prior_attempts": prior_attempts,
                "label": label,
                "reward": reward,
                "high_risk": label in {"cheat_detected", "replay_blocked"},
                "provenance": "physical_sandbox_transition",
            })
            real_actionable += 1
            patch = _extract_patch(action)
            digest = action.get("patch_sha256") or hashlib.sha256(patch.encode("utf-8")).hexdigest()
            history[ticket].append(digest)
        elif operation in {"compile", "compile_request"} and actual.get("status") == "MISSING_CONSTRAINT":
            examples.append({
                "sample_id": f"campaign:{ticket}:{row['step_idx']}",
                "group_id": f"ticket:{ticket}",
                "state": state,
                "action": action,
                "context": action.get("context", {}),
                "prior_attempts": prior_attempts,
                "label": "missing_constraint",
                "reward": float(row["reward"]),
                "high_risk": False,
                "provenance": "physical_sandbox_transition",
            })
            real_actionable += 1

    expected_source_count = 1050 + 47
    if synthetic_count != 1050 or len(physical_rows) != 47 or real_actionable != 11:
        raise ValueError({"synthetic": synthetic_count, "physical": len(physical_rows), "labeled_physical": real_actionable})
    if len(examples) != 1061 or len(examples) + (len(physical_rows) - real_actionable) != expected_source_count:
        raise ValueError("unexpected causal dataset accounting")
    return examples, {
        "raw_unique_records": expected_source_count,
        "synthetic_action_outcomes": synthetic_count,
        "physical_transitions": len(physical_rows),
        "physical_actionable_outcomes": real_actionable,
        "physical_lifecycle_rows_excluded": len(physical_rows) - real_actionable,
        "training_examples": len(examples),
        "source_files": {
            "sft_mediator_train.jsonl": hashlib.sha256((root / "datasets" / SFT_FILES[0]).read_bytes()).hexdigest(),
            "sft_mediator_eval.jsonl": hashlib.sha256((root / "datasets" / SFT_FILES[1]).read_bytes()).hexdigest(),
            "campaign/campaign_traces.jsonl": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        },
    }


def split_by_group(examples):
    groups = {row["group_id"] for row in examples}
    if not TEST_GROUPS <= groups or not VALIDATION_GROUPS <= groups:
        raise ValueError({"missing_test_groups": sorted(TEST_GROUPS - groups),
                          "missing_validation_groups": sorted(VALIDATION_GROUPS - groups)})
    if TEST_GROUPS & VALIDATION_GROUPS:
        raise ValueError("test/validation group overlap")
    train = [i for i, row in enumerate(examples) if row["group_id"] not in TEST_GROUPS | VALIDATION_GROUPS]
    validation = [i for i, row in enumerate(examples) if row["group_id"] in VALIDATION_GROUPS]
    test = [i for i, row in enumerate(examples) if row["group_id"] in TEST_GROUPS]
    group_sets = [{examples[i]["group_id"] for i in split} for split in (train, validation, test)]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise ValueError("group leakage across train, validation and test")
    return train, validation, test


def _init_weights(input_size, hidden_size, seed):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / (input_size + hidden_size)), (input_size, hidden_size)),
        "b1": np.zeros(hidden_size),
        "Ws": rng.normal(0, 0.025, (hidden_size, len(STATE_LABELS))),
        "bs": np.zeros(len(STATE_LABELS)),
        "Wr": rng.normal(0, 0.025, (hidden_size, 1)),
        "br": np.zeros(1),
        "Wrisk": rng.normal(0, 0.025, (hidden_size, 1)),
        "brisk": np.zeros(1),
    }


def _forward(x, weights):
    hidden = np.tanh(x @ weights["W1"] + weights["b1"])
    logits = hidden @ weights["Ws"] + weights["bs"]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -60, 60))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    reward = (hidden @ weights["Wr"] + weights["br"]).reshape(-1)
    risk_logits = (hidden @ weights["Wrisk"] + weights["brisk"]).reshape(-1)
    risk = 1 / (1 + np.exp(-np.clip(risk_logits, -60, 60)))
    return hidden, probabilities, reward, risk_logits, risk


def _loss(x, labels, reward_targets, risk_targets, weights, class_weights):
    _, probabilities, reward, risk_logits, _ = _forward(x, weights)
    sample_weights = class_weights[labels]
    denom = max(float(sample_weights.sum()), 1e-12)
    state_loss = float((-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)) * sample_weights).sum() / denom)
    reward_loss = float(np.mean((reward - reward_targets) ** 2))
    risk_loss = float(np.mean(np.logaddexp(0, risk_logits) - risk_targets * risk_logits))
    return state_loss + 0.5 * reward_loss + 0.5 * risk_loss


def _train_epoch(x, labels, reward_targets, risk_targets, weights, optimizer, class_weights,
                 *, batch_size=64, learning_rate=0.005):
    m, v, step = optimizer
    order = np.random.permutation(len(labels))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        xb, yb = x[indices], labels[indices]
        rb, kb = reward_targets[indices], risk_targets[indices]
        hidden, probabilities, reward, risk_logits, risk = _forward(xb, weights)
        sample_weights = class_weights[yb]
        denom = max(float(sample_weights.sum()), 1e-12)
        d_state = probabilities.copy()
        d_state[np.arange(len(yb)), yb] -= 1
        d_state *= (sample_weights / denom)[:, None]
        d_reward = (2.0 * 0.5 / len(yb)) * (reward - rb)[:, None]
        d_risk = (0.5 / len(yb)) * (risk - kb)[:, None]
        grads = {
            "Ws": hidden.T @ d_state,
            "bs": d_state.sum(axis=0),
            "Wr": hidden.T @ d_reward,
            "br": d_reward.sum(axis=0),
            "Wrisk": hidden.T @ d_risk,
            "brisk": d_risk.sum(axis=0),
        }
        d_hidden = d_state @ weights["Ws"].T + d_reward @ weights["Wr"].T + d_risk @ weights["Wrisk"].T
        d_hidden *= 1.0 - hidden ** 2
        grads["W1"] = xb.T @ d_hidden
        grads["b1"] = d_hidden.sum(axis=0)
        step += 1
        for name, grad in grads.items():
            grad = np.clip(grad, -5, 5)
            m[name] = 0.9 * m[name] + 0.1 * grad
            v[name] = 0.999 * v[name] + 0.001 * grad * grad
            m_hat = m[name] / (1 - 0.9 ** step)
            v_hat = v[name] / (1 - 0.999 ** step)
            weights[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    optimizer[2] = step


def _binary_metrics(targets, predictions):
    tp = sum(bool(t) and bool(p) for t, p in zip(targets, predictions))
    fp = sum(not bool(t) and bool(p) for t, p in zip(targets, predictions))
    fn = sum(bool(t) and not bool(p) for t, p in zip(targets, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "true_positive": tp,
            "false_positive": fp, "false_negative": fn, "support": sum(bool(t) for t in targets)}


def _evaluate(x, examples, indices, label_indices, reward_mean, reward_scale, weights, class_weights):
    chosen = [examples[i] for i in indices]
    labels = label_indices[indices]
    rewards = np.asarray([examples[i]["reward"] for i in indices], dtype=float)
    reward_targets = (rewards - reward_mean) / reward_scale
    risk_targets = np.asarray([examples[i]["high_risk"] for i in indices], dtype=float)
    subset_x = x[indices]
    total_loss = _loss(subset_x, labels, reward_targets, risk_targets, weights, class_weights)
    _, probabilities, predicted_reward_scaled, risk_logits, risk_probability = _forward(subset_x, weights)
    raw_predictions = probabilities.argmax(axis=1)
    guarded_predictions = []
    guard_predictions = []
    for row, raw in zip(chosen, raw_predictions):
        reason, _ = preblock_reason(row["action"], row["prior_attempts"])
        guard_predictions.append(reason is not None)
        if reason:
            guarded_predictions.append(STATE_LABELS.index("replay_blocked" if reason == "repeated_patch" else "cheat_detected"))
        else:
            guarded_predictions.append(int(raw))
    predicted_rewards = np.clip(predicted_reward_scaled * reward_scale + reward_mean, -2, 1.5)
    actual_names = [STATE_LABELS[index] for index in labels]
    raw_names = [STATE_LABELS[index] for index in raw_predictions]
    guarded_names = [STATE_LABELS[index] for index in guarded_predictions]
    prob_true = probabilities[np.arange(len(labels)), labels]
    risk_binary = risk_probability >= 0.5
    true_risk = [row["high_risk"] for row in chosen]
    correct_guarded = sum(a == b for a, b in zip(actual_names, guarded_names))
    return {
        "loss": total_loss,
        "state_accuracy": sum(a == b for a, b in zip(actual_names, raw_names)) / max(1, len(labels)),
        "state_log_loss": float(-np.log(np.clip(prob_true, 1e-12, 1)).mean()),
        "guarded_state_accuracy": correct_guarded / max(1, len(labels)),
        "reward_mae": float(np.abs(predicted_rewards - rewards).mean()),
        "reward_rmse": float(np.sqrt(np.mean((predicted_rewards - rewards) ** 2))),
        "causal_prediction_error": {
            "state_mismatch_rate": 1.0 - correct_guarded / max(1, len(labels)),
            "reward_absolute_error_mean": float(np.abs(predicted_rewards - rewards).mean()),
        },
        "risk_head_precision_recall": _binary_metrics(true_risk, risk_binary.tolist()),
        "preblock_precision_recall": _binary_metrics(true_risk, guard_predictions),
        "state_support": dict(Counter(actual_names)),
        "high_risk_support": sum(true_risk),
        "samples": len(indices),
    }


def _serialize_weights(weights, input_size, reward_mean, reward_scale, reward_interval_radius, version_seed):
    payload = {
        "schema_version": "aegis.world_model.weights.v1",
        "state_labels": list(STATE_LABELS),
        "input_size": int(input_size),
        "hidden_size": int(weights["W1"].shape[1]),
        "reward_mean": float(reward_mean),
        "reward_scale": float(reward_scale),
        "reward_interval_radius": float(reward_interval_radius),
        "weights": {
            "W1": weights["W1"].tolist(), "b1": weights["b1"].tolist(),
            "Ws": weights["Ws"].tolist(), "bs": weights["bs"].tolist(),
            "Wr": weights["Wr"].reshape(-1).tolist(), "br": weights["br"].reshape(-1).tolist(),
            "Wrisk": weights["Wrisk"].reshape(-1).tolist(), "brisk": weights["brisk"].reshape(-1).tolist(),
        },
        "seed": version_seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload["model_version"] = hashlib.sha256(canonical).hexdigest()
    return payload


def train_and_evaluate(*, root: Path = ROOT, output_dir: Path | None = None,
                       epochs: int = 100, seed: int = 20261117):
    start_time = time.monotonic()
    output_dir = output_dir or root / "artifacts" / "causal_world_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples, corpus_info = build_corpus(root)
    train_indices, validation_indices, test_indices = split_by_group(examples)
    train_rows = [examples[i] for i in train_indices]
    encoder = FeatureEncoder(max_vocab=768).fit(train_rows)
    x = np.asarray([encoder.encode(row["state"], row["action"], row["context"], row["prior_attempts"])
                    for row in examples], dtype=np.float64)
    label_indices = np.asarray([STATE_LABELS.index(row["label"]) for row in examples], dtype=np.int64)
    rewards = np.asarray([row["reward"] for row in examples], dtype=np.float64)
    reward_mean = float(rewards[train_indices].mean())
    reward_scale = float(rewards[train_indices].std()) or 1.0
    reward_targets = (rewards - reward_mean) / reward_scale
    risk_targets = np.asarray([row["high_risk"] for row in examples], dtype=np.float64)
    counts = np.bincount(label_indices[train_indices], minlength=len(STATE_LABELS))
    present = max(1, int((counts > 0).sum()))
    class_weights = np.ones(len(STATE_LABELS), dtype=np.float64)
    for index, count in enumerate(counts):
        if count:
            class_weights[index] = min(8.0, len(train_indices) / (present * count))
    weights = _init_weights(encoder.input_size, 48, seed)
    baseline_weights = copy.deepcopy(weights)
    baseline = _evaluate(x, examples, test_indices, label_indices, reward_mean, reward_scale,
                         baseline_weights, class_weights)
    moments = [{key: np.zeros_like(value) for key, value in weights.items()},
               {key: np.zeros_like(value) for key, value in weights.items()}, 0]
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(weights)
    patience = 22
    stale_epochs = 0
    np.random.seed(seed)
    for epoch in range(1, epochs + 1):
        _train_epoch(x[train_indices], label_indices[train_indices], reward_targets[train_indices],
                     risk_targets[train_indices], weights, moments, class_weights)
        train_loss = _loss(x[train_indices], label_indices[train_indices], reward_targets[train_indices],
                           risk_targets[train_indices], weights, class_weights)
        validation_loss = _loss(x[validation_indices], label_indices[validation_indices],
                                reward_targets[validation_indices], risk_targets[validation_indices],
                                weights, class_weights)
        history.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": validation_loss})
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(weights)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    validation_hidden, _, validation_reward_scaled, _, _ = _forward(x[validation_indices], best_weights)
    del validation_hidden
    validation_reward = np.clip(validation_reward_scaled * reward_scale + reward_mean, -2, 1.5)
    validation_residuals = np.abs(validation_reward - rewards[validation_indices])
    interval_rank = min(len(validation_residuals), max(1, math.ceil((len(validation_residuals) + 1) * 0.90)))
    reward_interval_radius = float(np.sort(validation_residuals)[interval_rank - 1])
    interval_coverage = float(np.mean(validation_residuals <= reward_interval_radius))
    trained = _evaluate(x, examples, test_indices, label_indices, reward_mean, reward_scale,
                        best_weights, class_weights)
    weight_payload = _serialize_weights(best_weights, encoder.input_size, reward_mean, reward_scale,
                                        reward_interval_radius, seed)
    (output_dir / "model_weights.json").write_text(json.dumps(weight_payload, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "feature_encoder.json").write_text(json.dumps(encoder.to_dict(), ensure_ascii=False,
        separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "model_weights.npz", **best_weights,
                        reward_mean=np.asarray([reward_mean]), reward_scale=np.asarray([reward_scale]))
    split_groups = {
        "train": sorted({examples[i]["group_id"] for i in train_indices}),
        "validation": sorted({examples[i]["group_id"] for i in validation_indices}),
        "test": sorted({examples[i]["group_id"] for i in test_indices}),
    }
    split_manifest = {
        "schema_version": "aegis.world_model.split.v1",
        "strategy": "hold out complete template families and ticket IDs; no group may cross splits",
        "seed": seed,
        "groups": split_groups,
        "sample_ids": {
            name: [examples[i]["sample_id"] for i in indices]
            for name, indices in (("train", train_indices), ("validation", validation_indices), ("test", test_indices))
        },
        "counts": {"train": len(train_indices), "validation": len(validation_indices), "test": len(test_indices)},
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    group_leakage = bool(set(split_groups["train"]) & set(split_groups["validation"])
                         or set(split_groups["train"]) & set(split_groups["test"])
                         or set(split_groups["validation"]) & set(split_groups["test"]))
    metrics = {
        "schema_version": "aegis.world_model.metrics.v1",
        "hardware": {"platform": platform.platform(), "python": sys.version.split()[0],
                     "cpu_count": os.cpu_count(), "device": "CPU", "cuda_used": False,
                     "numpy_version": np.__version__},
        "dataset": {**corpus_info,
                    "train_examples": len(train_indices), "validation_examples": len(validation_indices),
                    "holdout_test_examples": len(test_indices),
                    "label_counts": dict(Counter(row["label"] for row in examples)),
                    "split_label_counts": {
                        name: dict(Counter(examples[i]["label"] for i in indices))
                        for name, indices in (("train", train_indices), ("validation", validation_indices), ("test", test_indices))
                    },
                    "group_leakage": group_leakage},
        "training": {"architecture": "CausalWorldNet(shared tanh hidden layer; state softmax, reward regression, risk sigmoid heads)",
                     "optimizer": "Adam", "learning_rate": 0.005, "batch_size": 64, "hidden_size": 48,
                     "vocabulary_size": len(encoder.vocabulary), "input_size": encoder.input_size,
                     "seed": seed, "epochs_requested": epochs, "epochs_completed": len(history),
                     "best_epoch": best_epoch, "best_validation_loss": best_loss,
                     "reward_interval": {"method": "90% validation absolute-residual conformal radius",
                                         "radius": reward_interval_radius,
                                         "validation_coverage": interval_coverage,
                                         "validation_samples": len(validation_residuals)},
                     "elapsed_seconds": time.monotonic() - start_time,
                     "weights_sha256": hashlib.sha256((output_dir / "model_weights.json").read_bytes()).hexdigest()},
        "before_training": baseline,
        "after_training": trained,
        "history": history,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"artifact_dir": str(output_dir), "examples": len(examples),
               "train": len(train_indices), "validation": len(validation_indices),
               "test": len(test_indices), "best_epoch": best_epoch,
               "before_training": baseline, "after_training": trained,
               "group_leakage": group_leakage}
    return metrics, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    _, summary = train_and_evaluate(root=args.root.resolve(), output_dir=args.output, epochs=args.epochs, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
