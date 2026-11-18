"""Small replay updates anchored to the shipped training corpus."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile

from .ledger import Ledger
from .experience_bank import ExperienceBank
from .runner import classify_causal_outcome
from .world_model import (CausalWorldModel, STATE_LABELS, bundled_replay_anchors,
                          encoder_digest, load_repository_model, prediction_telemetry)


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_numpy_weights(raw):
    return {
        "W1": np.asarray(raw["W1"], dtype=np.float64),
        "b1": np.asarray(raw["b1"], dtype=np.float64),
        "Ws": np.asarray(raw["Ws"], dtype=np.float64),
        "bs": np.asarray(raw["bs"], dtype=np.float64),
        "Wr": np.asarray(raw["Wr"], dtype=np.float64).reshape(-1, 1),
        "br": np.asarray(raw["br"], dtype=np.float64).reshape(-1),
        "Wrisk": np.asarray(raw["Wrisk"], dtype=np.float64).reshape(-1, 1),
        "brisk": np.asarray(raw["brisk"], dtype=np.float64).reshape(-1),
    }


def _anchors(root, model, seed):
    pack = bundled_replay_anchors()
    if pack["encoder_sha256"] == encoder_digest(model.encoder):
        return pack["training"], pack["validation"]
    # A custom encoder needs anchors encoded in that encoder, from its own corpus.
    if not (root / "datasets" / "sft_mediator_train.jsonl").is_file():
        return [], []
    try:
        from scripts.train_and_eval_world_model import build_corpus, split_by_group
    except ImportError:
        return [], []
    examples, _ = build_corpus(root)
    train, validation, _ = split_by_group(examples)
    rng = random.Random(seed)
    training_indices = []
    for label in STATE_LABELS:
        candidates = [i for i in train if examples[i]["label"] == label]
        rng.shuffle(candidates)
        training_indices.extend(candidates[:24])

    def rows(indices):
        return [{**examples[index], "features": _encode(model, examples[index])} for index in indices]

    return rows(training_indices), rows(validation[:96])


def _encode(model, row):
    return model.encoder.encode(row["state"], row["action"], row.get("context"), row.get("prior_attempts", []))


def learn_from_ledger(*, repo: str | Path, db: str | Path, min_surprise: float = 0.3,
                      artifact_dir: str | Path | None = None, limit: int = 256,
                      epochs: int = 4, seed: int = 20261118) -> dict:
    """Fine-tune on high-error real traces while replaying grouped corpus anchors."""
    if type(epochs) is not int or not 1 <= epochs <= 20:
        raise ValueError("epochs must be 1..20")
    root = Path(repo).resolve(strict=True)
    artifacts = Path(artifact_dir or root / "artifacts" / "causal_world_model").resolve()
    traces, recoveries, coverage = Ledger(db).learning_selection(min_surprise, limit=limit)
    model = (CausalWorldModel.load_if_available(artifacts, model_source="artifact_dir")
             if artifact_dir is not None else None)
    if model is None:
        model = load_repository_model(root)
    experience_bank = ExperienceBank(db)
    surprise_report = experience_bank.ingest_traces(traces)
    experience_report = experience_bank.ingest_verified_recoveries(recoveries)
    experience_report.update(added=surprise_report["added"] + experience_report["added"],
                             surprise_added=surprise_report["added"], recovery_added=experience_report["added"])
    telemetry = prediction_telemetry(model, scope="ledger_repair_attempts", **coverage,
                                     verified_recoveries=[
                                         {"failed_trace_id": i["failed_trace"]["trace_id"],
                                          "passed_trace_id": i["passed_trace_id"]} for i in recoveries])
    summary = {"threshold": min_surprise, "selected_traces": len(traces), "updated": False,
               "world_model_version": model.model_version if model else None,
               "telemetry": telemetry, "experience_bank": experience_report}
    if not traces:
        return {"status": "no_surprises", **summary}
    if model is None:
        return {"status": "model_unavailable", **summary}
    global np
    try:
        import numpy as np
    except ImportError:
        return {"status": "training_dependency_unavailable", **summary}
    from .replay_math import RISK_LABELS, _loss, _serialize_weights, _train_epoch

    anchor_weights = _load_numpy_weights(model.network.weights)
    anchors, calibration_rows = _anchors(root, model, seed)
    if not anchors or not calibration_rows:
        return {"status": "anchor_corpus_unavailable", **summary}
    replay = []
    for trace in traces:
        actual = trace["actual_state_s1"]
        label = actual.get("world_model_outcome") or classify_causal_outcome(actual)
        if label not in STATE_LABELS:
            continue
        action = trace["action_a0"]
        replay.append({"features": _encode(model, {"state": trace["state_s0"], "action": action,
                                                   "context": action.get("context", {})}),
                       "label": label, "reward": float(trace["reward"]), "high_risk": label in RISK_LABELS})
    if not replay:
        return {"status": "no_usable_surprises", **summary}

    rows = anchors + replay * 2
    feature_matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray([STATE_LABELS.index(row["label"]) for row in rows], dtype=np.int64)
    reward_mean, reward_scale = model.network.reward_mean, model.network.reward_scale
    reward_targets = np.asarray([(row["reward"] - reward_mean) / reward_scale for row in rows], dtype=np.float64)
    risk_targets = np.asarray([row["high_risk"] for row in rows], dtype=np.float64)
    class_counts = np.bincount(labels, minlength=len(STATE_LABELS))
    replay_risk_count = int(risk_targets.sum())
    risk_positive_weight = min(64.0, (len(rows) - replay_risk_count) / max(1, replay_risk_count)) if replay_risk_count else 1.0
    present = max(1, int(np.count_nonzero(class_counts)))
    class_weights = np.ones(len(STATE_LABELS), dtype=np.float64)
    for index, count in enumerate(class_counts):
        if count:
            class_weights[index] = min(8.0, len(rows) / (present * count))
    before_loss = _loss(feature_matrix, labels, reward_targets, risk_targets,
                        anchor_weights, class_weights, risk_positive_weight)
    validation_x = np.asarray([row["features"] for row in calibration_rows], dtype=np.float64)
    validation_labels = np.asarray([STATE_LABELS.index(row["label"]) for row in calibration_rows], dtype=np.int64)
    validation_rewards = np.asarray([(row["reward"] - reward_mean) / reward_scale for row in calibration_rows])
    validation_risks = np.asarray([row["high_risk"] for row in calibration_rows], dtype=np.float64)
    validation_before = _loss(validation_x, validation_labels, validation_rewards,
                              validation_risks, anchor_weights, class_weights,
                              risk_positive_weight)
    weights = {name: value.copy() for name, value in anchor_weights.items()}
    optimizer = [{key: np.zeros_like(value) for key, value in weights.items()},
                 {key: np.zeros_like(value) for key, value in weights.items()}, 0]
    np.random.seed(seed)
    history = []
    best_weights = None
    best_loss = before_loss
    best_validation_loss = validation_before
    best_epoch = None
    anchor_tolerance = 0.02
    retained_displacement = 0.05
    for epoch in range(1, epochs + 1):
        _train_epoch(feature_matrix, labels, reward_targets, risk_targets, weights, optimizer,
                     class_weights, batch_size=32, learning_rate=0.0008,
                     risk_positive_weight=risk_positive_weight)
        # Proximal anchor: constrain each epoch's displacement toward the
        # original checkpoint so small surprise batches cannot erase it.
        for name in weights:
            weights[name] = anchor_weights[name] + retained_displacement * (weights[name] - anchor_weights[name])
        replay_loss = _loss(feature_matrix, labels, reward_targets, risk_targets,
                            weights, class_weights, risk_positive_weight)
        validation_loss = _loss(validation_x, validation_labels, validation_rewards,
                                validation_risks, weights, class_weights,
                                risk_positive_weight)
        accepted = (validation_loss <= validation_before + anchor_tolerance
                    and replay_loss < best_loss - 1e-8)
        history.append({"epoch": epoch, "replay_loss": replay_loss,
                        "anchor_validation_loss": validation_loss, "accepted": accepted})
        if accepted:
            best_loss = replay_loss
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_weights = {name: value.copy() for name, value in weights.items()}
    if best_weights is None:
        report = {**summary, "schema_version": "aegis.online_replay_report.v1",
                  "status": "anchor_protection_rejected", "replay_examples": len(replay),
                  "previous_world_model_version": model.model_version,
                  "world_model_version": model.model_version,
                  "replay_loss_before": before_loss, "anchor_validation_loss_before": validation_before,
                  "anchor_tolerance": anchor_tolerance, "history": history}
        _atomic_bytes(artifacts / "online_replay_report.json",
                      (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
        return report
    weights = best_weights
    after_loss = best_loss
    validation_after = best_validation_loss
    new_payload = _serialize_weights(weights, model.encoder.input_size, reward_mean,
                                     reward_scale, model.network.reward_interval_radius,
                                     model.network.reward_interval_radii, seed)
    weight_bytes = (json.dumps(new_payload, ensure_ascii=False, separators=(",", ":"),
                               allow_nan=False) + "\n").encode("utf-8")
    weight_sha = hashlib.sha256(weight_bytes).hexdigest()
    version = new_payload["model_version"]
    telemetry.update(initial_world_model_version=model.model_version, world_model_version=version)
    report = {
        **summary,
        "schema_version": "aegis.online_replay_report.v1",
        "status": "updated",
        "threshold": float(min_surprise),
        "selected_traces": len(traces),
        "replay_examples": len(replay),
        "anchor_examples": len(anchors),
        "anchor_policy": "up to 24 ticket/template-disjoint training anchors per state; two replay copies per surprise",
        "regularization": {"kind": "proximal_to_previous_weights",
                            "displacement_retained_per_epoch": retained_displacement,
                            "anchor_validation_tolerance": anchor_tolerance,
                            "selected_epoch": best_epoch},
        "trace_ids": [trace["trace_id"] for trace in traces],
        "ticket_ids": sorted({trace["ticket_id"] for trace in traces}),
        "surprise_by_outcome": {label: sum(row["label"] == label for row in replay) for label in STATE_LABELS},
        "replay_loss_before": before_loss,
        "replay_loss_after": after_loss,
        "anchor_validation_loss_before": validation_before,
        "anchor_validation_loss_after": validation_after,
        "history": history,
        "previous_world_model_version": model.model_version,
        "world_model_version": version,
        "weights_sha256": weight_sha,
        "experience_bank": experience_report,
    }
    version_path = artifacts / "versions" / f"{version}.json"
    encoder_bytes = (json.dumps(model.encoder.to_dict(), ensure_ascii=False, separators=(",", ":"),
                                allow_nan=False) + "\n").encode("utf-8")
    _atomic_bytes(artifacts / "feature_encoder.json", encoder_bytes)
    _atomic_bytes(version_path, weight_bytes)
    _atomic_bytes(artifacts / "model_weights.json", weight_bytes)
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **weights,
                        reward_mean=np.asarray([reward_mean]), reward_scale=np.asarray([reward_scale]))
    _atomic_bytes(artifacts / "model_weights.npz", npz_buffer.getvalue())
    _atomic_bytes(artifacts / "online_replay_report.json",
                  (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    metrics_path = artifacts / "training_metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["online_replay"] = report
        metrics.setdefault("training", {})["weights_sha256"] = weight_sha
        _atomic_bytes(metrics_path, (json.dumps(metrics, ensure_ascii=False, indent=2,
                                                allow_nan=False) + "\n").encode())
    return report
