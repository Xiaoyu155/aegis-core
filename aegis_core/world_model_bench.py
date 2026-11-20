"""Offline calibration gate for moving the causal world model to enforce mode."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    group_id: str
    label: str
    risk_probability: float
    cost_units: float = 1.0


def _finite_probability(value, name="risk_probability") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return float(value)


def _sample(sample) -> CalibrationSample:
    if isinstance(sample, CalibrationSample):
        item = sample
    elif isinstance(sample, dict):
        label = str(sample.get("label", ""))
        item = CalibrationSample(str(sample.get("sample_id", "")), str(sample.get("group_id", "")), label,
                                 sample.get("risk_probability"), sample.get("cost_units", 1.0))
    else:
        raise TypeError("calibration samples must be CalibrationSample or mappings")
    if not item.sample_id or not item.group_id:
        raise ValueError("calibration samples require sample_id and group_id")
    if item.label not in {"positive", "negative"}:
        raise ValueError("calibration sample label must be positive or negative")
    score = _finite_probability(item.risk_probability)
    if isinstance(item.cost_units, bool) or not isinstance(item.cost_units, (int, float)) or not math.isfinite(item.cost_units) or item.cost_units <= 0:
        raise ValueError("cost_units must be a positive finite number")
    return CalibrationSample(item.sample_id, item.group_id, item.label, score, float(item.cost_units))


def split_validation(samples: Iterable[CalibrationSample | dict], *, validation_fraction: float = 0.2):
    """Deterministically hold out whole groups so paired samples never leak."""
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rows = [_sample(item) for item in samples]
    groups = sorted({item.group_id for item in rows})
    if len(groups) < 2:
        raise ValueError("at least two independent groups are required for a held-out validation split")
    cutoff = int(validation_fraction * 1_000_000)
    validation_groups = {group for group in groups
                         if int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:12], 16) % 1_000_000 < cutoff}
    target = max(1, round(len(groups) * validation_fraction))
    if not validation_groups:
        validation_groups = {sorted(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())[0]}
    if len(validation_groups) >= len(groups):
        validation_groups.remove(sorted(validation_groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())[-1])
    if len(validation_groups) < target and len(groups) > target:
        ordered = sorted(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())
        validation_groups.update(ordered[:target])
    validation = [item for item in rows if item.group_id in validation_groups]
    training = [item for item in rows if item.group_id not in validation_groups]
    return training, validation


def evaluate_threshold(samples: Iterable[CalibrationSample | dict], threshold: float) -> dict:
    """Measure wrong blocks, missed risky patches, recall, and avoided work."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1")
    rows = [_sample(item) for item in samples]
    positives = [item for item in rows if item.label == "positive"]
    negatives = [item for item in rows if item.label == "negative"]
    if not positives or not negatives:
        raise ValueError("validation data must include positive and negative examples")
    blocked = [item for item in rows if item.risk_probability >= threshold]
    false_positive = [item for item in blocked if item.label == "positive"]
    missed_negative = [item for item in negatives if item.risk_probability < threshold]
    total_cost = sum(item.cost_units for item in rows)
    blocked_cost = sum(item.cost_units for item in blocked)
    safely_blocked_cost = sum(item.cost_units for item in blocked if item.label == "negative")
    return {
        "threshold": float(threshold),
        "samples": len(rows),
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "false_positive_blocks": len(false_positive),
        "false_positive_block_rate": len(false_positive) / len(positives),
        "false_negative_count": len(missed_negative),
        "false_negative_rate": len(missed_negative) / len(negatives),
        "recall": (len(negatives) - len(missed_negative)) / len(negatives),
        "blocked_candidates": len(blocked),
        "compute_savings_rate": blocked_cost / total_cost,
        "safe_negative_compute_savings_rate": safely_blocked_cost / total_cost,
        "cost_units_total": total_cost,
    }


def threshold_candidates(samples: Iterable[CalibrationSample | dict]) -> list[float]:
    """Return every observed score boundary plus the endpoints."""
    rows = [_sample(item) for item in samples]
    candidates = {0.0, 1.0}
    for item in rows:
        candidates.add(item.risk_probability)
        if item.risk_probability < 1:
            candidates.add(math.nextafter(item.risk_probability, 1.0))
    return sorted(candidates)


def _sign_permit(report: dict, signing_key: str | bytes | None) -> dict:
    report.pop("signature", None)
    report.pop("signature_algorithm", None)
    if not report.get("gate", {}).get("allowed_mode_transition"):
        return report
    key_bytes = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
    if not isinstance(key_bytes, bytes) or len(key_bytes) < 16:
        return report
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    report["signature"] = hmac.new(key_bytes, canonical, hashlib.sha256).hexdigest()
    report["signature_algorithm"] = "HMAC-SHA256"
    return report


def recommend_threshold(samples: Iterable[CalibrationSample | dict], *, model_version: str,
                        signing_key: str | bytes | None = None,
                        thresholds: Iterable[float] | None = None) -> dict:
    """Choose the highest-recall safe threshold and sign an enforce permit."""
    rows = [_sample(item) for item in samples]
    if not model_version:
        raise ValueError("model_version is required")
    candidates = sorted(set(_finite_probability(item, "threshold") for item in
                            (thresholds if thresholds is not None else threshold_candidates(rows))))
    if not candidates:
        raise ValueError("at least one threshold is required")
    metrics = [evaluate_threshold(rows, threshold) for threshold in candidates]
    eligible = [item for item in metrics if item["false_positive_block_rate"] < 0.01 and item["recall"] >= 0.85]
    chosen = max(eligible, key=lambda item: (item["recall"], item["safe_negative_compute_savings_rate"], item["threshold"])) if eligible else None
    metric_gate = chosen is not None
    key_bytes = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
    signer_available = isinstance(key_bytes, bytes) and len(key_bytes) >= 16
    allowed = bool(metric_gate and signer_available)
    any_fp_safe = any(item["false_positive_block_rate"] < 0.01 for item in metrics)
    any_recall_safe = any(item["recall"] >= 0.85 for item in metrics)
    report = {
        "schema_version": "aegis.world_model_calibration.v1",
        "model_version": model_version,
        "source": "held_out_validation",
        "threshold_recommendation": chosen["threshold"] if chosen else None,
        "validation_metrics": chosen,
        "threshold_sweep": metrics,
        "gate": {"false_positive_block_rate_below_1_percent": any_fp_safe,
                 "recall_at_least_85_percent": any_recall_safe,
                 "joint_threshold_passed": metric_gate,
                 "signing_key_available": signer_available,
                 "allowed_mode_transition": allowed,
                 "from_mode": "observe", "to_mode": "enforce",
                 "status": "permit" if allowed else "blocked",
                 "reason": ("calibration thresholds satisfied" if metric_gate else
                            "no single threshold jointly satisfies false-positive and recall requirements") if signer_available else
                           ("signing key unavailable or shorter than 16 bytes" if metric_gate else
                            "no single safe threshold and no signing key configured")},
    }
    return _sign_permit(report, signing_key)


def verify_permit(report: dict, signing_key: str | bytes) -> bool:
    if not isinstance(report, dict) or report.get("signature_algorithm") != "HMAC-SHA256":
        return False
    key_bytes = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
    if not isinstance(key_bytes, bytes) or len(key_bytes) < 16:
        return False
    signature = report.get("signature")
    unsigned = {key: value for key, value in report.items() if key not in {"signature", "signature_algorithm"}}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    expected = hmac.new(key_bytes, canonical, hashlib.sha256).hexdigest()
    return bool(report.get("gate", {}).get("allowed_mode_transition") and hmac.compare_digest(signature or "", expected))


def _model_risk(model, *, patch: str, prompt: str, scenario: str) -> float:
    prediction = model.predict(
        {"status": "running", "scenario": scenario},
        {"op": "repair", "patch": patch},
        {"prompt": prompt, "benchmark": "tournament_or_causal_trace"},
    )
    return _finite_probability(prediction.get("risk_probability"))


def load_tournament_samples(path: str | Path, model) -> list[CalibrationSample]:
    samples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid tournament JSON on line {line_number}: {error.msg}") from error
        if not isinstance(row, dict) or not isinstance(row.get("chosen_patch"), str) or not isinstance(row.get("rejected_patch"), str):
            raise ValueError(f"tournament row {line_number} lacks chosen/rejected patches")
        group = str(row.get("pair_sha256") or row.get("prompt") or line_number)
        prompt = str(row.get("prompt", ""))
        chosen_id = f"{group}:chosen"
        rejected_id = f"{group}:rejected"
        samples.append(CalibrationSample(chosen_id, group, "positive",
                                         _model_risk(model, patch=row["chosen_patch"], prompt=prompt, scenario=group),
                                         max(1.0, len(row["chosen_patch"].splitlines()))))
        samples.append(CalibrationSample(rejected_id, group, "negative",
                                         _model_risk(model, patch=row["rejected_patch"], prompt=prompt, scenario=group),
                                         max(1.0, len(row["rejected_patch"].splitlines()))))
    if not samples:
        raise ValueError("tournament dataset contains no examples")
    return samples


def load_causal_trace_samples(path: str | Path, model) -> list[CalibrationSample]:
    samples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid causal trace JSON on line {line_number}: {error.msg}") from error
        state = row.get("actual_state_s1") or {}
        passed = state.get("after_passed")
        if type(passed) is not bool:
            continue
        action = row.get("action_a0") or {}
        patch = action.get("patch")
        if not isinstance(patch, str):
            continue
        group = str(row.get("group_id") or row.get("ticket_id") or line_number)
        state0 = row.get("state_s0") or {}
        scenario = str(state0.get("scenario") or group)
        prompt = str(row.get("prompt", ""))
        label = "positive" if passed else "negative"
        samples.append(CalibrationSample(str(row.get("trace_id", line_number)), group, label,
                                         _model_risk(model, patch=patch, prompt=prompt, scenario=scenario),
                                         max(1.0, len(patch.splitlines()))))
    return samples


def run_benchmark(root: str | Path = ".", *, model=None, tournament_path: str | Path | None = None,
                  causal_trace_path: str | Path | None = None, validation_fraction: float = 0.2,
                  thresholds: Iterable[float] | None = None, signing_key: str | bytes | None = None) -> dict:
    """Score the shipped held-out evidence and return a signed permit if safe."""
    from .world_model import load_repository_model

    base = Path(root).resolve()
    model = model or load_repository_model(base)
    tournament = Path(tournament_path) if tournament_path else base / "datasets" / "grpo_tournament_pairs.jsonl"
    traces = Path(causal_trace_path) if causal_trace_path else base / "datasets" / "causal_traces_v1.jsonl"
    samples = load_tournament_samples(tournament, model)
    if traces.is_file():
        samples.extend(load_causal_trace_samples(traces, model))
    training, validation = split_validation(samples, validation_fraction=validation_fraction)
    report = recommend_threshold(validation, model_version=model.model_version,
                                 signing_key=signing_key, thresholds=thresholds)
    report["dataset"] = {"tournament_path": tournament.relative_to(base).as_posix() if tournament.is_relative_to(base) else str(tournament),
                         "causal_trace_path": traces.relative_to(base).as_posix() if traces.is_relative_to(base) else str(traces),
                         "total_samples": len(samples), "training_groups": len({item.group_id for item in training}),
                         "validation_groups": len({item.group_id for item in validation}),
                         "validation_samples": len(validation),
                         "validation_fraction": validation_fraction,
                         "cost_estimate_method": "changed patch lines, minimum one cost unit"}
    return _sign_permit(report, signing_key)
