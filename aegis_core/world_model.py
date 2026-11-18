"""Small, portable causal outcome model used by the Aegis control tower.

Training uses NumPy in ``scripts/train_and_eval_world_model.py``. Inference is
stdlib-only so installing the model does not add a heavyweight runtime stack to
the core package.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


STATE_LABELS = (
    "gate_passed",
    "assertion_error",
    "cheat_detected",
    "replay_blocked",
    "missing_constraint",
)
PREBLOCK_REWARD_INTERVALS = {
    # These are the bounded outcomes from the deterministic reward contract.
    # Keep them separate from the learned regression head: a hard policy block
    # is not a successful repair, regardless of lexical reward correlations.
    # The current reward contract can emit -2.0 for an immediate pre-block;
    # retain that endpoint while covering the full negative policy penalty.
    "cheat_detected": (-2.0, -1.30),
    "replay_blocked": (-2.0, -1.30),
}


def preblock_reward_distribution(outcome: str) -> tuple[float, list[float]]:
    interval = PREBLOCK_REWARD_INTERVALS.get(outcome)
    if interval is None:
        raise ValueError("no deterministic reward profile for outcome")
    return sum(interval) / 2.0, [float(interval[0]), float(interval[1])]

TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")
PATCH_PATH_RE = re.compile(r"^(?:\+\+\+ b/|--- a/)([^\t\r\n]+)", re.MULTILINE)
NUMERIC_FEATURES = (
    "text_token_count",
    "text_char_count",
    "patch_char_count",
    "patch_line_count",
    "added_line_count",
    "removed_line_count",
    "test_path_count",
    "assertion_edit_count",
    "prior_attempt_count",
    "prior_patch_count",
    "ast_item_count",
    "ast_source_char_count",
    "target_file_count",
    "patch_hunk_count",
    "added_numeric_mean",
    "removed_numeric_mean",
    "numeric_mean_delta",
    "added_negation_count",
    "removed_negation_count",
    "added_true_count",
    "added_false_count",
    "has_patch",
    "has_contract",
    "has_ast_context",
    "has_repeated_patch",
)


def _json_text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _extract_patch(action) -> str:
    if isinstance(action, str):
        try:
            decoded = json.loads(action)
        except (json.JSONDecodeError, TypeError):
            return action
        return _extract_patch(decoded) if decoded is not action else action
    if isinstance(action, dict):
        for key in ("patch", "completion"):
            candidate = action.get(key)
            if isinstance(candidate, str):
                if key == "completion":
                    try:
                        candidate = json.loads(candidate).get("patch", candidate)
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass
                if isinstance(candidate, str):
                    return candidate
    return ""


def _record_text(state, action, context) -> str:
    return "\n".join(_json_text(value) for value in (state, action, context))[:50000]


def _record_tokens(state, action, context) -> list[str]:
    state_text = _json_text(state)
    action_text = _json_text(action)
    context_text = _json_text(context)
    tokens = []
    for prefix, text in (("s0_", state_text), ("a0_", action_text), ("ast_", context_text)):
        tokens.extend(prefix + token for token in TOKEN_RE.findall(text.lower()))
    patch = _extract_patch(action)
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            tokens.extend("add_" + token for token in TOKEN_RE.findall(line[1:].lower()))
        elif line.startswith("-") and not line.startswith("---"):
            tokens.extend("del_" + token for token in TOKEN_RE.findall(line[1:].lower()))
    return tokens


def _ast_counts(context) -> tuple[int, int]:
    if not isinstance(context, dict):
        return 0, 0
    items = context.get("items", [])
    if not isinstance(items, list):
        items = []
    source_chars = sum(len(item.get("source", "")) for item in items if isinstance(item, dict))
    return len(items), source_chars


def _prior_patch_hashes(prior_attempts) -> set[str]:
    hashes = set()
    if not isinstance(prior_attempts, (list, tuple)):
        return hashes
    for prior in prior_attempts:
        if not isinstance(prior, dict):
            continue
        digest = prior.get("patch_sha256")
        if isinstance(digest, str):
            hashes.add(digest)
        action = prior.get("action_a0", prior.get("action", {}))
        if isinstance(action, dict):
            digest = action.get("patch_sha256")
            if isinstance(digest, str):
                hashes.add(digest)
            previous_patch = _extract_patch(action)
            if previous_patch:
                hashes.add(hashlib.sha256(previous_patch.encode("utf-8")).hexdigest())
        previous_patch = _extract_patch(prior)
        if previous_patch:
            hashes.add(hashlib.sha256(previous_patch.encode("utf-8")).hexdigest())
    return hashes


def _prior_patch_counts(prior_attempts) -> Counter:
    counts = Counter()
    if not isinstance(prior_attempts, (list, tuple)):
        return counts
    for prior in prior_attempts:
        if not isinstance(prior, dict):
            continue
        action = prior.get("action_a0", prior.get("action", {}))
        digest = prior.get("patch_sha256")
        if isinstance(action, dict):
            digest = action.get("patch_sha256", digest)
            previous_patch = _extract_patch(action)
            if previous_patch:
                digest = hashlib.sha256(previous_patch.encode("utf-8")).hexdigest()
        if not isinstance(digest, str):
            previous_patch = _extract_patch(prior)
            if previous_patch:
                digest = hashlib.sha256(previous_patch.encode("utf-8")).hexdigest()
        if isinstance(digest, str):
            counts[digest] += 1
    return counts


def preblock_reason(action, prior_attempts=()) -> tuple[str | None, list[str]]:
    """Detect exact policy violations before creating an execution worktree."""
    patch = _extract_patch(action)
    paths = sorted({path.replace("\\", "/") for path in PATCH_PATH_RE.findall(patch)
                    if path != "/dev/null"})
    for path in paths:
        parts = path.split("/")
        from .ast_slicer import is_protected_test_path
        if is_protected_test_path(path):
            return "test_modification", paths
        if ".." in parts or path.startswith("/"):
            return "out_of_scope_path", paths
    if patch:
        digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        if _prior_patch_counts(prior_attempts)[digest] >= 2:
            return "repeated_patch", paths
    if isinstance(action, dict) and action.get("repeated_patch") is True:
        return "repeated_patch", paths
    return None, paths


def _numeric_values(state, action, context, prior_attempts=()) -> dict[str, float]:
    patch = _extract_patch(action)
    text = _record_text(state, action, context)
    tokens = TOKEN_RE.findall(text.lower())
    patch_lines = patch.splitlines()
    added = [line for line in patch_lines if line.startswith("+") and not line.startswith("+++")]
    removed = [line for line in patch_lines if line.startswith("-") and not line.startswith("---")]
    number_re = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")
    added_numbers = [float(value) for line in added for value in number_re.findall(line)]
    removed_numbers = [float(value) for line in removed for value in number_re.findall(line)]
    _, paths = preblock_reason(action, prior_attempts)
    test_paths = [path for path in paths if ("tests" in path.lower().split("/")
                  or path.lower().split("/")[-1].startswith("test_")
                  or path.lower().split("/")[-1].endswith("_test.py"))]
    ast_items, ast_chars = _ast_counts(context)
    targets = []
    if isinstance(action, dict):
        contract = action.get("contract") or {}
        if not isinstance(contract, dict):
            contract = {}
        targets = action.get("target_files", []) or contract.get("target_files", [])
    if not isinstance(targets, list):
        targets = []
    prior_hashes = _prior_patch_counts(prior_attempts)
    digest = hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else ""
    repeated = prior_hashes[digest] >= 2 or (isinstance(action, dict) and action.get("repeated_patch") is True)
    return {
        "text_token_count": math.log1p(len(tokens)),
        "text_char_count": math.log1p(len(text)),
        "patch_char_count": math.log1p(len(patch)),
        "patch_line_count": math.log1p(len(patch_lines)),
        "added_line_count": math.log1p(len(added)),
        "removed_line_count": math.log1p(len(removed)),
        "test_path_count": float(len(test_paths)),
        "assertion_edit_count": math.log1p(sum("assert " in line for line in added + removed)),
        "prior_attempt_count": math.log1p(len(prior_attempts) if isinstance(prior_attempts, (list, tuple)) else 0),
        "prior_patch_count": math.log1p(len(prior_hashes)),
        "ast_item_count": math.log1p(ast_items),
        "ast_source_char_count": math.log1p(ast_chars),
        "target_file_count": math.log1p(len(targets)),
        "patch_hunk_count": math.log1p(sum(line.startswith("@@") for line in patch_lines)),
        "added_numeric_mean": math.log1p(statistics.fmean(added_numbers)) if added_numbers else 0.0,
        "removed_numeric_mean": math.log1p(statistics.fmean(removed_numbers)) if removed_numbers else 0.0,
        "numeric_mean_delta": math.copysign(math.log1p(abs(statistics.fmean(added_numbers) - statistics.fmean(removed_numbers))),
                                             statistics.fmean(added_numbers) - statistics.fmean(removed_numbers))
                              if added_numbers and removed_numbers and statistics.fmean(added_numbers) != statistics.fmean(removed_numbers) else 0.0,
        "added_negation_count": math.log1p(sum(len(re.findall(r"(?<![=<>!])-(?=\s*[A-Za-z_(0-9])", line)) for line in added)),
        "removed_negation_count": math.log1p(sum(len(re.findall(r"(?<![=<>!])-(?=\s*[A-Za-z_(0-9])", line)) for line in removed)),
        "added_true_count": float(sum(bool(re.search(r"\btrue\b", line, re.I)) for line in added)),
        "added_false_count": float(sum(bool(re.search(r"\bfalse\b", line, re.I)) for line in added)),
        "has_patch": float(bool(patch)),
        "has_contract": float(isinstance(action, dict) and bool(action.get("contract"))),
        "has_ast_context": float(ast_items > 0 or ast_chars > 0),
        "has_repeated_patch": float(bool(repeated)),
    }


class FeatureEncoder:
    """Training-only vocabulary and numeric normalization for causal inputs."""

    def __init__(self, vocabulary=None, numeric_mean=None, numeric_scale=None, *, max_vocab=384):
        self.vocabulary = list(vocabulary or [])
        self.numeric_mean = list(numeric_mean or [0.0] * len(NUMERIC_FEATURES))
        self.numeric_scale = list(numeric_scale or [1.0] * len(NUMERIC_FEATURES))
        self.max_vocab = int(max_vocab)
        self._indices = {token: index for index, token in enumerate(self.vocabulary)}

    @property
    def input_size(self):
        return len(self.vocabulary) + len(NUMERIC_FEATURES)

    def fit(self, records):
        counts = Counter()
        numeric_rows = []
        for record in records:
            counts.update(_record_tokens(record["state"], record["action"], record.get("context")))
            numeric_rows.append(_numeric_values(record["state"], record["action"],
                                                record.get("context"), record.get("prior_attempts", [])))
        self.vocabulary = [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:self.max_vocab]]
        self._indices = {token: index for index, token in enumerate(self.vocabulary)}
        self.numeric_mean = []
        self.numeric_scale = []
        for name in NUMERIC_FEATURES:
            values = [row[name] for row in numeric_rows]
            mean = sum(values) / max(1, len(values))
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
            self.numeric_mean.append(mean)
            self.numeric_scale.append(math.sqrt(variance) or 1.0)
        return self

    def encode(self, state, action, context=None, prior_attempts=()) -> list[float]:
        text = _record_text(state, action, context)
        tokens = _record_tokens(state, action, context)
        token_counts = Counter(tokens)
        scale = math.sqrt(max(1, len(tokens)))
        features = [0.0] * self.input_size
        for token, count in token_counts.items():
            index = self._indices.get(token)
            if index is not None:
                features[index] = count / scale
        numeric = _numeric_values(state, action, context, prior_attempts)
        offset = len(self.vocabulary)
        for index, name in enumerate(NUMERIC_FEATURES):
            normalized = (numeric[name] - self.numeric_mean[index]) / self.numeric_scale[index]
            features[offset + index] = max(-8.0, min(8.0, normalized))
        return features

    def to_dict(self):
        return {
            "schema_version": "aegis.world_model.encoder.v1",
            "max_vocab": self.max_vocab,
            "vocabulary": self.vocabulary,
            "numeric_features": list(NUMERIC_FEATURES),
            "numeric_mean": self.numeric_mean,
            "numeric_scale": self.numeric_scale,
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != "aegis.world_model.encoder.v1":
            raise ValueError("unsupported world-model encoder")
        if payload.get("numeric_features") != list(NUMERIC_FEATURES):
            raise ValueError("world-model numeric feature schema mismatch")
        return cls(payload["vocabulary"], payload["numeric_mean"], payload["numeric_scale"],
                   max_vocab=payload.get("max_vocab", 384))


class CausalWorldNet:
    """Portable forward pass for a shared hidden layer and three task heads."""

    def __init__(self, weights):
        self.weights = weights["weights"]
        self.state_labels = tuple(weights.get("state_labels", STATE_LABELS))
        self.reward_mean = float(weights.get("reward_mean", 0.0))
        self.reward_scale = float(weights.get("reward_scale", 1.0)) or 1.0
        self.reward_interval_radius = max(0.0, float(weights.get("reward_interval_radius", 0.0)))
        self.reward_interval_radii = {
            str(label): max(0.0, float(radius))
            for label, radius in weights.get("reward_interval_radii", {}).items()
        }
        self.input_size = int(weights["input_size"])

    def predict(self, features):
        if len(features) != self.input_size:
            raise ValueError("world-model feature size mismatch")
        w = self.weights
        hidden = []
        for column, bias in zip(zip(*w["W1"]), w["b1"]):
            hidden.append(math.tanh(bias + sum(value * weight for value, weight in zip(features, column))))
        state_logits = [bias + sum(value * weight for value, weight in zip(hidden, column))
                        for column, bias in zip(zip(*w["Ws"]), w["bs"])]
        peak = max(state_logits)
        exp = [math.exp(max(-60.0, min(60.0, value - peak))) for value in state_logits]
        total = sum(exp) or 1.0
        probabilities = [value / total for value in exp]
        reward_scaled = w["br"][0] + sum(value * weight for value, weight in zip(hidden, w["Wr"]))
        reward = max(-2.0, min(1.5, reward_scaled * self.reward_scale + self.reward_mean))
        risk_logit = w["brisk"][0] + sum(value * weight for value, weight in zip(hidden, w["Wrisk"]))
        risk_probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, risk_logit))))
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        radius = self.reward_interval_radii.get(self.state_labels[best], self.reward_interval_radius)
        reward_interval = [max(-2.0, reward - radius), min(1.5, reward + radius)]
        return {
            "outcome": self.state_labels[best],
            "probabilities": {label: probabilities[index] for index, label in enumerate(self.state_labels)},
            "confidence": probabilities[best],
            "predicted_reward": reward,
            "predicted_reward_interval": reward_interval,
            "risk_probability": risk_probability,
        }


class CausalWorldModel:
    """Load trained artifacts and predict one state transition before execution."""

    def __init__(self, encoder: FeatureEncoder, network: CausalWorldNet, *, model_version: str):
        self.encoder = encoder
        self.network = network
        self.model_version = model_version

    @classmethod
    def load(cls, artifact_dir: str | Path):
        path = Path(artifact_dir)
        encoder_payload = json.loads((path / "feature_encoder.json").read_text(encoding="utf-8"))
        weight_payload = json.loads((path / "model_weights.json").read_text(encoding="utf-8"))
        if weight_payload.get("schema_version") != "aegis.world_model.weights.v1":
            raise ValueError("unsupported world-model weights")
        encoder = FeatureEncoder.from_dict(encoder_payload)
        network = CausalWorldNet(weight_payload)
        if encoder.input_size != network.input_size:
            raise ValueError("world-model artifact dimensions disagree")
        version = weight_payload.get("model_version") or hashlib.sha256(
            json.dumps(weight_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(encoder, network, model_version=version)

    @classmethod
    def load_if_available(cls, artifact_dir: str | Path):
        path = Path(artifact_dir)
        if not (path / "model_weights.json").is_file():
            return None
        return cls.load(path)

    def predict(self, state, action, context=None, prior_attempts=()) -> dict:
        features = self.encoder.encode(state, action, context, prior_attempts)
        prediction = self.network.predict(features)
        reason, paths = preblock_reason(action, prior_attempts)
        outcome = prediction["outcome"]
        confidence = prediction["confidence"]
        risk_probability = prediction["risk_probability"]
        source = "causal_world_net"
        probabilities = prediction["probabilities"]
        predicted_reward = prediction["predicted_reward"]
        reward_interval = prediction["predicted_reward_interval"]
        if reason:
            outcome = "replay_blocked" if reason == "repeated_patch" else "cheat_detected"
            confidence = 1.0
            risk_probability = 1.0
            source = "deterministic_pre_block"
            probabilities = {label: float(label == outcome) for label in STATE_LABELS}
            predicted_reward, reward_interval = preblock_reward_distribution(outcome)
        return {
            "predicted_next_state": {
                "outcome": outcome,
                "probabilities": probabilities,
                "decision_source": source,
                "high_risk_pre_block": bool(reason),
                "pre_block_reason": reason,
                "changed_paths": paths,
            },
            "predicted_reward": predicted_reward,
            "predicted_reward_interval": reward_interval,
            "prediction_confidence": confidence,
            "risk_probability": risk_probability,
            "model_version": self.model_version,
        }


def load_repository_model(root):
    """Load the repository model, honoring the attached-repository override."""
    import os

    path = Path(os.environ.get("AEGIS_WORLD_MODEL_DIR", Path(root) / "artifacts" / "causal_world_model"))
    return CausalWorldModel.load_if_available(path.resolve())
