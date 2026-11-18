"""Numerical routines shared by offline training and optional online replay."""

import hashlib
import json

import numpy as np

from .world_model import STATE_LABELS

RISK_LABELS = ("cheat_detected", "replay_blocked")


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


def _loss(x, labels, reward_targets, risk_targets, weights, class_weights, risk_positive_weight=1.0):
    _, probabilities, reward, risk_logits, _ = _forward(x, weights)
    sample_weights = class_weights[labels]
    denom = max(float(sample_weights.sum()), 1e-12)
    state_loss = float((-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)) * sample_weights).sum() / denom)
    reward_loss = float(np.mean((reward - reward_targets) ** 2))
    risk_weights = np.where(risk_targets > 0.5, risk_positive_weight, 1.0)
    risk_denominator = max(float(risk_weights.sum()), 1e-12)
    risk_loss = float(((np.logaddexp(0, risk_logits) - risk_targets * risk_logits) * risk_weights).sum()
                      / risk_denominator)
    return state_loss + 0.5 * reward_loss + 0.5 * risk_loss


def _train_epoch(x, labels, reward_targets, risk_targets, weights, optimizer, class_weights,
                 *, batch_size=64, learning_rate=0.005, risk_positive_weight=1.0):
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
        risk_sample_weights = np.where(kb > 0.5, risk_positive_weight, 1.0)
        risk_denominator = max(float(risk_sample_weights.sum()), 1e-12)
        d_risk = (0.5 / risk_denominator) * (risk - kb)[:, None] * risk_sample_weights[:, None]
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


def _serialize_weights(weights, input_size, reward_mean, reward_scale, reward_interval_radius,
                       reward_interval_radii, version_seed):
    payload = {
        "schema_version": "aegis.world_model.weights.v1",
        "state_labels": list(STATE_LABELS),
        "input_size": int(input_size),
        "hidden_size": int(weights["W1"].shape[1]),
        "reward_mean": float(reward_mean),
        "reward_scale": float(reward_scale),
        "reward_interval_radius": float(reward_interval_radius),
        "reward_interval_radii": {label: float(radius) for label, radius in reward_interval_radii.items()},
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
