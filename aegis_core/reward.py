"""Deterministic bounded dense reward; no model-based judge."""
import math


def compute_step_reward(*, baseline_failed: bool, after_passed: bool,
                        target_files: list[str], changed_files: list[str], diff_lines: int,
                        elapsed_sec: float, latency_budget_sec: float,
                        concise_budget_lines: int = 80, cheat_detected: bool = False,
                        repeated_patch: bool = False, timeout_killed: bool = False) -> float:
    for flag in (baseline_failed, after_passed, cheat_detected, repeated_patch, timeout_killed):
        if type(flag) is not bool:
            raise ValueError("gate flags must be bool")
    for number in (elapsed_sec, latency_budget_sec):
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError("finite timing required")
    if elapsed_sec < 0 or latency_budget_sec <= 0 or type(diff_lines) is not int or diff_lines < 0 or type(concise_budget_lines) is not int or concise_budget_lines <= 0:
        raise ValueError("invalid reward budgets")
    from .mediator import canonical_path
    targets = {canonical_path(path) for path in target_files}
    changed = {canonical_path(path) for path in changed_files}
    gate = 1.0 if baseline_failed and after_passed else (-0.5 if baseline_failed else -1.0)
    union = targets | changed
    scope = len(targets & changed) / len(union) if union else 0.0
    concise = max(0.0, 1.0 - diff_lines / concise_budget_lines)
    latency = min(1.0, elapsed_sec / latency_budget_sec)
    penalty = (1.0 if cheat_detected or repeated_patch else 0.0) + (1.25 if timeout_killed else 0.0)
    score = gate + 0.3 * scope + 0.2 * concise - 0.2 * latency - penalty
    if timeout_killed:
        score = min(score, -0.25)
    return max(-2.0, min(1.5, score))
