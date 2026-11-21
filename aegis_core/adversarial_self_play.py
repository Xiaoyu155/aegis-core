"""Deterministic red/blue robustness evaluation over the verified v2 release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Callable

from .invariant_guard import inspect_test_assertion_delta
from .security_rules import inspect_javascript_test_assertion_delta
from .security_gate import scan_text


TRANSFORMS = ("rename", "split_lines", "helper_wrapper", "true_condition", "comment_noise")
ROUNDS = 3


def _rename(source: str) -> str:
    pairs = ((r"\bindex\b", "cursor"), (r"\bvalues\b", "items"),
             (r"\bidempotency_key\b", "retry_token"),
             (r"\bidempotencyKey\b", "retryToken"), (r"\bcount\b", "shared_total"),
             (r"\brequestId\b", "requestToken"), (r"\buser_id\b", "subject_id"),
             (r"\buserId\b", "subjectId"), (r"\bcase_(\d+)\b", r"operation_\1"))
    for pattern, replacement in pairs:
        source = re.sub(pattern, replacement, source)
    declaration = re.search(r"\b(?:async\s+def|def|class|function|const|let|var)\s+([A-Za-z_$][\w$]*)", source)
    if declaration:
        original = declaration.group(1)
        renamed = "aegis_" + original
        source = re.sub(rf"\b{re.escape(original)}\b", renamed, source)
    return source


def _split_lines(source: str, language: str) -> str:
    if language == "python":
        # Split a real syntactic boundary using Python's implicit continuation.
        source, count = re.subn(r"(\b(?:async\s+)?def\s+[\w]+)\(([^()]*)\)",
                                lambda match: match.group(1) + "(\n    " +
                                match.group(2).replace(", ", ",\n    ") + "\n)", source, count=1)
        if not count:
            source = "# split-line red-team marker\n" + source
    else:
        source = re.sub(r"\{\s*", "{\n  ", source)
        source = re.sub(r";\s*", ";\n  ", source)
    return source


def _helper_wrapper(source: str, language: str) -> str:
    if language == "python":
        return "def _aegis_dispatch():\n" + "".join("    " + line if line.strip() else line
                                                          for line in source.splitlines(keepends=True))
    return "function _aegisDispatch() {\n" + "\n".join("  " + line for line in source.splitlines()) + "\n}\n"


def _true_condition(source: str, language: str) -> str:
    if language == "python":
        return "if True:\n" + "".join("    " + line if line.strip() else line
                                         for line in source.splitlines(keepends=True))
    return "if (true) {\n" + "\n".join("  " + line for line in source.splitlines()) + "\n}\n"


def _comment_noise(source: str, language: str) -> str:
    if language == "python":
        return "# except: pass; fetch(url); if True: return 200\n" + source
    return "// catch (error) {} fetch(url); if (true) return 200;\n" + source


_TRANSFORM: dict[str, Callable[[str, str], str]] = {
    "rename": lambda source, _language: _rename(source),
    "split_lines": _split_lines,
    "helper_wrapper": _helper_wrapper,
    "true_condition": _true_condition,
    "comment_noise": _comment_noise,
}


def transform_sample(source: str, language: str, transform: str, round_index: int = 1) -> str:
    if language not in {"python", "javascript"}:
        raise ValueError("language must be python or javascript")
    if transform not in _TRANSFORM:
        raise ValueError(f"unknown red-team transform: {transform}")
    # A different order each round prevents accidental dependence on one pass.
    order = TRANSFORMS[round_index % len(TRANSFORMS):] + TRANSFORMS[:round_index % len(TRANSFORMS)]
    current = source
    if transform == "rename":
        return _TRANSFORM[transform](current, language)
    if transform == "split_lines":
        return _TRANSFORM[transform](current, language)
    if transform == "helper_wrapper":
        return _TRANSFORM[transform](current, language)
    if transform == "true_condition":
        return _TRANSFORM[transform](current, language)
    if transform == "comment_noise":
        return _TRANSFORM[transform](current, language)
    for name in order:
        current = _TRANSFORM[name](current, language)
    return current


def _detect(row: dict, source: str) -> bool:
    language = row["language"]
    if row["defect_type"] == "weaken_test_assertion":
        before = row["frozen_test"]
        after = row["rejected_test"]
        if language == "python":
            return not inspect_test_assertion_delta(before, after, path="tests/generated.py")["passed"]
        return not inspect_javascript_test_assertion_delta(before, after, path="tests/generated.js")["passed"]
    suffix = ".py" if language == "python" else ".js"
    expected = _RULE_CODE[row["defect_type"]]
    return any(item["code"] == expected for item in scan_text(source, row["sample_id"] + suffix))


_RULE_CODE = {
    "off_by_one_boundary": "OFF_BY_ONE_BOUNDARY", "swallow_exception": "EXCEPTION_SWALLOWED",
    "remove_timeout_validation": "MISSING_TIMEOUT",
    "break_idempotency_key": "NONDETERMINISTIC_IDEMPOTENCY_KEY",
    "unbounded_memory_growth": "UNBOUNDED_GROWTH",
    "concurrency_lock_loss": "SHARED_STATE_WITHOUT_LOCK", "authorization_bypass": "AUTHORIZATION_BYPASS",
    "failure_as_success": "FAILURE_AS_SUCCESS", "sql_command_injection": "SQL_COMMAND_INJECTION",
    "weaken_test_assertion": "TEST_ASSERTION_ALWAYS_TRUE",
}


def run_self_play(dataset_path: str | Path, *, report_path: str | Path | None = None,
                  regression_dir: str | Path | None = None,
                  baseline_report_path: str | Path | None = None) -> dict:
    dataset = Path(dataset_path)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    negatives = [row for row in rows if row.get("sample_kind") == "hard_negative"]
    positives = [row for row in rows if row.get("sample_kind") == "hard_positive"]
    per_round = []
    first_round_escapes: dict[tuple[str, str, str], dict] = {}
    final_escapes: list[dict] = []
    for round_index in range(ROUNDS):
        attempts = escapes = 0
        class_stats = {name: {"attempts": 0, "detected": 0,
                              "by_language_transform": {f"{language}:{transform}": {"attempts": 0, "detected": 0}
                                                        for language in ("python", "javascript")
                                                        for transform in TRANSFORMS}}
                       for name in _RULE_CODE}
        round_escapes = []
        for row in negatives:
            source = row["rejected_source"]
            for transform in TRANSFORMS:
                candidate = transform_sample(source, row["language"], transform, round_index)
                detected = _detect(row, candidate)
                attempts += 1
                stats = class_stats[row["defect_type"]]
                stats["attempts"] += 1
                stats["detected"] += int(detected)
                sub = stats["by_language_transform"][f"{row['language']}:{transform}"]
                sub["attempts"] += 1
                sub["detected"] += int(detected)
                if not detected:
                    escapes += 1
                    record = {"sample_id": row["sample_id"], "defect_type": row["defect_type"],
                              "language": row["language"], "transform": transform,
                              "round": round_index + 1, "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                              "source": candidate}
                    round_escapes.append(record)
                    if round_index == 0:
                        first_round_escapes[(row["sample_id"], transform, row["language"])] = record
        positive_findings = []
        for row in positives:
            # Valid refactors are evaluated against the original policy signatures.
            findings = scan_text(row["source"], row["sample_id"] + (".py" if row["language"] == "python" else ".js"))
            if findings:
                positive_findings.append({"sample_id": row["sample_id"], "findings": findings})
        per_round.append({"round": round_index + 1, "attempts": attempts, "escapes": escapes,
                          "escape_rate": escapes / attempts if attempts else 0.0,
                          "recall": (attempts - escapes) / attempts if attempts else 0.0,
                          "transforms": [f"{language}:{transform}" for language in ("python", "javascript")
                                         for transform in TRANSFORMS],
                          "per_class": {name: {"attempts": value["attempts"], "detected": value["detected"],
                                              "recall": value["detected"] / value["attempts"] if value["attempts"] else 0.0,
                                              "transform_recall": {key: item["detected"] / item["attempts"]
                                                                    if item["attempts"] else 0.0
                                                                    for key, item in value["by_language_transform"].items()}}
                                        for name, value in class_stats.items()},
                          "hard_positive_count": len(positives), "hard_positive_false_positives": len(positive_findings),
                          "false_positive_rate": len(positive_findings) / len(positives) if positives else 0.0})
        final_escapes = round_escapes
    if regression_dir is None:
        regression_dir = dataset.parent / "evasion_regressions"
    regression_path = Path(regression_dir)
    regression_path.mkdir(parents=True, exist_ok=True)
    # Preserve every observed first-pass evasion, including ones later fixed by a rule update.
    for record in first_round_escapes.values():
        key = hashlib.sha256((record["sample_id"] + record["transform"]).encode()).hexdigest()[:16]
        (regression_path / f"{key}.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    final = per_round[-1]
    all_class_recall = all(row["recall"] >= .92 for row in final["per_class"].values())
    baseline = None
    if baseline_report_path and Path(baseline_report_path).is_file():
        try:
            before = json.loads(Path(baseline_report_path).read_text(encoding="utf-8"))
            prior_rounds = before.get("rounds", [])
            if prior_rounds:
                baseline = {"report": Path(baseline_report_path).as_posix(),
                            "rounds": len(prior_rounds),
                            "escape_rate": prior_rounds[-1].get("escape_rate"),
                            "false_positive_rate": prior_rounds[-1].get("false_positive_rate")}
        except (OSError, json.JSONDecodeError):
            baseline = {"report": Path(baseline_report_path).as_posix(), "status": "invalid"}
    regression_count = len(list(regression_path.glob("*.json")))
    report = {"schema_version": "aegis.adversarial_report.v1", "rounds": per_round,
              "final_false_positive_rate": final["false_positive_rate"],
              "final_recall": final["recall"], "final_per_class_min_recall": min(
                  row["recall"] for row in final["per_class"].values()),
              "final_escape_rate": final["escape_rate"],
              "first_round_regressions_persisted": len(first_round_escapes),
              "permanent_regression_case_count": regression_count,
              "pre_hardening_baseline": baseline,
              "regression_directory": regression_path.as_posix(),
              "converged": (final["false_positive_rate"] == 0.0 and final["recall"] >= .975
                            and all_class_recall and final["escape_rate"] < .03)}
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report
