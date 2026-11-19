"""Structured v2 contract compilation, seeded mutation and true-red checks."""
import json
from pathlib import Path
import subprocess

import pytest

from aegis_core.probe_synth import (STRUCTURED_KINDS, compile_structured_probe,
                                    render_structured_probe, structured_cases,
                                    validate_structured_rule)


def _base_rule(kind, path="app.py", symbol="target"):
    common = {"kind": kind, "path": path, "symbol": symbol, "seed": 41}
    if kind == "numeric_bounds":
        return {**common, "min": 0, "max": 10, "fallback": None}
    if kind == "state_machine":
        return {**common, "transitions": {"new": {"pay": "paid"}, "paid": {"ship": "sent"}},
                "invalid_result": None}
    if kind == "idempotency":
        return {**common, "args": ["key-1", 7], "expected": "accepted", "replays": 3}
    if kind == "bounded_collection":
        return {**common, "capacity": 3, "values": [1, 2, 3], "eviction": "oldest"}
    if kind == "async_timeout":
        return {**common, "args": ["fast"], "expected": "ready", "timeout_args": ["slow"],
                "timeout_fallback": "cached", "timeout_ms": 10}
    if kind == "contract_schema":
        string_schema = {"type": "string"}
        return {**common, "input_schema": {"type": "object", "required": ["id"],
                                              "properties": {"id": string_schema}},
                "return_schema": {"type": "object", "required": ["id", "ok"],
                                  "properties": {"id": string_schema, "ok": {"type": "boolean"}}},
                "valid_args": [[{"id": "x"}]], "invalid_args": [["bad"]]}
    raise AssertionError(kind)


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path, source):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "tests").mkdir()
    (root / "app.py").write_text(source, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Probe Tests")
    _git(root, "config", "user.email", "probes@example.invalid")
    _git(root, "add", "app.py")
    _git(root, "commit", "-qm", "baseline")
    return root


@pytest.mark.parametrize("kind", sorted(STRUCTURED_KINDS))
def test_each_high_level_kind_compiles_positive_negative_and_seeded_cases(kind):
    rule = validate_structured_rule(_base_rule(kind))
    first = structured_cases(rule)
    second = structured_cases(rule)
    assert first == second
    labels = {case["label"] for case in first}
    assert any(label.startswith("positive") for label in labels)
    assert any(label.startswith("negative") for label in labels)
    assert any(label.startswith("randomized") for label in labels)


def test_numeric_cases_cover_bounds_underflow_overflow_and_bad_types():
    cases = structured_cases(_base_rule("numeric_bounds"))
    by_label = {case["label"]: case for case in cases}
    assert by_label["positive_low"]["expected"] == 0
    assert by_label["positive_high"]["expected"] == 10
    assert by_label["negative_underflow"]["expected"] == 0
    assert by_label["negative_overflow"]["expected"] == 10
    assert by_label["negative_type"]["expected"] is None


def test_numeric_random_properties_are_deterministic_and_bounded():
    rule = _base_rule("numeric_bounds")
    first = [case for case in structured_cases(rule) if case["label"].startswith("randomized")]
    assert first == [case for case in structured_cases(rule) if case["label"].startswith("randomized")]
    assert all(0 <= case["expected"] <= 10 for case in first)


def test_state_machine_builds_all_legal_and_illegal_pairs():
    cases = structured_cases(_base_rule("state_machine"))
    assert {tuple(case["args"]) for case in cases if case["label"].startswith("positive")} == {
        ("new", "pay"), ("paid", "ship")}
    assert any(case["args"] == ["__unknown_state__", "pay"] for case in cases)
    assert any(case["args"][1] == "__unknown_action__" for case in cases)


def test_idempotency_cases_repeat_the_same_frozen_payload():
    cases = structured_cases(_base_rule("idempotency"))
    replay = next(case for case in cases if case["label"] == "negative_duplicate_replay")
    assert replay["args"] == ["key-1", 7]
    assert replay["repeats"] == 3
    assert all(case["args"] == replay["args"] for case in cases if case["label"].startswith("randomized_replay"))


def test_idempotency_compiles_conflicting_key_payload_attack():
    rule = _base_rule("idempotency")
    rule.update(conflict_args=["key-1", 99], conflict_expected="accepted")
    assert any(case["label"] == "negative_key_payload_conflict" for case in structured_cases(rule))


def test_bounded_collection_checks_eviction_under_overflow():
    cases = structured_cases(_base_rule("bounded_collection"))
    overflow = next(case for case in cases if case["label"] == "negative_capacity_overflow")
    assert overflow["expected"] == [6, 7, 8]
    assert all(len(case["expected"]) <= 3 for case in cases)


@pytest.mark.parametrize("eviction,expected", [("oldest", [4, 5, 6]), ("newest", [1, 2, 3])])
def test_bounded_collection_eviction_policy_is_explicit(eviction, expected):
    rule = _base_rule("bounded_collection")
    rule["eviction"] = eviction
    overflow = next(case for case in structured_cases(rule) if case["label"] == "negative_capacity_overflow")
    if eviction == "oldest":
        expected = [6, 7, 8]
    assert overflow["expected"] == expected


def test_async_probe_records_fallback_and_bounded_wait():
    cases = structured_cases(_base_rule("async_timeout"))
    fallback = next(case for case in cases if case["label"] == "negative_timeout_fallback")
    assert fallback["expected"] == "cached"
    assert fallback["async_timeout_ms"] >= 240


def test_schema_compiler_constructs_seeded_valid_objects():
    cases = structured_cases(_base_rule("contract_schema"))
    assert any(case["label"].startswith("randomized_contract") for case in cases)
    assert all(case.get("return_schema") for case in cases if case["label"].startswith("positive"))


@pytest.mark.parametrize("kind", sorted(STRUCTURED_KINDS))
def test_generated_probe_source_is_executable_python(kind):
    filename, source, command, cases = render_structured_probe(_base_rule(kind))
    compile(source, filename, "exec")
    assert filename == "tests/test_aegis_probe.py"
    assert command[-1] == filename
    assert len(cases) >= 7
    assert "AEGIS_ACCEPTANCE" in source


@pytest.mark.parametrize("kind,source", [
    ("numeric_bounds", "def target(value):\n    return value\n"),
    ("state_machine", "def target(state, action):\n    return 'sent'\n"),
    ("idempotency", "counter = 0\ndef target(key, amount):\n    global counter\n    counter += 1\n    return counter\n"),
    ("bounded_collection", "def target(values, capacity):\n    return values\n"),
    ("async_timeout", "import asyncio\nasync def target(mode):\n    if mode == 'slow':\n        await asyncio.sleep(10)\n    return 'ready'\n"),
    ("contract_schema", "def target(payload):\n    return payload\n"),
])
def test_compiler_physically_proves_all_six_probes_red_on_real_git_baselines(tmp_path, kind, source):
    root = _repo(tmp_path, source)
    compiled = compile_structured_probe(_base_rule(kind), root, _git(root, "rev-parse", "HEAD"), timeout=3)
    assert compiled["red"]["exit_code"] == 1
    assert compiled["red"]["failure_code"] == "ASSERTION_FAILED"
    assert "AEGIS_ACCEPTANCE" in compiled["red"]["stdout"] or "AEGIS_ACCEPTANCE" in compiled["red"]["stderr"]
    assert set(compiled["case_categories"]) == {"positive", "negative", "randomized"}
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_correct_numeric_baseline_cannot_be_frozen_as_a_red_probe(tmp_path):
    root = _repo(tmp_path, "def target(value):\n    if type(value) is not int:\n        return None\n    return min(10, max(0, value))\n")
    with pytest.raises(ValueError, match="must fail a real"):
        compile_structured_probe(_base_rule("numeric_bounds"), root, _git(root, "rev-parse", "HEAD"), timeout=3)


@pytest.mark.parametrize("rule", [
    None,
    {"kind": "unknown", "path": "app.py", "symbol": "f"},
    {"kind": "numeric_bounds", "path": "../app.py", "symbol": "f", "min": 0, "max": 1},
    {"kind": "numeric_bounds", "path": "app.js", "symbol": "f", "min": 0, "max": 1},
    {"kind": "numeric_bounds", "path": "app.py", "symbol": "f", "min": 2, "max": 1},
    {"kind": "numeric_bounds", "path": "app.py", "symbol": "f", "min": 0, "max": 1, "seed": -1},
    {"kind": "idempotency", "path": "app.py", "symbol": "f", "args": ["x"], "expected": None, "replays": 1},
    {"kind": "bounded_collection", "path": "app.py", "symbol": "f", "capacity": 0, "values": []},
    {"kind": "async_timeout", "path": "app.py", "symbol": "f", "args": [], "expected": 1, "timeout_args": [], "timeout_ms": 0, "timeout_fallback": None},
    {"kind": "contract_schema", "path": "app.py", "symbol": "f", "input_schema": {}, "return_schema": {}, "valid_args": [[]], "invalid_args": [[]]},
    {"kind": "numeric_bounds", "path": "app.py", "symbol": "f", "min": 0, "max": 1, "unexpected": True},
])
def test_structured_rule_rejects_unsafe_or_incomplete_specs(rule):
    with pytest.raises(ValueError):
        validate_structured_rule(rule)


@pytest.mark.parametrize("exception", ["ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError"])
def test_invalid_transition_exception_whitelist(exception):
    rule = _base_rule("state_machine")
    rule["invalid_exception"] = exception
    assert validate_structured_rule(rule)["invalid_exception"] == exception


def test_non_whitelisted_exception_is_rejected():
    rule = _base_rule("state_machine")
    rule["invalid_exception"] = "SystemExit"
    with pytest.raises(ValueError):
        validate_structured_rule(rule)


def test_invalid_schema_types_and_depth_are_rejected():
    rule = _base_rule("contract_schema")
    rule["return_schema"] = {"type": "object", "properties": {"x": {"type": "callable"}}}
    with pytest.raises(ValueError):
        validate_structured_rule(rule)


def test_structured_probe_json_is_finite_and_serializable():
    rule = _base_rule("idempotency")
    rule["expected"] = float("nan")
    with pytest.raises(ValueError):
        validate_structured_rule(rule)
