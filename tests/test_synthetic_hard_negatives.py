"""Mutation operators, evidence extraction, and real red/green subprocess proofs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aegis_core.synthetic_hard_negatives import (
    HardNegativeError,
    MUTATION_TYPES,
    _canonical_line,
    _diff_file,
    _initialize_fixture,
    _message_patch,
    _run_test,
    _safe_relative,
    _sha256_file,
    extract_record_payload,
    generate_release,
    load_official_tokenizer,
    mutate_source,
    mutate_test_assertions,
    tokenizer_length,
    verify_zero_truncation,
)
from aegis_core.invariant_guard import inspect_test_assertion_delta


MUTATION_CASES = [
    ("off_by_one_boundary", "def allowed(value):\n    return value <= 10\n", "return value < 10"),
    ("off_by_one_boundary", "def cap(value):\n    return min(10, value)\n", "min(9, value)"),
    ("swallow_exception", "def read():\n    try:\n        return load()\n    except ValueError:\n        raise\n", "except ValueError:\n        pass"),
    ("remove_timeout_validation", "def run(timeout):\n    if not math.isfinite(timeout):\n        raise ValueError()\n", "if not True"),
    ("remove_timeout_validation", "def run(timeout):\n    if timeout <= 0:\n        raise ValueError()\n", "if False:\n        raise ValueError()"),
    ("break_idempotency_key", "def send():\n    idempotency_key = make_key(request)\n", "idempotency_key = uuid.uuid4().hex"),
    ("break_idempotency_key", "def send():\n    payload['idempotency_key'] = create_key()\n", "payload['idempotency_key'] = uuid.uuid4().hex"),
    ("unbounded_memory_growth", "from collections import deque\nhistory = deque(maxlen=8)\n", "history = deque()"),
    ("unbounded_memory_growth", "def append(history, item):\n    history.append(item)\n    del history[:-128]\n", "history.append(item)"),
    ("weaken_test_assertion", "def test_value():\n    assert value() == 4\n", "assert True"),
    ("weaken_test_assertion", "def test_value(self):\n    self.assertEqual(value(), 4)\n", "assert True"),
]


@pytest.mark.parametrize("mutation_type,source,expected", MUTATION_CASES)
def test_each_hard_negative_operator_changes_the_named_real_source_pattern(mutation_type, source, expected):
    result = mutate_source(source, mutation_type)
    assert result is not None and result["changed"] is True
    assert expected in result["source"]
    assert result["reason"]


@pytest.mark.parametrize("mutation_type", MUTATION_TYPES)
def test_mutation_operators_fail_closed_when_the_source_has_no_matching_defect_site(mutation_type):
    assert mutate_source("def safe(value):\n    return value\n", mutation_type) is None


@pytest.mark.parametrize("source", [
    "def allowed(value):\n    return value <= 5\n",
    "def allowed(value):\n    if value >= 2:\n        return True\n    return False\n",
    "def allowed(value):\n    return min(4, value)\n",
    "def allowed(value):\n    return range(value + 1)\n",
])
def test_boundary_mutations_are_deterministic_and_keep_python_source_parseable(source):
    result = mutate_source(source, "off_by_one_boundary")
    assert result is not None
    compile(result["source"], "<mutation>", "exec")
    assert result == mutate_source(source, "off_by_one_boundary")


@pytest.mark.parametrize("original,candidate,code", [
    ("def test_x():\n    assert calculate() == 8\n", "def test_x():\n    assert True\n", "TEST_ASSERTION_ALWAYS_TRUE"),
    ("def test_x():\n    assert first()\n    assert second()\n", "def test_x():\n    assert first()\n", "TEST_ASSERTION_REMOVED"),
    ("def test_x():\n    assert first() and second()\n", "def test_x():\n    assert first()\n", "TEST_ASSERTION_SIMPLIFIED"),
])
def test_invariant_guard_reports_removed_or_weakened_test_assertions(original, candidate, code):
    report = inspect_test_assertion_delta(original, candidate, path="tests/test_guard.py")
    assert report["passed"] is False
    assert code in {item["code"] for item in report["findings"]}


def test_invariant_guard_accepts_preserved_assertions_and_reports_counts():
    source = "def test_value():\n    assert value() == 4\n"
    report = inspect_test_assertion_delta(source, source)
    assert report["passed"] is True
    assert report["baseline_assertions"] == report["candidate_assertions"] == 1


def test_test_assertion_mutator_enumerates_each_statement_including_multiline_unittest_calls():
    source = ("# 断言文本 assert ignored\n"
              "def test_values():\n"
              "    assert value(\"雪\") == 4\n"
              "    self.assertEqual(\n"
              "        calculate(), 8, msg=\"expected value\"\n"
              "    )\n"
              "    assert other()\n")
    mutations = mutate_test_assertions(source)
    assert len(mutations) == 3
    assert len({item["source"] for item in mutations}) == 3
    for item in mutations:
        proof = inspect_test_assertion_delta(source, item["source"])
        assert proof["passed"] is False
        assert "TEST_ASSERTION_ALWAYS_TRUE" in {finding["code"] for finding in proof["findings"]}
        compile(item["source"], "<assertion-mutation>", "exec")


@pytest.mark.parametrize("record", [
    {"prompt": [{"role": "user", "content": json.dumps({"source": {"a.py": "x=1\n"}, "frozen_test": "assert True"})}]},
    {"prompt": json.dumps({"source": {"b.py": "x=2\n"}, "frozen_test": "assert True"})},
    {"prompt": [{"role": "system", "content": "ignored"},
                 {"role": "user", "content": json.dumps({"source": {"pkg/c.py": "x=3\n"}, "frozen_test": "assert True"})}]},
])
def test_executed_repair_record_payload_is_parsed_from_supported_prompt_shapes(record):
    payload = extract_record_payload(record)
    assert payload is not None
    assert payload["source"]
    assert payload["frozen_test"] == "assert True"


@pytest.mark.parametrize("record", [
    {}, {"prompt": "not-json"}, {"prompt": json.dumps({"source": {}})},
    {"prompt": json.dumps({"source": {"../escape.py": "bad"}, "frozen_test": "assert True"})},
])
def test_invalid_or_incomplete_record_payload_is_not_admitted(record):
    payload = extract_record_payload(record)
    if payload:
        with pytest.raises(ValueError, match="unsafe source path"):
            for path in payload["source"]:
                _safe_relative(path)
    else:
        assert payload is None


@pytest.mark.parametrize("path", ["../secret.py", "/tmp/secret.py", "", "pkg/../../secret.py"])
def test_source_path_escape_attempts_are_rejected(path):
    with pytest.raises(ValueError, match="unsafe source path"):
        _safe_relative(path)


@pytest.mark.parametrize("text", [
    "hello", "Unicode 雪 and emoji 🐈", "\n".join(["patch line"] * 40),
])
def test_serialized_pair_canonicalization_is_stable_and_sha256_ready(text):
    row = {"pair_sha256": hashlib.sha256(text.encode()).hexdigest(), "prompt": text}
    first = _canonical_line(row)
    assert first == _canonical_line(dict(reversed(list(row.items()))))
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(first.encode()).hexdigest()


def test_real_frozen_unittest_passes_for_chosen_patch_and_fails_after_boundary_mutation(tmp_path):
    package = tmp_path / "aegis_case"
    package.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = "def clamp(value):\n    if value <= 10:\n        return value\n    return 9\n"
    (package / "limits.py").write_text(source, encoding="utf-8")
    test = tests / "test_limits.py"
    test.write_text("import unittest\nfrom aegis_case.limits import clamp\n\nclass Contract(unittest.TestCase):\n    def test_edge(self):\n        self.assertEqual(clamp(10), 10)\n", encoding="utf-8")
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_limits.py"]
    chosen = _run_test(tmp_path, "tests/test_limits.py", command, 20)
    assert chosen["passed"] is True
    mutant = mutate_source(source, "off_by_one_boundary")
    (package / "limits.py").write_text(mutant["source"], encoding="utf-8")
    redlight = _run_test(tmp_path, "tests/test_limits.py", command, 20)
    assert redlight["passed"] is False
    assert redlight["exit_code"] == 1


def test_fixture_builder_creates_a_real_git_baseline_and_frozen_test(tmp_path):
    probe = _initialize_fixture(tmp_path, {"app.py": "VALUE = 3\n"},
                                "def test_value():\n    assert 3 == 3\n")
    assert probe == "tests/test_aegis_frozen_probe.py"
    assert (tmp_path / ".git").is_dir()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", probe], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_token_budget_rejects_nonpositive_or_noninteger_limits_before_counting(limit):
    with pytest.raises(ValueError, match="max_tokens"):
        verify_zero_truncation(None, "sample", limit)


def test_official_tokenizer_failure_is_explicit_and_release_does_not_create_output(tmp_path):
    with pytest.raises(HardNegativeError, match="official tokenizer"):
        load_official_tokenizer("aegis-test/missing-tokenizer")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "release_v13_augmented"
    with pytest.raises(HardNegativeError):
        generate_release(tmp_path, source_dataset=source, output_dir=output, target_pairs=1)
    assert not output.exists()


def test_completion_parser_accepts_only_json_patch_payloads():
    assert _message_patch({"completion": json.dumps({"patch": "diff --git x"})}) == "diff --git x"
    assert _message_patch({"completion": "not-json"}) is None
    assert _message_patch({"completion": json.dumps({"patch": 4})}) is None


def test_source_diff_is_a_real_unified_patch_with_deterministic_hash(tmp_path):
    patch = _diff_file("app.py", "def value():\n    return 1\n", "def value():\n    return 2\n")
    assert patch.startswith("--- a/app.py\n+++ b/app.py\n")
    assert "-    return 1" in patch and "+    return 2" in patch
    file = tmp_path / "manifest.jsonl"
    file.write_text(patch, encoding="utf-8")
    assert _sha256_file(file) == hashlib.sha256(file.read_bytes()).hexdigest()


def test_mutation_type_catalog_covers_ten_distinct_hidden_defect_classes():
    assert len(MUTATION_TYPES) == 10
    assert len(set(MUTATION_TYPES)) == 10
    assert {"off_by_one_boundary", "swallow_exception", "remove_timeout_validation",
            "break_idempotency_key", "unbounded_memory_growth", "weaken_test_assertion",
            "concurrency_lock_loss", "authorization_bypass", "failure_as_success",
            "sql_command_injection"} == set(MUTATION_TYPES)
