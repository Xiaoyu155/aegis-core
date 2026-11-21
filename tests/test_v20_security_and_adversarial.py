import ast
import hashlib
import json
from pathlib import Path

import pytest

from aegis_core.adversarial_self_play import TRANSFORMS, transform_sample
from aegis_core.invariant_guard import inspect_test_assertion_delta
from aegis_core.security_gate import scan_text
from aegis_core.security_rules import inspect_javascript_test_assertion_delta
from aegis_core.synthetic_hard_negatives import MUTATION_TYPES, mutate_source
from aegis_core.v20_samples import build_case


LANGUAGES = ("python", "javascript")


@pytest.mark.parametrize("defect,language", [(defect, language) for defect in MUTATION_TYPES for language in LANGUAGES])
def test_each_bilingual_mutator_creates_a_scanner_or_guard_redlight(defect, language):
    case = build_case(defect, language, 17)
    if defect == "weaken_test_assertion":
        candidates = (inspect_test_assertion_delta(case["frozen_test"], "assert True\n", path="tests/case.py")
                     if language == "python" else
                     inspect_javascript_test_assertion_delta(case["frozen_test"], "assert.ok(true);\n", path="tests/case.js"))
        assert not candidates["passed"]
        return
    mutation = mutate_source(case["source"], defect, language=language)
    assert mutation and mutation["changed"]
    extension = ".py" if language == "python" else ".js"
    expected = {
        "off_by_one_boundary": "OFF_BY_ONE_BOUNDARY", "swallow_exception": "EXCEPTION_SWALLOWED",
        "remove_timeout_validation": "MISSING_TIMEOUT", "break_idempotency_key": "NONDETERMINISTIC_IDEMPOTENCY_KEY",
        "unbounded_memory_growth": "UNBOUNDED_GROWTH", "concurrency_lock_loss": "SHARED_STATE_WITHOUT_LOCK",
        "authorization_bypass": "AUTHORIZATION_BYPASS", "failure_as_success": "FAILURE_AS_SUCCESS",
        "sql_command_injection": "SQL_COMMAND_INJECTION",
    }[defect]
    assert expected in {finding["code"] for finding in scan_text(mutation["source"], "src/case" + extension)}


@pytest.mark.parametrize("defect,language,index", [(defect, language, index)
    for defect in MUTATION_TYPES for language in LANGUAGES for index in range(10)])
def test_sample_factory_is_deterministic_and_varies_across_seeds(defect, language, index):
    first = build_case(defect, language, index)
    second = build_case(defect, language, index)
    assert first == second
    assert first["source"] and first["frozen_test"]
    assert first["defect_type"] == defect and first["language"] == language


@pytest.mark.parametrize("defect,language", [(defect, language) for defect in MUTATION_TYPES for language in LANGUAGES])
def test_hard_positive_refactors_have_no_security_findings(defect, language):
    case = build_case(defect, language, 29, positive=True)
    extension = ".py" if language == "python" else ".js"
    assert scan_text(case["source"], "src/positive" + extension) == []


@pytest.mark.parametrize("defect,language,transform", [(defect, language, transform)
    for defect in MUTATION_TYPES for language in LANGUAGES for transform in TRANSFORMS])
def test_red_team_transformations_are_repeatable_and_well_formed(defect, language, transform):
    case = build_case(defect, language, 41)
    source = case["source"]
    if defect != "weaken_test_assertion":
        mutation = mutate_source(source, defect, language=language)
        assert mutation is not None
        source = mutation["source"]
    first = transform_sample(source, language, transform, round_index=1)
    assert first == transform_sample(source, language, transform, round_index=1)
    assert first != source
    if language == "python":
        ast.parse(first)


@pytest.mark.parametrize("round_index", range(15))
@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_adversarial_transform_cycle_keeps_bounded_outputs(language, transform, round_index):
    case = build_case("break_idempotency_key", language, round_index)
    output = transform_sample(case["source"], language, transform, round_index)
    assert isinstance(output, str) and len(output) < 10_000
    if language == "python":
        ast.parse(output)


_REGRESSION_CASES = sorted((Path(__file__).resolve().parents[1] / "evasion_regressions").glob("*.json"))


@pytest.mark.parametrize("regression_path", _REGRESSION_CASES, ids=lambda path: path.stem)
def test_fixed_red_team_escapes_remain_permanent_detected_regressions(regression_path):
    row = json.loads(regression_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(row["source"].encode()).hexdigest() == row["candidate_sha256"]
    suffix = ".py" if row["language"] == "python" else ".js"
    expected = {"unbounded_memory_growth": "UNBOUNDED_GROWTH"}[row["defect_type"]]
    assert expected in {item["code"] for item in scan_text(row["source"], "evasion" + suffix)}
