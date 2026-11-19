"""State-machine, role authority, concurrent proposals and AST merge arbitration."""
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest

from aegis_core.swarm_arbiter import ROLES, SwarmArbiter, semantic_three_way_merge


def _arbiter():
    swarm = SwarmArbiter("ticket-14", "c3a892c3963c24ddee1d359898ce22213e53fff9")
    swarm.plan("enforce account limits", ["trace request flow", "preserve quota invariant"],
               invariants=["remaining quota never goes below zero"])
    return swarm


def _candidate(swarm, candidate_id="safe", patch="diff --git a/app.py b/app.py\n+pass\n"):
    swarm.submit(candidate_id, "bounded defensive change", patch)
    swarm.critique(candidate_id, [])
    return candidate_id


@pytest.mark.parametrize("role", ROLES)
def test_swarm_advertises_the_four_required_roles(role):
    swarm = _arbiter()
    assert role in swarm.status()["roles"]


def test_architect_plan_records_tasks_and_shared_invariants():
    swarm = SwarmArbiter("task", "abcdef0")
    result = swarm.plan("repair contract", ["analyze", "implement"], invariants=["MAX_ITEMS >= 0"])
    assert result["state"] == "CODING"
    assert result["breakdown"] == ["analyze", "implement"]
    assert result["invariants"] == ["MAX_ITEMS >= 0"]
    assert [event["role"] for event in result["events"]] == ["Architect", "Coder"]


@pytest.mark.parametrize("task,breakdown", [
    ("", ["step"]), ("task", []), ("task", [""]), ("task", [" "]),
    ("task", "not-a-list"), (None, ["step"]),
])
def test_architect_rejects_missing_or_unbounded_work_breakdown(task, breakdown):
    swarm = SwarmArbiter("task", "abcdef0")
    with pytest.raises(ValueError): swarm.plan(task, breakdown)
    assert swarm.state == "NEW"


@pytest.mark.parametrize("candidate_id", ["", "has space", "../escape", "x" * 97, "a/b", "💥"])
def test_coder_rejects_unsafe_candidate_identifiers(candidate_id):
    swarm = _arbiter()
    with pytest.raises(ValueError): swarm.submit(candidate_id, "strategy", "patch")


@pytest.mark.parametrize("strategy,patch", [("", "patch"), ("strategy", ""), ("  ", "patch"),
                                              ("strategy", "\n"), (None, "patch")])
def test_coder_requires_bounded_strategy_and_nonempty_patch(strategy, patch):
    swarm = _arbiter()
    with pytest.raises(ValueError): swarm.submit("candidate", strategy, patch)


def test_coder_patch_digest_is_stable_and_duplicate_ids_are_blocked():
    swarm = _arbiter()
    patch = "diff --git a/app.py b/app.py\n+return 1\n"
    result = swarm.submit("safe", "conservative", patch)
    assert result["patch_sha256"] == hashlib.sha256(patch.encode()).hexdigest()
    with pytest.raises(ValueError): swarm.submit("safe", "another", patch)


def test_critic_must_review_before_verifier_can_decide():
    swarm = _arbiter(); swarm.submit("candidate", "safe", "patch")
    with pytest.raises(RuntimeError): swarm.verify("candidate", static_passed=True,
        dynamic_passed=True, l3_approved=True)


@pytest.mark.parametrize("severity", ["info", "low", "medium", "high", "critical"])
def test_critic_normalizes_findings_and_tracks_blocking_severity(severity):
    swarm = _arbiter(); swarm.submit("candidate", "safe", "patch")
    result = swarm.critique("candidate", [{"severity": severity, "message": "reviewed", "file": "app.py", "line": 4}])
    assert result["blocking_findings"] == int(severity in {"high", "critical"})


@pytest.mark.parametrize("finding", ["not-a-dict", {"severity": "unknown", "message": "bad"},
                                      {"severity": "high"}, {"message": "missing severity"}])
def test_critic_rejects_malformed_findings(finding):
    swarm = _arbiter(); swarm.submit("candidate", "safe", "patch")
    with pytest.raises(ValueError): swarm.critique("candidate", [finding])


@pytest.mark.parametrize("static,dynamic,l3,blocking", [
    (False, False, False, False), (False, True, True, False), (True, False, True, False),
    (True, True, False, False), (True, True, True, True), (True, True, True, False),
    (False, True, False, True), (True, False, False, True),
])
def test_verifier_requires_all_three_gate_receipts_and_no_red_team_blocker(static, dynamic, l3, blocking):
    swarm = _arbiter(); _candidate(swarm)
    findings = ([{"severity": "critical", "message": "invariant break"}] if blocking else [])
    if findings:
        # Replace the initial empty critique with the actual adversarial finding.
        swarm.candidates["safe"].critiqued = False
        swarm.critique("safe", findings)
    result = swarm.verify("safe", static_passed=static, dynamic_passed=dynamic, l3_approved=l3)
    assert result["accepted"] is (static and dynamic and l3 and not blocking)


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_verifier_accepts_only_finite_score_in_unit_interval(value):
    swarm = _arbiter(); _candidate(swarm)
    assert swarm.verify("safe", static_passed=True, dynamic_passed=True, l3_approved=True,
                        score=value)["accepted"]


@pytest.mark.parametrize("value", [-1, 1.01, float("nan"), "1"])
def test_verifier_rejects_invalid_score(value):
    swarm = _arbiter(); _candidate(swarm)
    with pytest.raises(ValueError): swarm.verify("safe", static_passed=True,
        dynamic_passed=True, l3_approved=True, score=value)


def test_swarm_can_evaluate_multiple_candidates_then_select_best_verified_result():
    swarm = _arbiter()
    for candidate_id, score in (("fast", .7), ("safe", .95), ("unsafe", .99)):
        swarm.submit(candidate_id, "strategy " + candidate_id, "patch " + candidate_id)
        swarm.critique(candidate_id, [{"severity": "high", "message": "unsafe"}] if candidate_id == "unsafe" else [])
        swarm.verify(candidate_id, static_passed=True, dynamic_passed=True,
                     l3_approved=True, score=score)
    assert swarm.status()["candidate_count"] == 3
    assert swarm.select()["candidate_id"] == "safe"


def test_rejected_swarm_round_can_reopen_coding_for_targeted_retry():
    swarm = _arbiter(); _candidate(swarm)
    swarm.verify("safe", static_passed=False, dynamic_passed=True, l3_approved=True)
    swarm.submit("retry", "fix observed assertion", "corrective patch")
    assert swarm.state == "CODING"
    assert swarm.status()["candidate_count"] == 2


def test_verifier_receipts_are_preserved_in_role_event_log():
    swarm = _arbiter(); _candidate(swarm)
    result = swarm.verify("safe", static_passed=True, dynamic_passed=True, l3_approved=True,
                          receipts={"mutation": "pass", "l3": "approved"})
    event = swarm.status()["events"][-1]
    assert result["accepted"] and event["role"] == "Verifier"
    assert event["payload"]["receipts"]["mutation"] == "pass"


def test_concurrent_candidate_submissions_are_serialized_and_unique():
    swarm = _arbiter()
    def submit(index): return swarm.submit(f"worker-{index}", "parallel strategy", f"patch-{index}")["candidate_id"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(submit, range(8)))
    assert len(set(values)) == 8
    assert swarm.status()["candidate_count"] == 8


@pytest.mark.parametrize("name", ["", "not-a-hash", "g" * 40, "-abc1234"])
def test_session_requires_a_hexadecimal_commit_identity(name):
    with pytest.raises(ValueError): SwarmArbiter("task", name)


@pytest.mark.parametrize("left_body,right_body", [
    ("return value + 1", "return value - 1"), ("return min(value, 5)", "return max(value, 0)"),
    ("return True", "return False"), ("return [value]", "return {value}"),
])
def test_same_function_changes_require_semantic_arbitration(left_body, right_body):
    base = "def compute(value):\n    return value\n\ndef preserve(value):\n    return value\n"
    left = base.replace("return value\n\n", left_body + "\n\n", 1)
    right = base.replace("return value\n\n", right_body + "\n\n", 1)
    result = semantic_three_way_merge(base, left, right)
    assert result["status"] == "conflict"
    assert result["conflicts"][0]["kind"] == "same_symbol"


@pytest.mark.parametrize("left_value,right_value", [(2, 4), (3, 5), (0, 6), (-1, 8), (7, 9), (11, 12)])
def test_distinct_python_functions_merge_by_disjoint_ast_symbol(left_value, right_value):
    base = f"def first():\n    return 1\n\ndef second():\n    return 2\n\nVALUE = {left_value}\n"
    left = base.replace("return 1", f"return {left_value}")
    right = base.replace("return 2", f"return {right_value}")
    result = semantic_three_way_merge(base, left, right)
    assert result["status"] == "merged"
    assert f"return {left_value}" in result["content"] and f"return {right_value}" in result["content"]
    assert result["left_symbols"] == ["first"] and result["right_symbols"] == ["second"]


@pytest.mark.parametrize("name", ["MAX_ITEMS", "REQUEST_LIMIT", "CACHE_CAP", "MIN_STATE"])
def test_shared_global_contract_invariants_force_red_team_arbitration(name):
    base = f"{name} = 10\ndef put():\n    return {name}\ndef read():\n    return {name}\n"
    left = base.replace(f"return {name}", f"return {name} - 1", 1)
    right = base.replace(f"return {name}", f"return {name} + 1", 1)
    result = semantic_three_way_merge(base, left, right)
    assert result["status"] == "conflict"
    assert any(item["kind"] == "shared_invariant" for item in result["conflicts"])


@pytest.mark.parametrize("language,base,left,right", [
    ("javascript", "function first(){return 1;}\nfunction second(){return 2;}\n",
     "function first(){return 3;}\nfunction second(){return 2;}\n",
     "function first(){return 1;}\nfunction second(){return 4;}\n"),
    ("typescript", "export function first(): number { return 1; }\nexport function second(): number { return 2; }\n",
     "export function first(): number { return 3; }\nexport function second(): number { return 2; }\n",
     "export function first(): number { return 1; }\nexport function second(): number { return 4; }\n"),
    ("js", "function a(){return 0;}\nfunction b(){return 1;}\n", "function a(){return 9;}\nfunction b(){return 1;}\n",
     "function a(){return 0;}\nfunction b(){return 8;}\n"),
])
def test_javascript_and_typescript_function_edits_merge_independently(language, base, left, right):
    result = semantic_three_way_merge(base, left, right, language=language)
    assert result["status"] == "merged"
    assert "return 3" in result["content"] or "return 9" in result["content"]
    assert "return 4" in result["content"] or "return 8" in result["content"]


@pytest.mark.parametrize("left,right", [
    ("def a():\n    return 3\n", "def a():\n    return 4\n"),
    ("def a():\n    return 3\n", "def a():\n    return 5\n"),
    ("def a():\n    return 3\n", "def a():\n    return 6\n"),
])
def test_same_python_symbol_different_coder_proposals_conflict(left, right):
    base = "def a():\n    return 1\n"
    result = semantic_three_way_merge(base, left, right)
    assert result["status"] == "conflict" and result["conflicts"][0]["kind"] == "same_symbol"


def test_identical_proposals_are_deduplicated_without_false_conflict():
    merged = semantic_three_way_merge("def value(): return 1\n", "def value(): return 2\n",
                                      "def value(): return 2\n")
    assert merged["status"] == "merged" and merged["strategy"] == "identical_proposals"


@pytest.mark.parametrize("source", ["def broken(:\n", "class A(:\n    pass\n", "return )\n"])
def test_invalid_python_proposal_is_rejected_before_merge(source):
    result = semantic_three_way_merge("def good(): return 1\n", source, "def good(): return 2\n")
    assert result["status"] == "conflict" and result["conflicts"][0]["kind"] == "parse_error"


def test_merge_rejects_unsupported_language_and_non_text_inputs():
    with pytest.raises(ValueError): semantic_three_way_merge("", "", "", language="sql")
    with pytest.raises(TypeError): semantic_three_way_merge("", None, "")
