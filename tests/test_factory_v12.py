"""Deterministic contract, ledger, and physical worktree coverage for v12."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from aegis_core.budget_guard import BUDGET_EXCEEDED_HALT, BudgetExceeded, BudgetGuard, budget_limits
from aegis_core.factory_queue import run_factory_queue
from aegis_core.ledger import Ledger
from aegis_core.probe_mutator import mutation_cases, validate_target
from aegis_core.probe_synth import acceptance_rule, expected_value, is_true_red, static_cases
from aegis_core.sandbox import validate_probe_patch

CLAMP = "src/limits.py::clamp clamps integers to [0, 10] and returns null for invalid inputs."
EVENTS = "src/events.py::count counts distinct string events and returns -1 for non-string events."


@pytest.mark.parametrize(("value", "expected"), [
    (-1000, 0), (-1, 0), (0, 0), (1, 1), (5, 5), (10, 10), (11, 10), (10**15, 10),
    (True, None), (False, None), (0.5, None), ("5", None), ([], None), ({}, None), (None, None),
])
def test_clamp_acceptance_oracle(value, expected):
    assert expected_value(acceptance_rule(CLAMP), value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    ([], 0), (["open"], 1), (["open", "close", "open"], 2), ([""] * 50, 1),
    (["open", "Open"], 2), (["open", {}], -1), ([[]], -1), ([None], -1), ("open", -1),
    ({"event": "open"}, -1), (123, -1), (None, -1),
])
def test_distinct_events_acceptance_oracle(value, expected):
    assert expected_value(acceptance_rule(EVENTS), value) == expected


@pytest.mark.parametrize("intent", [CLAMP, EVENTS])
def test_blind_cases_are_reproducible_and_disjoint_from_static_cases(intent):
    rule = acceptance_rule(intent)
    first = mutation_cases(rule, seed=11)
    assert first == mutation_cases(rule, seed=11)
    assert first != mutation_cases(rule, seed=12)
    assert not set(map(repr, first)) & set(map(repr, static_cases(rule)))
    assert len(first) == len({repr(value) for value, _ in first})


@pytest.mark.parametrize("function", [
    "def other(value: int): pass",
    "async def clamp(value: int): pass",
    "def clamp(*args): pass",
    "def clamp(value, *, maximum): pass",
    "def clamp(value: str): pass",
])
def test_baseline_ast_rejects_incompatible_target(tmp_path, function):
    source = tmp_path / "limits.py"
    source.write_text(function, encoding="utf-8")
    with pytest.raises(ValueError, match="mutation target"):
        validate_target(source, acceptance_rule(CLAMP))


@pytest.mark.parametrize("options", [
    None, [], {"unknown": 1}, {"max_token_budget": True}, {"max_token_budget": 1.5},
    {"max_cost_usd": -1}, {"max_cost_usd": float("nan")}, {"max_wall_clock_sec": float("inf")},
])
def test_budget_limits_reject_invalid_values(options):
    with pytest.raises(ValueError):
        budget_limits(options)


@pytest.mark.parametrize("limits, single, final, cap", [
    ({"max_token_budget": 3}, 2, 2, "max_token_budget"),
    ({"max_cost_usd": 0.00003}, 2, 2, "max_cost_usd"),
    ({"max_wall_clock_sec": 0}, 0, 0, "max_wall_clock_sec"),
])
def test_parent_budget_circuit_breaker_is_durable(tmp_path, limits, single, final, cap):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    parent = BudgetGuard(ledger, "dag", limits)
    first = BudgetGuard(ledger, "task-a", {}, parent=parent)
    second = BudgetGuard(ledger, "task-b", {}, parent=parent)
    first.record("provider_request", tokens=single, reward=2)
    with pytest.raises(BudgetExceeded, match=BUDGET_EXCEEDED_HALT):
        second.check(final)
    with pytest.raises(BudgetExceeded):
        first.check()
    assert parent.report()["halted_by"] == cap
    assert parent.report()["token_consumption"] == single
    assert [row["operation"] for row in ledger.budget_events(parent.scope)] == ["provider_request", "halt"]


def test_child_limit_does_not_halt_independent_sibling(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    parent = BudgetGuard(ledger, "dag", {"max_token_budget": 100})
    child = BudgetGuard(ledger, "task", {"max_token_budget": 1}, parent=parent)
    sibling = BudgetGuard(ledger, "sibling", {}, parent=parent)
    with pytest.raises(BudgetExceeded, match="max_token_budget"):
        child.check(2)
    sibling.check(2)
    assert sibling.report()["halted_by"] is None


def test_parent_cost_accounting_under_parallel_writers(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    parent = BudgetGuard(ledger, "dag", {})
    children = [BudgetGuard(ledger, str(index), {}, parent=parent) for index in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda child: child.record("provider_request", tokens=1, reward=1), children))
    assert parent.report()["token_consumption"] == 8
    assert sum(event["reward"] for event in ledger.budget_events(parent.scope)) == 8
    assert all(len(ledger.budget_events(child.scope)) == 1 for child in children)


@pytest.mark.parametrize(("code", "failure", "output"), [
    (1, "ASSERTION_FAILED", "SyntaxError: invalid syntax"),
    (1, "MODULE_NOT_IMPLEMENTED", "AssertionError: AEGIS_ACCEPTANCE"),
    (1, "ASSERTION_FAILED", "test failed"),
    (0, None, "AssertionError: AEGIS_ACCEPTANCE"),
    (2, "ASSERTION_FAILED", "AssertionError: AEGIS_ACCEPTANCE"),
])
def test_true_red_requires_real_assertion_receipt(code, failure, output):
    assert not is_true_red({"exit_code": code, "failure_code": failure, "changed": False,
                            "stdout": output, "stderr": ""})


@pytest.mark.parametrize("script", ["src/limits.py", "tests/other.py", "tests/test_aegis_probe.mjs"])
def test_compiled_probe_patch_rejects_out_of_scope_or_missing_assert(script):
    patch = f"--- /dev/null\n+++ b/{script}\n+console.log('hello')\n"
    with pytest.raises(ValueError):
        validate_probe_patch(patch)


def test_factory_isolates_errors_and_retains_stable_priority_order(tmp_path):
    config = tmp_path / "factory.json"
    config.write_text(json.dumps({"schema_version": "aegis.factory.v1", "jobs": [
        {"id": name, "priority": priority, "profile": "missing.json", "dag": "missing-plan.json"}
        for name, priority in [("late", 1), ("first", 3), ("second", 3)]
    ]}), encoding="utf-8")
    report, code = run_factory_queue(config)
    assert code == 1 and report["failed"] == 3
    assert [job["id"] for job in report["jobs"]] == ["first", "second", "late"]
    assert all(job["error_type"] == "FileNotFoundError" for job in report["jobs"])


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js 22 or newer is required")
def test_factory_v12_drill():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/run_factory_v12_drill.py")],
                            cwd=root, text=True, encoding="utf-8", capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    for line, evidence in zip(lines, ("MUTATION_PROBE_FAILED", "ASSERTION_FAILED", "BUDGET_EXCEEDED_HALT",
                                     "repositories=2")):
        assert evidence in line


@pytest.mark.parametrize("baseline", [
    "def clamp(value):\n    return min(10, max(0, value)) if type(value) is int else None\n",
    "import missing_factory_v12_module\ndef clamp(value):\n    return value\n",
    "def clamp(value):\n    return (\n",
])
def test_synthesis_rejects_green_import_and_syntax_baselines(tmp_path, baseline):
    from aegis_core.pm_expander import expand_requirement
    from aegis_core.probe_synth import compile_probe
    from aegis_core.sandbox import ProbeNotFailingError

    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src/limits.py").write_text(baseline, encoding="utf-8")
    _commit_repository(root)
    with pytest.raises(ProbeNotFailingError, match="real acceptance assertion"):
        compile_probe(expand_requirement(CLAMP, root), root, "HEAD")
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], encoding="utf-8")


def _commit_repository(root):
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Tests", "-c", "user.email=test@example.invalid", "commit", "-qm", "Baseline")


def test_distinct_events_mutation_gate_physically_rejects_replay_bug(tmp_path):
    import difflib

    from aegis_core.pm_expander import expand_requirement
    from aegis_core.probe_synth import compile_probe
    from aegis_core.sandbox import Sandbox

    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    target = "src/events.py"
    original = "def count(events: list[str]):\n    return len(events) if isinstance(events, list) else -1\n"
    (root / target).write_text(original, encoding="utf-8")
    _commit_repository(root)
    base = _git(root, "rev-parse", "HEAD").strip()
    spec = expand_requirement(EVENTS, root)
    probe = compile_probe(spec, root, base)
    patched = ("def count(events: list[str]):\n"
               "    if not isinstance(events, list) or any(type(event) is not str for event in events):\n"
               "        return -1\n"
               "    if events == ['open', 'open']:\n"
               "        return 1\n"
               "    return len(events)\n")
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), patched.splitlines(keepends=True),
                                        f"a/{target}", f"b/{target}"))
    sandbox = Sandbox(root)
    sandbox.create("events", base=base)
    try:
        command = [sys.executable if arg == "{python}" else arg for arg in probe["command"]]
        result = sandbox.repair("events", patch, fail_before_cmd=command, pass_after_cmd=command,
                                source_dirs=("src",), probe_patch=probe["patch"], probe_script_sha256=probe["hashes"],
                                mutation_rule=probe["rule"], require_assertion_red=True)
        assert result["after_passed"] and not result["passed"]
        assert result["gate_failure"] == "MUTATION_PROBE_FAILED"
    finally:
        sandbox.cleanup("events")
