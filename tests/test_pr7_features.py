"""PR #7 planner, mixed-language indexing, and online replay coverage."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from aegis_core.ast_slicer import ImportGraph, is_protected_test_path
from aegis_core.ledger import Ledger
from aegis_core.mediator import is_business_source
from aegis_core.planner import _topological_layers, build_task_dag, execute_task_dag
from aegis_core.repo_adapter import attach_repository, inspect_repository
from aegis_core.world_model import CausalWorldModel, preblock_reward_distribution


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", [
    "tests/test_gate.py", "test/gate.py", "src/__tests__/gate.js", "src/test/gate.ts",
    "test_a.py", "check_gate.py", "verify_gate.py", "validate_gate.py", "foo_test.py",
    "foo_test.js", "foo_test.tsx", "foo.test.js", "foo.test.jsx", "foo.test.ts",
    "foo.test.tsx", "foo.spec.js", "foo.spec.tsx", "foo.spec.mts",
])
def test_test_path_policy_covers_python_javascript_and_typescript(path):
    assert is_protected_test_path(path)
    assert not is_business_source(path)


@pytest.mark.parametrize("outcome,expected", [
    ("cheat_detected", [-2.0, -1.3]),
    ("replay_blocked", [-2.0, -1.3]),
])
def test_deterministic_preblock_reward_profiles_are_negative_and_bounded(outcome, expected):
    reward, interval = preblock_reward_distribution(outcome)
    assert interval == expected
    assert interval[0] <= reward <= interval[1] < 0


@pytest.mark.parametrize("reason,outcome", [
    ("test_modification", "cheat_detected"),
    ("repeated_patch", "replay_blocked"),
])
def test_preblocked_model_state_and_reward_are_calibrated_together(reason, outcome):
    model = CausalWorldModel.load(ROOT / "artifacts" / "causal_world_model")
    patch = ("--- a/tests/gate.test.ts\n+++ b/tests/gate.test.ts\n@@ -1 +1 @@\n-a\n+b\n"
             if reason == "test_modification" else
             "--- a/src/guard.py\n+++ b/src/guard.py\n@@ -1 +1 @@\n-a\n+b\n")
    action = {"patch": patch, "repeated_patch": reason == "repeated_patch"}
    prediction = model.predict({"status": "running"}, action)
    low, high = prediction["predicted_reward_interval"]
    assert prediction["predicted_next_state"]["outcome"] == outcome
    assert low <= prediction["predicted_reward"] <= high < 0
    assert prediction["predicted_next_state"]["decision_source"] == "deterministic_pre_block"


def _grouped_rows(*, per_risk=3, normal=30):
    rows = []
    for label in ("gate_passed", "assertion_error"):
        for index in range(normal):
            rows.append({"label": label, "group_id": f"ticket:{label}-{index}",
                         "group_keys": [f"ticket:{label}-{index}", f"template:{label}-{index}"],
                         "sample_id": f"{label}-{index}"})
    for label in ("cheat_detected", "replay_blocked"):
        for index in range(per_risk):
            rows.append({"label": label, "group_id": f"ticket:{label}-{index}",
                         "group_keys": [f"ticket:{label}-{index}", f"template:{label}-{index}"],
                         "sample_id": f"{label}-{index}"})
    return rows


def test_stratified_group_split_covers_each_risk_label_when_three_independent_groups_exist():
    from scripts.train_and_eval_world_model import split_by_group
    rows = _grouped_rows()
    train, validation, test, report = split_by_group(rows, return_summary=True)
    assert report["group_leakage"] is False
    assert report["unrepresented_risk_labels"] == []
    assert all(report["risk_support"][split][label] >= 1
               for split in ("train", "validation", "test")
               for label in ("cheat_detected", "replay_blocked"))
    sets = [set(i for i in indices) for indices in (train, validation, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


@pytest.mark.parametrize("count", [0, 1, 2])
def test_stratified_group_split_reports_unavoidable_rare_risk_gaps(count):
    from scripts.train_and_eval_world_model import split_by_group
    report = split_by_group(_grouped_rows(per_risk=count), return_summary=True)[-1]
    if count < 3:
        assert report["unrepresented_risk_labels"]
    assert report["group_leakage"] is False


def test_stratified_group_split_keeps_linked_ticket_template_component_together():
    from scripts.train_and_eval_world_model import split_by_group
    rows = _grouped_rows()
    rows += [{"label": "cheat_detected", "group_id": "linked", "group_keys": [
        "ticket:linked-cheat", "template:shared-cheat-family"], "sample_id": "linked"}]
    rows += [{"label": "assertion_error", "group_id": "other", "group_keys": [
        "ticket:other", "template:shared-cheat-family"], "sample_id": "other"}]
    train, validation, test, report = split_by_group(rows, return_summary=True)
    location = next(name for name, values in zip(("train", "validation", "test"), (train, validation, test))
                    if rows[-1] in [rows[index] for index in values])
    assert rows[-2] in [rows[index] for index in {"train": train, "validation": validation, "test": test}[location]]
    assert report["group_leakage"] is False


def test_planner_orders_quant_dependencies_before_rebalancer_and_validation():
    plan = build_task_dag(
        "Fix quant engine risk_guard.py leverage clamp and regime_gate.py recovery deadlock; validate rebalancer behavior",
        ROOT,
    )
    by_target = {task["contract"]["target_files"][0]: task for task in plan["tasks"] if task["kind"] == "repair"}
    guard = by_target["examples/quant_engine/risk_guard.py"]["task_id"]
    regime = by_target["examples/quant_engine/regime_gate.py"]["task_id"]
    validation = next(task for task in plan["tasks"] if task["kind"] == "validation")
    assert "examples/quant_engine/rebalancer.py" in validation["contract"]["integration_modules"]
    assert {guard, regime} <= set(validation["depends_on"])
    layers = plan["topological_layers"]
    assert layers.index([item for item in layers if validation["task_id"] in item][0]) > layers.index([item for item in layers if guard in item][0])
    assert layers[-1] == ["validate-integration"]
    assert plan["dependency_edges"]
    assert all(task["contract"].get("success_criteria") for task in by_target.values())


@pytest.mark.parametrize("tasks,error", [
    ([], "at least one task"),
    ([{"task_id": "a", "depends_on": ["missing"]}], "unknown dependency"),
    ([{"task_id": "a", "depends_on": ["a"]}], "self dependency"),
    ([{"task_id": "a", "depends_on": ["b"]}, {"task_id": "b", "depends_on": ["a"]}], "cycle"),
    ([{"task_id": "a", "depends_on": []}, {"task_id": "a", "depends_on": []}], "unique"),
])
def test_planner_rejects_invalid_dag_shapes(tasks, error):
    with pytest.raises(ValueError, match=error):
        _topological_layers(tasks)


@pytest.mark.parametrize("source,expected", [
    ("export function draw(x) { return x; }", "draw"),
    ("export const tick = (x) => { return x + 1; };", "tick"),
    ("export class Cat { run() { return 1; } }", "Cat.run"),
    ("export const resize = (width: number): number => { return width; }", "resize"),
    ("export class Canvas { resize(width: number): number { return width; } }", "Canvas.resize"),
    ("export async function walk() { return 1; }", "walk"),
])
def test_js_ts_symbol_extractor_indexes_functions_arrows_and_methods(tmp_path, source, expected):
    path = tmp_path / "src" / "main.ts"
    path.parent.mkdir()
    path.write_text(source, encoding="utf-8")
    graph = ImportGraph(tmp_path)
    assert f"src/main.ts:{expected}" in graph.js_symbols
    sliced = graph.slice("src/main.ts", expected)
    assert sliced["items"][0]["source"]


def test_js_symbol_extractor_ignores_declarations_inside_comments_and_strings(tmp_path):
    (tmp_path / "main.js").write_text(
        "// function ghost() {}\nconst text = 'function phantom() {}';\nfunction actual() { return 1; }\n",
        encoding="utf-8")
    symbols = ImportGraph(tmp_path).js_symbols
    assert set(symbols) == {"main.js:actual"}


def test_js_import_dependencies_feed_ast_graph(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "guard.ts").write_text("export function guard() { return true; }\n")
    (tmp_path / "src" / "app.ts").write_text("import { guard } from './guard';\nexport function run() { return guard(); }\n")
    graph = ImportGraph(tmp_path)
    assert graph.file_dependencies["src/app.ts"] == ["src/guard.ts"]
    assert graph.slice("src/app.ts", "run")["target"] == "src/app.ts:run"


def test_retrieval_can_select_javascript_symbols(tmp_path):
    from aegis_core.runner import retrieve
    (tmp_path / "engine.js").write_text("export function sortDepth(items) { return items; }\n")
    result = retrieve(tmp_path, "fix depth sorting")
    assert result["candidates"][0]["file"] == "engine.js"
    assert result["context"]["items"][0]["symbol"] == "engine.js:sortDepth"


def test_adapter_detects_mixed_project_and_test_runners(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): return 1\n")
    (tmp_path / "src" / "canvas.js").write_text("export function render() {}\n")
    (tmp_path / "src" / "state.ts").write_text("export function step() {}\n")
    (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass\n")
    (tmp_path / "tests" / "canvas.test.js").write_text("test('render', () => {});\n")
    profile = inspect_repository(tmp_path)
    assert profile["languages"] == {
        "python": {"files": 2, "present": True},
        "javascript": {"files": 2, "present": True},
        "typescript": {"files": 1, "present": True},
    }
    assert set(profile["test_frameworks"]) == {"pytest", "node --test"}
    assert set(profile["test_files"]) == {"tests/test_main.py", "tests/canvas.test.js"}


def test_adapter_uses_npm_test_script_and_package_entry_point(tmp_path):
    (tmp_path / "index.js").write_text("export function main() {}\n")
    (tmp_path / "package.json").write_text(json.dumps({"main": "index.js", "scripts": {"test": "node --test"}}))
    profile = inspect_repository(tmp_path)
    assert profile["entry_modules"] == ["index.js"]
    assert set(profile["test_frameworks"]) == {"node --test", "npm test"}


def test_attach_writes_standard_profile_atomically(tmp_path):
    (tmp_path / "main.py").write_text("def main(): pass\n")
    result = attach_repository(tmp_path)
    saved = json.loads(Path(result["profile_path"]).read_text(encoding="utf-8"))
    assert saved == result["profile"]
    assert saved["schema_version"] == "aegis.repo_profile.v1"
    assert saved["protected_test_patterns"]


def _init_repo(root: Path, *, failing: bool):
    root.mkdir(parents=True)
    (root / "app.py").write_text("def value(): return 0\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(
        "from app import value\ndef test_value():\n    assert value() == 1\n", encoding="utf-8")
    for args in (("init", "-b", "main"), ("add", "."),
                 ("-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-m", "baseline")):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    if not failing:
        return


def _dag_plan(root: Path, patch: Path):
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_app.py"]
    return {"schema_version": "aegis.task_dag.v1", "dag_id": "test-dag",
            "repo_root": str(root), "base_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True,
                text=True).stdout.strip(), "requirement": "Fix value", "topological_layers": [],
            "dependency_edges": [], "retrieval": {}, "tasks": [
                {"task_id": "repair-value", "kind": "repair", "title": "Fix value",
                 "depends_on": [], "contract": {"intent": "Fix value", "target_files": ["app.py"],
                                                   "success_criteria": ["value returns one"]},
                 "risk_estimate": 0.1, "patches": [str(patch)], "source_dirs": ["."],
                 "fail_before_cmd": command, "pass_after_cmd": command, "timeout_sec": 60},
                {"task_id": "validate", "kind": "validation", "title": "Validate", "depends_on": ["repair-value"],
                 "contract": {"intent": "test value"}, "risk_estimate": 0.05,
                 "command": command, "timeout_sec": 60},
            ]}


@pytest.mark.parametrize("fixes,expected", [(True, "succeeded"), (False, "failed")])
def test_run_dag_executes_aegis_repair_and_fuses_dependent_validation(tmp_path, fixes, expected):
    from difflib import unified_diff
    root = tmp_path / "repo"
    _init_repo(root, failing=True)
    before = (root / "app.py").read_text().splitlines(keepends=True)
    after = ["def value(): return 1\n"] if fixes else ["def value():\n    # unchanged behavior\n    return 0\n"]
    patch_path = tmp_path / "fix.diff"
    patch_path.write_text("".join(unified_diff(before, after, fromfile="a/app.py", tofile="b/app.py")),
                          encoding="utf-8")
    plan = _dag_plan(root, patch_path)
    dag_path = tmp_path / "plan.json"
    dag_path.write_text(json.dumps(plan), encoding="utf-8")
    result, code = execute_task_dag(dag_path, db=tmp_path / "ledger.sqlite", workers=2)
    assert result["status"] == expected
    assert (code == 0) is fixes
    if fixes:
        assert result["task_statuses"] == {"repair-value": "passed", "validate": "passed"}
        assert "return 0" in (root / "app.py").read_text()  # validated DAG changes stay in the isolated staging worktree
    else:
        assert result["task_statuses"]["repair-value"] == "failed"
        assert result["task_statuses"]["validate"] == "blocked"
        assert "return 0" in (root / "app.py").read_text()
    assert subprocess.run(["git", "status", "--porcelain"], cwd=root,
                          check=True, capture_output=True, text=True).stdout == ""


def _add_surprise(ledger: Ledger, ticket: str, *, error: float, preblocked=False, error_type=False):
    ledger.create_ticket(ticket, "test surprise", {"status": "queued"}, "def repair() -> None: ...")
    token = ledger.acquire_lease(ticket, "test")
    state = ledger.get_state(ticket)
    compiled = {"status": "COMPILED"}
    ledger.transition(ticket, token, "queued", "COMPILED", state_s0=state,
                      action_a0={"op": "compile"}, actual_state_s1=compiled, reward=0,
                      compiled_intent=compiled, contract_stub="def repair() -> None: ...")
    ledger.transition(ticket, token, "COMPILED", "running", state_s0=compiled,
                      action_a0={"op": "start_run"}, actual_state_s1={"status": "running"}, reward=0)
    running = ledger.get_state(ticket)
    actual = {"status": "running", "passed": False, "after_passed": False,
              "cheat_detected": False, "repeated_patch": False,
              "pre_blocked": preblocked, "world_model_outcome": "assertion_error"}
    if error_type:
        actual["error_type"] = "CalledProcessError"
    action = {"op": "repair_attempt", "patch": "--- a/src/guard.py\n+++ b/src/guard.py\n@@ -1 +1 @@\n-a\n+b\n",
              "context": {"items": []}, "target_files": ["src/guard.py"],
              "contract": {"target_files": ["src/guard.py"]}}
    ledger.transition(ticket, token, "running", "running", state_s0=running,
                      action_a0=action, actual_state_s1=actual, reward=-0.5,
                      predicted_next_state={"outcome": "gate_passed"}, predicted_reward=1.0,
                      prediction_confidence=0.8, prediction_error={"composite_error": error})
    ledger.transition(ticket, token, "running", "failed", state_s0=actual,
                      action_a0={"op": "finish_run"},
                      actual_state_s1={"status": "failed", "reason": "probe"}, reward=0)


def test_ledger_high_surprise_query_filters_threshold_and_preblocked_rows(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    _add_surprise(ledger, "high", error=1.2)
    _add_surprise(ledger, "low", error=0.1)
    _add_surprise(ledger, "preblock", error=1.1, preblocked=True)
    _add_surprise(ledger, "patch-not-applied", error=1.3, error_type=True)
    rows = ledger.high_surprise_traces(0.3)
    assert [row["ticket_id"] for row in rows] == ["high"]
    assert rows[0]["composite_error"] == 1.2


def test_learn_without_surprises_preserves_existing_model_version(tmp_path):
    from aegis_core.learning import learn_from_ledger
    artifacts = tmp_path / "artifacts"
    shutil.copytree(ROOT / "artifacts" / "causal_world_model", artifacts)
    version = CausalWorldModel.load(artifacts).model_version
    result = learn_from_ledger(repo=ROOT, db=tmp_path / "empty.sqlite", artifact_dir=artifacts)
    assert {key: result[key] for key in ("status", "threshold", "selected_traces", "world_model_version", "updated")} == {
        "status": "no_surprises", "threshold": 0.3, "selected_traces": 0, "world_model_version": version,
        "updated": False,
    }
    assert result["telemetry"]["model_loaded"] and result["telemetry"]["predictions_made"] == 0


def test_learn_updates_weights_and_writes_versioned_online_calibration_report(tmp_path):
    from aegis_core.learning import learn_from_ledger
    artifacts = tmp_path / "artifacts"
    shutil.copytree(ROOT / "artifacts" / "causal_world_model", artifacts)
    ledger = Ledger(tmp_path / "ledger.sqlite")
    _add_surprise(ledger, "surprise", error=1.4)
    old_version = CausalWorldModel.load(artifacts).model_version
    result = learn_from_ledger(repo=ROOT, db=ledger.path, artifact_dir=artifacts, epochs=1)
    assert result["status"] == "updated"
    assert result["updated"] is True
    assert result["world_model_version"] != old_version
    assert result["selected_traces"] == 1
    assert result["anchor_examples"] > 0
    assert (artifacts / "versions" / f"{result['world_model_version']}.json").is_file()
    report = json.loads((artifacts / "online_replay_report.json").read_text())
    assert report["updated"] is True
    assert report["trace_ids"] and report["regularization"]["kind"] == "proximal_to_previous_weights"
    assert CausalWorldModel.load(artifacts).model_version == result["world_model_version"]


def test_cli_plan_writes_formal_dag_json(tmp_path):
    from aegis_core import cli
    output = tmp_path / "plan.json"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert cli.main(["plan", "Fix quant engine risk_guard.py and regime_gate.py; integrate rebalancer",
                         "--repo", str(ROOT), "--out", str(output)]) == 0
    result = json.loads(stdout.getvalue())
    assert result["status"] == "planned"
    assert json.loads(output.read_text())["schema_version"] == "aegis.task_dag.v1"


def test_cli_attach_writes_mixed_repo_profile(tmp_path):
    from aegis_core import cli
    (tmp_path / "main.py").write_text("def main(): pass\n")
    (tmp_path / "index.ts").write_text("export function main() {}\n")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert cli.main(["attach", "--repo", str(tmp_path)]) == 0
    result = json.loads(stdout.getvalue())
    assert Path(result["profile_path"]).is_file()
    assert result["profile"]["languages"]["typescript"]["present"]
