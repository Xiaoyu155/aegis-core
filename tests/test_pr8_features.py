"""PR #8 lifecycle, cross-repository planning, and causal experience coverage."""
from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from aegis_core.experience_bank import ExperienceBank
from aegis_core.ledger import Ledger
from aegis_core.mediator import compile_wish, validate_compilation
from aegis_core.planner import build_task_dag, execute_task_dag
from aegis_core.reward import compute_step_reward
from aegis_core.runner import retrieve
from aegis_core.sandbox import run_exam_gate


ROOT = Path(__file__).resolve().parents[1]


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def _repo(root: Path, files: dict[str, str]):
    root.mkdir(parents=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "add", ".")
    subprocess.run(["git", "-c", "user.name=PR8 Test", "-c", "user.email=pr8@test",
                    "commit", "-m", "fixture"], cwd=root, check=True, capture_output=True)
    return root


def _diff(before: str, after: str, path: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile="a/" + path, tofile="b/" + path))


def _plan(root: Path, tasks: list[dict], name="pr8-dag"):
    plan = {"schema_version": "aegis.task_dag.v1", "dag_id": name, "repo_root": str(root),
            "base_commit": _git(root, "rev-parse", "HEAD"), "requirement": name,
            "topological_layers": [], "dependency_edges": [], "retrieval": {}, "tasks": tasks}
    path = root.parent / (name + ".json")
    path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", [".py", ".js", ".ts", ".tsx"])
def test_ast_index_skips_oversized_generated_sources_without_parsing(tmp_path, suffix):
    from aegis_core.ast_slicer import ImportGraph
    path = tmp_path / ("generated" + suffix)
    path.write_text("function ignored() {}\n" + "x" * (512 * 1024), encoding="utf-8")
    graph = ImportGraph(tmp_path)
    assert graph.diagnostics == [{"file": path.name, "error": "SourceTooLarge"}]
    assert path.name not in graph.paths
    assert path.name not in graph.js_modules


def test_plan_creates_missing_js_helper_before_exact_core_consumer(tmp_path):
    root = _repo(tmp_path / "repo", {
        "game/catcafe/src/core.js": "const THREE = {};\nfunction pointerDown() {}\n",
        "miniprogram/game/core.js": "function unrelatedCore() {}\n",
    })
    plan = build_task_dag(
        "仅修改 game/catcafe/src/core.js 并新增 game/catcafe/src/interaction.js，"
        "加入拖拽置顶和多猫排斥交互探针。", root, max_tasks=1)
    repairs = {task["task_id"]: task for task in plan["tasks"] if task["kind"] == "repair"}
    create = next(task for task in repairs.values() if task["contract"]["target_files"] == [
        "game/catcafe/src/interaction.js"])
    core = next(task for task in repairs.values() if task["contract"]["target_files"] == [
        "game/catcafe/src/core.js"])
    assert core["depends_on"] == [create["task_id"]]
    assert plan["topological_layers"][0] == [create["task_id"]]
    assert plan["topological_layers"][-1] == ["validate-integration"]
    assert all(task["contract"].get("success_criteria") for task in repairs.values())


@pytest.mark.parametrize("filename", ["src/new.py", "src/new.js", "src/new.ts", "src/new.mjs"])
def test_contract_can_anchor_a_new_business_module_beside_existing_source(filename):
    payload = {"kind": "CompiledContract", "ticket_id": "new-source", "status": "COMPILED",
               "target_files": ["src/current.py", filename],
               "contract_stub": "def apply_change() -> None: ...", "constraints": {}}
    assert validate_compilation(payload, ["src/current.py"])["target_files"] == ["src/current.py", filename]


@pytest.mark.parametrize("filename", ["outside/new.js", "tests/new.js", "src/../outside.js", "src/new.json"])
def test_contract_rejects_unanchored_or_protected_new_target_paths(filename):
    payload = {"kind": "CompiledContract", "ticket_id": "new-source", "status": "COMPILED",
               "target_files": ["src/current.py", filename],
               "contract_stub": "def apply_change() -> None: ...", "constraints": {}}
    with pytest.raises(ValueError):
        validate_compilation(payload, ["src/current.py"])


def test_dag_materializes_only_the_candidate_that_passed_both_probes(tmp_path):
    root = _repo(tmp_path / "repo", {
        "app.py": "def value(): return 0\n",
        "tests/test_app.py": "from app import value\ndef test_value():\n    assert value() == 1\n",
    })
    baseline = (root / "app.py").read_text(encoding="utf-8")
    bad = tmp_path / "bad.diff"
    good = tmp_path / "good.diff"
    bad.write_text(_diff(baseline, "def value(): return 2\n", "app.py"), encoding="utf-8")
    good.write_text(_diff(baseline, "def value(): return 1\n", "app.py"), encoding="utf-8")
    command = [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_app.py"]
    task = {"task_id": "repair", "kind": "repair", "title": "diagnosis repair", "depends_on": [],
            "contract": {"intent": "Fix value", "target_files": ["app.py"],
                         "success_criteria": ["value returns one"]},
            "risk_estimate": 0.2, "patches": [str(bad), str(good)], "max_attempts": 2,
            "source_dirs": ["."], "fail_before_cmd": command, "pass_after_cmd": command,
            "timeout_sec": 30}
    plan_path = _plan(root, [task])
    result, code = execute_task_dag(plan_path, db=tmp_path / "ledger.sqlite", workers=1)
    assert code == 0 and result["status"] == "succeeded"
    attempt_rows = result["tasks"][0]["aegis_result"]["results"]
    assert attempt_rows[0]["diagnosis"]["kind"] == "FailureDiagnosis"
    assert attempt_rows[1]["passed"] is True
    assert "return 0" in (root / "app.py").read_text(encoding="utf-8")
    assert _git(root, "status", "--porcelain") == ""


def test_dag_marks_downstream_as_cascade_skipped_after_gate_failure(tmp_path):
    root = _repo(tmp_path / "repo", {
        "app.py": "def value(): return 0\n",
        "tests/test_app.py": "from app import value\ndef test_value():\n    assert value() == 1\n",
    })
    patch = tmp_path / "still-broken.diff"
    patch.write_text(_diff("def value(): return 0\n", "def value(): return 2\n", "app.py"), encoding="utf-8")
    command = [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_app.py"]
    marker = tmp_path / "downstream.marker"
    downstream = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"]
    tasks = [
        {"task_id": "upstream", "kind": "repair", "title": "expected failure", "depends_on": [],
         "contract": {"intent": "Fix value", "target_files": ["app.py"], "success_criteria": ["pass"]},
         "risk_estimate": 0.4, "patches": [str(patch)], "source_dirs": ["."],
         "fail_before_cmd": command, "pass_after_cmd": command, "timeout_sec": 30},
        {"task_id": "downstream", "kind": "command", "title": "must skip", "depends_on": ["upstream"],
         "contract": {"intent": "must not run"}, "risk_estimate": 0.9,
         "command": downstream, "timeout_sec": 2},
    ]
    plan_path = _plan(root, tasks, "cascade-dag")
    result, code = execute_task_dag(plan_path, db=tmp_path / "ledger.sqlite", workers=1)
    row = next(task for task in result["tasks"] if task["task_id"] == "downstream")
    assert code == 1
    assert row["status"] == "blocked"
    assert row["cascade_status"] == "SKIPPED_UPSTREAM_FAILED"
    assert row["failed_dependencies"] == ["upstream"]
    assert not marker.exists()


def test_gate_timeout_kills_the_whole_child_process_group_and_reaps_descendants(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_file = tmp_path / "child.pid"
    command = [sys.executable, "-c",
               "import pathlib,subprocess,sys; "
               f"child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
               f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
               "exec('while True:\\n    pass')"]
    result = run_exam_gate(repo, command, timeout=0.25)
    assert result["exit_code"] == 124
    assert result["timed_out"] and result["timeout_killed"] and result["killed_process_group"]
    assert result["termination_reason"] == "TIMEOUT_KILLED"
    assert result["passed"] is False
    pid = int(pid_file.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and Path(f"/proc/{pid}").exists():
        time.sleep(0.02)
    if sys.platform.startswith("linux"):
        assert not Path(f"/proc/{pid}").exists()


@pytest.mark.parametrize("baseline_failed,after_passed", [(True, False), (False, False), (True, True)])
def test_timeout_kill_reward_is_always_negative(baseline_failed, after_passed):
    reward = compute_step_reward(baseline_failed=baseline_failed, after_passed=after_passed,
                                 target_files=["src/guard.py"], changed_files=["src/guard.py"],
                                 diff_lines=0, elapsed_sec=1, latency_budget_sec=10,
                                 timeout_killed=True)
    assert reward <= -0.25


def _trace(trace_id, pattern="cat cafe drag interaction", surprise=1.2, outcome="assertion_error"):
    return {"trace_id": trace_id, "ticket_id": f"ticket-{trace_id}", "composite_error": surprise,
            "prediction_error": {"composite_error": surprise, "actual_outcome": outcome},
            "action_a0": {"op": "repair_attempt", "target_files": ["game/catcafe/src/core.js"],
                          "contract": {"intent": pattern, "target_files": ["game/catcafe/src/core.js"]}},
            "actual_state_s1": {"status": "failed", "world_model_outcome": outcome,
                                "diagnosis": {"category": "assertion"}}}


def test_experience_bank_ingests_immutable_trace_summaries_and_deduplicates(tmp_path):
    bank = ExperienceBank(tmp_path / "ledger.sqlite")
    trace = _trace(17)
    first = bank.ingest_traces([trace])
    second = bank.ingest_traces([trace])
    assert first["added"] == 1 and first["experience_count"] == 1
    assert second["added"] == 0 and second["version"] == first["version"]
    assert first["version_sha256"]
    matches = bank.search("cat cafe drag interaction", top_k=2)
    assert matches[0]["trace_id"] == 17
    assert matches[0]["source_kind"] == "real_trace"
    assert matches[0]["cause"] == "assertion"
    assert "patch" not in matches[0]["pattern"]


@pytest.mark.parametrize("statement", [
    "UPDATE experience_records SET outcome='overwritten'",
    "DELETE FROM experience_records",
])
def test_experience_bank_records_are_append_only(statement, tmp_path):
    bank = ExperienceBank(tmp_path / "ledger.sqlite")
    bank.ingest_traces([_trace(9)])
    with sqlite3.connect(bank.path) as db, pytest.raises(sqlite3.IntegrityError):
        db.execute(statement)


def test_experience_bank_human_arbitration_creates_a_new_version_and_top_k_match(tmp_path):
    bank = ExperienceBank(tmp_path / "ledger.sqlite")
    first = bank.ingest_traces([_trace(1)])
    arbitration = bank.record_arbitration(pattern="cat cafe drag interaction stable review",
                                          rule={"ambiguity_threshold": 0.07},
                                          cause="reviewed_module_selection", resolution="human approved")
    assert arbitration["added"] == 1
    assert arbitration["version"] == first["version"] + 1
    matches = bank.search("cat cafe drag interaction stable review", top_k=5, min_similarity=0.01)
    assert matches[0]["source_kind"] == "human_arbitration"
    assert matches[0]["human_rule"]["ambiguity_threshold"] == 0.07


@pytest.mark.parametrize("threshold", [0.0, 0.049, 0.151, 1.0, True])
def test_experience_bank_rejects_unsafe_ambiguity_threshold_rules(threshold, tmp_path):
    bank = ExperienceBank(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError):
        bank.record_arbitration(pattern="module ambiguity", rule={"ambiguity_threshold": threshold})


@pytest.mark.parametrize("top_k", [0, 51, True])
def test_experience_bank_rejects_unbounded_top_k(top_k, tmp_path):
    bank = ExperienceBank(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError):
        bank.search("known issue", top_k=top_k)


def test_mediator_uses_reviewed_experience_to_lower_only_the_ambiguity_threshold(tmp_path):
    root = _repo(tmp_path / "repo", {
        "alpha.py": "def start():\n    return 1\n",
        "beta.py": "def start():\n    return 2\n",
    })
    db = tmp_path / "memory.sqlite"
    ExperienceBank(db).record_arbitration(pattern="select routine module start", 
                                          rule={"ambiguity_threshold": 0.06})
    compiled = compile_wish(root, "select routine module start", targets=["alpha.py"], experience_db=db)
    memory = compiled["context"]["experience_bank"]
    assert memory["top_k"]
    assert memory["ambiguity_threshold"] == 0.06
    assert memory["safety_constraints_auto_filled"] is False


def test_human_experience_does_not_fill_financial_safety_constraints(tmp_path):
    root = _repo(tmp_path / "repo", {"risk_guard.py": "def guard():\n    return True\n"})
    db = tmp_path / "memory.sqlite"
    ExperienceBank(db).record_arbitration(
        pattern="实盘资金风控",
        rule={"ambiguity_threshold": 0.05, "max_drawdown": 0.1,
              "close_method": "market", "unlock_condition": "next_utc_midnight"})
    compiled = compile_wish(root, "实盘资金风控", targets=["risk_guard.py"], experience_db=db)
    assert {"max_drawdown", "unlock_condition", "close_method"} <= set(compiled["missing"])
    assert compiled["context"]["experience_bank"]["safety_constraints_auto_filled"] is False


def test_online_learning_returns_experience_bank_version_with_surprise_update(tmp_path):
    from aegis_core.learning import learn_from_ledger
    from aegis_core.world_model import CausalWorldModel
    artifacts = tmp_path / "artifacts"
    import shutil
    shutil.copytree(ROOT / "artifacts" / "causal_world_model", artifacts)
    ledger = Ledger(tmp_path / "ledger.sqlite")
    _add_trace(ledger, "online", 1.4)
    before = CausalWorldModel.load(artifacts).model_version
    report = learn_from_ledger(repo=ROOT, db=ledger.path, artifact_dir=artifacts, epochs=1)
    assert report["status"] == "updated"
    assert report["world_model_version"] != before
    assert report["experience_bank"]["added"] >= 1
    assert report["experience_bank"]["version_sha256"]


def _add_trace(ledger: Ledger, ticket: str, error: float):
    ledger.create_ticket(ticket, "cat cafe drag interaction", {}, "def repair() -> None: ...")
    token = ledger.acquire_lease(ticket, "pr8-test")
    state = ledger.get_state(ticket)
    compiled = {"status": "COMPILED"}
    ledger.transition(ticket, token, "queued", "COMPILED", state_s0=state,
                      action_a0={"op": "compile"}, actual_state_s1=compiled, reward=0,
                      compiled_intent=compiled, contract_stub="def repair() -> None: ...")
    ledger.transition(ticket, token, "COMPILED", "running", state_s0=compiled,
                      action_a0={"op": "start_run"}, actual_state_s1={"status": "running"}, reward=0)
    running = ledger.get_state(ticket)
    action = {"op": "repair_attempt", "target_files": ["game/catcafe/src/core.js"],
              "contract": {"intent": "cat cafe drag interaction",
                           "target_files": ["game/catcafe/src/core.js"]}}
    actual = {"status": "running", "passed": False, "world_model_outcome": "assertion_error",
              "diagnosis": {"category": "assertion"}}
    ledger.transition(ticket, token, "running", "running", state_s0=running, action_a0=action,
                      actual_state_s1=actual, reward=-0.5,
                      predicted_next_state={"outcome": "gate_passed"}, predicted_reward=1.0,
                      prediction_confidence=0.8,
                      prediction_error={"composite_error": error, "actual_outcome": "assertion_error"})
    ledger.transition(ticket, token, "running", "failed", state_s0=actual,
                      action_a0={"op": "finish_run"},
                      actual_state_s1={"status": "failed", "reason": "assertion"}, reward=0)


def test_retrieval_reuses_an_existing_ast_graph_for_javascript(tmp_path):
    (tmp_path / "core.js").write_text("export function pointerDown() { return 'cat'; }\n", encoding="utf-8")
    result = retrieve(tmp_path, "cat drag pointerDown")
    assert result["candidates"][0]["file"] == "core.js"


@pytest.mark.parametrize("path", ["test_capture.png", "tests/test_render.html", "test_asset.glb"])
def test_adapter_does_not_classify_non_source_test_assets_as_test_code(path):
    from aegis_core.repo_adapter import _test_name
    assert _test_name(path) is False


@pytest.mark.parametrize("path", ["tests/test_render.js", "src/render.test.ts", "test_engine.py"])
def test_adapter_classifies_supported_test_source_paths(path):
    from aegis_core.repo_adapter import _test_name
    assert _test_name(path) is True

