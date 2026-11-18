"""PR #9 cross-repository iteration and ExperienceBank v3 regression coverage."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from aegis_core.experience_bank import ExperienceBank
from aegis_core.ledger import Ledger
from aegis_core.mediator import compile_wish
from aegis_core.planner import _topological_layers, execute_task_dag

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


def _repo(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "add", ".")
    subprocess.run(["git", "-c", "user.name=PR9 Test", "-c", "user.email=pr9@test",
                    "commit", "-m", "fixture"], cwd=root, check=True, capture_output=True)
    return root


def _diff(before: str, after: str, path: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile="a/" + path, tofile="b/" + path))


def _plan(root: Path, tasks: list[dict], name: str) -> Path:
    document = {"schema_version": "aegis.task_dag.v1", "dag_id": name,
                "repo_root": str(root), "base_commit": _git(root, "rev-parse", "HEAD"),
                "requirement": name, "tasks": tasks}
    path = root.parent / (name + ".json")
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _repair_task(task_id: str, patch: Path, command: list[str], title: str) -> dict:
    return {"task_id": task_id, "kind": "repair", "title": title, "depends_on": [],
            "contract": {"intent": title, "target_files": ["core.py"],
                         "success_criteria": ["preserve the module interface", "pass the frozen assertion"]},
            "risk_estimate": 0.2, "patches": [str(patch)], "source_dirs": ["."],
            "fail_before_cmd": command, "pass_after_cmd": command, "timeout_sec": 30}


def _memory_trace(trace_id: int, target: str, *, intent="cat cafe interaction state", surprise=1.2):
    return {"trace_id": trace_id, "ticket_id": f"catcafe-{trace_id}", "composite_error": surprise,
            "prediction_error": {"composite_error": surprise, "actual_outcome": "gate_passed"},
            "action_a0": {"op": "repair_attempt", "patch": "excluded patch content",
                          "target_files": [target],
                          "contract": {"intent": intent, "target_files": [target]}},
            "actual_state_s1": {"status": "succeeded", "passed": True,
                                "world_model_outcome": "gate_passed", "changed_files": [target],
                                "diagnosis": {"category": "gate_passed"}}}


def test_experience_bank_versions_three_real_trace_batches_and_deduplicates(tmp_path):
    bank = ExperienceBank(tmp_path / "experience.sqlite")
    reports = [bank.ingest_traces([_memory_trace(i, "game/catcafe/src/core.js")]) for i in (1, 2, 3)]
    assert [item["version"] for item in reports] == [1, 2, 3]
    assert [item["added"] for item in reports] == [1, 1, 1]
    duplicate = bank.ingest_traces([_memory_trace(3, "game/catcafe/src/core.js")])
    assert duplicate["added"] == 0
    assert duplicate["version"] == 3
    assert duplicate["experience_count"] == 3
    assert duplicate["version_sha256"] == reports[-1]["version_sha256"]


def test_experience_bank_v3_search_returns_ranked_source_trace_summaries(tmp_path):
    bank = ExperienceBank(tmp_path / "experience.sqlite")
    bank.ingest_traces([_memory_trace(21, "game/catcafe/src/core.js")])
    bank.ingest_traces([_memory_trace(22, "game/catcafe/src/render_cache.js", intent="cat cafe frame cache")])
    bank.ingest_traces([_memory_trace(23, "game/catcafe/src/save_state.js", intent="cat cafe save state")])
    matches = bank.search("game catcafe src core js render cache save state", top_k=3, min_similarity=0.01)
    assert len(matches) == 3
    assert matches[0]["version"] == 3
    assert all(item["source_kind"] == "real_trace" for item in matches)
    assert all("excluded patch content" not in item["pattern"] for item in matches)
    assert matches == sorted(matches, key=lambda item: (-item["similarity"], -item["experience_id"]))


@pytest.mark.parametrize("top_k", [0, -1, 51, True])
def test_experience_bank_v3_keeps_top_k_bounded(top_k, tmp_path):
    with pytest.raises(ValueError):
        ExperienceBank(tmp_path / "experience.sqlite").search("cat cafe", top_k=top_k)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, True])
def test_experience_bank_v3_rejects_invalid_similarity_thresholds(threshold, tmp_path):
    with pytest.raises(ValueError):
        ExperienceBank(tmp_path / "experience.sqlite").search("cat cafe", min_similarity=threshold)


def test_mediator_retrieves_v3_catcafe_experience_for_a_similar_game_request(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "core.py").write_text("def update_state(state):\n    return state\n", encoding="utf-8")
    (root / "other.py").write_text("def render_frame(state):\n    return state\n", encoding="utf-8")
    db = tmp_path / "experience.sqlite"
    bank = ExperienceBank(db)
    for trace_id, target in [(31, "game/catcafe/src/core.js"),
                             (32, "game/catcafe/src/render_cache.js"),
                             (33, "game/catcafe/src/save_state.js")]:
        bank.ingest_traces([_memory_trace(trace_id, target)])
    request = "game/catcafe/src/core.js cat cafe interaction state feed pet play cooldown"
    result = compile_wish(root, request, targets=["core.py"], experience_db=db)
    memory = result["context"]["experience_bank"]
    assert memory["top_k"]
    v3_hits = [item for item in memory["top_k"] if item["version"] == 3]
    assert v3_hits
    assert "game/catcafe/src/" in v3_hits[0]["pattern"]
    assert memory["safety_constraints_auto_filled"] is False
    assert memory["ambiguity_threshold"] == 0.15


def test_mediator_does_not_turn_experience_hits_into_missing_safety_constraints(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "risk.py").write_text("def assess():\n    return True\n", encoding="utf-8")
    db = tmp_path / "experience.sqlite"
    bank = ExperienceBank(db)
    for trace_id in (41, 42, 43):
        bank.ingest_traces([_memory_trace(trace_id, "risk.py", intent="live finance risk drawdown")])
    bank.record_arbitration(pattern="live finance risk drawdown", rule={
        "ambiguity_threshold": 0.05, "max_drawdown": 0.1,
        "close_method": "market", "unlock_condition": "next_utc_midnight"})
    result = compile_wish(root, "实盘资金回撤风控", targets=["risk.py"], experience_db=db)
    assert {"max_drawdown", "unlock_condition", "close_method"} <= set(result["missing"])
    assert result["context"]["experience_bank"]["safety_constraints_auto_filled"] is False


def test_planner_builds_dependency_layers_for_multistep_game_workorders():
    tasks = [
        {"task_id": "state-module", "depends_on": []},
        {"task_id": "game-core", "depends_on": ["state-module"]},
        {"task_id": "renderer", "depends_on": ["state-module"]},
        {"task_id": "integration", "depends_on": ["game-core", "renderer"]},
    ]
    assert _topological_layers(tasks) == [["state-module"], ["game-core", "renderer"], ["integration"]]


@pytest.mark.parametrize("tasks,message", [
    ([], "at least one task"),
    ([{"task_id": "a", "depends_on": ["missing"]}], "unknown dependency"),
    ([{"task_id": "a", "depends_on": ["a"]}], "self dependency"),
    ([{"task_id": "a", "depends_on": []}, {"task_id": "a", "depends_on": []}], "unique"),
    ([{"task_id": "a", "depends_on": ["b"]}, {"task_id": "b", "depends_on": ["a"]}], "cycle"),
])
def test_planner_v3_rejects_invalid_task_dags(tasks, message):
    with pytest.raises(ValueError, match=message):
        _topological_layers(tasks)


def test_two_real_gated_workorders_continue_from_the_prior_staged_commit(tmp_path):
    root = _repo(tmp_path / "external", {"core.py": "def value():\n    return 0\n"})
    main_head = _git(root, "rev-parse", "HEAD")
    command = [sys.executable, "-c", "from core import value; assert value() == 1"]
    first_patch = tmp_path / "workorder-a.diff"
    first_patch.write_text(_diff("def value():\n    return 0\n", "def value():\n    return 1\n", "core.py"), encoding="utf-8")
    first_plan = _plan(root, [_repair_task("ticket-a", first_patch, command, "apply state transition")], "ticket-a")
    first, first_code = execute_task_dag(first_plan, db=tmp_path / "ledger.sqlite", workers=1)
    assert first_code == 0 and first["status"] == "succeeded"
    assert first["tasks"][0]["aegis_result"]["results"][0]["baseline_failed"] is True
    assert first["tasks"][0]["aegis_result"]["results"][0]["after_passed"] is True
    assert "return 0" in (root / "core.py").read_text(encoding="utf-8")
    assert _git(root, "status", "--porcelain") == ""
    _git(root, "branch", "pr9-feature", first["staged_commit"])
    _git(root, "switch", "pr9-feature")
    second_command = [sys.executable, "-c", "from core import value; assert value() == 2"]
    second_patch = tmp_path / "workorder-b.diff"
    second_patch.write_text(_diff("def value():\n    return 1\n", "def value():\n    return 2\n", "core.py"), encoding="utf-8")
    second_plan = _plan(root, [_repair_task("ticket-b", second_patch, second_command, "add bounded state cache")], "ticket-b")
    second, second_code = execute_task_dag(second_plan, db=tmp_path / "ledger.sqlite", workers=1)
    assert second_code == 0 and second["status"] == "succeeded"
    assert second["tasks"][0]["aegis_result"]["results"][0]["baseline_failed"] is True
    staged_source = _git(root, "show", second["staged_commit"] + ":core.py")
    assert "return 2" in staged_source
    assert _git(root, "rev-parse", "main") == main_head
    assert _git(root, "status", "--porcelain") == ""


def _add_predicted_trace(ledger: Ledger, ticket: str, surprise: float):
    ledger.create_ticket(ticket, "cat cafe interaction replay", {}, "def repair() -> None: ...")
    token = ledger.acquire_lease(ticket, "pr9-test")
    state = ledger.get_state(ticket)
    compiled = {"status": "COMPILED"}
    ledger.transition(ticket, token, "queued", "COMPILED", state_s0=state,
                      action_a0={"op": "compile"}, actual_state_s1=compiled, reward=0,
                      compiled_intent=compiled, contract_stub="def repair() -> None: ...")
    ledger.transition(ticket, token, "COMPILED", "running", state_s0=compiled,
                      action_a0={"op": "start_run"}, actual_state_s1={"status": "running"}, reward=0)
    running = ledger.get_state(ticket)
    action = {"op": "repair_attempt", "target_files": ["game/catcafe/src/core.js"],
              "contract": {"intent": "cat cafe interaction replay", "target_files": ["game/catcafe/src/core.js"]}}
    actual = {"status": "running", "passed": False, "world_model_outcome": "assertion_error",
              "diagnosis": {"category": "assertion"}}
    ledger.transition(ticket, token, "running", "running", state_s0=running, action_a0=action,
                      actual_state_s1=actual, reward=-0.5,
                      predicted_next_state={"outcome": "gate_passed"}, predicted_reward=1.0,
                      prediction_confidence=0.8,
                      prediction_error={"composite_error": surprise, "actual_outcome": "assertion_error"})


def test_online_replay_report_and_experience_bank_advance_together_to_v3(tmp_path):
    from aegis_core.learning import learn_from_ledger
    from aegis_core.world_model import CausalWorldModel
    artifacts = tmp_path / "artifacts"
    shutil.copytree(ROOT / "artifacts" / "causal_world_model", artifacts)
    ledger = Ledger(tmp_path / "ledger.sqlite")
    for index in range(3):
        _add_predicted_trace(ledger, f"online-{index}", 1.0 + index * 0.2)
    bank = ExperienceBank(ledger.path)
    rows = ledger.high_surprise_traces(0.25)
    bank.ingest_traces([rows[0]])
    bank.ingest_traces([rows[1]])
    before = CausalWorldModel.load(artifacts).model_version
    report = learn_from_ledger(repo=ROOT, db=ledger.path, artifact_dir=artifacts,
                               min_surprise=0.25, epochs=1)
    assert report["status"] == "updated"
    assert report["world_model_version"] != before
    assert report["experience_bank"]["added"] == 1
    assert report["experience_bank"]["version"] == 3
    assert report["experience_bank"]["version_sha256"]
    tolerance = report["regularization"]["anchor_validation_tolerance"]
    assert report["anchor_validation_loss_after"] <= report["anchor_validation_loss_before"] + tolerance
    assert ExperienceBank(ledger.path).status()["version"] == 3
