import copy
from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from aegis_core.ledger import Ledger
from aegis_core.mediator import SYSTEM_PROMPT, compile_request, validate_compilation
from aegis_core.reward import compute_step_reward
from aegis_core.router import Router, load_models, normalize_provider, request_provider
from aegis_core.sandbox import Sandbox, detect_source_cheat
from aegis_core.simulator import FAULTS, create_arena, fixture_patch, run_arena, export_dataset

ROOT = Path(__file__).resolve().parents[1]
SKELETON = ["signal_engine.py", "intraday_guardian.py", "tests/test_signal.py", "engine/render.py"]
PROPOSAL = {"target_files": ["signal_engine.py"], "contract_stub": 'def place(symbol: str, *, reduce_only: bool = False) -> dict:\n    """Respect frozen risk bounds."""\n    ...'}


def test_mediator_credentials_finance_and_ledger_pause_resume(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    ledger.create_ticket("ticket", "实盘交易", {}, "def pending(): ...")
    token = ledger.acquire_lease("ticket", "compiler")
    decision = compile_request("ticket", "实盘交易", SKELETON, PROPOSAL, ledger=ledger, lease_token=token)
    assert decision["status"] == "MISSING_CONSTRAINT"
    assert {q["code"] for q in decision["questions"]} == {"A_CREDENTIALS", "B_FINANCE"}
    assert ledger.get_ticket("ticket")["status"] == "MISSING_CONSTRAINT"
    assert validate_compilation(decision, SKELETON) == decision
    assert all(q["recommended"] in {o["id"] for o in q["options"]} for q in decision["questions"])
    # Credential reference exists but irreversible finance still has no hard bounds.
    constraints = {"execution_mode": "live", "credential_ref": "env:EXCHANGE_KEY", "credential_available": True}
    decision = compile_request("ticket", "实盘交易", SKELETON, PROPOSAL, constraints=constraints)
    assert [q["code"] for q in decision["questions"]] == ["B_FINANCE"]
    constraints.update(max_drawdown=0.05, max_notional=100, position_mode="one_way")
    decision = compile_request("ticket", "实盘交易", SKELETON, PROPOSAL, constraints=constraints, ledger=ledger, lease_token=token)
    assert decision["status"] == "COMPILED"
    assert ledger.get_ticket("ticket")["contract_stub"] == PROPOSAL["contract_stub"]
    assert ledger.get_ticket("ticket")["compiled_intent"] == decision
    assert ledger.export_jsonl(tmp_path / "traces.jsonl") == 3
    with pytest.raises(ValueError):
        compile_request("t", "实盘交易", SKELETON, PROPOSAL, constraints={"api_key": "never-log-this"})


def test_mediator_routine_dry_run_and_migration():
    assert compile_request("t", "实盘交易仿真", SKELETON, PROPOSAL,
                           constraints={"execution_mode": "dry_run"})["status"] == "COMPILED"
    decision = compile_request("t", "执行破坏性数据迁移", SKELETON, PROPOSAL)
    assert decision["questions"][0]["code"] == "B_MIGRATION"
    assert compile_request("t", "执行破坏性数据迁移", SKELETON, PROPOSAL,
                           constraints={"migration_allowlist": ["signal_engine.py"], "rollback_boundary": "snapshot-before-migration"})["status"] == "COMPILED"


@pytest.mark.parametrize("gap,needs_question", [(0.149, True), (0.15, False), (0.151, False)])
def test_ambiguity_boundary_and_selection(gap, needs_question):
    candidates = [{"path": "signal_engine.py", "score": 0.9}, {"path": "intraday_guardian.py", "score": 0.9 - gap}]
    result = compile_request("t", "修复策略", SKELETON, PROPOSAL, candidates=candidates)
    assert (result["status"] == "MISSING_CONSTRAINT") == needs_question
    result = compile_request("t", "修复策略", SKELETON, PROPOSAL, candidates=candidates,
                             constraints={"selected_module": "signal_engine.py"})
    assert result["status"] == "COMPILED"


@pytest.mark.parametrize("targets", [["unknown.py"], ["tests/test_signal.py"], ["../signal_engine.py"], ["signal_engine.py", "signal_engine.py"]])
def test_path_anchoring_and_test_only_rejection(targets):
    with pytest.raises(ValueError):
        compile_request("t", "fix", SKELETON, {**PROPOSAL, "target_files": targets})


@pytest.mark.parametrize("stub", ["def f():\n    return 1", "@execute()\ndef f(): ...", "def f(x=execute()): ...", "print('oops')", "x: int"])
def test_executable_stubs_rejected(stub):
    with pytest.raises(ValueError):
        compile_request("t", "fix", SKELETON, {**PROPOSAL, "contract_stub": stub})


def test_questions_strict_json():
    result = compile_request("t", "实盘交易", SKELETON, PROPOSAL)
    result["questions"][0]["recommended"] = "nonexistent"
    with pytest.raises(ValueError):
        validate_compilation(result, SKELETON)
    result = compile_request("t", "实盘交易", SKELETON, PROPOSAL)
    result["unknown"] = 1
    with pytest.raises(ValueError):
        validate_compilation(result, SKELETON)
    assert "0.15" in SYSTEM_PROMPT


def _reward(**kwargs):
    values = dict(baseline_failed=True, after_passed=True, target_files=["app.py"], changed_files=["app.py"],
                  diff_lines=0, elapsed_sec=0, latency_budget_sec=60)
    values.update(kwargs)
    return compute_step_reward(**values)


def test_reward_formula_boundaries_and_exact_components():
    assert _reward() == 1.5
    assert _reward(after_passed=False) == pytest.approx(0.0)
    assert _reward(baseline_failed=False) == pytest.approx(-0.5)
    assert _reward(cheat_detected=True, repeated_patch=True) == 0.5
    assert _reward(target_files=["app.py", "lib.py"], changed_files=["app.py", "other.py"],
                   diff_lines=40, elapsed_sec=30) == pytest.approx(1.1)
    assert _reward(baseline_failed=False, after_passed=False, target_files=[], changed_files=[],
                   diff_lines=1000, elapsed_sec=1000, cheat_detected=True) == -2
    assert _reward(target_files=[], changed_files=[]) == 1.2


@pytest.mark.parametrize("kwargs", [{"diff_lines": -1}, {"elapsed_sec": float("nan")}, {"latency_budget_sec": 0}, {"diff_lines": True}])
def test_reward_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        _reward(**kwargs)


def test_router_normalization_deadline_and_concurrency_dam():
    config = load_models(ROOT / "config/models.yaml")
    router = Router(config)
    assert normalize_provider("codex") == "codex_cli"
    assert normalize_provider("sub2api") == "openai_compatible"
    tier = "L1_LOCAL_CAUSAL"
    assert router.acquire(tier, deadline=130, now=100)["status"] == "blocked_needs_human"
    admission = router.acquire(tier, deadline=140, now=100)
    assert admission["timeout_sec"] == 10
    assert admission["deadline"] == 140
    second = router.acquire(tier, deadline=500, now=100)
    assert second["tier"] == "L2_WORKER_POOL"  # 3B/9B share one GPU lock.
    router.release(admission["lease_id"])
    router.release(second["lease_id"])
    with pytest.raises(ValueError):
        router.release(second["lease_id"])
    with ThreadPoolExecutor(max_workers=12) as pool:
        admissions = list(pool.map(lambda _: router.acquire(tier, deadline=500, now=100), range(20)))
    active = [a for a in admissions if a["status"] == "admitted"]
    assert sum(a["tier"] == tier for a in active) == 1
    assert sum(a["tier"] == "L2_WORKER_POOL" for a in active) == 3
    for item in active:
        router.release(item["lease_id"])


def test_router_breaker_resets_then_quarantines_and_escalates():
    router = Router(load_models(ROOT / "config/models.yaml"))
    slot = "local_3b"
    digest = router.record_rejection(slot, "bad patch")
    assert digest == hashlib.sha256(b"bad patch").hexdigest()
    assert not router.record_output(slot, "bad patch")["repeated_patch"]
    assert not router.record_output(slot, "different patch")["repeated_patch"]
    assert router.record_output(slot, "bad patch")["repeat_count"] == 1
    result = router.record_output(slot, "bad patch")
    assert result["repeated_patch"] and result["next_tier"] == "L2_WORKER_POOL"
    for _ in range(4):
        admitted = router.acquire("L1_LOCAL_CAUSAL", deadline=500, now=100)
        assert admitted["slot_id"] != slot
        router.release(admitted["lease_id"])
    router.record_rejection("codex_direct", "x")
    router.record_output("codex_direct", "x")
    assert router.record_output("codex_direct", "x")["status"] == "blocked_needs_human"


def test_router_transport_budget_propagation_failure_fallback_and_release():
    router = Router(load_models(ROOT / "config/models.yaml"))
    deadline = time.monotonic() + 40
    calls = []
    def transport(node, messages, *, timeout, deadline):
        calls.append(node["id"])
        assert 0 < timeout <= 10
        assert deadline > time.monotonic()
        if node["id"].startswith("local_"):
            raise TimeoutError("simulated")
        return '{"ok":true}'
    result = router.call("L1_LOCAL_CAUSAL", [{"role": "user", "content": "x"}], deadline=deadline, transport=transport)
    assert result["tier"] == "L2_WORKER_POOL"
    assert result["output"] == '{"ok":true}'
    assert len(calls) == 3
    assert all(value == 0 for value in router._used.values())


def test_sub2api_payload_protocol_without_network(monkeypatch):
    router = Router(load_models(ROOT / "config/models.yaml"))
    node = router.nodes["qwen_28b"]
    monkeypatch.setenv("SUB2API_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("SUB2API_QWEN28B_MODEL", "upstream-model-id")
    monkeypatch.setenv("SUB2API_API_KEY", "test-only-placeholder")
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"choices":[{"message":{"content":"ok"}}]}'
    def open_request(request, timeout):
        assert request.full_url == "https://gateway.example/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-only-placeholder"
        data = json.loads(request.data)
        assert data["model"] == "upstream-model-id" and data["stream"] is False
        assert timeout <= 2
        return Response()
    monkeypatch.setattr("aegis_core.router.urllib.request.urlopen", open_request)
    assert request_provider(node, [{"role": "user", "content": "hello"}], timeout=2, deadline=time.monotonic() + 3) == "ok"


@pytest.mark.parametrize("scenario", sorted(FAULTS))
def test_two_business_arenas_failure_fix_and_trace_export(tmp_path, scenario):
    ledger = Ledger(tmp_path / "ledger.db")
    result = run_arena(tmp_path / "arena", scenario, ledger)
    assert [row["passed"] for row in result] == [False, True]
    assert result[0]["reward"] < 0 < result[1]["reward"]
    exported = tmp_path / "traces.jsonl"
    assert ledger.export_jsonl(exported) == 8
    rows = [json.loads(line) for line in exported.read_text().splitlines()]
    terminal = [row for row in rows if row["action_a0"].get("op") == "repair"]
    assert len(terminal) == 2
    assert all(row["state_s1"]["baseline_failed"] for row in terminal)
    assert terminal[0]["state_s1"]["exit_code"] == 1
    if scenario == "quant_posside":
        assert "code=51000" in terminal[0]["state_s1"]["stdout"]


def _diff(path, old, new):
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile="a/" + path, tofile="b/" + path))


def test_cheat_test_only_mixed_config_and_legacy_gate_bypass(tmp_path):
    root = tmp_path / "arena"
    arena = create_arena(root, "cat_blank")
    manager = Sandbox(root)
    path = manager.create("cheat")
    testfile = "tests/test_invariant.py"
    original = (path / testfile).read_text()
    cheat = _diff(testfile, original, "def test_fake():\n    assert True\n")
    result = manager.repair("cheat", cheat, fail_before_cmd=arena["test_cmd"], pass_after_cmd=arena["test_cmd"], source_dirs=("engine",))
    assert result["cheat_detected"] and not result["passed"]
    assert result["source_changed_lines"] == 0
    with pytest.raises(ValueError):
        manager.exam("cheat", [sys.executable, "-c", "pass"])
    with pytest.raises(ValueError):
        manager.merge("cheat", "cheat")
    manager.cleanup("cheat")
    path = manager.create("mixed")
    manager.apply_patch("mixed", fixture_patch("cat_blank", positive=True) + cheat)
    assert detect_source_cheat(path, ("engine",))["cheat_detected"]
    manager.cleanup("mixed")


def test_double_probe_invalid_baselines_noop_and_command_not_found(tmp_path):
    root = tmp_path / "arena"
    arena = create_arena(root, "cat_blank")
    manager = Sandbox(root)
    for ticket, command in (("baseline_pass", [sys.executable, "-c", "pass"]), ("missing", ["aegis-does-not-exist"])):
        manager.create(ticket)
        result = manager.repair(ticket, fixture_patch("cat_blank", positive=True), fail_before_cmd=command,
                                pass_after_cmd=arena["test_cmd"], source_dirs=("engine",))
        assert not result["baseline_failed"] and not result["passed"] and result["final"] is None
        manager.cleanup(ticket)
    manager.create("noop")
    result = manager.repair("noop", fixture_patch("cat_blank", positive=False), fail_before_cmd=arena["test_cmd"],
                            pass_after_cmd=arena["test_cmd"], source_dirs=("engine",))
    assert result["baseline_failed"] and not result["after_passed"]
    manager.cleanup("noop")


def test_export_dataset_training_schema_and_paired_rewards(tmp_path):
    destination = tmp_path / "datasets/causal_traces_v1.jsonl"
    assert export_dataset(destination, seed=42) == 32
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    repair = [row for row in rows if row["trainable"]]
    assert len(repair) == 8
    for scenario in FAULTS:
        group = [row for row in repair if row["group_id"] == scenario]
        assert len(group) == 2 and min(row["reward"] for row in group) < 0 < max(row["reward"] for row in group)
    for row in rows:
        assert row["state_s1"] == row["actual_state_s1"]
        assert row["messages"][-1]["role"] == "assistant"
        assert json.loads(row["completion"]) == row["action_a0"]
        assert row["provenance"] == "offline_original_simulation"
        assert -2 <= row["reward"] <= 1.5


def test_config_only_and_untracked_source_detection(tmp_path):
    root = tmp_path / "arena"
    create_arena(root, "cat_blank")
    (root / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts="--ignore=tests"\n')
    assert detect_source_cheat(root, ("engine",))["cheat_detected"]
    (root / "pyproject.toml").unlink()
    (root / "engine/new.py").write_text("value = 1\n")
    result = detect_source_cheat(root, ("engine",))
    assert result["source_changed_lines"] == 1
    assert result["changed_files"] == ["engine/new.py"]


def test_baseline_timeout_not_failure_and_detached_result_credential(tmp_path):
    root = tmp_path / "arena"
    arena = create_arena(root, "cat_blank")
    manager = Sandbox(root)
    manager.create("timeout")
    result = manager.repair("timeout", fixture_patch("cat_blank", positive=True),
                            fail_before_cmd=[sys.executable, "-c", "import time;time.sleep(3)"],
                            pass_after_cmd=arena["test_cmd"], source_dirs=("engine",), timeout=0.05)
    assert not result["baseline_failed"] and result["baseline"]["exit_code"] == 124
    result["passed"] = True
    result["baseline"]["exit_code"] = 1
    with pytest.raises(ValueError):
        manager.merge("timeout", "cannot forge returned result")
    manager.cleanup("timeout")


def test_router_weighted_selection_and_quota_recovery():
    config = load_models(ROOT / "config/models.yaml")
    router = Router(config)
    counts = {"qwen_28b": 0, "deepseek": 0, "glm": 0}
    for _ in range(60):
        admitted = router.acquire("L2_WORKER_POOL", deadline=1000, now=0)
        counts[admitted["slot_id"]] += 1
        router.release(admitted["lease_id"])
    assert counts == {"qwen_28b": 30, "deepseek": 20, "glm": 10}


def test_no_actual_model_id_is_not_silently_invented(monkeypatch):
    router = Router(load_models(ROOT / "config/models.yaml"))
    monkeypatch.delenv("SUB2API_QWEN28B_MODEL", raising=False)
    with pytest.raises(ValueError, match="actual upstream model ID"):
        request_provider(router.nodes["qwen_28b"], [], timeout=1, deadline=time.monotonic() + 2)
    with pytest.raises(ValueError):
        compile_request("t", "fix", SKELETON, {**PROPOSAL, "contract_stub": "def f(x: evil()): ..."})


def test_exported_examples_are_executable_and_anchored():
    examples = json.loads((ROOT / "examples/mediator_cases.json").read_text(encoding="utf-8"))
    assert len(examples) == 4
    for example in examples:
        args = example["request"]
        initial = compile_request(**args)
        assert initial == example["medium_question"]
        resolved_args = {**args, "constraints": example["answered_constraints"]}
        compiled = compile_request(**resolved_args)
        assert compiled == example["compiled_contract"]


def test_router_call_automatically_blocks_replay_and_escalates():
    config = load_models(ROOT / "config/models.yaml")
    config["tiers"]["L1_LOCAL_CAUSAL"][1]["enabled"] = False
    router = Router(config)
    router.record_rejection("local_3b", "bad")
    deadline = time.monotonic() + 120
    seen = []
    def transport(node, messages, *, timeout, deadline):
        seen.append(node["id"])
        return json.dumps({"patch": "bad"}) if node["id"] == "local_3b" else json.dumps({"patch": "new"})
    first = router.call("L1_LOCAL_CAUSAL", [], deadline=deadline, transport=transport)
    assert first["breaker"]["repeat_count"] == 1
    second = router.call("L1_LOCAL_CAUSAL", [], deadline=deadline, transport=transport)
    assert second["tier"] == "L2_WORKER_POOL" and second["deadline"] == deadline
    assert "local_3b" in router._quarantined
    assert all(value == 0 for value in router._used.values())


def test_codex_and_anthropic_adapters_parse_actual_protocol_shapes(monkeypatch):
    router = Router(load_models(ROOT / "config/models.yaml"))
    monkeypatch.setenv("CODEX_DIRECT_MODEL", "configured-model")
    def subprocess_run(command, **kwargs):
        assert command == ["codex", "exec", "--json", "--model", "configured-model", "-"]
        assert json.loads(kwargs["input"])[0]["role"] == "user"
        return subprocess.CompletedProcess(command, 0, stdout='{"type":"thread.started"}\n{"type":"item.completed","item":{"type":"agent_message","text":"final answer"}}\n')
    monkeypatch.setattr("aegis_core.router.subprocess.run", subprocess_run)
    assert request_provider(router.nodes["codex_direct"], [{"role": "user", "content": "hello"}],
                            timeout=1, deadline=time.monotonic() + 2) == "final answer"
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    monkeypatch.setenv("CLAUDE_MODEL", "configured-model")
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'{"content":[{"type":"text","text":"ok"}]}'
    def open_request(request, timeout):
        body = json.loads(request.data)
        assert body["system"] == "system rules"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert request.full_url.endswith('/v1/messages')
        assert request.get_header('Anthropic-version') == '2023-06-01'
        return Response()
    monkeypatch.setattr("aegis_core.router.urllib.request.urlopen", open_request)
    assert request_provider(router.nodes["claude_architect"], [{"role": "system", "content": "system rules"},
                            {"role": "user", "content": "hello"}], timeout=1, deadline=time.monotonic() + 2) == "ok"


@pytest.mark.parametrize("filename", ["config.py", ".gitignore"])
def test_protected_python_config_and_git_exclusions_cannot_bypass_gate(tmp_path, filename):
    root = tmp_path / "arena"
    create_arena(root, "cat_blank")
    (root / filename).write_text("# modified policy\n")
    result = detect_source_cheat(root, ("engine",))
    assert result["cheat_detected"] and filename in result["protected_changed"]
