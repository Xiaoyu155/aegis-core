import gzip
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aegis_core.ast_slicer import ImportGraph
from aegis_core.cli import main as aegis
from aegis_core.contract_lock import InterfaceContractLock
from aegis_core.pm_expander import expand_requirement
from aegis_core.reviewer import audit_candidate
from aegis_core.reward import compute_step_reward
from aegis_core.router import route_complexity
from aegis_core.simulator import freeze_arena
from training.common import ByteTokenizer, load_preflight, lora_settings, verify_dataset
from training.train_grpo import load_groups, main as grpo, prepare_groups, score_patch
from training.train_qlora_sft import load_conversations, main as sft

ROOT = Path(__file__).resolve().parents[1]
BASE = """from typing import TypedDict

class State(TypedDict):
    ready: bool

STATE = {'ready': False, 'nested': {'count': 0}}

def value(x: int = 1, /, *, scale: int = 1) -> dict:
    return {'result': x * scale}

def _private(arg):
    return arg
"""


def _sources(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _jsonl(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(("before", "after", "symbol", "kind"), [
    ("x: int", "x: float", "value", "signature"),
    ("scale: int = 1", "scale: int = 2", "value", "signature"),
    (", /,", ",", "value", "signature"),
    ("*, scale", "scale", "value", "signature"),
    ("-> dict", "-> State", "value", "signature"),
    ("def value", "async def value", "value", "signature"),
    ("'result':", "'value':", "value", "return_keys"),
    ("'count':", "'size':", "STATE", "state_keys"),
    ("ready: bool", "available: bool", "State", "state_keys"),
    ("ready: bool", "ready: int", "State.ready", "annotation"),
])
def test_lock_rejects_visible_interface_drift(tmp_path, before, after, symbol, kind):
    root = _sources(tmp_path, {"api.py": BASE, "consumer.py": "import api\n\ndef call():\n    return api.value()\n"})
    lock = InterfaceContractLock.capture(ImportGraph(root), "baseline", ["api.py"])
    (root / "api.py").write_text(BASE.replace(before, after, 1), encoding="utf-8")
    differences = lock.violations(root, ["api.py"])
    assert any(row["symbol"] == symbol and row["kind"] == kind and row["observed"] != row["expected"]
               for row in differences)


def test_lock_allows_internal_implementation_and_private_helper_changes(tmp_path):
    root = _sources(tmp_path, {"api.py": BASE, "consumer.py": "from api import value\n"})
    lock = InterfaceContractLock.capture(ImportGraph(root), "baseline", ["api.py"])
    changed = BASE.replace("x * scale", "(x * scale) + 1").replace("_private(arg)", "_private(first, second=0)")
    (root / "api.py").write_text(changed, encoding="utf-8")
    assert lock.violations(root, ["api.py"]) == []


def test_imported_private_python_function_is_locked(tmp_path):
    root = _sources(tmp_path, {"api.py": BASE, "consumer.py": "from api import _private as helper\n"})
    lock = InterfaceContractLock.capture(ImportGraph(root), "baseline", ["api.py"])
    assert any(entry.symbol == "_private" for entry in lock.entries)


@pytest.mark.parametrize(("source", "new", "kind"), [
    ("STATE = {}\n\ndef set_state():\n    STATE['enabled'] = True\n", "'available'", "written_state_keys"),
    ("from typing import TypeAlias\n\nResult: TypeAlias = list[int]\n", "list[str]", "type_alias"),
    ("from dataclasses import dataclass\n\n@dataclass\nclass State:\n    enabled: bool = True\n",
     "enabled: bool = False", "dataclass_fields"),
])
def test_extended_python_state_contract(tmp_path, source, new, kind):
    symbols = {"written_state_keys": "STATE", "type_alias": "Result", "dataclass_fields": "State"}
    previous = {"written_state_keys": "'enabled'", "type_alias": "list[int]", "dataclass_fields": "enabled: bool = True"}
    root = _sources(tmp_path, {"api.py": source, "consumer.py": f"from api import {symbols[kind]}\n"})
    lock = InterfaceContractLock.capture(ImportGraph(root), "baseline", ["api.py"])
    (root / "api.py").write_text(source.replace(previous[kind], new), encoding="utf-8")
    assert any(row["kind"] == kind for row in lock.violations(root, ["api.py"]))


def test_lock_is_frozen_and_fails_digest_tampering(tmp_path):
    root = _sources(tmp_path, {"api.py": BASE, "consumer.py": "from api import value\n"})
    lock = InterfaceContractLock.capture(ImportGraph(root), "baseline", ["api.py"])
    restored = InterfaceContractLock.from_dict(lock.to_dict())
    assert restored == lock
    with pytest.raises(FrozenInstanceError):
        restored.baseline = "new"
    document = lock.to_dict()
    document["entries"][0]["expected"] = "modified"
    with pytest.raises(ValueError, match="digest"):
        InterfaceContractLock.from_dict(document)


@pytest.mark.parametrize(("change", "kind", "symbol"), [
    (("value: number", "value: string"), "signature", "publish"),
    (("count: 0", "items: 0"), "state_keys", "state"),
    (("enabled: boolean", "enabled: number"), "annotation", "State"),
    (("module.exports = {publish, state}", "module.exports = {state}"), "export", "publish"),
])
def test_typescript_contract_with_relative_extension(tmp_path, change, kind, symbol):
    source = """interface State { enabled: boolean; }
const state = {count: 0};
function publish(value: number): number { return value; }
module.exports = {publish, state};
"""
    root = _sources(tmp_path, {"app/api.ts": source, "app/client/use.ts": "import {publish} from '../api.ts';\n"})
    graph = ImportGraph(root)
    assert graph.file_dependencies["app/client/use.ts"] == ["app/api.ts"]
    lock = InterfaceContractLock.capture(graph, "baseline", ["app/api.ts"])
    (root / "app/api.ts").write_text(source.replace(*change), encoding="utf-8")
    assert any(row["kind"] == kind and row["symbol"] == symbol for row in lock.violations(root, ["app/api.ts"]))


@pytest.mark.parametrize(("length", "risk", "level", "radius"), [
    (0, None, "L1", 0),
    (1, None, "L2", 1),
    (3, None, "L3", 3),
    (0, .34, "L1", 0),
    (0, .35, "L2", 0),
    (0, .70, "L3", 0),
])
def test_complexity_uses_transitive_callers_and_risk(tmp_path, length, risk, level, radius):
    files = {"node0.py": "def fn():\n    return 1\n",
             "tests/test_nodes.py": "from node0 import fn\n"}
    for index in range(1, length + 1):
        files[f"node{index}.py"] = f"from node{index - 1} import fn as prev\n\ndef fn():\n    return prev()\n"
    root = _sources(tmp_path, files)
    route = route_complexity(ImportGraph(root), ["node0.py"], risk_probability=risk)
    assert route["level"] == level and route["tier"].startswith(level)
    assert route["impact_radius"] == radius and route["coupling"] == bool(length)
    assert len(route["impacted_files"]) == length + 1


@pytest.mark.parametrize("risk", [-.1, 1.1, True, float("nan")])
def test_router_rejects_invalid_risk(tmp_path, risk):
    with pytest.raises(ValueError, match="risk probability"):
        route_complexity(ImportGraph(tmp_path), [], risk_probability=risk)


@pytest.mark.parametrize(("new", "finding"), [
    ("def thing(n):\n    return 4 / n\n", "UNGUARDED_DIVISOR"),
    ("def thing(n):\n    return 4 / len(n)\n", "UNGUARDED_DIVISOR"),
    ("def thing(n):\n    return 4 // 0\n", "DIVISION_BY_ZERO"),
    ("cache = []\n\ndef thing(n):\n    cache.append(n)\n", "UNBOUNDED_GROWTH"),
    ("class App:\n    def thing(self, n):\n        self.history.append(n)\n", "UNBOUNDED_GROWTH"),
    ("def thing(n):\n    try:\n        return int(n)\n    except Exception:\n        pass\n", "EXCEPTION_SWALLOWED"),
    ("def thing(n):\n    try:\n        return int(n)\n    except:\n        return None\n", "EXCEPTION_SWALLOWED"),
    ("def thing(n):\n    try:\n        return int(n)\n    except ValueError:\n        pass\n", "EXCEPTION_SWALLOWED"),
    ("def thing(n):\n    return connect(api_key='literal-value')\n", "HARDCODED_SECRET"),
    ("def thing(n):\n    return {'access_token': 'literal-value'}\n", "HARDCODED_SECRET"),
    ("def thing(n):\n    apiKey = 'literal-value'\n    return apiKey\n", "HARDCODED_SECRET"),
    ("def risk_guard(n):\n    return True\n", "CONTROL_GUARD_REMOVED"),
    ("def risk_guard(n):\n    if n < 0 and False:\n        return False\n    return True\n", "CONTROL_GUARD_REMOVED"),
    ("def risk_guard(n):\n    return True\n    if n < 0:\n        return False\n", "CONTROL_GUARD_REMOVED"),
])
def test_independent_audit_rejects_new_regressions(tmp_path, new, finding):
    root = tmp_path / "repo"
    freeze_arena(root, "audit", {"service.py": "def risk_guard(n):\n    if n < 0:\n        return False\n    return True\n"},
                 "", target="service.py", source_dirs=(".",))
    (root / "service.py").write_text(new, encoding="utf-8")
    review = audit_candidate(root, "HEAD", ["service.py"])
    assert review["passed"] is False
    assert finding in {row["code"] for row in review["findings"]}
    assert "literal-value" not in json.dumps(review)


@pytest.mark.parametrize("new", [
    "def thing(n):\n    if n == 0:\n        return 0\n    return 4 / n\n",
    "def thing(n):\n    if n:\n        return 4 / n\n    return 0\n",
    "def thing(n):\n    try:\n        return 4/n\n    except ZeroDivisionError:\n        return 0\n",
    "def thing(n):\n    local = []\n    local.append(n)\n    return local\n",
    "from collections import deque\n\nclass App:\n    def __init__(self):\n        self.history = deque(maxlen=5)\n    def thing(self, n):\n        self.history.append(n)\n",
    "class App:\n    def thing(self, n):\n        if len(self.history) >= 5:\n            self.history.pop(0)\n        self.history.append(n)\n",
    "def thing(n):\n    secret_env = 'REQUIRED_TOKEN'\n    return secret_env\n",
])
def test_audit_accepts_guarded_or_local_behavior(tmp_path, new):
    root = tmp_path / "repo"
    freeze_arena(root, "audit", {"service.py": "def thing(n):\n    return n\n"},
                 "", target="service.py", source_dirs=(".",))
    (root / "service.py").write_text(new, encoding="utf-8")
    assert audit_candidate(root, "HEAD", ["service.py"])["passed"]


def test_existing_audit_problem_does_not_block_unrelated_edit(tmp_path):
    old = "def thing(n):\n    return 4/n\n"
    root = tmp_path / "repo"
    freeze_arena(root, "audit", {"service.py": old}, "", target="service.py", source_dirs=(".",))
    (root / "service.py").write_text("# Keep current behavior.\n" + old, encoding="utf-8")
    assert audit_candidate(root, "HEAD", ["service.py"])["findings"] == []


@pytest.mark.parametrize("after_passed", [False, True])
def test_blocked_semantic_gate_reward_is_negative(after_passed):
    assert compute_step_reward(baseline_failed=True, after_passed=after_passed, target_files=["module.py"],
                               changed_files=["module.py"], diff_lines=1, elapsed_sec=0, latency_budget_sec=10,
                               semantic_gate_rejected=True) <= -.5


@pytest.mark.parametrize(("text", "answers", "questions"), [
    ("Keep cache memory bounded", {}, []),
    ("Add drawdown protection", {}, ["max_drawdown"]),
    ("Add drawdown protection at 8%", {}, []),
    ("Add drawdown protection", {"max_drawdown": .08}, []),
    ("Add drawdown protection with circuit break and liquidate positions", {},
     ["max_drawdown", "close_method", "unlock_condition"]),
    ("Use {{max_items}} items in the response", {}, ["max_items"]),
    ("Use {{max_items}} items in the response", {"max_items": 7}, []),
])
def test_expanded_spec_only_returns_unresolved_business_questions(tmp_path, text, answers, questions):
    root = _sources(tmp_path, {"app.py": "def cache(drawdown):\n    return drawdown\n"})
    spec = expand_requirement(text, root, answers=answers).to_dict()
    assert [item["parameter"] for item in spec["questions"]] == questions
    assert [row["kind"] for row in spec["acceptance_criteria"]] == ["normal", "boundary", "fallback"]
    assert bool(spec["engineering_invariants"] and spec["edge_cases"])
    assert spec["status"] == ("MISSING_CONSTRAINT" if questions else "READY")


@pytest.mark.parametrize("answers", [{"max_drawdown": 0}, {"max_drawdown": True}, {"unknown": 1}])
def test_spec_rejects_invalid_business_answers(tmp_path, answers):
    with pytest.raises(ValueError):
        expand_requirement("Add drawdown protection", tmp_path, answers=answers)


def test_spec_uses_an_existing_explicitly_preserved_threshold(tmp_path):
    root = _sources(tmp_path, {"guard.py": "MAX_DRAWDOWN = .05\n\ndef drawdown(value):\n    return value > MAX_DRAWDOWN\n"})
    spec = expand_requirement("Keep current drawdown protection in guard.py", root).to_dict()
    assert spec["business_parameters"] == {"max_drawdown": .05}
    assert spec["context"]["business_parameter_sources"] == {"max_drawdown": ["guard.py"]}
    assert not spec["questions"]


def test_spec_does_not_solicit_credentials(tmp_path):
    with pytest.raises(ValueError, match="cannot contain credentials"):
        expand_requirement("Connect using {{api_key}}", tmp_path)


def test_parallel_plan_rejects_overlapping_source_targets(tmp_path, capsys):
    root = tmp_path / "repo"
    freeze_arena(root, "overlap", {"app.py": "def value():\n    return 0\n"}, "",
                 target="app.py", source_dirs=(".",))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": "aegis.task_dag.v1", "repo_root": str(root),
        "base_commit": head, "tasks": [
            {"task_id": item, "kind": "repair", "depends_on": [], "contract": {"target_files": ["app.py"]}}
            for item in ("left", "right")]}), encoding="utf-8")
    assert aegis(["run", "--dag", str(plan)]) == 1
    assert "parallel repair targets overlap" in capsys.readouterr().err
    assert not (root / ".worktrees").exists()


def test_plan_refuses_execution_before_business_answer(tmp_path, capsys):
    root = tmp_path / "repo"
    freeze_arena(root, "policy", {"policy.py": "def drawdown(n):\n    return n\n"}, "",
                 target="policy.py", source_dirs=(".",))
    plan = tmp_path / "plan.json"
    assert aegis(["plan", "Add policy.py drawdown protection", "--repo", str(root), "--out", str(plan)]) == 2
    plan_result = json.loads(capsys.readouterr().out)
    assert plan_result["spec"]["questions"][0]["parameter"] == "max_drawdown"
    assert aegis(["run", "--dag", str(plan)]) == 1
    assert "outstanding business parameters" in capsys.readouterr().err
    answers = tmp_path / "answers.json"
    answers.write_text('{"max_drawdown": 0.08}', encoding="utf-8")
    assert aegis(["plan", "Add policy.py drawdown protection", "--repo", str(root), "--out", str(plan),
                  "--answers", str(answers)]) == 0
    result = json.loads(plan.read_text(encoding="utf-8"))
    repair = next(task for task in result["tasks"] if task["kind"] == "repair")
    assert repair["contract"]["business_parameters"] == {"max_drawdown": .08}


@pytest.mark.parametrize("rank", [4, 8, 16])
def test_lora_preflight_validates_rank(rank):
    settings = lora_settings(rank)
    assert settings["r"] == rank and settings["lora_alpha"] == 2 * rank
    assert settings["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


@pytest.mark.parametrize("rank", [0, True, 32])
def test_lora_preflight_rejects_invalid_rank(rank):
    with pytest.raises(ValueError, match="LoRA"):
        lora_settings(rank)


@pytest.mark.parametrize("entry,file,key,count", [
    (sft, "mediator_sft_train.jsonl", "rows", 472),
    (grpo, "grpo_repair_pairs.jsonl", "groups", 264),
])
def test_latest_dataset_runs_cpu_training_preflight(tmp_path, capsys, entry, file, key, count):
    assert entry(["--dry-run", "--dataset", str(ROOT / "datasets" / file), "--output", str(tmp_path)]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert not metrics["gpu_used"] and metrics["checkpoint_verified"] and metrics["prompt_truncations"] > 0
    checkpoint = load_preflight(tmp_path / metrics["checkpoint"])
    assert len(checkpoint[key]) == count and len(checkpoint["dataset_sha256"]) == 2
    assert checkpoint["lora"]["r"] == metrics["lora_rank"]
    if key == "rows":
        for row in checkpoint["rows"]:
            assert len(row["input_ids"]) == len(row["labels"]) == len(row["attention_mask"]) <= 1024
            assert row["labels"][0] == -100 and row["labels"][-1] == row["input_ids"][-1]
    else:
        for group in checkpoint["groups"]:
            assert len(group["prompt_ids"]) <= 768
            assert sum(row["advantage"] for row in group["candidates"]) == pytest.approx(0)
            assert all(0 < len(row["completion_ids"]) <= 256 for row in group["candidates"])


def test_preflight_checkpoint_detects_corruption(tmp_path, capsys):
    dataset = tmp_path / "custom.jsonl"
    _jsonl(dataset, {"messages": [{"role": "user", "content": "request"}, {"role": "assistant", "content": "{}"}]})
    assert sft(["--dry-run", "--dataset", str(dataset), "--output", str(tmp_path)]) == 0
    path = tmp_path / json.loads(capsys.readouterr().out)["checkpoint"]
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    envelope["payload"]["lora"]["r"] = 32
    path.write_bytes(gzip.compress(json.dumps(envelope).encode()))
    with pytest.raises(ValueError, match="checksum"):
        load_preflight(path)


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": "system"}, {"role": "assistant", "content": "{}"}],
    [{"role": "user", "content": "request"}, {"role": "assistant", "content": "[]"}],
    [{"role": "user", "content": ""}, {"role": "assistant", "content": "{}"}],
])
def test_sft_rejects_invalid_conversations(tmp_path, messages):
    path = tmp_path / "dataset.jsonl"
    _jsonl(path, {"messages": messages})
    with pytest.raises(ValueError):
        load_conversations(path)


def _rows(right=None):
    prompt = [{"role": "user", "content": "fix the output"}]
    row = {"group_id": "group", "scenario": "case", "prompt": prompt, "completion": '{"patch":"diff"}'}
    return {**row, "reward": -1.0}, {**row, "reward": 2.0, **(right or {})}


@pytest.mark.parametrize("corruption", [
    {"reward": True},
    {"reward": float("inf")},
    {"reward": 0.0},
    {"completion": "{}"},
    {"patch_sha256": "bad"},
    {"scenario": "another"},
    {"prompt": [{"role": "user", "content": "different"}]},
])
def test_grpo_rejects_invalid_candidate_pairs(tmp_path, corruption):
    path = tmp_path / "dataset.jsonl"
    _jsonl(path, *_rows(corruption))
    with pytest.raises(ValueError):
        load_groups(path)


def test_normalization_uses_group_sample_variance(tmp_path):
    path = tmp_path / "dataset.jsonl"
    _jsonl(path, *_rows())
    groups, metrics = prepare_groups(load_groups(path), ByteTokenizer(), 128, 128)
    advantages = [row["advantage"] for row in groups[0]["candidates"]]
    assert advantages[1] == pytest.approx(1.5 / ((4.5 ** .5) + 1e-4))
    assert advantages[0] == pytest.approx(-advantages[1])
    assert metrics["reward_groups_normalized"] == 1


def test_training_manifest_rejects_modified_supplement(tmp_path):
    baseline = tmp_path / "base.jsonl"
    supplement = tmp_path / "cats.jsonl"
    baseline.write_text("{}\n", encoding="utf-8")
    supplement.write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest_v3.json").write_text(json.dumps({"sha256": {"cats.jsonl": "incorrect"},
        "cat_cafe_replay": {"supplements": {"base.jsonl": "cats.jsonl"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="cats.jsonl"):
        verify_dataset(baseline)


def test_new_cat_cafe_grpo_scenario_uses_physical_reward():
    path = ROOT / "datasets" / "cat_cafe_replay_grpo.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows[:2]:
        outcome = score_patch(row["scenario"], row["completion"])
        assert (outcome > 0) == row["positive"]


def test_whitepaper_drill_with_all_four_sections(tmp_path):
    import shutil

    dataset = tmp_path / "datasets"
    dataset.mkdir()
    for filename in ("manifest_v3.json", "mediator_sft_train.jsonl", "mediator_sft_eval.jsonl",
                     "grpo_repair_pairs.jsonl"):
        shutil.copyfile(ROOT / "datasets" / filename, dataset / filename)
    output = subprocess.run([sys.executable, str(ROOT / "scripts/run_whitepaper_v10_drill.py"),
                             "--dataset-dir", str(dataset), "--output", str(tmp_path / "checkpoints")],
                            cwd=ROOT, check=True, capture_output=True, encoding="utf-8")
    sections = dict(line.split(" ", 1) for line in output.stdout.splitlines())
    assert json.loads(sections["CONTRACT"])["reason"] == "INTERFACE_CONTRACT_VIOLATION"
    assert json.loads(sections["REVIEW"])["green_exit"] == 0
    assert json.loads(sections["REVIEW_PASS"])["passed"]
    assert json.loads(sections["PM_SPEC"])["experience_bank_version"] == 3
    assert json.loads(sections["DATASET_V3"])["physical_attempts"] == 8
    assert json.loads(sections["RESULT"]) == {"status": "PASS", "sections_passed": 4}
    for filename in ("cat_cafe_replay_sft.jsonl", "cat_cafe_replay_grpo.jsonl", "manifest_v3.json"):
        assert (dataset / filename).read_bytes() == (ROOT / "datasets" / filename).read_bytes()
