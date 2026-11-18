"""Offline, original business fixtures and paired causal/SFT dataset exporter.

python -m aegis_core.simulator --output datasets/causal_traces_v1.jsonl
No exchange, renderer dependency, model endpoint or training framework is invoked.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .ledger import Ledger
from .mediator import compile_request
from .reward import compute_step_reward
from .sandbox import Sandbox, tree_hash

REPAIR_SYSTEM_PROMPT = """You are an offline Aegis repair worker. Return exactly the
next repair action as JSON, with op, patch, patch_sha256, target_files,
changed_files, diff_lines, reward_elapsed_sec and latency_budget_sec.
Use only the frozen fixture source and failing invariant. Preserve tests and
configuration. Never contact an exchange or a live model. Do not invent outcome
observations: predicted outcomes are separate from actual sandbox feedback."""
AUDIT_SYSTEM_PROMPT = "Predict the next recorded control-plane audit action as a JSON object."

QUANT = {
    "signal_engine.py": '''from execution_router import submit_order
EXECUTION_PERIOD_SEC = 8 * 60 * 60

def make_order(symbol, position_mode="one_way"):
    return {"symbol": symbol, "posSide": "long", "reduceOnly": False}

def execute_signal(symbol, position_mode="one_way"):
    return submit_order(make_order(symbol, position_mode), position_mode)
''',
    "execution_router.py": '''class ExchangeAPIError(RuntimeError):
    pass

def submit_order(order, position_mode="one_way"):
    if position_mode == "one_way" and order.get("posSide") not in (None, "net"):
        raise ExchangeAPIError('code=51000: posSide invalid in One-Way mode')
    return {"accepted": True, "reduceOnly": order.get("reduceOnly", False)}
''',
    "intraday_guardian.py": '''from execution_router import submit_order
GUARD_INTERVAL_SEC = 60
MAX_DRAWDOWN = 0.05

def guard(peak, equity, elapsed_sec):
    if peak <= 0 or equity < 0:
        raise ValueError("invalid equity")
    drawdown = (peak - equity) / peak
    if elapsed_sec >= GUARD_INTERVAL_SEC and drawdown >= MAX_DRAWDOWN:
        order = {"symbol": "BTC-USDT-SWAP", "reduceOnly": False}
        return submit_order(order, "one_way")
    return {"accepted": False, "reduceOnly": False}
''',
}
CAT = {
    "engine/__init__.py": "",
    "engine/cat_kinematics.py": '''from dataclasses import dataclass

@dataclass(frozen=True)
class Cat:
    name: str
    foot_anchor_y: float
    body_anchor_y: float
''',
    "engine/iso_renderer.py": '''from statistics import pvariance
from .cat_kinematics import Cat

def depth_order(cats: list[Cat]) -> list[str]:
    return [cat.name for cat in sorted(cats, key=lambda cat: cat.body_anchor_y)]

def render_center_roi() -> list[int]:
    return [120] * 64

def visual_gate() -> bool:
    return pvariance(render_center_roi()) >= 18.5
''',
}
TESTS = {
    "quant_posside": '''from signal_engine import execute_signal, EXECUTION_PERIOD_SEC

def test_order_mode():
    assert EXECUTION_PERIOD_SEC == 28800
    assert execute_signal("BTC-USDT-SWAP", "one_way")["accepted"] is True
''',
    "quant_guard": '''from intraday_guardian import guard, GUARD_INTERVAL_SEC

def test_guard_threshold():
    assert GUARD_INTERVAL_SEC == 60
    assert guard(1000, 950, 60)["reduceOnly"] is True
    assert guard(1000, 951, 60)["accepted"] is False
    assert guard(1000, 950, 59)["accepted"] is False
''',
    "cat_depth": '''from engine.cat_kinematics import Cat
from engine.iso_renderer import depth_order

def test_depth_anchor():
    cats = [Cat("back", 10, 99), Cat("front", 20, 1)]
    assert depth_order(cats) == ["back", "front"]
''',
    "cat_blank": '''from statistics import pvariance
from engine.iso_renderer import render_center_roi, visual_gate

def test_center_roi():
    assert pvariance(render_center_roi()) >= 18.5
    assert visual_gate() is True
''',
}
# Each tuple: source path, buggy token, corrected token, plausible but wrong token.
FAULTS = {
    "quant_posside": ("signal_engine.py", '"posSide": "long"', '"posSide": "net" if position_mode == "one_way" else "long"', '"posSide": "short"'),
    "quant_guard": ("intraday_guardian.py", 'order = {"symbol": "BTC-USDT-SWAP", "reduceOnly": False}', 'order = {"symbol": "BTC-USDT-SWAP", "reduceOnly": True}', 'order = {"symbol": "BTC-USDT-SWAP", "reduceOnly": 0}'),
    "cat_depth": ("engine/iso_renderer.py", 'key=lambda cat: cat.body_anchor_y', 'key=lambda cat: cat.foot_anchor_y', 'key=lambda cat: -cat.foot_anchor_y'),
    "cat_blank": ("engine/iso_renderer.py", 'return [120] * 64', 'return [100, 140] * 32', 'return [121] * 64'),
}


# Additional templates for the combined stage 3/4 dataset. V2 FAULTS and its
# 32-row export intentionally remain unchanged for backwards compatibility.
_EXTRA_QUANT_SIZE = "\ndef within_notional(amount, cap):\n    return amount >= cap\n"
_EXTRA_CAT_ANCHOR = "\ndef foot_depth(cat):\n    return cat.body_anchor_y\n"
_EXTRA_CAT_PROJECT = "\ndef project_foot(x, y, camera_offset):\n    return (x - y, (x + y) / 2 - camera_offset)\n"
ALL_FAULTS = {**FAULTS,
    "quant_notional_cap": ("execution_router.py", "return amount >= cap", "return 0 < amount <= cap", "return amount < 0"),
    "quant_drawdown_boundary": ("intraday_guardian.py", "drawdown > MAX_DRAWDOWN", "drawdown >= MAX_DRAWDOWN", "drawdown >= MAX_DRAWDOWN + 0.01"),
    "quant_guard_period": ("intraday_guardian.py", "GUARD_INTERVAL_SEC = 120", "GUARD_INTERVAL_SEC = 60", "GUARD_INTERVAL_SEC = 180"),
    "quant_signal_period": ("signal_engine.py", "EXECUTION_PERIOD_SEC = 24 * 60 * 60", "EXECUTION_PERIOD_SEC = 8 * 60 * 60", "EXECUTION_PERIOD_SEC = 12 * 60 * 60"),
    "cat_anchor_contract": ("engine/cat_kinematics.py", "return cat.body_anchor_y", "return cat.foot_anchor_y", "return -cat.foot_anchor_y"),
    "cat_depth_ties": ("engine/iso_renderer.py", "key=lambda cat: (cat.foot_anchor_y, cat.name)", "key=lambda cat: cat.foot_anchor_y", "key=lambda cat: cat.body_anchor_y"),
    "cat_visual_threshold": ("engine/iso_renderer.py", ">= 1000", ">= 18.5", ">= 2000"),
    "cat_camera_offset": ("engine/cat_kinematics.py", "/ 2 - camera_offset", "/ 2 + camera_offset", "/ 2 - camera_offset * 2"),
}
TESTS.update({
    "quant_notional_cap": "from execution_router import within_notional\ndef test_cap():\n    assert within_notional(100, 100)\n    assert not within_notional(101, 100)\n    assert not within_notional(0, 100)\n",
    "quant_drawdown_boundary": TESTS["quant_guard"],
    "quant_guard_period": TESTS["quant_guard"],
    "quant_signal_period": "from signal_engine import EXECUTION_PERIOD_SEC\ndef test_period():\n    assert EXECUTION_PERIOD_SEC == 28800\n",
    "cat_anchor_contract": "from engine.cat_kinematics import Cat, foot_depth\ndef test_anchor():\n    assert foot_depth(Cat('cat', 20, 99)) == 20\n",
    "cat_depth_ties": "from engine.cat_kinematics import Cat\nfrom engine.iso_renderer import depth_order\ndef test_ties():\n    assert depth_order([Cat('z', 20, 99), Cat('a', 20, 1)]) == ['z', 'a']\n",
    "cat_visual_threshold": "from engine.iso_renderer import visual_gate\ndef test_threshold():\n    assert visual_gate() is True\n",
    "cat_camera_offset": "from engine.cat_kinematics import project_foot\ndef test_camera():\n    assert project_foot(20, 10, 10) == (10, 25)\n",
})


def fixture_files(scenario):
    files = dict(QUANT if scenario.startswith("quant_") else CAT)
    if scenario == "quant_notional_cap":
        files["execution_router.py"] += _EXTRA_QUANT_SIZE
    elif scenario in {"quant_drawdown_boundary", "quant_guard_period"}:
        files["intraday_guardian.py"] = files["intraday_guardian.py"].replace('"reduceOnly": False}', '"reduceOnly": True}', 1)
        if scenario == "quant_drawdown_boundary":
            files["intraday_guardian.py"] = files["intraday_guardian.py"].replace("drawdown >= MAX_DRAWDOWN", "drawdown > MAX_DRAWDOWN")
        else:
            files["intraday_guardian.py"] = files["intraday_guardian.py"].replace("GUARD_INTERVAL_SEC = 60", "GUARD_INTERVAL_SEC = 120")
    elif scenario == "quant_signal_period":
        files["signal_engine.py"] = files["signal_engine.py"].replace("8 * 60 * 60", "24 * 60 * 60")
    elif scenario == "cat_anchor_contract":
        files["engine/cat_kinematics.py"] += _EXTRA_CAT_ANCHOR
    elif scenario == "cat_camera_offset":
        files["engine/cat_kinematics.py"] += _EXTRA_CAT_PROJECT
    elif scenario == "cat_depth_ties":
        files["engine/iso_renderer.py"] = files["engine/iso_renderer.py"].replace("key=lambda cat: cat.body_anchor_y", "key=lambda cat: (cat.foot_anchor_y, cat.name)")
    elif scenario == "cat_visual_threshold":
        files["engine/iso_renderer.py"] = files["engine/iso_renderer.py"].replace("return [120] * 64", "return [100, 140] * 32").replace(">= 18.5", ">= 1000")
    return files


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def create_arena(root: Path, scenario: str) -> dict:
    if scenario not in ALL_FAULTS:
        raise ValueError("unknown arena scenario")
    return freeze_arena(root, scenario, fixture_files(scenario), TESTS[scenario],
                        target=ALL_FAULTS[scenario][0],
                        source_dirs=(".",) if scenario.startswith("quant_") else ("engine",))


def freeze_arena(root: Path, scenario, sources, probe, *, target, source_dirs):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name, source in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="\n")
    (root / "tests").mkdir()
    (root / "tests/test_invariant.py").write_text(probe, encoding="utf-8", newline="\n")
    (root / ".gitignore").write_text(".worktrees/\n__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Aegis Offline Simulator")
    _git(root, "config", "user.email", "simulator@localhost")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Freeze original offline arena contract and failing invariant")
    return {"scenario": scenario, "fixture_sha256": tree_hash(root), "source_dirs": source_dirs,
            "target_files": [target], "test_cmd": [sys.executable, "-B", "-m", "pytest", "-q", "--tb=short"]}


def fixture_patch(scenario: str, *, positive: bool) -> str:
    path, old, good, bad = ALL_FAULTS[scenario]
    source = fixture_files(scenario)[path]
    replacement = source.replace(old, good if positive else bad)
    return "".join(difflib.unified_diff(source.splitlines(keepends=True), replacement.splitlines(keepends=True),
                                       fromfile="a/" + path, tofile="b/" + path))


def run_arena(root: Path, scenario: str, ledger: Ledger) -> list[dict]:
    arena = create_arena(root, scenario)
    manager = Sandbox(root)
    records = []
    for label, positive in (("negative", False), ("positive", True)):
        ticket = scenario + "-" + label
        patch = fixture_patch(scenario, positive=positive)
        ledger.create_ticket(ticket, "修复离线靶场 " + scenario,
                             {"scenario": scenario}, "def repair() -> None: ...")
        token = ledger.acquire_lease(ticket, "offline-simulator", ttl=300)
        path = manager.create(ticket)
        try:
            compile_request(ticket, "修复离线仿真", list(fixture_files(scenario)),
                            {"target_files": arena["target_files"], "contract_stub": "def repair() -> None: ..."},
                            constraints={"execution_mode": "dry_run"}, ledger=ledger, lease_token=token)
            state = ledger.get_state(ticket)
            running = {"status": "running", "tree_hash": tree_hash(path), "scenario": scenario,
                       "fixture_sha256": arena["fixture_sha256"]}
            ledger.transition(ticket, token, "COMPILED", "running", state_s0=state,
                              action_a0={"op": "create_worktree"}, actual_state_s1=running, reward=0)
            result = manager.repair(ticket, patch, fail_before_cmd=arena["test_cmd"],
                                    pass_after_cmd=arena["test_cmd"], source_dirs=arena["source_dirs"], timeout=60)
            # Synthetic cost clock for reproducible offline labels; actual wall time
            # is retained in measured_elapsed_sec for operational analysis.
            reward = compute_step_reward(baseline_failed=result["baseline_failed"], after_passed=result["after_passed"],
                                         target_files=arena["target_files"], changed_files=result["changed_files"],
                                         diff_lines=result["diff_lines"], elapsed_sec=1.0, latency_budget_sec=60,
                                         cheat_detected=result["cheat_detected"])
            terminal = "succeeded" if result["passed"] else "failed"
            feedback = {"status": terminal, "tree_hash": result["after_hash"],
                        "baseline_failed": result["baseline_failed"], "after_passed": result["after_passed"],
                        "cheat_detected": result["cheat_detected"], "fixture_sha256": arena["fixture_sha256"],
                        "baseline_exit_code": result["baseline"]["exit_code"],
                        "baseline_stdout": result["baseline"]["stdout"],
                        "baseline_stderr": result["baseline"]["stderr"],
                        "exit_code": result["final"]["exit_code"] if result["final"] else None,
                        "stdout": result["final"]["stdout"] if result["final"] else "",
                        "stderr": result["final"]["stderr"] if result["final"] else "",
                        "measured_elapsed_sec": result["elapsed_sec"]}
            action = {"op": "repair", "patch": patch, "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                      "target_files": arena["target_files"], "changed_files": result["changed_files"],
                      "diff_lines": result["diff_lines"], "reward_elapsed_sec": 1.0, "latency_budget_sec": 60}
            ledger.transition(ticket, token, "running", terminal, state_s0=running, action_a0=action,
                              actual_state_s1=feedback, predicted_next_state={"status": "succeeded"}, reward=reward)
            if positive != result["passed"] or not result["baseline_failed"]:
                raise AssertionError("arena outcome disagrees with frozen fault contract")
            if positive:
                manager.merge(ticket, "Fix offline " + scenario)
            records.append({"ticket_id": ticket, "group_id": scenario, "reward": reward, "passed": result["passed"]})
        finally:
            manager.cleanup(ticket)
    return records


def export_dataset(destination: Path, *, seed: int = 0) -> int:
    """Export all lifecycle transitions, with paired repair samples for GRPO.

    This is a framework-neutral dataset. QLoRA/GRPO trainer-specific loading and
    online policy sampling still require an external ML stack, not this package.
    """
    if type(seed) is not int:
        raise ValueError("integer dataset seed required")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aegis-arenas-") as directory:
        workspace = Path(directory)
        ledger = Ledger(workspace / "ledger.sqlite")
        for scenario in sorted(FAULTS):
            run_arena(workspace / scenario, scenario, ledger)
        raw = workspace / "raw.jsonl"
        ledger.export_jsonl(raw)
        rows = []
        for line in raw.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            scenario = row["ticket_id"].rsplit("-", 1)[0]
            repair = row["action_a0"].get("op") == "repair"
            prompt = json.dumps({"state_s0": row["state_s0"], "scenario": scenario,
                                 "fixture_source": QUANT if scenario.startswith("quant_") else CAT,
                                 "frozen_test": TESTS[scenario]}, ensure_ascii=False, sort_keys=True)
            completion = json.dumps(row["action_a0"], ensure_ascii=False, sort_keys=True)
            row.update({"schema_version": "aegis.causal.v2", "provenance": "offline_original_simulation",
                        "seed": seed, "group_id": scenario, "sample_role": "repair" if repair else "lifecycle",
                        "trainable": repair, "prompt": prompt, "completion": completion,
                        "messages": [{"role": "system", "content": REPAIR_SYSTEM_PROMPT if repair else AUDIT_SYSTEM_PROMPT},
                                     {"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]})
            rows.append(row)
        # Same-filesystem atomic publish, including when destination is on another disk.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent,
                                         prefix=".causal-", suffix=".jsonl", delete=False) as output:
            temporary = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/causal_traces_v1.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    count = export_dataset(args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output), "transitions": count, "provenance": "offline_original_simulation"}))


if __name__ == "__main__":
    main()
