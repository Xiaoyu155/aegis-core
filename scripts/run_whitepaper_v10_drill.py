"""Run the PR #10 acceptance drill with authored patches and actual Git worktrees."""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from aegis_core.ast_slicer import ImportGraph
from aegis_core.cat_cafe_replay import update_dataset
from aegis_core.experience_bank import ExperienceBank
from aegis_core.ledger import Ledger
from aegis_core.simulator import freeze_arena

ROOT = Path(__file__).resolve().parents[1]


def _report(section, value):
    print(f"{section} " + json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _command(*arguments, cwd=ROOT, expected=0):
    env = {**os.environ, "AEGIS_WORLD_MODEL_DIR": str(ROOT / "artifacts/causal_world_model"),
           "PYTHONUTF8": "1"}
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd, capture_output=True,
                            encoding="utf-8", env=env, check=False)
    if result.returncode != expected:
        raise RuntimeError(f"expected exit {expected}, got {result.returncode}\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def _git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, encoding="utf-8").strip()


def _patch(directory, root, file, symbol, before, after):
    context = ImportGraph(root).slice(file, symbol)
    target = next(item["source"] for item in context["items"] if item["relation"] == "target")
    assert before in target
    source = (root / file).read_text(encoding="utf-8")
    assert source.count(before) == 1
    replacement = source.replace(before, after, 1)
    patch = directory / (Path(file).stem + ".diff")
    patch.write_text("".join(difflib.unified_diff(source.splitlines(keepends=True), replacement.splitlines(keepends=True),
                                               fromfile=f"a/{file}", tofile=f"b/{file}")),
                     encoding="utf-8", newline="\n")
    return patch


def _plan(directory, root, db, requirement, repairs, *, expected=1):
    path = directory / (root.name + ".json")
    _command("-m", "aegis_core", "plan", requirement, "--repo", root, "--db", db, "--out", path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    for task in plan["tasks"]:
        if task["kind"] == "repair":
            patch, probe = repairs[task["contract"]["target_files"][0]]
            task.update(patches=[str(patch)], fail_before_cmd=probe, pass_after_cmd=probe)
        else:
            # Includes the independent integration test in the frozen fixture.
            task["command"] = [sys.executable, "-B", "-m", "pytest", "-q"]
    path.write_text(json.dumps(plan), encoding="utf-8")
    head = _git(root, "rev-parse", "HEAD")
    result = _command("-m", "aegis_core", "run", "--dag", path, "--db", db, "--workers", "2", expected=expected)
    assert _git(root, "rev-parse", "HEAD") == head
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1
    return plan, result


def _attempt(result, target):
    return next(row["aegis_result"]["results"][0] for row in result["tasks"]
                if row.get("aegis_result") and target in row["aegis_result"]["results"][0]["changed_files"])


def contract_drill(directory, db):
    root = directory / "parallel"
    sources = {
        "backend.py": "def read_frame(index: int) -> int:\n    return index - 1\n",
        "cadence.py": "def cadence(delay: int) -> int:\n    return delay - 1\n",
        "consumer.py": "from backend import read_frame\nfrom cadence import cadence\n\ndef sample(value):\n    return read_frame(value), cadence(value)\n",
    }
    probe = """from backend import read_frame
from cadence import cadence

def test_backend():
    assert read_frame(4) == 4

def test_cadence():
    assert cadence(4) == 4
"""
    freeze_arena(root, "parallel", sources, probe, target="backend.py", source_dirs=(".",))
    patch_a = _patch(directory, root, "backend.py", "read_frame", sources["backend.py"].rstrip(),
                     "def read_frame(index: int, offset: int = 0) -> int:\n    return index + offset")
    patch_b = _patch(directory, root, "cadence.py", "cadence", "return delay - 1", "return delay")
    probe_cmd = [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_invariant.py"]
    plan, result = _plan(directory, root, db, "Fix backend.py frame reading and cadence.py timing", {
        "backend.py": (patch_a, [*probe_cmd[:-1], probe_cmd[-1] + "::test_backend"]),
        "cadence.py": (patch_b, [*probe_cmd[:-1], probe_cmd[-1] + "::test_cadence"]),
    })
    rejected = _attempt(result, "backend.py")
    accepted = _attempt(result, "cadence.py")
    lock = result["interface_locks"]["0"]
    assert len(plan["topological_layers"][0]) == 2
    assert plan["interface_locks"]["0"] == lock
    assert rejected["gate_failure"] == "INTERFACE_CONTRACT_VIOLATION" and rejected["reward"] < 0
    assert rejected["baseline"]["exit_code"] == 1 and rejected["final"] is None
    assert accepted["passed"] and accepted["final"]["exit_code"] == 0
    ledger = Ledger(db)
    for statuses in result["tasks"]:
        if statuses.get("aegis_result"):
            attempt = statuses["aegis_result"]["results"][0]
            approved = ledger.approved_patch(statuses["aegis_result"]["ticket_id"], attempt["patch_sha256"])
            assert (approved is not None) == attempt["passed"]
    assert result["task_statuses"]["validate-integration"] == "blocked"
    _report("CONTRACT", {"fanout": len(plan["topological_layers"][0]), "locked_entries": len(lock["entries"]),
                         "reason": rejected["gate_failure"], "reward_negative": rejected["reward"] < 0,
                         "red_exit": rejected["baseline"]["exit_code"], "green_executed": rejected["final"] is not None,
                         "signature_diff": rejected["interface_contract_differences"], "sibling_green_exit": 0,
                         "downstream": "SKIPPED_UPSTREAM_FAILED", "original_clean": True})


def reviewer_drill(directory, db):
    root = directory / "review"
    sources = {
        "frame.py": "class FrameBuffer:\n    def __init__(self):\n        self.history = []\n\n    def record(self, value: int) -> int:\n        return value - 1\n",
        "renderer.py": "from frame import FrameBuffer\n\ndef render(value):\n    return FrameBuffer().record(value)\n",
        "game.py": "from renderer import render\n\ndef tick(value):\n    return render(value)\n",
        "loop.py": "from game import tick\n\ndef loop(value):\n    return tick(value)\n",
    }
    probe = """from frame import FrameBuffer

def test_frame_value():
    buffer = FrameBuffer()
    assert buffer.record(4) == 4
"""
    freeze_arena(root, "review", sources, probe, target="frame.py", source_dirs=(".",))
    patch = _patch(directory, root, "frame.py", "FrameBuffer.record", "return value - 1",
                   "self.history.append(value)\n        return value")
    command = [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_invariant.py"]
    _, result = _plan(directory, root, db, "Fix frame.py cache history recording result", {"frame.py": (patch, command)})
    attempt = _attempt(result, "frame.py")
    assert attempt["complexity"]["level"] == "L3" and attempt["complexity"]["impact_radius"] == 3
    assert attempt["gate_failure"] == "ADVERSARIAL_AUDIT_REJECTED" and attempt["reward"] < 0
    assert attempt["baseline"]["exit_code"] == 1 and attempt["final"]["exit_code"] == 0
    assert any(finding["code"] == "UNBOUNDED_GROWTH" for finding in attempt["review"]["findings"])
    _report("REVIEW", {"route": attempt["complexity"]["level"], "impact_radius": attempt["complexity"]["impact_radius"],
                       "world_model_version": attempt["world_model_version"], "red_exit": attempt["baseline"]["exit_code"],
                       "green_exit": attempt["final"]["exit_code"], "reason": attempt["gate_failure"],
                       "findings": attempt["review"]["findings"], "reward_negative": attempt["reward"] < 0,
                       "original_clean": True})
    # A separate candidate demonstrates that a passing L3 review can stage a commit.
    good = _patch(directory, root, "frame.py", "FrameBuffer.record", "return value - 1",
                  "self.history[:] = [value]\n        return value")
    _, accepted = _plan(directory, root, db, "Fix frame.py cache history recording result with bounded storage",
                        {"frame.py": (good, command)}, expected=0)
    good_attempt = _attempt(accepted, "frame.py")
    assert good_attempt["review"]["passed"] and good_attempt["passed"]
    _report("REVIEW_PASS", {"route": good_attempt["complexity"]["level"], "passed": good_attempt["passed"],
                            "findings": good_attempt["review"]["findings"], "integration": "passed"})
    return root


def pm_drill(directory, db, root):
    ledger = Ledger(db)
    traces = sorted(ledger.high_surprise_traces(0), key=lambda trace: trace["trace_id"])
    bank = ExperienceBank(db)
    for category in ("INTERFACE_CONTRACT_VIOLATION", "ADVERSARIAL_AUDIT_REJECTED", None):
        selected = [trace for trace in traces if trace["actual_state_s1"].get("gate_failure") == category]
        assert selected
        bank.ingest_traces(selected)
    state = bank.status()
    spec = _command("-m", "aegis_core", "spec", "Keep frame cache history bounded", "--repo", root, "--db", db)
    assert state["version"] == 3 and spec["experience"] and spec["context"]["items"]
    assert {item["kind"] for item in spec["acceptance_criteria"]} == {"normal", "boundary", "fallback"}
    assert spec["questions"] == []
    wish = "Add drawdown protection"
    pending = _command("-m", "aegis_core", "spec", wish, "--repo", root, "--db", db, expected=2)
    assert [question["parameter"] for question in pending["questions"]] == ["max_drawdown"]
    answers = directory / "answers.json"
    answers.write_text('{"max_drawdown": 0.08}', encoding="utf-8")
    resolved = _command("-m", "aegis_core", "spec", wish, "--repo", root, "--db", db, "--answers", answers)
    assert resolved["questions"] == [] and resolved["business_parameters"] == {"max_drawdown": 0.08}
    _report("PM_SPEC", {"experience_bank_version": state["version"], "physical_experiences": state["experience_count"],
                        "experience_matches": len(spec["experience"]), "ast_target": spec["context"]["target"],
                        "acceptance_criteria": spec["acceptance_criteria"], "edge_cases": spec["edge_cases"],
                        "questions": spec["questions"], "asked_parameter": "max_drawdown",
                        "resolved_parameters": resolved["business_parameters"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--tokenizer-dir", type=Path, help="Use the official model tokenizer from this local directory")
    parser.add_argument("--output", type=Path, default=ROOT / "training/runs/whitepaper_v10")
    args = parser.parse_args(argv)
    dataset = args.dataset_dir.resolve()
    output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="aegis-whitepaper-v10-") as temporary:
        directory = Path(temporary)
        db = directory / "ledger.sqlite"
        _report("AEGIS_WHITEPAPER_V10", {"patch_author": "Codex", "entry_points": ["plan", "run --dag", "spec"]})
        contract_drill(directory, db)
        root = reviewer_drill(directory, db)
        pm_drill(directory, db, root)
        manifest, evidence = update_dataset(dataset)
        _report("DATASET_V3", {**manifest["cat_cafe_replay"], "evidence": evidence, "sha256": manifest["sha256"]})
        tokenizer_args = ["--tokenizer-dir", args.tokenizer_dir.resolve()] if args.tokenizer_dir else []
        for name, module, file, stress_args in (
            ("SFT", "training.train_qlora_sft", "mediator_sft_train.jsonl", ["--max-length", "384"]),
            ("GRPO", "training.train_grpo", "grpo_repair_pairs.jsonl",
             ["--max-prompt-length", "384", "--max-completion-length", "128"]),
        ):
            result = _command("-m", module, "--dry-run", "--dataset", dataset / file,
                              "--output", output, *tokenizer_args)
            assert result["checkpoint_verified"] and not result["gpu_used"]
            _report(name, result)
            stress = _command("-m", module, "--dry-run", "--dataset", dataset / file,
                              "--output", output / "truncation", *tokenizer_args, *stress_args)
            assert stress["checkpoint_verified"] and stress["prompt_truncations"] > 0
            _report(name + "_TRUNCATION", stress)
    _report("RESULT", {"status": "PASS", "sections_passed": 4})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
