"""Actual CLI subprocess tests; no vendor endpoints or GPU dependencies."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from aegis_core import cli
from aegis_core.dataset_builder import build_datasets
from aegis_core.mediator import compile_request
from aegis_core.simulator import create_arena, fixture_patch, ALL_FAULTS
from training.train_grpo import score_patch

ROOT = Path(__file__).resolve().parents[1]
PROBE = json.dumps(["{python}", "-B", "-m", "pytest", "-q"])


def call(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-m", "aegis_core", *map(str, args)], cwd=ROOT,
                            capture_output=True, encoding="utf-8", env=env)
    return result.returncode, json.loads(result.stdout if result.stdout.strip() else result.stderr)


def wish(root, tmp_path, patches, *, ticket="task", text="修复", target=None, extra=()):
    manifest = tmp_path / (ticket + ".json")
    db = tmp_path / "ledger.sqlite"
    args = ["wish", text, "--repo", root, "--db", db, "--ticket-id", ticket, "--out", manifest,
            "--fail-before", PROBE, "--pass-after", PROBE, "--timeout", "120", "--max-attempts", "5"]
    if target:
        args += ["--target", target]
    for patch in patches:
        args += ["--patch", patch]
    return args + list(extra), manifest, db


def test_wish_question_resume_run_stats_and_export_full_cli(tmp_path):
    root = tmp_path / "target"
    create_arena(root, "quant_posside")
    patch = tmp_path / "good.diff"
    patch.write_text(fixture_patch("quant_posside", positive=True), encoding="utf-8")
    args, contract, db = wish(root, tmp_path, [patch])
    code, initial = call(*args)
    assert code == 2 and initial["compilation"]["status"] == "MISSING_CONSTRAINT"
    code, resolved = call(*args, "--selected-module", "signal_engine.py")
    assert code == 0 and resolved["compilation"]["target_files"] == ["signal_engine.py"]
    code, result = call("run", "--contract", contract, "--db", db)
    assert code == 0 and result["status"] == "succeeded"
    assert result["results"][0]["baseline_failed"] and result["results"][0]["passed"]
    assert '"posSide": "long"' in (root / "signal_engine.py").read_text()
    assert not list((root / ".worktrees").iterdir())
    code, stats = call("stats", "--repo", root, "--db", db)
    assert code == 0 and stats["success_rate"] == 1 and stats["cheat_interceptions"] == 0
    assert stats["mean_reward"] > 0
    out = tmp_path / "latest.jsonl"
    code, exported = call("export-traces", "--repo", root, "--db", db, "--out", out)
    assert code == 0 and exported["exported"] == 7
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for a, b in zip(rows, rows[1:]):
        assert a["state_s1"] == b["state_s0"]
    assert rows[-1]["state_s1"]["status"] == "succeeded"


def test_cli_cheat_and_exact_replay_breaker_are_ledgered(tmp_path):
    root = tmp_path / "target"
    create_arena(root, "cat_blank")
    testfile = root / "tests/test_invariant.py"
    patch = tmp_path / "cheat.diff"
    patch.write_text("".join(difflib.unified_diff(testfile.read_text().splitlines(True),
                                               ["def test_fake():\n", "    assert True\n"],
                                               fromfile="a/tests/test_invariant.py", tofile="b/tests/test_invariant.py")), encoding="utf-8")
    args, manifest, db = wish(root, tmp_path, [patch, patch, patch], target="engine/iso_renderer.py")
    assert call(*args)[0] == 0
    code, result = call("run", "--contract", manifest, "--db", db)
    assert code == 1 and result["reason"] == "blocked_needs_human"
    assert len(result["results"]) == 3
    assert result["results"][0]["cheat_detected"]
    assert result["results"][-1]["repeated_patch"]
    stats = call("stats", "--repo", root, "--db", db)[1]
    assert stats["cheat_interceptions"] == 2 and stats["mean_reward"] < 0


def test_tampered_or_stale_manifest_does_not_start_ticket(tmp_path):
    root = tmp_path / "target"
    create_arena(root, "cat_blank")
    patch = tmp_path / "good.diff"
    patch.write_text(fixture_patch("cat_blank", positive=True), encoding="utf-8")
    args, manifest, db = wish(root, tmp_path, [patch], target="engine/iso_renderer.py")
    assert call(*args)[0] == 0
    original = json.loads(manifest.read_text())
    modified = {**original, "pass_after_cmd": [sys.executable, "-c", "pass"]}
    # Rehashing a modified file cannot bypass its ledger binding.
    modified["contract_sha256"] = cli._digest(modified)
    manifest.write_text(json.dumps(modified), encoding="utf-8")
    assert call("run", "--contract", manifest, "--db", db)[0] == 1
    stats = call("stats", "--repo", root, "--db", db)[1]
    assert stats["statuses"] == {"COMPILED": 1}


def test_malformed_patch_is_a_negative_attempt_then_good_patch_can_pass(tmp_path):
    root = tmp_path / "target"
    create_arena(root, "cat_blank")
    bad, good = tmp_path / "bad.diff", tmp_path / "good.diff"
    bad.write_text("not a patch", encoding="utf-8")
    good.write_text(fixture_patch("cat_blank", positive=True), encoding="utf-8")
    args, manifest, db = wish(root, tmp_path, [bad, good], target="engine/iso_renderer.py")
    assert call(*args)[0] == 0
    code, result = call("run", "--contract", manifest, "--db", db)
    assert code == 0 and len(result["results"]) == 2
    assert result["results"][0]["reward"] < 0 < result["results"][1]["reward"]


def test_explicit_model_route_is_injected_and_uses_same_gate(tmp_path, monkeypatch, capsys):
    root = tmp_path / "target"
    create_arena(root, "cat_blank")
    args, manifest, db = wish(root, tmp_path, [], target="engine/iso_renderer.py",
                              extra=["--allow-model", "--models", str(ROOT / "config/models.yaml")])
    assert call(*args)[0] == 0
    def fake_call(self, tier, messages, *, deadline):
        assert "source" in json.loads(messages[-1]["content"])
        return {"status": "admitted", "slot_id": "local_3b", "output": json.dumps({"patch": fixture_patch("cat_blank", positive=True)}),
                "breaker": {"repeated_patch": False}}
    monkeypatch.setattr(cli.Router, "call", fake_call)
    assert cli.main(["run", "--contract", str(manifest), "--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_console_entrypoint_and_no_heavy_imports():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["scripts"]["aegis"] == "aegis_core.cli:main"
    exe = Path(sys.executable).parent / ("aegis.exe" if os.name == "nt" else "aegis")
    assert subprocess.run([str(exe), "--help"], capture_output=True).returncode == 0
    command = "import aegis_core.cli, training.train_sft, training.train_grpo, sys; assert not any(x in sys.modules for x in ('torch','transformers','peft','trl'))"
    assert subprocess.run([sys.executable, "-c", command], cwd=ROOT, capture_output=True).returncode == 0


def test_twelve_fault_modes_520_samples_split_and_training_preflight(tmp_path):
    manifest = build_datasets(tmp_path, verify_templates=True)
    assert len(ALL_FAULTS) == 12 and manifest["executed_templates"] == 12
    assert manifest["physical_template_attempts"] == 24
    train = [json.loads(line) for line in (tmp_path / "mediator_sft_train.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluation = [json.loads(line) for line in (tmp_path / "mediator_sft_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 468 and len(evaluation) == 52
    assert not {r["id"] for r in train} & {r["id"] for r in evaluation}
    for row in train + evaluation:
        assert compile_request(**row["request"]) == row["expected"]
    pairs = [json.loads(line) for line in (tmp_path / "grpo_repair_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pairs) == 520 and all(row["template_verified"] for row in pairs)
    assert {row["scenario"] for row in pairs} == set(ALL_FAULTS)
    for name, checksum in manifest["sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == checksum
        assert (tmp_path / name).read_bytes() == (ROOT / "datasets" / name).read_bytes()
    for module, data in (("training.train_sft", "mediator_sft_train.jsonl"), ("training.train_grpo", "grpo_repair_pairs.jsonl")):
        result = subprocess.run([sys.executable, "-m", module, "--dataset", str(tmp_path / data), "--validate-only"], cwd=ROOT, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_online_grpo_reward_reexecutes_patch_not_self_reported_score():
    assert score_patch("cat_blank", json.dumps({"patch": fixture_patch("cat_blank", positive=True), "reward": -100})) > 0
    assert score_patch("cat_blank", json.dumps({"patch": fixture_patch("cat_blank", positive=False), "reward": 100})) < 0
    assert score_patch("cat_blank", '{"reward":100}') == -2


def test_uncommitted_business_edits_cannot_drift_frozen_context(tmp_path):
    root = tmp_path / "target"
    create_arena(root, "cat_blank")
    patch = tmp_path / "good.diff"
    patch.write_text(fixture_patch("cat_blank", positive=True), encoding="utf-8")
    args, manifest, db = wish(root, tmp_path, [patch], target="engine/iso_renderer.py")
    assert call(*args)[0] == 0
    (root / "engine/iso_renderer.py").write_text("# uncommitted drift\n")
    code, error = call("run", "--contract", manifest, "--db", db)
    assert code == 1 and "working tree changed" in error["message"]
    assert call("stats", "--repo", root, "--db", db)[1]["repair_attempts"] == 0
