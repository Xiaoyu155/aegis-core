"""CPU-only release data cleaning, tokenizer accounting and immutable source ingestion."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from aegis_core.distill_pipeline import DistillPipeline
from aegis_core.experience_bank import ExperienceBank
from aegis_core.ledger import Ledger
from training.common import ByteTokenizer


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return path


def _sft(prompt="Implement the boundary check safely.", answer='{"patch":"return max(0, value)"}'):
    return {"messages": [{"role": "system", "content": "Return structured repair JSON."},
                         {"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}]}


def _pair(prompt="Fix the quota boundary with an explicit lower clamp.", chosen=None, rejected=None,
          chosen_score=.92, rejected_score=.31):
    chosen = chosen or "diff --git a/app.py b/app.py\n+return max(0, value)\n"
    rejected = rejected or "diff --git a/app.py b/app.py\n+return value\n"
    return {"prompt": prompt, "chosen_patch": chosen, "rejected_patch": rejected,
            "chosen_score": chosen_score, "rejected_score": rejected_score,
            "chosen_sha256": hashlib.sha256(chosen.encode()).hexdigest(),
            "rejected_sha256": hashlib.sha256(rejected.encode()).hexdigest(),
            "source": "candidate_tournament"}


def _pipeline(tmp_path):
    root = Path(__file__).resolve().parents[1]
    return DistillPipeline(root, tokenizer=ByteTokenizer(), tokenizer_id="utf8-byte-preflight")


def _build(tmp_path, *, sft=(), pairs=(), max_tokens=4096, min_quality=.35, ledger=(), experience=()):
    sft_path = _jsonl(tmp_path / "sft.jsonl", sft)
    pair_path = _jsonl(tmp_path / "pairs.jsonl", pairs)
    output = tmp_path / "release_v14"
    result = _pipeline(tmp_path).build_bundle(output, sft_paths=[sft_path],
        tournament_paths=[pair_path], ledger_paths=ledger, experience_paths=experience,
        max_tokens=max_tokens, min_quality=min_quality)
    return output, result


@pytest.mark.parametrize("delta", [.01, .1, .25, .5, .9, 1.0])
def test_preference_pairs_are_normalized_with_exact_patch_digests(delta, tmp_path):
    row = _pair(chosen_score=.5 + delta / 2, rejected_score=.5 - delta / 2)
    pipeline = _pipeline(tmp_path)
    normalized = pipeline._normalize_pair(row, tmp_path / "pairs.jsonl")
    assert normalized["chosen_sha256"] == hashlib.sha256(row["chosen_patch"].encode()).hexdigest()
    assert normalized["rejected_sha256"] == hashlib.sha256(row["rejected_patch"].encode()).hexdigest()
    assert normalized["chosen_score"] > normalized["rejected_score"]


@pytest.mark.parametrize("row", [
    {}, {"prompt": "", "chosen_patch": "x", "rejected_patch": "y", "chosen_score": 1, "rejected_score": 0},
    _pair(chosen_score=.1, rejected_score=.2), _pair(chosen_score=float("nan")),
    _pair(chosen_score=float("inf")), _pair(chosen_score="1"),
    {**_pair(), "chosen_sha256": "0" * 64}, {**_pair(), "rejected_sha256": "f" * 64},
    {**_pair(), "chosen_patch": ""}, {**_pair(), "rejected_patch": ""},
])
def test_invalid_or_unranked_tournament_pairs_are_rejected(row, tmp_path):
    assert _pipeline(tmp_path)._normalize_pair(row, tmp_path / "pairs.jsonl") is None


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(prompt=""), lambda row: row.update(chosen_patch=""),
    lambda row: row.update(rejected_patch=""), lambda row: row.update(chosen_score=0, rejected_score=0),
    lambda row: row.update(rejected_score=2),
])
def test_pair_quality_filters_missing_prompt_patch_or_margin(mutate, tmp_path):
    row = _pair(); mutate(row)
    path = _jsonl(tmp_path / "pairs.jsonl", [row])
    _, result = _build(tmp_path, pairs=[row])
    assert result["counts"]["preference_pairs"] == 0


def test_pipeline_deduplicates_sft_and_preference_rows_deterministically(tmp_path):
    first = _sft()
    duplicate = _sft()
    output, result = _build(tmp_path, sft=[first, duplicate], pairs=[_pair(), _pair()])
    assert result["counts"]["sft_examples"] == 1
    assert result["counts"]["preference_pairs"] == 1
    assert result["counts"]["duplicate_sft_rows"] == 1
    assert result["counts"]["duplicate_tournament_pairs"] == 1
    sft_text = (output / "datasets/sft_train.jsonl").read_bytes()
    assert sft_text == (output / "datasets/sft_train.jsonl").read_bytes()


@pytest.mark.parametrize("long_text", ["x" * 200, "oversized " * 30, "token " * 40, "schema " * 40])
def test_oversized_rows_are_rejected_without_truncating_any_training_text(long_text, tmp_path):
    oversized_pair = _pair(prompt=long_text)
    _, result = _build(tmp_path, sft=[_sft(long_text)], pairs=[oversized_pair], max_tokens=128)
    assert result["counts"]["truncated_rows"] == 0
    assert result["counts"]["overlength_sft_rows"] == 1
    assert result["counts"]["overlength_tournament_pairs"] == 1


@pytest.mark.parametrize("min_quality", [.7, .8, .9, .99])
def test_quality_threshold_filters_weak_samples_and_records_policy(min_quality, tmp_path):
    _, result = _build(tmp_path, sft=[_sft("Fix it", '{"x":1}')], pairs=[_pair()], min_quality=min_quality)
    assert result["counts"]["sft_examples"] <= 1
    assert result["counts"]["preference_pairs"] <= 1
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["quality_policy"]["min_quality"] == min_quality


@pytest.mark.parametrize("row", [
    {}, {"messages": []}, {"messages": [{"role": "user", "content": "ask"}]},
    {"messages": [{"role": "admin", "content": "ask"}, {"role": "assistant", "content": "{}"}]},
    {"messages": [{"role": "user", "content": "ask"}, {"role": "assistant", "content": "not json"}]},
    {"messages": [{"role": "user", "content": "ask"}, {"role": "assistant", "content": "[]"}]},
    {"messages": [{"role": "user", "content": " "}, {"role": "assistant", "content": "{}"}]},
])
def test_invalid_sft_conversations_are_dropped(row, tmp_path):
    _, result = _build(tmp_path, sft=[row])
    assert result["counts"]["sft_examples"] == 0
    assert result["counts"]["invalid_sft_rows"] == 1


def test_release_contains_preserved_training_and_windows_bootstrap_sources(tmp_path):
    output, result = _build(tmp_path, sft=[_sft()], pairs=[_pair()])
    assert (output / "training/train_qlora_sft.py").is_file()
    assert (output / "training/requirements-train.txt").is_file()
    assert (output / "scripts/bootstrap_local.ps1").read_bytes() == (
        Path(__file__).resolve().parents[1] / "scripts/bootstrap_local.ps1").read_bytes()
    assert (output / "RUN_LOCAL_WINDOWS.md").is_file()
    assert result["counts"]["chosen_positive"] == result["counts"]["rejected_negative"] == 1


def test_bundle_cpu_readback_checks_hashes_row_counts_and_zero_truncation(tmp_path):
    output, result = _build(tmp_path, sft=[_sft()], pairs=[_pair()])
    verified = DistillPipeline.verify_bundle(output / "manifest.json")
    assert verified["valid"] and verified["sft_rows"] == verified["preference_pairs"] == 1
    sibling = Path(result["manifest"])
    assert DistillPipeline.verify_bundle(sibling)["valid"]


@pytest.mark.parametrize("path", ["../outside", "/absolute", "datasets/../../escape"])
def test_bundle_verifier_rejects_manifest_path_traversal(path, tmp_path):
    output, _ = _build(tmp_path, sft=[_sft()])
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = path
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): DistillPipeline.verify_bundle(manifest_path)


def test_bundle_verifier_detects_post_build_file_tampering(tmp_path):
    output, _ = _build(tmp_path, sft=[_sft()], pairs=[_pair()])
    target = output / "datasets/sft_train.jsonl"
    target.write_text(target.read_text() + "{}\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        DistillPipeline.verify_bundle(output / "manifest.json")


def test_actual_sqlite_experience_bank_rows_are_distilled_and_deduplicated(tmp_path):
    database = tmp_path / "aegis.sqlite"
    bank = ExperienceBank(database)
    bank.record_arbitration(pattern="python bounded retry", rule={"ambiguity_threshold": .1},
                            cause="timeout", resolution="bounded retry verified")
    output, result = _build(tmp_path, experience=[database])
    assert result["counts"]["experience_rows_seen"] == 1
    rows = [json.loads(line) for line in (output / "datasets/sft_train.jsonl").read_text().splitlines()]
    assert any("python bounded retry" in item["messages"][0]["content"] for item in rows)


def _record_real_recovery(database):
    ledger = Ledger(database)
    ledger.create_ticket("distill-recovery", "keep the quota bounded", {}, "def repair(): ...")
    lease = ledger.acquire_lease("distill-recovery", "test", ttl=60)
    state = ledger.get_state("distill-recovery")
    ledger.transition("distill-recovery", lease, "queued", "running", state_s0=state,
                      action_a0={"op": "start"}, actual_state_s1={"status": "running"}, reward=0)
    patch_bad, patch_good = "diff --git bad\n", "diff --git good\n"
    failed = {"status": "running", "baseline_failed": True, "after_passed": False,
              "passed": False, "gate_failure": "ASSERTION_FAILED",
              "diagnosis": {"kind": "FailureDiagnosis", "failure_code": "ASSERTION_FAILED"}}
    state = ledger.get_state("distill-recovery")
    ledger.transition("distill-recovery", lease, "running", "running", state_s0=state,
        action_a0={"op": "repair_attempt", "patch": patch_bad,
                   "patch_sha256": hashlib.sha256(patch_bad.encode()).hexdigest(),
                   "contract": "keep the quota bounded", "tournament_score": .2},
        actual_state_s1=failed, reward=-1)
    success = {"status": "succeeded", "baseline_failed": True, "after_passed": True,
               "passed": True, "diagnosis": {"kind": "FailureDiagnosis", "failure_code": "RECOVERED"}}
    state = ledger.get_state("distill-recovery")
    ledger.transition("distill-recovery", lease, "running", "succeeded", state_s0=state,
        action_a0={"op": "repair_attempt", "patch": patch_good,
                   "patch_sha256": hashlib.sha256(patch_good.encode()).hexdigest(),
                   "contract": "keep the quota bounded", "tournament_score": .9},
        actual_state_s1=success, reward=1)


def test_real_wal_recovery_pair_becomes_one_balanced_grpo_sample(tmp_path):
    database = tmp_path / "ledger.sqlite"
    _record_real_recovery(database)
    output, result = _build(tmp_path, ledger=[database])
    assert result["counts"]["ledger_traces_seen"] >= 4
    rows = [json.loads(line) for line in (output / "datasets/grpo_preference_pairs.jsonl").read_text().splitlines()]
    assert any(row["source"] == "ledger_verified_recovery" for row in rows)
    assert result["counts"]["chosen_positive"] == result["counts"]["rejected_negative"]


def test_training_release_does_not_claim_official_tokenizer_for_byte_preflight(tmp_path):
    _, result = _build(tmp_path, sft=[_sft()], pairs=[_pair()])
    assert result["tokenizer"]["tokenizer_id"] == "utf8-byte-preflight"
    assert result["tokenizer"]["official_tokenizer"] is False
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["tokenizer"]["zero_truncation"] is True


def test_build_rejects_nonfinite_quality_and_out_of_profile_token_budgets(tmp_path):
    pipeline = _pipeline(tmp_path)
    with pytest.raises(ValueError): pipeline.build_bundle(tmp_path / "release", max_tokens=64)
    with pytest.raises(ValueError): pipeline.build_bundle(tmp_path / "release2", min_quality=float("nan"))
