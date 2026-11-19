"""Versioned Phase 15 bundle paths preserve v14 while auditing every release file."""
import hashlib
import json
import hashlib
from pathlib import Path

import pytest

from aegis_core.distill_pipeline import DistillPipeline
from training.common import ByteTokenizer


def _pipeline():
    return DistillPipeline(Path(__file__).resolve().parents[1], tokenizer=ByteTokenizer(),
                           tokenizer_id="utf8-byte-release-test")


def _labeled(group, positive, patch, reward, *, digest=None, prompt="Repair the verified business defect."):
    encoded = json.dumps({"patch": patch}, sort_keys=True)
    return {"group_id": group, "prompt": prompt,
            "positive": positive, "completion": encoded, "reward": reward,
            "patch_sha256": digest or hashlib.sha256(patch.encode()).hexdigest(),
            "template_verified": True}


def test_v15_release_writes_versioned_sidecar_and_verifies_bundle(tmp_path):
    root = tmp_path / "datasets"
    output = root / "release_v15"
    result = _pipeline().build_bundle(output, release_version=15)

    manifest_path = root / "release_v15_bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["status"] == "verified"
    assert manifest["schema_version"] == "aegis.training.release.v15"
    assert result["manifest"] == str(manifest_path)
    assert DistillPipeline.verify_bundle(manifest_path)["valid"] is True
    assert "v15" in (output / "RUN_LOCAL_WINDOWS.md").read_text(encoding="utf-8")


def test_v15_build_does_not_overwrite_v14_manifest_or_files(tmp_path):
    root = tmp_path / "datasets"
    pipeline = _pipeline()
    v14 = pipeline.build_bundle(root / "release_v14")
    v14_path = root / "release_v14_bundle_manifest.json"
    before = hashlib.sha256(v14_path.read_bytes()).hexdigest()

    v15 = pipeline.build_bundle(root / "release_v15", release_version=15)

    assert v14["manifest"] == str(v14_path)
    assert v15["manifest"] == str(root / "release_v15_bundle_manifest.json")
    assert hashlib.sha256(v14_path.read_bytes()).hexdigest() == before
    assert DistillPipeline.verify_bundle(v14_path)["valid"] is True
    assert DistillPipeline.verify_bundle(v15["manifest"])["valid"] is True


@pytest.mark.parametrize("version", [13, 16, True, "15", 14.0])
def test_release_version_rejects_unsupported_or_noninteger_values(tmp_path, version):
    with pytest.raises(ValueError, match="release_version"):
        _pipeline().build_bundle(tmp_path / "release", release_version=version)


def test_verifier_rejects_unknown_release_schema(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": "aegis.training.release.v99"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        DistillPipeline.verify_bundle(path)


def test_distill_pairs_verified_positive_and_negative_rollouts(tmp_path):
    source = tmp_path / "labeled.jsonl"
    rows = [_labeled("case-1", False, "bad", -.2),
            _labeled("case-1", True, "good-low", .8),
            _labeled("case-1", True, "good-high", 1.2)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    result = _pipeline().build_bundle(tmp_path / "datasets" / "release_v15", release_version=15,
                                      sft_paths=[empty], tournament_paths=[source])
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    pair = json.loads((tmp_path / "datasets/release_v15/datasets/grpo_preference_pairs.jsonl")
                      .read_text(encoding="utf-8"))
    assert manifest["counts"]["labeled_rollout_groups_paired"] == 1
    assert manifest["counts"]["preference_pairs"] == 1
    assert pair["chosen_patch"] == "good-high"
    assert pair["rejected_patch"] == "bad"
    assert pair["chosen_score"] > pair["rejected_score"]


@pytest.mark.parametrize("rows", [
    [_labeled("one-sided", True, "good", 1.0)],
    [_labeled("bad-digest", True, "good", 1.0, digest="0" * 64),
     _labeled("bad-digest", False, "bad", 0.0)],
    [_labeled("bad-json", True, "good", 1.0),
     {**_labeled("bad-json", False, "bad", 0.0), "completion": "not-json"}],
    [_labeled("bad-order", True, "good", 0.0), _labeled("bad-order", False, "bad", 0.0)],
])
def test_distill_drops_unverifiable_labeled_rollout_groups(tmp_path, rows):
    source = tmp_path / "labeled.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = _pipeline().build_bundle(tmp_path / "datasets" / "release_v15", release_version=15,
                                      sft_paths=[empty], tournament_paths=[source])
    assert result["counts"]["preference_pairs"] == 0
    assert result["counts"].get("labeled_rollout_groups_paired", 0) == 0


def test_labeled_chat_prompt_is_preserved_as_text_when_pairing_grpo_rows(tmp_path):
    prompt = [{"role": "system", "content": "Return a patch."},
              {"role": "user", "content": "Fix the broken retry."}]
    source = tmp_path / "labeled.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in [
        _labeled("chat-1", True, "correct", 1.0, prompt=prompt),
        _labeled("chat-1", False, "incorrect", 0.0, prompt=prompt),
    ]), encoding="utf-8")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    output = tmp_path / "datasets/release_v15"
    result = _pipeline().build_bundle(output, release_version=15, sft_paths=[empty],
                                      tournament_paths=[source])
    pair = json.loads((output / "datasets/grpo_preference_pairs.jsonl").read_text(encoding="utf-8"))
    assert result["counts"]["preference_pairs"] == 1
    assert pair["prompt"] == "system: Return a patch.\nuser: Fix the broken retry."
