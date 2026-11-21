import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aegis_core.env_doctor import EnvDoctor
from aegis_core.synthetic_hard_negatives import _canonical_line
from aegis_core.v20_performance import run_performance_profile


@pytest.mark.parametrize("free_bytes", range(20))
def test_low_disk_preflight_enters_sparse_mode_at_boundary(tmp_path, free_bytes):
    total = 1000
    doctor = EnvDoctor(tmp_path, min_free_bytes=20,
                       usage_provider=lambda _path: SimpleNamespace(total=total,
                                                                     used=total - free_bytes,
                                                                     free=free_bytes),
                       cleanup=False)
    result = doctor.check(sparse_mode="auto", changed_files=["aegis_core/changed.py"])
    assert result["sparse_mode"]["enabled"] is True
    assert result["status"] == "SPARSE_MODE"
    assert "disk_below_threshold" in result["sparse_mode"]["reasons"]


def test_release_v20_manifest_has_the_required_balanced_sample_matrix():
    root = Path(__file__).resolve().parents[1] / "datasets" / "release_v20"
    manifest = json.loads((root / "manifest_v2.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 2000
    assert manifest["hard_negative_count"] == 1600
    assert manifest["hard_positive_count"] == 400
    assert manifest["zero_truncation_verified"] is True
    assert all(item["hard_negative"] == 160 and item["hard_positive"] == 40
               and item["python"] == item["javascript"] == 100
               for item in manifest["category_counts"].values())


def test_release_v20_full_jsonl_sha_matches_manifest():
    root = Path(__file__).resolve().parents[1] / "datasets" / "release_v20"
    dataset = (root / "hard_samples_v2.jsonl").read_bytes()
    manifest = json.loads((root / "manifest_v2.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(dataset).hexdigest() == manifest["dataset_sha256"]


def test_three_round_adversarial_report_meets_zero_fpr_and_recall_thresholds():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "artifacts/v20_adversarial_report.json").read_text(encoding="utf-8"))
    assert len(report["rounds"]) == 3
    assert report["converged"] is True
    assert report["final_false_positive_rate"] == 0.0
    assert report["final_recall"] >= 0.975
    assert report["final_per_class_min_recall"] >= 0.92
    assert report["final_escape_rate"] < 0.03
    assert all(row["false_positive_rate"] == 0.0 for row in report["rounds"])
    assert all(row["recall"] >= 0.92 for row in report["rounds"][-1]["per_class"].values())


def test_blue_team_report_preserves_pre_hardening_escape_and_regression_count():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "artifacts/v20_adversarial_report.json").read_text(encoding="utf-8"))
    assert report["pre_hardening_baseline"]["escape_rate"] == 0.01
    assert report["permanent_regression_case_count"] == 80


@pytest.mark.parametrize("index", range(10))
def test_release_v20_sample_rows_keep_hash_and_tokenizer_proofs(index):
    root = Path(__file__).resolve().parents[1] / "datasets" / "release_v20"
    with (root / "hard_samples_v2.jsonl").open(encoding="utf-8") as handle:
        row = json.loads(next(line for offset, line in enumerate(handle) if offset == index))
    tokenizer = row["tokenizer"]
    assert tokenizer["zero_truncation"] is True
    assert tokenizer["token_count"] <= 4096
    body = {key: value for key, value in row.items() if key not in {"sample_sha256", "tokenizer"}}
    assert hashlib.sha256(_canonical_line(body).encode()).hexdigest() == row["sample_sha256"]


def test_repository_scale_performance_profile_measures_500_files_50_changes(tmp_path):
    report = run_performance_profile(output_path=tmp_path / "v20-performance.json")
    assert report["simulated_repository_files"] == 500
    assert report["changed_files"] == 50
    assert report["diff_lines"] == 5000
    assert report["files_scanned"] == 50
    assert report["p95_ms"] < 80
    assert report["wall_ms"] < 3000
    assert report["peak_memory_bytes"] < 256 * 1024**2
    assert json.loads((tmp_path / "v20-performance.json").read_text())["passed"]
