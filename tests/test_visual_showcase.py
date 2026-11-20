"""Integrity checks for the visual-only engineering showcase."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aegis_showcase_builder", ROOT / "scripts/build_aegis_showcase.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_showcase_metrics_match_current_status_and_release_manifest():
    data = BUILDER.load_data()
    status = json.loads((ROOT / "STATUS.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "datasets/release_v15/manifest.json").read_text(encoding="utf-8"))
    fleet = json.loads((ROOT / "artifacts/fleet_dashboard_v16.json").read_text(encoding="utf-8"))

    assert data["version"] == status["version"] == "1.1.0"
    assert data["tests"]["collected"] == data["tests"]["passed"] == 1338
    assert data["ci"]["jobs_expected"] == 8
    assert data["release"]["sft_examples"] == manifest["counts"]["sft_examples"] == 1462
    assert data["release"]["preference_pairs"] == manifest["counts"]["preference_pairs"] == 342
    assert data["fleet"]["summary"] == fleet["summary"]
    assert data["fleet"]["main_pollution"] == 0
    assert data["tests"]["history"][0] == {"phase": 13, "collected": 656, "passed": 656}


def test_release_rows_and_checksums_match_manifest():
    manifest = json.loads((ROOT / "datasets/release_v15/manifest.json").read_text(encoding="utf-8"))
    for name, expected_count in (
        ("datasets/sft_train.jsonl", manifest["counts"]["sft_examples"]),
        ("datasets/grpo_preference_pairs.jsonl", manifest["counts"]["preference_pairs"]),
    ):
        path = ROOT / "datasets/release_v15" / name
        content = path.read_bytes()
        entry = next(item for item in manifest["files"] if item["path"] == name)
        assert len(content.splitlines()) == expected_count
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    assert manifest["counts"]["truncated_rows"] == 0


def test_candidate_simulation_uses_four_same_task_preference_rows():
    data = BUILDER.load_data()
    rows = BUILDER.read_jsonl("datasets/grpo_tournament_pairs.jsonl")
    job_rows = [row for row in rows if row.get("chosen_candidate") == "job10_correct"]
    assert len(job_rows) == data["tournament"]["pair_rows"] == 4
    assert {row["rejected_candidate"] for row in job_rows} == {
        "job10_wrong",
        "job10_signature",
        "job10_test_cheat",
        "job10_memory_leak",
    }
    assert len({row["prompt"] for row in job_rows}) == 1
    rejected = {row["id"]: row for row in data["tournament"]["rejected"]}
    assert rejected["job10_test_cheat"]["audit_passed"] is False
    assert rejected["job10_memory_leak"]["audit_findings"][0]["code"] == "UNBOUNDED_GROWTH"
    assert data["scans"]["raw_rule_hits"] == 715
    assert data["scans"]["post_dedup_hits"] is None
    assert data["world"]["mode"] == "observe"
    assert data["world"]["false_positive_rate"] is None


def test_dashboard_embeds_verified_data_without_network_dependencies():
    data = BUILDER.load_data()
    page = BUILDER.render_html(data)
    embedded = re.search(r'<script id="aegis-data" type="application/json">(.*?)</script>', page, re.S)
    assert embedded
    assert json.loads(embedded.group(1)) == data
    assert "aegis_core/semantic_contract.py" in page
    assert "aegis_core/pr_orchestrator.py" in page
    assert "../showcase/preview.gif" in page
    assert "../showcase/aegis_demo.mp4" in page
    assert not re.search(r'<script\b[^>]*\bsrc\s*=', page, re.I)
    assert not re.search(r'<link\b[^>]*\bhref\s*=\s*["\']https?://', page, re.I)


def test_displayed_module_test_counts_match_pytest_collection():
    data = BUILDER.load_data()
    for module in data["modules"]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", module["test_source"]],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        match = re.search(r"(\d+) tests? collected", result.stdout)
        assert match, f"No collection count reported for {module['test_source']}"
        assert int(match.group(1)) == module["tests"], module["test_source"]
