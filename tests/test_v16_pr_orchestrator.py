"""Read-only three-way merge simulations over real temporary Git repositories."""
import json
import subprocess
import sys

import pytest

from aegis_core.pr_orchestrator import (
    audit_github_repository,
    audit_local_repository,
    render_dashboard,
    write_fleet_artifacts,
)


def git(root, *args, check=True):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=check).stdout.strip()


def repository(root):
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fleet Test")
    git(root, "config", "user.email", "fleet@example.invalid")
    (root / "shared.txt").write_text("base line\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")


@pytest.mark.parametrize("case", range(30))
def test_dry_run_finds_real_three_way_conflicts_without_mutating_any_ref(case, tmp_path):
    root = tmp_path / f"conflict-{case}"
    repository(root)
    git(root, "branch", "feat/aegis-delivery-v1")
    (root / "shared.txt").write_text(f"default edit {case}\n", encoding="utf-8")
    git(root, "commit", "-qam", "default edit")
    default_head = git(root, "rev-parse", "main")
    git(root, "checkout", "-q", "feat/aegis-delivery-v1")
    (root / "shared.txt").write_text(f"feature edit {case}\n", encoding="utf-8")
    git(root, "commit", "-qam", "feature edit")
    feature_head = git(root, "rev-parse", "feat/aegis-delivery-v1")
    git(root, "checkout", "-q", "main")

    report = audit_local_repository(root, default_branch="main")

    assert report["summary"]["conflicts"] == 1
    delivery = report["deliveries"][0]
    assert delivery["branch"] == "feat/aegis-delivery-v1"
    assert delivery["status"] == "conflict"
    assert delivery["priority"] == "P0"
    assert delivery["changes"]["files"] == ["shared.txt"]
    assert delivery["default_branch_zero_pollution"] is True
    assert git(root, "rev-parse", "main") == default_head
    assert git(root, "rev-parse", "feat/aegis-delivery-v1") == feature_head
    assert git(root, "status", "--porcelain") == ""


@pytest.mark.parametrize("case", range(8))
def test_mergeable_delivery_reports_real_diff_totals_and_risk_priority(case, tmp_path):
    root = tmp_path / f"clean-{case}"
    repository(root)
    git(root, "checkout", "-q", "-b", "feat/aegis-delivery-v1")
    file_name = "auth/session.py" if case % 2 else "src/feature.py"
    target = root / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"def feature_{case}():\n    return {case}\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "delivery")
    feature_head = git(root, "rev-parse", "HEAD")

    report = audit_local_repository(root, default_branch="main")

    delivery = report["deliveries"][0]
    assert delivery["status"] == "mergeable"
    assert delivery["priority"] == ("P1" if case % 2 else "P2")
    assert delivery["changes"]["changed_files"] == 1
    assert delivery["changes"]["lines_added"] == 2
    assert delivery["changes"]["lines_deleted"] == 0
    assert delivery["default_branch_zero_pollution"] is True
    assert git(root, "rev-parse", "HEAD") == feature_head


def test_audit_limits_candidates_to_aegis_delivery_prefix(tmp_path):
    root = tmp_path / "filtered"
    repository(root)
    for branch in ("feat/aegis-one", "feat/other", "feature/aegis-two"):
        git(root, "branch", branch)
    report = audit_local_repository(root)
    assert [item["branch"] for item in report["deliveries"]] == ["feat/aegis-one"]


@pytest.mark.parametrize("case", range(5))
def test_remote_bare_repo_audit_reads_feature_refs_and_preserves_remote_heads(case, tmp_path):
    work = tmp_path / f"work-{case}"
    repository(work)
    remote = tmp_path / f"remote-{case}.git"
    subprocess.run(["git", "clone", "--bare", "-q", str(work), str(remote)], check=True)
    git(work, "checkout", "-q", "-b", "feat/aegis-delivery-v1")
    (work / f"delivery-{case}.txt").write_text(f"payload {case}\n", encoding="utf-8")
    git(work, "add", ".")
    git(work, "commit", "-qm", "delivery")
    git(work, "push", "-q", str(remote), "feat/aegis-delivery-v1")
    before = subprocess.run(["git", "--git-dir", str(remote), "show-ref"], check=True,
                            capture_output=True, text=True).stdout

    report = audit_github_repository({"full_name": f"test/repo-{case}", "clone_url": str(remote),
                                      "default_branch": "main"})

    after = subprocess.run(["git", "--git-dir", str(remote), "show-ref"], check=True,
                           capture_output=True, text=True).stdout
    assert report["deliveries"][0]["status"] == "mergeable"
    assert report["deliveries"][0]["branch"] == "feat/aegis-delivery-v1"
    assert report["deliveries"][0]["changes"]["files"] == [f"delivery-{case}.txt"]
    assert before == after


def test_dashboard_and_json_ledger_are_written_as_atomic_artifacts(tmp_path):
    report = {"schema_version": "aegis.fleet_audit.v1", "owner": "sample",
              "repositories": [{"repository": "sample/repo", "deliveries": [{
                  "branch": "feat/aegis-v1", "status": "mergeable", "priority": "P2",
                  "default_branch_zero_pollution": True,
                  "changes": {"changed_files": 2, "lines_added": 4, "lines_deleted": 1}}]}],
              "summary": {"repositories_scanned": 1, "delivery_branches": 1, "conflicts": 0}}
    artifacts = write_fleet_artifacts(report, tmp_path / "artifacts")
    ledger = json.loads((tmp_path / "artifacts" / "fleet_dashboard_v16.json").read_text(encoding="utf-8"))
    dashboard = (tmp_path / "artifacts" / "fleet_dashboard_v16.md").read_text(encoding="utf-8")
    assert ledger == report
    assert "sample/repo" in dashboard and "feat/aegis-v1" in dashboard
    assert "4 / 1" in dashboard
    assert artifacts["json"].endswith("fleet_dashboard_v16.json")
    assert artifacts["markdown"].endswith("fleet_dashboard_v16.md")
    assert list((tmp_path / "artifacts").glob("*.tmp")) == []


def test_dashboard_escapes_missing_delivery_sets_and_read_failures():
    dashboard = render_dashboard({"repositories": [{"repository": "broken/repo", "status": "ref_read_failed",
                                                    "deliveries": []}],
                                  "summary": {"repositories_scanned": 1, "delivery_branches": 0, "conflicts": 0}})
    assert "broken/repo" in dashboard
    assert "ref_read_failed" in dashboard


def test_audit_fleet_cli_exposes_read_only_scope_and_skip_controls():
    result = subprocess.run([sys.executable, "-m", "aegis_core.cli", "audit-fleet", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--owner" in result.stdout
    assert "--exclude-repo" in result.stdout
    assert "read-only merge preflight" in result.stdout.lower()
