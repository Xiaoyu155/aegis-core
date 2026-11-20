"""Local multi-repository PR planning and idempotency tests."""
import json
import subprocess
import sys

import pytest

from aegis_core.fleet_shipper import (
    PROTECTED_REPOSITORIES,
    _build_delivery,
    _gate_state,
    execute_ship_plan,
    plan_local_fleet,
    plan_local_repository,
)


def git(root, *args, check=True):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=check)
    return result.stdout.strip()


def repository(root, name="repo"):
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Shipper Tests")
    git(root, "config", "user.email", "shipper@example.invalid")
    (root / "README.md").write_text(f"{name} base\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "baseline")
    git(root, "checkout", "-qb", "feat/aegis-delivery-v1")
    return root


def add_delivery(root, *, path="src/fix.py", content="def fixed():\n    return True\n", commits=1):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    for index in range(commits):
        target.write_text(content.replace("True", f"{index + 1}"), encoding="utf-8")
        git(root, "add", ".")
        git(root, "commit", "-qm", f"repair case {index + 1}")


def test_local_plan_collects_commit_history_change_stats_and_readiness(tmp_path):
    root = repository(tmp_path / "service")
    add_delivery(root, commits=2)
    report = plan_local_repository(root, default_branch="main",
                                   gate_statuses={"feat/aegis-delivery-v1": {"state": "success", "checks": ["success"]}})
    assert len(report) == 1
    item = report[0]
    assert item["commit_count"] == 2
    assert [row["subject"] for row in item["commits"]] == ["repair case 2", "repair case 1"]
    assert item["changes"]["changed_files"] == 1
    assert item["conflict_preflight"] == "mergeable"
    assert item["ready_to_execute"] is True
    assert item["risk_priority"] == "P2"
    assert item["action"] == "create_pr"


@pytest.mark.parametrize("case", range(10))
def test_pr_title_and_body_are_complete_and_submission_ready(case, tmp_path):
    root = repository(tmp_path / f"repo-{case}")
    add_delivery(root, path=f"src/fix_{case}.py", content=f"def fix_{case}():\n    return True\n")
    item = plan_local_repository(root, default_branch="main",
                                 gate_statuses={"feat/aegis-delivery-v1": {"state": "success", "summary": "8/8 CI green"}})[0]
    assert item["title"].startswith("Aegis: repair case")
    assert "## Defect and delivery summary" in item["body"]
    assert f"repair case 1" in item["body"]
    assert "## Red/green probe and CI results" in item["body"]
    assert "8/8 CI green" in item["body"]
    assert "## Conflict preflight" in item["body"]
    assert "## Manual review checklist" in item["body"]
    assert "- [ ]" in item["body"]
    assert item["idempotency_key"].endswith(item["head"])


@pytest.mark.parametrize("state", ["failure", "pending", "unknown", "cancelled", "error", "skipped", "action_required"])
def test_non_green_gates_hold_pr_creation(state, tmp_path):
    root = repository(tmp_path / f"gate-{state}")
    add_delivery(root)
    item = plan_local_repository(root, default_branch="main",
                                 gate_statuses={"feat/aegis-delivery-v1": {"state": state}})[0]
    assert item["action"] == "create_pr"
    assert item["ready_to_execute"] is False


@pytest.mark.parametrize("checks,expected", [
    ([{"name": "tests", "status": "completed", "conclusion": "success"}], "success"),
    ([{"name": "tests", "status": "completed", "conclusion": "failure"}], "failure"),
    ([{"name": "tests", "status": "in_progress"}], "pending"),
    (["success", "passed"], "success"),
    (["success", "timed_out"], "failure"),
])
def test_gate_aggregates_check_states_conservatively(checks, expected):
    state, normalized = _gate_state({"state": "unknown", "checks": checks})
    assert state == expected
    assert normalized["checks"] == checks


def test_existing_open_pr_is_reused_instead_of_created(tmp_path, monkeypatch):
    root = repository(tmp_path / "duplicate")
    add_delivery(root)
    existing = {"number": 42, "url": "https://github.com/o/r/pull/42", "title": "Already open"}
    item = plan_local_repository(root, default_branch="main",
                                 gate_statuses={"feat/aegis-delivery-v1": "success"},
                                 existing_prs={"feat/aegis-delivery-v1": existing})[0]
    assert item["action"] == "reuse_existing_pr"
    assert item["ready_to_execute"] is False
    plan = {"repositories": [{"deliveries": [item]}], "summary": {"conflicts": 0}}
    monkeypatch.setattr("aegis_core.fleet_shipper.urlopen", lambda *a, **k: pytest.fail("duplicate POST"))
    executed = execute_ship_plan(plan, token="long-enough-test-token")
    assert executed["execution_results"][0]["status"] == "reused"
    assert executed["execution_results"][0]["pull_request"] == existing


def test_local_dry_run_does_not_change_refs_or_worktree(tmp_path):
    root = repository(tmp_path / "dry-run")
    add_delivery(root)
    before = git(root, "show-ref")
    status = git(root, "status", "--porcelain")
    report = plan_local_repository(root, default_branch="main",
                                   gate_statuses={"feat/aegis-delivery-v1": "success"})
    assert report[0]["ready_to_execute"] is True
    assert git(root, "show-ref") == before
    assert git(root, "status", "--porcelain") == status == ""


@pytest.mark.parametrize("repo_name", sorted(PROTECTED_REPOSITORIES))
def test_protected_repositories_are_excluded_from_fleet_plans(repo_name, tmp_path):
    root = repository(tmp_path / repo_name, repo_name)
    add_delivery(root)
    report = plan_local_fleet([root], default_branch="main")
    assert report["repositories"] == []
    assert report["summary"]["repositories_scanned"] == 0
    assert report["excluded_repositories"] == [str(root.resolve())]


def test_multi_repository_plan_is_stable_and_separates_each_delivery(tmp_path):
    roots = []
    for name in ("alpha", "beta", "gamma"):
        root = repository(tmp_path / name, name)
        add_delivery(root, path=f"src/{name}.py")
        roots.append(root)
    report = plan_local_fleet(roots, default_branch="main", gate_statuses={
        str(root.resolve()): {"feat/aegis-delivery-v1": "success"} for root in roots})
    assert report["mode"] == "dry_run"
    assert report["executed"] is False
    assert report["summary"]["deliveries"] == 3
    assert report["summary"]["ready_to_execute"] == 3
    assert {item["repository"] for repo in report["repositories"] for item in repo["deliveries"]} == {
        "local/alpha", "local/beta", "local/gamma"}


def test_risk_priority_flags_sensitive_paths():
    item = _build_delivery("org/service", {"branch": "feat/aegis-fix", "head": "abc",
                                           "default_branch": "main", "status": "mergeable",
                                           "changes": {"files": ["auth/session.py"], "changed_files": 1,
                                                       "lines_added": 2, "lines_deleted": 1}},
                           [], "success")
    assert item["risk_priority"] == "P1"
    assert item["ready_to_execute"] is True


@pytest.mark.parametrize("branch", ["feat/aegis-fix", "feat/aegis-v1.2", "feat/aegis-delivery-v1"])
def test_idempotency_key_is_stable_for_the_same_branch_head(branch):
    delivery = {"branch": branch, "head": "abc123", "default_branch": "main", "status": "mergeable",
                "changes": {"files": [], "changed_files": 0, "lines_added": 0, "lines_deleted": 0}}
    first = _build_delivery("org/repo", delivery, [], "success")
    second = _build_delivery("org/repo", delivery, [], "success")
    assert first["idempotency_key"] == second["idempotency_key"] == f"org/repo:{branch}:abc123"


def test_execute_requires_token_and_holds_unready_delivery():
    plan = {"repositories": [{"deliveries": [{"repository": "org/repo", "branch": "feat/aegis-x",
                                                "action": "create_pr", "ready_to_execute": False}]}]}
    with pytest.raises(ValueError, match="GitHub token"):
        execute_ship_plan(plan, token=None)
    result = execute_ship_plan(plan, token="long-enough-test-token")
    assert result["execution_results"][0]["status"] == "held"


def _remote_ready_plan():
    delivery = {"repository": "org/repo", "branch": "feat/aegis-fix", "head": "abc123",
                "default_branch": "main", "action": "create_pr", "ready_to_execute": True,
                "title": "Aegis: repair", "body": "review this", "idempotency_key": "org/repo:branch:abc123"}
    return {"repositories": [{"deliveries": [delivery]}], "summary": {"conflicts": 0}}


def test_execute_rechecks_for_a_new_open_pr_before_posting(monkeypatch):
    monkeypatch.setattr("aegis_core.fleet_shipper._github_get_pull",
                        lambda *args: {"number": 7, "html_url": "https://github.com/org/repo/pull/7", "title": "Existing"})
    monkeypatch.setattr("aegis_core.fleet_shipper._github_json", lambda *args: pytest.fail("unneeded API read"))
    monkeypatch.setattr("aegis_core.fleet_shipper.urlopen", lambda *args, **kwargs: pytest.fail("duplicate POST"))
    result = execute_ship_plan(_remote_ready_plan(), token="long-enough-test-token")
    assert result["execution_results"][0]["status"] == "reused"


def test_execute_holds_when_branch_head_changed_since_dry_run(monkeypatch):
    monkeypatch.setattr("aegis_core.fleet_shipper._github_get_pull", lambda *args: None)
    monkeypatch.setattr("aegis_core.fleet_shipper._github_json", lambda *args: {"commit": {"sha": "new-head"}})
    monkeypatch.setattr("aegis_core.fleet_shipper.urlopen", lambda *args, **kwargs: pytest.fail("stale plan POST"))
    result = execute_ship_plan(_remote_ready_plan(), token="long-enough-test-token")
    assert result["execution_results"][0]["status"] == "held"
    assert "head changed" in result["execution_results"][0]["reason"]


def test_execute_holds_when_latest_checks_are_not_green(monkeypatch):
    monkeypatch.setattr("aegis_core.fleet_shipper._github_get_pull", lambda *args: None)
    monkeypatch.setattr("aegis_core.fleet_shipper._github_json", lambda *args: {"commit": {"sha": "abc123"}})
    monkeypatch.setattr("aegis_core.fleet_shipper._github_gate", lambda *args: {"state": "pending"})
    monkeypatch.setattr("aegis_core.fleet_shipper.urlopen", lambda *args, **kwargs: pytest.fail("pending gate POST"))
    result = execute_ship_plan(_remote_ready_plan(), token="long-enough-test-token")
    assert result["execution_results"][0]["status"] == "held"
    assert "not green" in result["execution_results"][0]["reason"]


def test_execute_rechecks_conflict_preflight_then_posts_once(monkeypatch):
    monkeypatch.setattr("aegis_core.fleet_shipper._github_get_pull", lambda *args: None)
    monkeypatch.setattr("aegis_core.fleet_shipper._github_json", lambda *args: {"commit": {"sha": "abc123"}})
    monkeypatch.setattr("aegis_core.fleet_shipper._github_gate", lambda *args: {"state": "success"})
    monkeypatch.setattr("aegis_core.fleet_shipper.audit_github_repository", lambda metadata, token=None: {
        "deliveries": [{"branch": "feat/aegis-fix", "head": "abc123", "status": "mergeable"}]})

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"number":9,"html_url":"https://github.com/org/repo/pull/9"}'

    requests = []
    monkeypatch.setattr("aegis_core.fleet_shipper.urlopen", lambda request, timeout: (requests.append(request), Response())[1])
    result = execute_ship_plan(_remote_ready_plan(), token="long-enough-test-token")
    assert result["execution_results"][0]["status"] == "created"
    assert result["execution_results"][0]["pull_request"]["number"] == 9
    assert len(requests) == 1
    assert requests[0].get_method() == "POST"


def test_aegis_ship_fleet_cli_documents_safe_default_and_explicit_execution():
    result = subprocess.run([sys.executable, "-m", "aegis_core.cli", "ship-fleet", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--owner" in result.stdout
    assert "--repo" in result.stdout
    assert "--execute" in result.stdout
    assert "Default is a dry run" in result.stdout


def test_aegis_ship_fleet_local_cli_defaults_to_dry_run(tmp_path):
    root = repository(tmp_path / "cli")
    add_delivery(root)
    result = subprocess.run([sys.executable, "-m", "aegis_core.cli", "ship-fleet", "--repo", str(root),
                             "--base-branch", "main"], capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["executed"] is False
