import json
import subprocess
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from luna.cli import main
from luna.dashboard import create_dashboard_server
from luna.pr_reviewer import _DEFAULT_RECEIPTS, review_fleet

SHA = "7" * 40
REPO = "Xiaoyu155/aegis-core"


@pytest.fixture
def github(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("remote", "add", "origin", f"https://github.com/{REPO}.git")
    pull = {"number": 23, "title": "test change", "html_url": f"https://github.com/{REPO}/pull/23",
            "head": {"sha": SHA}}
    detail = {**pull, "state": "open", "mergeable": True, "mergeable_state": "clean", "draft": False}
    receipts = [{"name": name, "app": {"slug": "github-actions"}, "head_sha": SHA, "status": "completed",
                 "conclusion": "success", "id": index}
                for index, name in enumerate(sorted(_DEFAULT_RECEIPTS))]
    responses = {
        f"repos/{REPO}/pulls?state=open&sort=created&direction=asc&per_page=100": [[pull]],
        f"repos/{REPO}/pulls/23": [detail],
        f"repos/{REPO}/commits/{SHA}/check-runs?filter=latest&per_page=100": [{"check_runs": receipts}],
        f"repos/{REPO}/commits/{SHA}/status?per_page=100": [{"total_count": 0, "state": "pending"}],
    }
    seen = []
    def run(argv, **kwargs):
        assert argv[:5] == ["gh", "api", "--method", "GET", "--paginate"]
        assert kwargs["cwd"] == tmp_path
        url = argv[5]
        seen.append(url)
        return subprocess.CompletedProcess(argv, 0, stdout="".join(json.dumps(page) for page in responses[url]), stderr="")
    import luna.pr_reviewer as reviewer
    import types
    monkeypatch.setattr(reviewer, "subprocess", types.SimpleNamespace(run=run))
    return tmp_path, detail, receipts, responses, seen


@pytest.mark.parametrize("key,value,issue", [
    ("mergeable", False, "merge_conflict"), ("mergeable", None, "mergeability_unconfirmed"),
    ("mergeable_state", "dirty", "github_branch_gate_not_clean"),
    ("mergeable_state", "behind", "github_branch_gate_not_clean"),
    ("mergeable_state", "blocked", "github_branch_gate_not_clean"),
    ("mergeable_state", "unstable", "github_branch_gate_not_clean"),
    ("draft", True, "draft"), ("state", "closed", "pull_changed_during_read"),
    ("head", {"sha": "1" * 40}, "pull_changed_during_read"),
])
def test_fleet_blocks_merge_on_changed_pr_or_gate(github, key, value, issue):
    root, detail, _, _, _ = github
    detail[key] = value
    item, = review_fleet(root)["pull_requests"]
    assert item["risk"] == "high"
    assert issue in item["issues"]
    assert item["merge_command"] is None


@pytest.mark.parametrize("field,value,issue", [
    ("name", "unrelated", "aegis_receipt_missing"),
    ("app", {"slug": "untrusted"}, "aegis_receipt_missing"),
    ("head_sha", "2" * 40, "aegis_receipt_missing"),
    ("status", "in_progress", "aegis_receipt_not_green"),
    ("status", "queued", "checks_not_green"),
    ("conclusion", "failure", "aegis_receipt_not_green"),
    ("conclusion", "cancelled", "checks_not_green"),
    ("conclusion", "skipped", "aegis_receipt_not_green"),
    ("conclusion", "neutral", "aegis_receipt_not_green"),
])
def test_fleet_requires_complete_trusted_receipts(github, field, value, issue):
    root, _, receipts, _, _ = github
    receipts[0][field] = value
    item, = review_fleet(root)["pull_requests"]
    assert issue in item["issues"]
    assert item["merge_command"] is None


@pytest.mark.parametrize("changed,added,deleted,risk", [(99, 0, 0, "low"), (100, 0, 0, "medium"),
                                                       (1, 1500, 1499, "low"), (1, 1500, 1500, "medium")])
def test_fleet_risk_and_head_pinned_command(github, changed, added, deleted, risk):
    root, detail, receipts, _, _ = github
    detail.update(changed_files=changed, additions=added, deletions=deleted)
    item, = review_fleet(root)["pull_requests"]
    assert item["risk"] == risk and not item["issues"]
    assert len(item["receipts"]) == len(receipts)
    assert item["merge_command"] == f"gh pr merge 23 --repo {REPO} --match-head-commit {SHA} --squash"


def test_fleet_pagination_and_cli_output(github, capsys):
    root, detail, _, responses, seen = github
    listing = next(iter(responses))
    responses[listing].append([{**detail, "number": 24}])
    responses[f"repos/{REPO}/pulls/24"] = [{**detail, "number": 24, "draft": True}]
    assert main(["review-fleet", "--repo", str(root)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["count"] == 2
    assert [item["number"] for item in data["pull_requests"]] == [24, 23]
    assert seen.count(listing) == 1


@pytest.mark.parametrize("context", ["pending", "failure", "error"])
def test_fleet_combined_commit_status(github, context):
    root, _, _, responses, _ = github
    responses[f"repos/{REPO}/commits/{SHA}/status?per_page=100"] = [{"total_count": 1, "state": context}]
    item, = review_fleet(root)["pull_requests"]
    assert item["issues"] == ["checks_not_green"]


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    monkeypatch.setattr("luna.dashboard.status_snapshot", lambda root, **kwargs: {
        "disk": {}, "ticket_timeline": [{"ticket_id": "sample"}],
        "interceptions": [{"reason": "rejected"}],
        "model_quota": {"models": [{"id": "coder", "tokens_remaining": 9}]},
    })
    with create_dashboard_server(tmp_path, port=0) as server:
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            worker.join()


@pytest.mark.parametrize("section", ["工单时间线", "风险拦截记录", "模型配额", "待验收清单",
                                      "width=device-width", "@media(max-width:640px)"])
def test_dashboard_serves_mobile_panels(dashboard, section):
    with urlopen(dashboard, timeout=2) as response:
        assert response.headers["Cache-Control"] == "no-store"
        assert section in response.read().decode()


@pytest.mark.parametrize("key", ["ticket_timeline", "interceptions", "model_quota"])
def test_dashboard_serves_data(dashboard, key):
    with urlopen(dashboard + "/api/status", timeout=2) as response:
        assert json.load(response)[key]


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_dashboard_remains_read_only(dashboard, method):
    with pytest.raises(HTTPError) as caught:
        urlopen(Request(dashboard + "/api/status", method=method), timeout=2)
    assert caught.value.code == 405
