"""Luna logging, product views, preflight and embedded end-to-end combat run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from luna.combat import _git_permissions, _model_gateway
from luna.cli import main as luna_main
from luna.dashboard import create_dashboard_server
from luna.flight_recorder import FlightRecorder, verify_jsonl
from luna.runner import _demo_candidates, _patch_for, run_requirement
from luna.status import format_status, status_snapshot


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


def _fixture_repo(tmp_path):
    root = tmp_path / "business"
    source = Path(__file__).parents[1] / "examples" / "luna_combat_sandbox"
    (root / "examples" / "luna_combat_sandbox").mkdir(parents=True)
    shutil.copyfile(source / "__init__.py", root / "examples" / "luna_combat_sandbox" / "__init__.py")
    shutil.copyfile(source / "inventory.py", root / "examples" / "luna_combat_sandbox" / "inventory.py")
    (root / ".gitignore").write_text(".worktrees/\nartifacts/luna_logs/\n__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Luna tests")
    _git(root, "config", "user.email", "luna-test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "embedded business baseline")
    return root


@pytest.mark.parametrize("step", ["doctor", "impact", "probe", "gate", "signature", "delivery"])
def test_flight_recorder_writes_hash_linked_product_events(tmp_path, step):
    recorder = FlightRecorder(tmp_path, run_id="chain-" + step, requirement="bounded stock")
    recorder.record(step, command=["luna", "run"], verdict="RECORDED", details={"n": 1})
    verified = verify_jsonl(recorder.machine_path)
    assert verified["valid"] and verified["entries"] == 1
    row = json.loads(recorder.machine_path.read_text(encoding="utf-8"))
    assert row["command"] == ["luna", "run"]
    assert row["previous_sha256"] == "0" * 64
    assert len(row["event_sha256"]) == 64


@pytest.mark.parametrize("field,value", [
    ("step", "changed"), ("command", "git push"), ("verdict", "SUCCEEDED"),
    ("details", {"tampered": True}), ("timestamp", "2000-01-01T00:00:00Z"),
    ("replay_handle", "forged-handle"), ("diff_sha256", "f" * 64),
    ("sequence", 8), ("previous_sha256", "1" * 64),
])
def test_flight_recorder_detects_tampered_event_fields(tmp_path, field, value):
    recorder = FlightRecorder(tmp_path, run_id="tamper-" + field)
    recorder.record("frozen", verdict="READY")
    row = json.loads(recorder.machine_path.read_text(encoding="utf-8"))
    row[field] = value
    recorder.machine_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert verify_jsonl(recorder.machine_path)["valid"] is False


@pytest.mark.parametrize("run_id", ["../escape", "with space", "a/b", "", "x" * 81, "ümlaut"])
def test_flight_recorder_rejects_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError):
        FlightRecorder(tmp_path, run_id=run_id)


@pytest.mark.parametrize("duration", [-1, float("nan"), "slow", True])
def test_flight_recorder_rejects_invalid_probe_timings(tmp_path, duration):
    recorder = FlightRecorder(tmp_path, run_id="timing-" + str(abs(hash(str(duration)))))
    with pytest.raises(ValueError):
        recorder.record("probe", probe_elapsed_ms=duration)


@pytest.mark.parametrize("steps", [[], ["需求理解"], ["计划", "门禁"], ["回放"]])
def test_flight_recorder_report_has_five_product_sections(tmp_path, steps):
    recorder = FlightRecorder(tmp_path, run_id="report-" + str(len(steps)), requirement="修复库存边界")
    for step in steps:
        recorder.record(step, verdict="RECORDED")
    report = recorder.render_report(understanding="修复库存边界", plan=["影响分析"],
                                    interceptions=["拦截错误候选"], evidence=["12 个用例通过"],
                                    deliverables=["交付分支 luna/delivery/run"], final_status="SUCCEEDED")
    content = report.read_text(encoding="utf-8")
    assert all(section in content for section in ("需求理解", "拆解方案", "拦截了哪些坑",
                                                   "真实测试证据", "最终交付物与合并建议"))
    assert "SHA-256 链头" in content


@pytest.mark.parametrize("line", [b"{\"sequence\":1}", b"not json\n", b"[]\n"])
def test_jsonl_verifier_rejects_incomplete_or_invalid_entries(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(line)
    assert verify_jsonl(path)["valid"] is False


@pytest.mark.parametrize("target", ["app.py", "pkg/mod.py", "examples/calc.py", "x/y/z.py"])
def test_diff_builder_is_scoped_to_named_source(tmp_path, target):
    file = tmp_path / target
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("def f():\n    return 1\n", encoding="utf-8")
    diff = _patch_for(tmp_path, target, "def f():\n    return 2\n")
    assert diff.startswith(f"diff --git a/{target} b/{target}\n")
    assert "-    return 1" in diff and "+    return 2" in diff


@pytest.mark.parametrize("suffix", ["", "?view=product", "#anchor"])
def test_luna_dashboard_serves_product_status_api(tmp_path, suffix):
    server = create_dashboard_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/status{suffix}", timeout=3) as response:
            payload = json.loads(response.read())
            assert response.status == 200
            assert payload["status"] != "HEALTHY"
            assert payload["project_health"] in {"READY", "READY_WITH_MISSING_CONFIG_TEMPLATES", "SPARSE_MODE"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.2", "example.com"])
def test_luna_dashboard_refuses_non_loopback_hosts(tmp_path, host):
    with pytest.raises(ValueError, match="loopback"):
        create_dashboard_server(tmp_path, host=host, port=0)


@pytest.mark.parametrize("port", [-1, 65536, 1.5])
def test_luna_dashboard_validates_port(tmp_path, port):
    with pytest.raises(ValueError):
        create_dashboard_server(tmp_path, port=port)


@pytest.mark.parametrize("endpoint,expected", [("/", 200), ("/api/status", 200), ("/missing", 404)])
def test_luna_dashboard_returns_expected_routes(tmp_path, endpoint, expected):
    server = create_dashboard_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            with urlopen(f"http://127.0.0.1:{server.server_address[1]}{endpoint}", timeout=3) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        assert status == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_luna_dashboard_is_read_only(tmp_path, method):
    server = create_dashboard_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{server.server_address[1]}/api/status", method=method)
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("text", ["", " " , "\n\t"])
def test_luna_run_rejects_empty_requirement(text, tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        run_requirement(text, tmp_path)


def test_product_status_counts_quarantine_and_formats_pending_acceptance(tmp_path):
    quarantine = tmp_path / ".aegis" / "quarantine"
    quarantine.mkdir(parents=True)
    (quarantine / "blocked.json").write_text(json.dumps({"findings": [{"code": "MISSING_TIMEOUT"}]}),
                                               encoding="utf-8")
    snapshot = status_snapshot(tmp_path)
    assert snapshot["risk_interceptions"] == 1
    assert snapshot["pending_acceptance"] == []
    assert "项目健康度" in format_status(snapshot)


@pytest.mark.parametrize("missing", ["missing.json", "bad.json"])
def test_model_gateway_fails_closed_without_a_valid_route_config(tmp_path, missing):
    path = tmp_path / missing
    if missing == "bad.json":
        path.write_text("{broken", encoding="utf-8")
    assert _model_gateway(path)["status"] == "blocked"


def test_model_gateway_requires_endpoint_and_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("LUNA_API_KEY", "")
    config = {"tiers": {"remote": [{"id": "test", "enabled": True, "model": "test-model",
                                     "model_env": "", "base_url": "https://localhost:9999/v1",
                                     "base_url_env": "", "api_key_env": "LUNA_API_KEY"}]}}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert _model_gateway(path)["status"] == "blocked"


def test_model_gateway_reports_ready_local_route_without_exposing_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("LUNA_API_KEY", "test-secret")
    import luna.combat as combat
    class Connected:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
    monkeypatch.setattr(combat.socket, "create_connection", lambda *_args, **_kwargs: Connected())
    config = {"tiers": {"remote": [{"id": "test", "enabled": True, "model": "test-model",
                                     "model_env": "", "base_url": "http://127.0.0.1:9999/v1",
                                     "base_url_env": "", "api_key_env": "LUNA_API_KEY"}]}}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = _model_gateway(path)
    assert report["status"] == "ready"
    assert "test-secret" not in json.dumps(report)


def test_git_permissions_are_blocked_without_remote_or_github_auth(monkeypatch, tmp_path):
    _git(tmp_path, "init", "-q")
    monkeypatch.setattr("luna.combat.shutil.which", lambda _name: None)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert _git_permissions(tmp_path)["status"] == "blocked"


def test_dirty_baseline_is_logged_and_blocked_before_any_candidate_runs(tmp_path):
    root = _fixture_repo(tmp_path)
    (root / "unreviewed.txt").write_text("dirty", encoding="utf-8")
    result = run_requirement("Fix stock bounds", root, run_id="dirty-baseline", demo=True)
    assert result["status"] == "BLOCKED"
    assert "clean" in result["reason"]
    assert result["machine_log"] is None
    assert not (root / "artifacts" / "luna_logs" / "dirty-baseline.jsonl").exists()


def test_luna_run_end_to_end_red_intercept_green_signature_and_delivery_branch(tmp_path, capsys):
    root = _fixture_repo(tmp_path)
    code = luna_main(["run", "将库存限制在 0 到 10，非法输入返回空值", "--repo", str(root),
                      "--run-id", "combat-e2e", "--demo", "--timeout", "8"])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "Luna 运行：SUCCEEDED" in output
    machine = root / "artifacts" / "luna_logs" / "combat-e2e.jsonl"
    result = verify_jsonl(machine)
    assert result["valid"]
    rows = [json.loads(line) for line in machine.read_text(encoding="utf-8").splitlines()]
    gate = next(row["details"] for row in rows if row["step"] == "隔离红绿探针与 L3 审查")
    assert gate["baseline_true_red"] == "ASSERTION_FAILED"
    assert len(gate["rounds"]) == 2
    assert gate["rounds"][-1]["winner"] == "bounded_stock"
    failed = [item for round_row in gate["rounds"]
              for item in round_row["tournament"]["evaluations"] if not item["accepted"]]
    assert failed and failed[0]["candidate_id"] == "floor_only"
    assert "Ed25519 验签通过" in output
    assert (root / "artifacts" / "luna_logs" / "combat-e2e_signature.json").is_file()
    package = root / "artifacts" / "luna_logs" / "combat-e2e_delivery.zip"
    assert package.is_file()
    branch = next(row["details"]["branch"] for row in rows if row["step"] == "生成隔离交付分支")
    assert branch.startswith("luna/delivery/combat-e2e")
    assert _git(root, "status", "--porcelain") == ""

