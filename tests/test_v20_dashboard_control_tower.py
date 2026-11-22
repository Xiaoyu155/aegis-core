from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from aegis_core.live_dashboard import DashboardReadModel, create_server


def test_dashboard_security_overview_reads_key_status_ledger_and_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    (repo / ".aegis").mkdir(parents=True)
    (repo / ".aegis/audit.jsonl").write_text("", encoding="utf-8")
    artifacts = repo / "artifacts"
    artifacts.mkdir()
    (artifacts / "v20_performance.json").write_text(json.dumps({"p95_ms": 12.5}), encoding="utf-8")
    sample_report = {"rounds": [{"round": 3, "escape_rate": 0.01,
                                  "transforms": ["python:rename", "javascript:rename"],
                                  "per_class": {"swallow_exception": {"transform_recall": {
                                      "python:rename": 1.0, "javascript:rename": 0.9}}}}]}
    (artifacts / "v20_adversarial_report.json").write_text(json.dumps(sample_report), encoding="utf-8")
    overview = DashboardReadModel(repo).security_overview()
    assert overview["enforce"]["ledger_integrity"] == "tampered"
    assert overview["p95_ms"] == 12.5
    assert overview["defect_matrix"][0]["recall"] == ["100.0%", "90.0%"]


def test_dashboard_quarantine_returns_metadata_and_bounded_diff_preview(tmp_path):
    directory = tmp_path / ".aegis/quarantine"
    directory.mkdir(parents=True)
    metadata = {"mode": "staged", "findings": [{"code": "MISSING_TIMEOUT", "path": "app.py", "line": 7}]}
    (directory / "entry.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "entry.diff").write_text("+" + "x" * 1400, encoding="utf-8")
    payload = DashboardReadModel(tmp_path).quarantine()
    assert payload["count"] == 1
    assert payload["items"][0]["findings"][0]["code"] == "MISSING_TIMEOUT"
    assert len(payload["items"][0]["preview"]) == 1200


@pytest.mark.parametrize("endpoint", ["/", "/api/overview", "/api/tickets", "/api/traces", "/api/fleet", "/api/quarantine"])
def test_dashboard_serves_local_page_and_all_json_apis(tmp_path, endpoint):
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}{endpoint}"
        with urlopen(url, timeout=3) as response:
            body = response.read()
            assert response.status == 200
            if endpoint == "/":
                assert "Aegis 控制台" in body.decode()
            else:
                payload = json.loads(body)
                assert payload["ok"] is True and isinstance(payload["data"], dict)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("query", ["?limit=0", "?limit=-1", "?limit=501", "?limit=no", "?limit=1.2"])
def test_dashboard_rejects_invalid_pagination(tmp_path, query):
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/tickets{query}"
        with pytest.raises(HTTPError) as error:
            urlopen(url, timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("index", range(20))
def test_dashboard_handles_concurrent_read_only_api_requests(tmp_path, index):
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/overview"
        def fetch(_):
            with urlopen(url, timeout=3) as response:
                return json.loads(response.read())["data"]["tickets"]
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(fetch, range(16)))
        assert values == [0] * 16
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_dashboard_is_loopback_only_and_read_only(tmp_path):
    with pytest.raises(ValueError):
        create_server(tmp_path, host="0.0.0.0", port=0)
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{server.server_address[1]}/api/overview", method="POST")
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)

