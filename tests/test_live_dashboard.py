"""Loopback HTTP API tests with WAL readers racing actual ledger writers."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
from urllib.parse import urlsplit

import pytest

from aegis_core.ledger import Ledger
from aegis_core.live_dashboard import DashboardReadModel, create_server


@pytest.fixture
def dashboard(tmp_path):
    database = tmp_path / ".aegis" / "ledger.sqlite3"
    database.parent.mkdir(parents=True)
    ledger = Ledger(database)
    for index in range(3):
        ledger.create_ticket(f"ticket-{index}", f"task {index}", {"index": index}, "frozen contract")
    with ledger._transaction() as db:
        Ledger._trace(db, "ticket-0", {"status": "queued"},
                      {"op": "repair_attempt", "patch_sha256": "abc"},
                      {"status": "running", "cheat_detected": True}, -1, None, 0, time.time())
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "fleet_dashboard_v13.json").write_text(json.dumps({
        "repositories": [{"repository": "owner/repo", "deliveries": []}],
        "summary": {"repositories_scanned": 1, "delivery_branches": 0, "conflicts": 0},
    }), encoding="utf-8")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    (datasets / "world_model_bench_v13.json").write_text(json.dumps({
        "dataset": {"total_samples": 800},
        "gate": {"status": "blocked", "allowed_mode_transition": False},
    }), encoding="utf-8")
    server = create_server(tmp_path, port=0, db_path=database)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    address = f"http://{server.server_address[0]}:{server.server_address[1]}"
    try:
        yield address, ledger, database, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def get(url):
    parsed = urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    connection.request("GET", parsed.path + ("?" + parsed.query if parsed.query else ""))
    response = connection.getresponse()
    payload = response.read()
    headers = dict(response.getheaders())
    status = response.status
    connection.close()
    return status, headers, payload


@pytest.mark.parametrize("endpoint,key", [
    ("/api/overview", "tickets"), ("/api/tickets", "items"),
    ("/api/traces", "items"), ("/api/fleet", "summary"),
])
def test_each_read_api_returns_structured_json_from_real_local_state(dashboard, endpoint, key):
    base, _ledger, _database, _server = dashboard
    status, headers, body = get(base + endpoint)
    response = json.loads(body)
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert response["ok"] is True
    assert key in response["data"]


def test_overview_reports_ticket_trace_interception_and_world_model_state(dashboard):
    base, _ledger, _database, _server = dashboard
    status, _headers, body = get(base + "/api/overview")
    data = json.loads(body)["data"]
    assert status == 200
    assert data["tickets"] == 3
    assert data["running"] == 3
    assert data["traces"] == 4
    assert data["interceptions"] == 1
    assert data["world_model"]["samples"] == 800
    assert data["world_model"]["allowed_mode_transition"] is False


def test_ticket_and_trace_endpoints_decode_real_json_ledger_fields(dashboard):
    base, _ledger, _database, _server = dashboard
    ticket_data = json.loads(get(base + "/api/tickets")[2])["data"]
    trace_data = json.loads(get(base + "/api/traces")[2])["data"]
    assert ticket_data["count"] == 3
    assert {item["ticket_id"] for item in ticket_data["items"]} == {"ticket-0", "ticket-1", "ticket-2"}
    repair = next(item for item in trace_data["items"] if item["action_a0"]["op"] == "repair_attempt")
    assert repair["actual_state_s1"]["cheat_detected"] is True
    assert repair["ticket_id"] == "ticket-0"


@pytest.mark.parametrize("limit", ["1", "2", "3", "10", "500"])
def test_api_limit_is_bounded_and_respected(limit, dashboard):
    base, _ledger, _database, _server = dashboard
    status, _headers, body = get(base + "/api/tickets?limit=" + limit)
    data = json.loads(body)["data"]
    assert status == 200
    assert len(data["items"]) <= min(int(limit), 3)


@pytest.mark.parametrize("query", ["0", "501", "-1", "many", "1.5"])
def test_bad_limit_values_return_client_errors_without_running_a_query(query, dashboard):
    base, _ledger, _database, _server = dashboard
    status, _headers, body = get(base + "/api/tickets?limit=" + query)
    assert status == 400
    assert json.loads(body)["ok"] is False


def test_dashboard_root_serves_chinese_dark_single_page_and_refreshes_api(dashboard):
    base, _ledger, _database, _server = dashboard
    status, headers, body = get(base + "/")
    text = body.decode("utf-8")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "Aegis 确定性软件工厂" in text
    assert "setInterval(refresh,3000)" in text
    assert "color-scheme:dark" in text
    assert "textContent" in text


def test_api_sets_no_store_and_nosniff_response_headers(dashboard):
    status, headers, _body = get(dashboard[0] + "/api/overview")
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_unknown_route_is_a_structured_404(dashboard):
    status, _headers, body = get(dashboard[0] + "/api/unknown")
    assert status == 404
    assert json.loads(body) == {"ok": False, "error": "not_found"}


def test_api_refuses_all_mutating_methods(dashboard):
    base = urlsplit(dashboard[0])
    for method in ("POST", "PUT", "DELETE"):
        connection = http.client.HTTPConnection(base.hostname, base.port, timeout=2)
        connection.request(method, "/api/overview", body=b"{}")
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        assert response.status == 405
        assert body["error"] == "read_only"


def test_read_model_connection_is_sqlite_query_only_and_cannot_write(dashboard):
    _base, _ledger, database, _server = dashboard
    model = DashboardReadModel(database.parent.parent, database)
    connection = model._connect()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM tickets")
    finally:
        connection.close()
    with _ledger._connection() as db:
        assert db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 3


def test_dashboard_rejects_non_loopback_bind_addresses(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        create_server(tmp_path, host="0.0.0.0")


def test_dashboard_accepts_ephemeral_loopback_port_for_integration_use(tmp_path):
    server = create_server(tmp_path, port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[1] > 0
    finally:
        server.server_close()


@pytest.mark.parametrize("port", [-1, 65536, 90000])
def test_invalid_dashboard_ports_are_rejected(port, tmp_path):
    with pytest.raises(ValueError, match="port"):
        create_server(tmp_path, port=port)


def test_concurrent_wal_writers_and_loopback_readers_complete_without_lockups(dashboard):
    base, ledger, _database, _server = dashboard
    failures = []

    def writer():
        for index in range(40):
            try:
                ledger.create_ticket(f"writer-{index}", "parallel", {"n": index}, "frozen")
            except Exception as error:  # a real writer failure is recorded for the assertion below
                failures.append(repr(error))

    def reader():
        for _ in range(40):
            status, _headers, body = get(base + "/api/overview")
            if status != 200:
                failures.append(f"HTTP {status}: {body[:200]!r}")
            else:
                json.loads(body)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = [pool.submit(writer), *(pool.submit(reader) for _ in range(5))]
        for task in tasks:
            task.result(timeout=15)
    assert failures == []
    assert time.monotonic() - started < 15
    with ledger._connection() as db:
        assert db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 43


def test_readers_continue_while_an_uncommitted_wal_writer_holds_reserved_lock(dashboard):
    base, _ledger, database, _server = dashboard
    writer = sqlite3.connect(database, timeout=2, isolation_level=None)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("BEGIN IMMEDIATE")
    try:
        status, _headers, body = get(base + "/api/overview")
        assert status == 200
        assert json.loads(body)["data"]["tickets"] == 3
    finally:
        writer.rollback()
        writer.close()


def test_fleet_api_reads_only_the_latest_local_fleet_artifact(dashboard):
    base, _ledger, _database, _server = dashboard
    status, _headers, body = get(base + "/api/fleet")
    data = json.loads(body)["data"]
    assert status == 200
    assert data["summary"]["repositories_scanned"] == 1
    assert data["repositories"][0]["repository"] == "owner/repo"
    assert data["source"] == "artifacts/fleet_dashboard_v13.json"


@pytest.mark.parametrize("limit", [1, 5, 20, 99])
def test_trace_api_returns_latest_records_with_each_json_column_decoded(limit, dashboard):
    base = dashboard[0]
    data = json.loads(get(base + f"/api/traces?limit={limit}")[2])["data"]
    assert data["count"] <= limit
    assert all(isinstance(item["action_a0"], dict) for item in data["items"])
    assert all(isinstance(item["actual_state_s1"], dict) for item in data["items"])


@pytest.mark.parametrize("endpoint", [
    "/api/overview", "/api/tickets", "/api/traces", "/api/fleet",
    "/api/overview?limit=1", "/api/tickets?limit=1", "/api/traces?limit=1", "/api/fleet?limit=1",
])
def test_api_routes_remain_readable_during_repeated_real_loopback_requests(endpoint, dashboard):
    base = dashboard[0]
    for _ in range(3):
        status, _headers, body = get(base + endpoint)
        assert status == 200
        assert json.loads(body)["ok"] is True


def test_empty_missing_ledger_returns_well_formed_zero_state_without_creating_database(tmp_path):
    model = DashboardReadModel(tmp_path, tmp_path / "missing.sqlite3")
    report = model.overview()
    assert report["tickets"] == 0
    assert report["traces"] == 0
    assert report["interceptions"] == 0
    assert model.tickets()["items"] == []
    assert model.traces()["items"] == []
    assert not (tmp_path / "missing.sqlite3").exists()


@pytest.mark.parametrize("command", ["doctor", "serve", "synthesize-hard-negatives"])
def test_new_cli_commands_publish_help_without_importing_optional_model_packages(command):
    result = subprocess.run([sys.executable, "-m", "aegis_core.cli", command, "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_doctor_cli_prints_machine_readable_runtime_report(tmp_path):
    result = subprocess.run([sys.executable, "-m", "aegis_core.cli", "doctor", "--repo", str(tmp_path),
                             "--min-free-bytes", "0"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == "aegis.env_doctor.v1"
    assert report["ready"] is True


def test_serve_cli_starts_a_real_loopback_http_process(tmp_path):
    database = tmp_path / ".aegis" / "ledger.sqlite3"
    database.parent.mkdir(parents=True)
    Ledger(database).create_ticket("cli-ticket", "serve", {}, "frozen")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([sys.executable, "-m", "aegis_core.cli", "serve", "--repo", str(tmp_path),
                                "--db", str(database), "--port", str(port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while True:
            if process.poll() is not None:
                detail = process.stderr.read() if process.stderr else ""
                raise AssertionError(f"serve command exited early: {detail}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/overview", timeout=0.2) as response:
                    data = json.loads(response.read())
                    assert response.status == 200
                    assert data["data"]["tickets"] == 1
                    break
            except URLError:
                if time.monotonic() >= deadline:
                    raise AssertionError("serve command did not open its loopback listener")
                time.sleep(0.05)
    finally:
        process.terminate()
        process.wait(timeout=5)
        if process.stderr:
            process.stderr.close()
