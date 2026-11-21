import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import time

import pytest

from luna.model_router import ModelRouter, QuotaLedger, gateway_environment, load_config


@pytest.fixture
def gateway():
    calls = []
    outcomes = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, self.headers.get("Authorization")))
            queue = outcomes.get(body["model"], [])
            kind = queue.pop(0) if queue else 200
            if kind == "timeout":
                time.sleep(0.28)
                kind = 200
            if kind == 307:
                self.send_response(307)
                self.send_header("Location", "/redirected")
                self.end_headers()
                return
            self.send_response(kind if isinstance(kind, int) else 200)
            if kind == 429:
                self.send_header("Retry-After", "0.5")
            self.end_headers()
            if kind == 200:
                body = {"usage": {"total_tokens": 9}, "choices": [{"message": {"content": "{}"}}]}
            elif kind == "invalid":
                body = {"choices": []}
            elif kind == "missing_usage":
                body = {"choices": [{"message": {"content": "{}"}}]}
            else:
                return
            try:
                self.wfile.write(json.dumps(body).encode())
            except OSError:
                pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", calls, outcomes
        finally:
            server.shutdown()
            worker.join()


@pytest.fixture
def data():
    return json.loads((Path(__file__).parents[1] / "config/luna-models.json").read_text(encoding="utf-8"))


def build(root, data, gateway, *, clock=time.time, monotonic=time.monotonic):
    url, _, _ = gateway
    for name, node in data["models"].items():
        for field in ("base_url_env", "model_env", "api_key_env"):
            node.pop(field, None)
        node.update(model=name, base_url=url + "/v1")
    path = root / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    config = load_config(path, root)
    ledger = QuotaLedger(root, config, clock=clock)
    delays = []
    return ModelRouter(config, ledger, sleep=delays.append, clock=monotonic), ledger, delays


def ask(router, role="planner", seconds=2):
    return router.call(role, [{"role": "user", "content": "测试"}], deadline=router.clock() + seconds)


@pytest.mark.parametrize("role,slot", [("planner", "reasoning"), ("reviewer", "reasoning"), ("coder", "coding")])
@pytest.mark.parametrize("outcomes,expected,requests,delays", [
    ([], "primary", 1, []),
    ([429], "primary", 2, [0.5]),
    ([429, 429], "local", 3, [0.5]),
    ([500], "local", 2, []),
    ([401], "local", 2, []),
    (["invalid"], "local", 2, []),
    ([307], "local", 2, []),
    (["missing_usage"], "primary", 1, []),
])
def test_http_routes(tmp_path, data, gateway, role, slot, outcomes, expected, requests, delays):
    _, calls, fault = gateway
    fault[slot] = list(outcomes)
    router, ledger, slept = build(tmp_path, data, gateway)
    response = ask(router, role)
    assert response["slot_id"] == (slot if expected == "primary" else "local")
    assert response["output"] == "{}"
    assert len(calls) == requests
    assert all(path == "/v1/chat/completions" and body["max_tokens"] == 4096 for path, body, _ in calls)
    assert slept == delays
    balances = {item["id"]: item for item in ledger.snapshot()["models"]}
    assert sum(item["requests_used"] for item in balances.values()) == requests
    assert balances[response["slot_id"]]["tokens_used"] >= 9


@pytest.mark.parametrize("field,limit", [("request_limit", 1), ("token_limit", 4136)])
@pytest.mark.parametrize("role,primary", [("planner", "reasoning"), ("reviewer", "reasoning"), ("coder", "coding")])
def test_endpoint_quota_falls_back(tmp_path, data, gateway, field, limit, role, primary):
    data["models"][primary][field] = limit
    router, ledger, _ = build(tmp_path, data, gateway)
    assert ask(router, role)["slot_id"] == primary
    second_router = ModelRouter(router.config, QuotaLedger(tmp_path, router.config))
    assert ask(second_router, role)["slot_id"] == "local"
    assert next(item for item in ledger.snapshot()["models"] if item["id"] == primary)["requests_used"] == 1


@pytest.mark.parametrize("value,field", [(0, "window_sec"), (False, "retries"), (6, "retries"),
                                         (-1, "task_request_limit"), (100001, "max_output_tokens"),
                                         (2, "version"), (1.0, "task_token_limit")])
def test_invalid_config(tmp_path, data, gateway, value, field):
    data[field] = value
    with pytest.raises(ValueError):
        build(tmp_path, data, gateway)


@pytest.mark.parametrize("role,primary", [("planner", "reasoning"), ("reviewer", "reasoning"), ("coder", "coding")])
def test_local_http_timeout_switches(tmp_path, data, gateway, role, primary, monkeypatch):
    from luna import model_router
    gateway[2][primary] = ["timeout", "timeout"]
    original = model_router._complete
    monkeypatch.setattr(model_router, "_complete",
                        lambda endpoint, payload, timeout: original(endpoint, payload, min(timeout, 0.12)))
    router, _, delays = build(tmp_path, data, gateway)
    assert ask(router, role)["slot_id"] == "local"
    assert delays == [0.25]
    assert len(gateway[1]) == 3


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_budget_counts_failed_requests(tmp_path, data, gateway, attempt):
    data["task_request_limit"] = attempt
    gateway[2].update(reasoning=[429, 429], local=[429, 429])
    router, _, _ = build(tmp_path, data, gateway)
    assert ask(router)["reason"] == "task_budget_exhausted"
    assert len(gateway[1]) == attempt


def test_tokens_reset_and_new_task_sees_quota(tmp_path, data, gateway):
    data["window_sec"] = 10
    now = [9.0]
    router, ledger, _ = build(tmp_path, data, gateway, clock=lambda: now[0])
    assert ask(router)["status"] == "admitted"
    assert ledger.snapshot()["models"][0]["tokens_used"] == 9
    now[0] = 10
    assert ledger.snapshot()["models"][0]["tokens_used"] == 0
    assert ledger.snapshot()["reset_at"] == 20


def test_concurrent_reservations_are_atomic(tmp_path, data, gateway):
    data["models"]["reasoning"]["request_limit"] = 4
    _, ledger, _ = build(tmp_path, data, gateway)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ledger.reserve("reasoning", 100), range(24)))
    assert sum(result is not None for result in results) == 4
    assert ledger.snapshot()["models"][0]["tokens_used"] == 400


def test_dotenv_environment_wins_without_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNA_MODEL", "process")
    (tmp_path / ".env.luna").write_text('# comment\nLUNA_MODEL=file\nLUNA_KEY="literal $VAR"\n', encoding="utf-8")
    env = gateway_environment(tmp_path)
    assert env["LUNA_MODEL"] == "process"
    assert env["LUNA_KEY"] == "literal $VAR"


def test_no_calls_after_deadline(tmp_path, data, gateway):
    router, ledger, _ = build(tmp_path, data, gateway)
    assert ask(router, seconds=-1)["reason"] == "deadline"
    assert not gateway[1]
    assert not ledger.path.exists()
