"""Role-based OpenAI-compatible routing and a shared, transactional quota ledger."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROLES = ("planner", "coder", "reviewer")
DEFAULT_CONFIG = Path("config/luna-models.json")


def config_path(root: Path, override=None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    production = root / DEFAULT_CONFIG
    return production if production.is_file() else root / "config/models.yaml"


def gateway_environment(root: Path) -> dict[str, str]:
    values = {}
    path = root / ".env.luna"
    if path.is_file():
        for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if not separator or not re.fullmatch(r"LUNA_[A-Z0-9_]+", name.strip()):
                raise ValueError(f".env.luna: invalid assignment on line {number}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            values[name.strip()] = value
    return values | dict(os.environ)


def _integer(data, key, minimum=1, maximum=1_000_000_000):
    value = data[key]
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be an integer in {minimum}..{maximum}")
    return value


def _valid_endpoint(url):
    parsed = urlsplit(url)
    return (bool(parsed.hostname) and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment
            and (parsed.scheme == "https" or
                 (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})))


@dataclass(frozen=True)
class Endpoint:
    model: str
    base_url: str
    credential: str
    configured: bool
    timeout_sec: int
    token_limit: int
    request_limit: int


@dataclass(frozen=True)
class RoutingConfig:
    routes: dict[str, list[str]]
    models: dict[str, Endpoint]
    window_sec: int
    task_token_limit: int
    task_request_limit: int
    max_output_tokens: int
    retries: int


def load_config(path: Path, root: Path) -> RoutingConfig | None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "tiers" in data and "roles" not in data:
        return None
    env = gateway_environment(root)

    def setting(node, name):
        value = env.get(node[f"{name}_env"], "") if f"{name}_env" in node else node.get(name, "")
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value.strip()

    try:
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported Luna model config version")
        models = {}
        for name, node in data["models"].items():
            if not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
                raise ValueError("invalid endpoint ID")
            model, url = setting(node, "model"), setting(node, "base_url")
            key = env.get(node["api_key_env"], "") if node.get("api_key_env") else ""
            if url and not _valid_endpoint(url):
                raise ValueError(f"{name}: use HTTPS or loopback HTTP for the model endpoint")
            models[name] = Endpoint(model, url.rstrip("/"), key,
                                    bool(model and url and (key or not node.get("api_key_env"))),
                                    _integer(node, "timeout_sec", maximum=600),
                                    _integer(node, "token_limit"), _integer(node, "request_limit"))
        routes = data["roles"]
        if set(routes) != set(ROLES):
            raise ValueError("roles must define planner, coder and reviewer")
        for role, names in routes.items():
            if not isinstance(names, list) or not names or len(set(names)) != len(names) or not all(
                name in models for name in names
            ):
                raise ValueError(f"{role}: list unique, defined endpoint IDs in fallback order")
        return RoutingConfig(routes, models, _integer(data, "window_sec", maximum=86400),
                             _integer(data, "task_token_limit"), _integer(data, "task_request_limit"),
                             _integer(data, "max_output_tokens", maximum=100000),
                             _integer(data, "retries", minimum=0, maximum=5))
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("invalid Luna model routing configuration") from error


class QuotaLedger:
    """Reserve before sending and reconcile against provider usage when available."""

    def __init__(self, root: Path, config: RoutingConfig, *, clock=time.time):
        self.path = root / ".aegis" / "luna-quota.sqlite"
        self.config = config
        self.clock = clock

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS quota (
                window INTEGER, slot TEXT, tokens INTEGER, requests INTEGER,
                PRIMARY KEY (window, slot))""")
        except BaseException:
            db.close()
            raise
        return db

    def _window(self):
        return int(self.clock()) // self.config.window_sec * self.config.window_sec

    def _balance(self, db, window, slot):
        return db.execute("SELECT tokens, requests FROM quota WHERE window=? AND slot=?",
                          (window, slot)).fetchone() or (0, 0)

    def reserve(self, slot, tokens):
        endpoint = self.config.models[slot]
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            window = self._window()
            db.execute("DELETE FROM quota WHERE window < ?", (window - 86400,))
            used, requests = self._balance(db, window, slot)
            if used + tokens > endpoint.token_limit or requests >= endpoint.request_limit:
                return None
            db.execute("INSERT OR REPLACE INTO quota VALUES (?, ?, ?, ?)",
                       (window, slot, used + tokens, requests + 1))
            return window

    def settle(self, window, slot, reserved, actual):
        with closing(self._connect()) as db, db:
            db.execute("UPDATE quota SET tokens=tokens+? WHERE window=? AND slot=?",
                       (actual - reserved, window, slot))

    def snapshot(self):
        window = self._window()
        balances = {}
        if self.path.is_file():
            with closing(self._connect()) as db:
                balances = {slot: (tokens, requests) for slot, tokens, requests in
                            db.execute("SELECT slot, tokens, requests FROM quota WHERE window=?", (window,))}
        return {"reset_at": window + self.config.window_sec, "models": [
            {"id": slot, "configured": endpoint.configured,
             "tokens_used": balances.get(slot, (0, 0))[0],
             "requests_used": balances.get(slot, (0, 0))[1],
             "tokens_remaining": max(0, endpoint.token_limit - balances.get(slot, (0, 0))[0]),
             "requests_remaining": max(0, endpoint.request_limit - balances.get(slot, (0, 0))[1])}
            for slot, endpoint in self.config.models.items()]}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _complete(endpoint, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if endpoint.credential:
        headers["Authorization"] = f"Bearer {endpoint.credential}"
    request = Request(endpoint.base_url + "/chat/completions", data=payload, headers=headers)
    with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
        data = json.loads(response.read(2_000_001))
    output = data["choices"][0]["message"]["content"]
    if not isinstance(output, str) or not isinstance(json.loads(output), dict):
        raise ValueError("expected a JSON object in model completion")
    usage = data.get("usage") or {}
    actual = usage.get("total_tokens")
    return output, actual if type(actual) is int and actual > 0 else None


class ModelRouter:
    """Create one router per task and share its ledger across processes."""

    def __init__(self, config: RoutingConfig, ledger: QuotaLedger, *, sleep=time.sleep, clock=time.monotonic):
        self.config = config
        self.ledger = ledger
        self.sleep = sleep
        self.clock = clock
        self.tokens = self.requests = 0

    def call(self, role, messages, *, deadline):
        messages = list(messages)
        prompt = sum(len(message["content"].encode("utf-8")) + 16 for message in messages) + 16
        reason = "no_configured_endpoint"
        for slot in self.config.routes[role]:
            endpoint = self.config.models[slot]
            if not endpoint.configured:
                continue
            for attempt in range(self.config.retries + 1):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    return {"status": "blocked", "reason": "deadline"}
                output_limit = min(self.config.max_output_tokens,
                                   self.config.task_token_limit - self.tokens - prompt,
                                   endpoint.token_limit - prompt)
                if self.requests >= self.config.task_request_limit or (
                    self.config.task_token_limit - self.tokens <= prompt
                ):
                    return {"status": "blocked", "reason": "task_budget_exhausted"}
                reserved = prompt + output_limit
                if output_limit <= 0 or (window := self.ledger.reserve(slot, reserved)) is None:
                    reason = "endpoint_quota_exhausted"
                    break
                self.requests += 1
                self.tokens += reserved
                payload = json.dumps({"model": endpoint.model, "messages": messages,
                                      "max_tokens": output_limit}).encode()
                retry_after = 0.0
                try:
                    output, actual = _complete(endpoint, payload, min(remaining, endpoint.timeout_sec))
                except HTTPError as error:
                    code = error.code
                    try:
                        retry_after = float(error.headers.get("Retry-After", 0))
                        if not math.isfinite(retry_after):
                            retry_after = 0
                    except (TypeError, ValueError):
                        pass
                    error.close()
                    reason = f"http_{code}"
                    if code != 429:
                        break
                except URLError as error:
                    reason = "timeout" if isinstance(error.reason, TimeoutError) else "connection_failed"
                    if reason != "timeout":
                        break
                except TimeoutError:
                    reason = "timeout"
                except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
                    reason = "invalid_or_unavailable_completion"
                    break
                else:
                    if actual is not None:
                        self.ledger.settle(window, slot, reserved, actual)
                        self.tokens += actual - reserved
                    if self.tokens > self.config.task_token_limit:
                        return {"status": "blocked", "reason": "task_budget_exhausted"}
                    return {"status": "admitted", "output": output, "tier": role, "slot_id": slot,
                            "provider": "openai-compatible"}
                delay = max(0.25 * 2**attempt, retry_after)
                if attempt == self.config.retries or delay >= deadline - self.clock():
                    break
                self.sleep(delay)
        return {"status": "blocked", "reason": reason}
