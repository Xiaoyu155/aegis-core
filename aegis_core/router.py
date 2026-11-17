"""Thread-safe in-process quotas, monotonic budgets and exact replay breaker."""
import hashlib
import json
import random
import re
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid

ALIASES = {"codex": "codex_cli", "sub2api": "openai_compatible", "openai": "openai_compatible",
           "claude": "anthropic", "gemini": "openai_compatible", "ollama": "openai_compatible"}
PROVIDERS = {"codex_cli", "openai_compatible", "anthropic"}


def normalize_provider(provider: str) -> str:
    result = ALIASES.get(provider, provider)
    if result not in PROVIDERS:
        raise ValueError("unknown provider")
    return result


def load_models(path: Path) -> dict:
    # JSON is a strict, unambiguous YAML 1.2 subset; do not install a YAML parser.
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    Router(result)  # Validate before the first job is admitted.
    return result


class Router:
    def __init__(self, config: dict, *, sleep=time.sleep, jitter=None) -> None:
        self.config = json.loads(json.dumps(config, allow_nan=False))
        if set(config) != {"version", "min_verify_reserve_sec", "groups", "tiers", "unattended_work_order"} or config["version"] != 1:
            raise ValueError("invalid routing config")
        reserve = config["min_verify_reserve_sec"]
        if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0:
            raise ValueError("positive verification reserve required")
        self.reserve = reserve
        self.groups = self.config["groups"]
        if any(type(v) is not int or v <= 0 for v in self.groups.values()):
            raise ValueError("positive concurrency quotas required")
        self.nodes = {}
        for tier, nodes in self.config["tiers"].items():
            if type(nodes) is not list or not nodes:
                raise ValueError("nonempty tier nodes required")
            for node in nodes:
                if set(node) != {"id", "provider", "model", "model_env", "base_url", "base_url_env", "api_key_env", "timeout_sec", "max_concurrency", "group", "weight", "enabled"}:
                    raise ValueError("missing/unknown node fields")
                if type(node["id"]) is not str or not node["id"] or node["id"] in self.nodes:
                    raise ValueError("unique slot IDs required")
                node["provider"] = normalize_provider(node["provider"])
                if node["group"] not in self.groups:
                    raise ValueError("unknown concurrency group")
                for field in ("weight", "max_concurrency"):
                    if type(node[field]) is not int or node[field] <= 0:
                        raise ValueError("positive weight/concurrency required")
                timeout = node["timeout_sec"]
                if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0 or type(node["enabled"]) is not bool:
                    raise ValueError("invalid node timeout/enabled")
                for field in ("model", "model_env", "base_url", "base_url_env", "api_key_env"):
                    if type(node[field]) is not str:
                        raise ValueError("node string fields required")
                node["tier"] = tier
                self.nodes[node["id"]] = node
        self.chain = self.config["unattended_work_order"]
        if not self.chain or self.chain[-1] != "blocked_needs_human" or len(set(self.chain)) != len(self.chain) or any(tier not in self.config["tiers"] for tier in self.chain[:-1]):
            raise ValueError("invalid unattended downgrade chain")
        self.sleep = sleep
        self.jitter = jitter or random.Random(0).random
        self.cooldowns = {}
        self.events = []
        self._lock = threading.RLock()
        self._leases = {}
        self._used = {slot: 0 for slot in self.nodes}
        self._group_used = {group: 0 for group in self.groups}
        self._served = {slot: 0 for slot in self.nodes}
        self._rejected = {}
        self._repeats = {}
        self._quarantined = set()

    def _blocked(self, reason):
        return {"status": "blocked_needs_human", "reason": reason}

    def acquire(self, tier: str, *, deadline: float, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (now, deadline)):
            raise ValueError("finite monotonic deadline required")
        if tier not in self.chain[:-1]:
            raise ValueError("tier outside unattended chain")
        remaining = deadline - now
        if remaining <= self.reserve:
            return self._blocked("verification_reserve")
        with self._lock:
            for candidate_tier in self.chain[self.chain.index(tier):-1]:
                nodes = [node for node in self.nodes.values() if node["tier"] == candidate_tier
                         and node["enabled"] and node["id"] not in self._quarantined
                         and self.cooldowns.get(node['id'], 0) <= now
                         and self._used[node["id"]] < node["max_concurrency"]
                         and self._group_used[node["group"]] < self.groups[node["group"]]]
                if not nodes:
                    continue
                node = min(nodes, key=lambda n: (self._served[n["id"]] / n["weight"], n["id"]))
                lease_id = uuid.uuid4().hex
                self._leases[lease_id] = node["id"]
                self._used[node["id"]] += 1
                self._group_used[node["group"]] += 1
                self._served[node["id"]] += 1
                return {"status": "admitted", "lease_id": lease_id, "slot_id": node["id"],
                        "tier": candidate_tier, "provider": node["provider"], "deadline": deadline,
                        "timeout_sec": min(node["timeout_sec"], remaining - self.reserve)}
        return self._blocked("quota_or_quarantine")

    def release(self, lease_id: str) -> None:
        with self._lock:
            slot = self._leases.pop(lease_id, None)
            if slot is None:
                raise ValueError("unknown/released lease")
            self._used[slot] -= 1
            self._group_used[self.nodes[slot]["group"]] -= 1

    @staticmethod
    def patch_hash(patch: str) -> str:
        return hashlib.sha256(patch.encode("utf-8")).hexdigest()

    def record_rejection(self, slot_id: str, patch: str) -> str:
        with self._lock:
            if slot_id not in self.nodes:
                raise KeyError(slot_id)
            digest = self.patch_hash(patch)
            if self._rejected.get(slot_id) != digest:
                self._repeats[slot_id] = 0
            self._rejected[slot_id] = digest
            return digest

    def record_output(self, slot_id: str, patch: str) -> dict:
        with self._lock:
            if slot_id not in self.nodes:
                raise KeyError(slot_id)
            same = self.patch_hash(patch) == self._rejected.get(slot_id)
            self._repeats[slot_id] = self._repeats.get(slot_id, 0) + 1 if same else 0
            repeated = self._repeats[slot_id] >= 2
            if repeated:
                self._quarantined.add(slot_id)
            tier = self.nodes[slot_id]["tier"]
            next_tier = self.chain[self.chain.index(tier) + 1] if repeated else tier
            return {"repeated_patch": repeated, "repeat_count": self._repeats[slot_id],
                    "status": "blocked_needs_human" if next_tier == "blocked_needs_human" else ("escalate" if repeated else "continue"),
                    "next_tier": next_tier, "patch_sha256": self.patch_hash(patch)}

    def call(self, tier: str, messages: list[dict], *, deadline: float, transport=None, require_json=False, max_retries=2) -> dict:
        """A transport is injected in tests; no network is used by the simulator."""
        attempted = set()
        while len(attempted) < len(self.nodes):
            admitted = self.acquire(tier, deadline=deadline)
            if admitted["status"] != "admitted":
                return admitted
            slot = admitted["slot_id"]
            try:
                if slot in attempted:
                    return self._blocked("providers_unavailable")
                attempted.add(slot)
                remaining = deadline - time.monotonic() - self.reserve
                if remaining <= 0:
                    return self._blocked("verification_reserve")
                # Copy: untrusted transports cannot mutate shared router configuration.
                node = dict(self.nodes[slot])
                timeout = min(admitted["timeout_sec"], remaining)
                for retry in range(max_retries + 1):
                    try:
                        result = (transport or request_provider)(node, messages, timeout=min(timeout, deadline-time.monotonic()-self.reserve), deadline=deadline)
                        if require_json:
                            result = json.dumps(repair_json(result), ensure_ascii=False)
                        break
                    except (OSError, ValueError, KeyError, TimeoutError, subprocess.SubprocessError) as error:
                        transient = not isinstance(error, urllib.error.HTTPError) or error.code in (408, 429, 500, 502, 503, 504)
                        delay = min(4, .1 * 2**retry + .1*self.jitter())
                        if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                            try:
                                delay = max(delay, float(error.headers.get('Retry-After', '1')))
                            except (ValueError, AttributeError):
                                delay = max(delay, 1)
                            with self._lock:
                                self.cooldowns[slot] = time.monotonic() + delay
                        with self._lock:
                            self.events.append({'slot': slot, 'retry': retry, 'error': type(error).__name__, 'penalty': -.1})
                        if not transient or retry >= max_retries or delay >= deadline-time.monotonic()-self.reserve:
                            raise
                        self.sleep(delay)
                if time.monotonic() > deadline - self.reserve:
                    return self._blocked("verification_reserve")
                if type(result) is not str:
                    raise ValueError("provider response must be text")
                candidate_patch = result
                try:
                    envelope = json.loads(result)
                    if isinstance(envelope, dict) and isinstance(envelope.get("patch"), str):
                        candidate_patch = envelope["patch"]
                except ValueError:
                    pass
                breaker = self.record_output(slot, candidate_patch)
                if breaker["repeated_patch"]:
                    if breaker["next_tier"] == "blocked_needs_human":
                        return {**self._blocked("repeated_patch"), **breaker}
                    tier = breaker["next_tier"]
                    continue
                return {**admitted, "output": result, "breaker": breaker, "route_penalty": -.1*len(self.events)}
            except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError):
                # Do not put vendor payloads, URLs with auth or API keys in traces.
                with self._lock:
                    self._quarantined.add(slot)
            finally:
                self.release(admitted["lease_id"])
        return self._blocked("providers_unavailable")


def request_provider(node: dict, messages: list[dict], *, timeout: float, deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or timeout <= 0:
        raise TimeoutError("deadline expired")
    timeout = min(timeout, remaining)
    model = os.environ.get(node["model_env"], node["model"]) if node["model_env"] else node["model"]
    if not model:
        raise ValueError("configure an actual upstream model ID")
    if node["provider"] == "codex_cli":
        prompt = json.dumps(messages, ensure_ascii=False, allow_nan=False)
        output = subprocess.run(["codex", "exec", "--sandbox", "read-only", "--json", "--model", model, "-"], input=prompt,
                                capture_output=True, text=True, timeout=timeout, check=True).stdout
        agent_messages = []
        for line in output.splitlines():
            event = json.loads(line)
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "agent_message":
                agent_messages.append(item["text"])
        if not agent_messages:
            raise ValueError("Codex CLI emitted no completed agent message")
        return agent_messages[-1]
    base = os.environ.get(node["base_url_env"], node["base_url"]) if node["base_url_env"] else node["base_url"]
    parsed = urllib.parse.urlsplit(base)
    if (not base or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.hostname or (parsed.scheme != "https" and not
                (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}))):
        raise ValueError("configure a credential-free TLS or loopback provider endpoint")
    key = os.environ.get(node["api_key_env"], "") if node["api_key_env"] else ""
    if node["api_key_env"] and not key:
        raise ValueError("provider credential reference is unavailable")
    headers = {"Content-Type": "application/json"}
    if node["provider"] == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        system = "\n".join(message["content"] for message in messages if message["role"] == "system")
        body = {"model": model, "max_tokens": 2048, "system": system,
                "messages": [message for message in messages if message["role"] != "system"]}
        suffix = "/messages"
    else:
        if key:
            headers["Authorization"] = "Bearer " + key
        body = {"model": model, "messages": messages, "temperature": 0, "stream": False}
        suffix = "/chat/completions"
    request = urllib.request.Request(base.rstrip("/") + suffix, data=json.dumps(body, allow_nan=False).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        if len(data) > 2_000_000:
            raise ValueError('model response too large')
        result = repair_json(data.decode('utf-8'))
    if node["provider"] == "anthropic":
        return "".join(block["text"] for block in result["content"] if block.get("type") == "text")
    return result["choices"][0]["message"]["content"]


def repair_json(text):
    """Bounded syntax repair only: fences, trailing commas, missing final braces."""
    if not isinstance(text, str) or len(text.encode()) > 2_000_000:
        raise ValueError('invalid JSON response size')
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    quoted, escaped, stack = False, False, []
    for char in text:
        if escaped:
            escaped = False
        elif char == '\\' and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted:
            if char in '{[':
                stack.append('}' if char == '{' else ']')
            elif char in '}]':
                if not stack or stack.pop() != char:
                    raise ValueError('mismatched JSON brackets')
    if quoted:
        raise ValueError('truncated JSON string')
    value = json.loads(text + ''.join(reversed(stack)), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(value, dict):
        raise ValueError('expected JSON object')
    return value
