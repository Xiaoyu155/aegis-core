"""Wall-clock deadlines and a durable, shared execution budget."""
from __future__ import annotations

import json
import math
from threading import RLock
import time
import uuid

BUDGET_EXCEEDED_HALT = "BUDGET_EXCEEDED_HALT"
LIMIT_KEYS = ("max_token_budget", "max_cost_usd", "max_wall_clock_sec")
# Estimated blended provider price. Offline patches and local processes have zero API cost.
USD_PER_TOKEN = 0.00001
MAX_OUTPUT_TOKENS = 2048


class BudgetExceeded(Exception):
    def __init__(self, limit):
        self.limit = limit
        super().__init__(BUDGET_EXCEEDED_HALT + ": " + limit)


def budget_limits(source):
    if not isinstance(source, dict) or set(source) - set(LIMIT_KEYS):
        raise ValueError("invalid budget limits")
    result = {}
    for key, value in source.items():
        if (type(value) not in (int, float) or (key == "max_token_budget" and type(value) is not int)
                or not 0 <= value <= 1e15 or not math.isfinite(value)):
            raise ValueError("invalid " + key)
        result[key] = value
    return result


def estimate_tokens(value):
    return len(value.encode("utf-8"))


class BudgetGuard:
    def __init__(self, ledger, scope, limits, *, parent=None):
        self.ledger = ledger
        self.scope = scope + ":" + uuid.uuid4().hex
        self.limits = budget_limits(limits)
        self.parent = parent
        self.lock = parent.lock if parent else RLock()
        self.started = time.monotonic()
        self.tokens = self.reserved = 0
        self.reward = 0.0
        self.halted_by = None

    @property
    def deadline(self):
        local = self.started + self.limits.get("max_wall_clock_sec", math.inf)
        return min(local, self.parent.deadline) if self.parent else local

    def scopes(self):
        return (self, *self.parent.scopes()) if self.parent else (self,)

    def check(self, extra=0):
        with self.lock:
            for guard in self.scopes():
                checks = {"max_token_budget": guard.tokens + guard.reserved + extra,
                          "max_cost_usd": (guard.tokens + guard.reserved + extra) * USD_PER_TOKEN,
                          "max_wall_clock_sec": time.monotonic() - guard.started}
                if guard.halted_by:
                    raise BudgetExceeded(guard.halted_by)
                for key, value in checks.items():
                    if key not in guard.limits:
                        continue
                    limit = guard.limits[key]
                    if value > limit or (key == "max_wall_clock_sec" and value >= limit):
                        guard.halt(key)

    def halt(self, key):
        with self.lock:
            if self.halted_by is None:
                self.ledger.record_budget_event([self.scope], "halt", 0, 0, 0, limit=key)
                self.halted_by = key
        raise BudgetExceeded(self.halted_by)

    def halt_wall(self):
        guard = self
        while guard.parent and guard.parent.deadline == self.deadline:
            guard = guard.parent
        guard.halt("max_wall_clock_sec")

    def record(self, operation, *, tokens=0, seconds=0, reward=0):
        with self.lock:
            scopes = self.scopes()
            self.ledger.record_budget_event([guard.scope for guard in scopes], operation,
                                            tokens, seconds, reward)
            for guard in scopes:
                guard.tokens += tokens
                guard.reward += reward

    def model_request(self, request, node, messages, *, timeout, deadline):
        prompt = estimate_tokens(json.dumps(messages, ensure_ascii=False))
        reservation = prompt + MAX_OUTPUT_TOKENS * 4
        with self.lock:
            if node.get("provider") == "codex_cli":
                for guard in self.scopes():
                    for key in ("max_token_budget", "max_cost_usd"):
                        if key in guard.limits:
                            guard.halt(key)
            self.check(reservation)
            node["budgeted"] = any(key in guard.limits for guard in self.scopes()
                                   for key in ("max_token_budget", "max_cost_usd"))
            node["_budget_collect_usage"] = True
            for guard in self.scopes():
                guard.reserved += reservation
        start = time.monotonic()
        result = None
        node.pop("_budget_token_usage", None)
        try:
            deadline = min(deadline, self.deadline)
            result = request(node, messages, timeout=min(timeout, deadline - start), deadline=deadline)
        finally:
            with self.lock:
                for guard in self.scopes():
                    guard.reserved -= reservation
                usage = node.pop("_budget_token_usage", None)
                if type(usage) is not int or usage < 0:
                    usage = prompt + (estimate_tokens(result) if isinstance(result, str) else 0)
                self.record("provider_request", tokens=usage,
                            seconds=time.monotonic() - start)
                self.check()
        return result

    def run_gate(self, runner, path, command, *, timeout, **kwargs):
        self.check()
        start = time.monotonic()
        try:
            return runner(path, command, timeout=min(timeout, self.deadline - start), **kwargs)
        finally:
            self.record("sandbox_gate", seconds=time.monotonic() - start)
            self.check()

    def report(self):
        cost = self.tokens * USD_PER_TOKEN
        return {"scope": self.scope, "limits": self.limits, "token_consumption": self.tokens,
                "estimated_cost_usd": cost, "estimated_usd_per_token": USD_PER_TOKEN,
                "wall_clock_sec": time.monotonic() - self.started, "reward": self.reward,
                "reward_per_1000_tokens": self.reward * 1000 / self.tokens if self.tokens else None,
                "reward_per_usd": self.reward / cost if cost else None,
                "halted_by": next((guard.halted_by for guard in self.scopes() if guard.halted_by), None)}
