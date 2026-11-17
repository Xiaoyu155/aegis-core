"""Transactional tickets, fenced leases and append-only causal training traces."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("A durable file is required")
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript('''
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY, wish_raw TEXT NOT NULL,
                    compiled_intent TEXT NOT NULL, contract_stub TEXT NOT NULL,
                    assigned_tier TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS leases (
                    ticket_id TEXT PRIMARY KEY REFERENCES tickets(ticket_id),
                    owner TEXT NOT NULL, token TEXT NOT NULL, expires_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS causal_traces (
                    trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    step_idx INTEGER NOT NULL, state_s0 TEXT NOT NULL,
                    action_a0 TEXT NOT NULL, predicted_next_state TEXT,
                    actual_state_s1 TEXT NOT NULL, reward REAL NOT NULL,
                    token_cost INTEGER NOT NULL, timestamp REAL NOT NULL,
                    UNIQUE(ticket_id, step_idx));
                CREATE TRIGGER IF NOT EXISTS traces_no_update BEFORE UPDATE ON causal_traces
                BEGIN SELECT RAISE(ABORT, 'traces are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS traces_no_delete BEFORE DELETE ON causal_traces
                BEGIN SELECT RAISE(ABORT, 'traces are append-only'); END;
            ''')

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=15000")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def create_ticket(self, ticket_id: str, wish_raw: str, compiled_intent: dict,
                      contract_stub: str, assigned_tier: str = "local") -> None:
        if not ticket_id or not contract_stub.strip():
            raise ValueError("ticket_id and frozen contract_stub are required")
        now = time.time()
        with self._transaction() as db:
            db.execute("INSERT INTO tickets VALUES (?,?,?,?,?,'queued',?,?)",
                       (ticket_id, wish_raw, _json(compiled_intent), contract_stub,
                        assigned_tier, now, now))
            self._trace(db, ticket_id, {}, {"op": "create_ticket"},
                        {"status": "queued"}, 0, None, 0, now)

    def acquire_lease(self, ticket_id: str, owner: str, ttl: float = 60) -> str | None:
        if not owner or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("owner and positive finite TTL are required")
        with self._transaction() as db:
            now = time.time()
            row = db.execute("SELECT status FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            if row is None:
                raise KeyError(ticket_id)
            if row["status"] in {"succeeded", "failed"}:
                return None
            lease = db.execute("SELECT expires_at FROM leases WHERE ticket_id=?", (ticket_id,)).fetchone()
            if lease and lease[0] > now:
                return None
            token = uuid.uuid4().hex
            db.execute("INSERT OR REPLACE INTO leases VALUES (?,?,?,?)", (ticket_id, owner, token, now + ttl))
            return token

    def get_ticket(self, ticket_id: str) -> dict:
        with self._connection() as db:
            row = db.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            if row is None:
                raise KeyError(ticket_id)
            result = dict(row)
            result["compiled_intent"] = json.loads(result["compiled_intent"])
            return result

    @staticmethod
    def _trace(db, ticket_id, s0, action, s1, reward, prediction, cost, now):
        if type(cost) is not int or cost < 0 or type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError("invalid reward or token_cost")
        if not all(isinstance(x, dict) for x in (s0, action, s1)):
            raise ValueError("causal states and action must be objects")
        if prediction is not None and not isinstance(prediction, dict):
            raise ValueError("prediction must be an object or null")
        step = db.execute("SELECT COALESCE(MAX(step_idx),-1)+1 FROM causal_traces WHERE ticket_id=?", (ticket_id,)).fetchone()[0]
        db.execute("INSERT INTO causal_traces VALUES (NULL,?,?,?,?,?,?,?,?,?)",
                   (ticket_id, step, _json(s0), _json(action), _json(prediction) if prediction is not None else None,
                    _json(s1), reward, cost, now))
        return step

    def transition(self, ticket_id: str, lease_token: str, expected_status: str,
                   new_status: str, *, state_s0: dict, action_a0: dict,
                   actual_state_s1: dict, reward: float,
                   predicted_next_state: dict | None = None, token_cost: int = 0) -> int:
        edges = {"queued": {"running"}, "running": {"succeeded", "failed"}}
        if new_status not in edges.get(expected_status, set()):
            raise ValueError("illegal status transition")
        with self._transaction() as db:
            now = time.time()
            lease = db.execute("SELECT * FROM leases WHERE ticket_id=?", (ticket_id,)).fetchone()
            if not lease or lease["token"] != lease_token or lease["expires_at"] <= now:
                raise PermissionError("missing, expired or stale lease")
            prior = db.execute("SELECT actual_state_s1 FROM causal_traces WHERE ticket_id=? ORDER BY step_idx DESC LIMIT 1", (ticket_id,)).fetchone()
            if not prior or json.loads(prior[0]) != state_s0:
                raise ValueError("state_s0 must equal the preceding actual state")
            if state_s0.get("status") != expected_status or actual_state_s1.get("status") != new_status:
                raise ValueError("causal states must match lifecycle status")
            changed = db.execute("UPDATE tickets SET status=?,updated_at=? WHERE ticket_id=? AND status=?", (new_status, now, ticket_id, expected_status)).rowcount
            if changed != 1:
                raise ValueError("status conflict")
            step = self._trace(db, ticket_id, state_s0, action_a0, actual_state_s1,
                               reward, predicted_next_state, token_cost, now)
            if new_status in {"succeeded", "failed"}:
                db.execute("DELETE FROM leases WHERE ticket_id=?", (ticket_id,))
            return step

    def export_jsonl(self, destination: Path, ticket_id: str | None = None) -> int:
        destination = Path(destination)
        if destination.resolve() in {Path(self.path + suffix).resolve() for suffix in ("", "-wal", "-shm", "-journal")}:
            raise ValueError("cannot export over database or its sidecars")
        with self._connection() as db:
            query = "SELECT * FROM causal_traces"
            args = ()
            if ticket_id is not None:
                query += " WHERE ticket_id=?"
                args = (ticket_id,)
            rows = db.execute(query + " ORDER BY ticket_id,step_idx", args)
            count = 0
            # Unique sibling file prevents concurrent exporters sharing a staging path.
            temp = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                with temp.open("w", encoding="utf-8", newline="\n") as output:
                    for row in rows:
                        item = dict(row)
                        for key in ("state_s0", "action_a0", "predicted_next_state", "actual_state_s1"):
                            item[key] = json.loads(item[key]) if item[key] is not None else None
                        item["state_s1"] = item["actual_state_s1"]
                        output.write(_json(item) + "\n")
                        count += 1
                temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)
            return count
