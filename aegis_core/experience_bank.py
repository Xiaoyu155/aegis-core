"""Append-only, versioned causal memories distilled from real gated traces."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ExperienceBank:
    """Store trace summaries and human arbitration rules beside the WAL ledger."""

    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        if self.path == ":memory:":
            raise ValueError("experience bank requires a durable SQLite file")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS experience_records (
                    experience_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL UNIQUE,
                    version INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    source_kind TEXT NOT NULL,
                    trace_id INTEGER,
                    surprise REAL NOT NULL,
                    pattern TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    cause TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    human_rule TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experience_versions (
                    version INTEGER PRIMARY KEY,
                    created_at REAL NOT NULL,
                    added_count INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS experiences_no_update
                BEFORE UPDATE ON experience_records BEGIN
                    SELECT RAISE(ABORT, 'experience records are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS experiences_no_delete
                BEFORE DELETE ON experience_records BEGIN
                    SELECT RAISE(ABORT, 'experience records are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS experience_versions_no_update
                BEFORE UPDATE ON experience_versions BEGIN
                    SELECT RAISE(ABORT, 'experience versions are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS experience_versions_no_delete
                BEFORE DELETE ON experience_versions BEGIN
                    SELECT RAISE(ABORT, 'experience versions are append-only');
                END;
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _validate_rule(rule):
        if type(rule) is not dict:
            raise ValueError("human arbitration rule must be a JSON object")
        threshold = rule.get("ambiguity_threshold")
        if threshold is not None and (type(threshold) not in (int, float) or not 0.05 <= threshold <= 0.15):
            raise ValueError("ambiguity_threshold must be between 0.05 and 0.15")
        # Prevent arbitrary code-like values and keep a bounded, serializable record.
        encoded = _json(rule)
        if len(encoded) > 4096:
            raise ValueError("human arbitration rule is too large")

    @staticmethod
    def _fingerprint(record):
        stable = {key: record[key] for key in (
            "source_kind", "trace_id", "surprise", "pattern", "outcome", "cause", "resolution", "human_rule")}
        return hashlib.sha256(_json(stable).encode("utf-8")).hexdigest()

    def _append(self, records: list[dict]) -> dict:
        if not records:
            return {"added": 0, **self.status()}
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                version = int(db.execute("SELECT COALESCE(MAX(version),0)+1 FROM experience_versions").fetchone()[0])
                inserted = []
                for record in records:
                    self._validate_rule(record["human_rule"])
                    fingerprint = self._fingerprint(record)
                    cursor = db.execute("""INSERT OR IGNORE INTO experience_records
                        (fingerprint,version,created_at,source_kind,trace_id,surprise,pattern,outcome,cause,resolution,human_rule)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (fingerprint, version, now, record["source_kind"], record["trace_id"],
                         float(record["surprise"]), record["pattern"], record["outcome"],
                         record["cause"], record["resolution"], _json(record["human_rule"])))
                    if cursor.rowcount:
                        inserted.append(fingerprint)
                if inserted:
                    digest = hashlib.sha256("\n".join(inserted).encode("ascii")).hexdigest()
                    db.execute("INSERT INTO experience_versions VALUES (?,?,?,?)",
                               (version, now, len(inserted), digest))
                db.commit()
            except BaseException:
                db.rollback()
                raise
        return {"added": len(inserted), **self.status()}

    def ingest_traces(self, traces: list[dict]) -> dict:
        """Distill high-surprise ledger rows without copying whole patches."""
        records = []
        for trace in traces:
            actual = trace.get("actual_state_s1") or {}
            action = trace.get("action_a0") or {}
            error = trace.get("prediction_error") or {}
            diagnosis = actual.get("diagnosis") or {}
            contract = action.get("contract") or {}
            paths = contract.get("target_files") or action.get("target_files") or actual.get("changed_files") or []
            intent = contract.get("intent") or actual.get("wish") or ""
            outcome = actual.get("world_model_outcome") or error.get("actual_outcome") or actual.get("status", "unknown")
            cause = diagnosis.get("category") or actual.get("pre_block_cause") or outcome
            pattern = " ".join(str(value) for value in [intent, *paths, cause, outcome] if value)
            records.append({"source_kind": "real_trace", "trace_id": int(trace["trace_id"]),
                            "surprise": float(trace.get("composite_error", error.get("composite_error", 0.0))),
                            "pattern": pattern[:10000], "outcome": str(outcome)[:120],
                            "cause": str(cause)[:120],
                            "resolution": "dual_probe_passed" if actual.get("passed") else "gate_rejected",
                            "human_rule": {}})
        return self._append(records)

    def record_arbitration(self, *, pattern: str, rule: dict, cause: str = "human_arbitration",
                           resolution: str = "reviewed") -> dict:
        if type(pattern) is not str or not pattern.strip() or len(pattern) > 10000:
            raise ValueError("a bounded nonempty arbitration pattern is required")
        self._validate_rule(rule)
        record = {"source_kind": "human_arbitration", "trace_id": None, "surprise": 0.0,
                  "pattern": pattern, "outcome": "arbitrated", "cause": cause[:120],
                  "resolution": resolution[:500], "human_rule": rule}
        return self._append([record])

    def ingest_verified_recoveries(self, pairs: list[dict]) -> dict:
        """Index diagnosed failures that were followed by a verified repair."""
        records = []
        for pair in pairs:
            failure = pair["failed_trace"]
            action = failure["action_a0"]
            diagnosis = failure["actual_state_s1"]["diagnosis"]
            cause = str(diagnosis.get("failure_code") or diagnosis["category"])
            paths = action.get("target_files", [])
            kind = action.get("probe_kind", "unit")
            finding_codes = sorted({item["code"] for item in diagnosis.get("evidence", [])
                                    if isinstance(item, dict) and isinstance(item.get("code"), str)})
            pattern = " ".join([*paths, cause, *finding_codes, kind, "verified recovery"])
            records.append({"source_kind": "verified_recovery", "trace_id": int(failure["trace_id"]),
                            "surprise": 0.0, "pattern": pattern[:10000], "outcome": "gate_passed",
                            "cause": cause[:120], "resolution": f"dual_probe_recovered:{int(pair['passed_trace_id'])}",
                            "human_rule": {}})
        return self._append(records)

    def search(self, query: str, *, top_k: int = 5, min_similarity: float = 0.05) -> list[dict]:
        if type(query) is not str or not query.strip():
            raise ValueError("experience query is required")
        if type(top_k) is not int or not 1 <= top_k <= 50:
            raise ValueError("top_k must be 1..50")
        if type(min_similarity) not in (int, float) or not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be in [0,1]")
        from .runner import terms
        query_terms = terms(query)
        with self._connect() as db:
            rows = db.execute("SELECT * FROM experience_records ORDER BY experience_id DESC").fetchall()
        ranked = []
        for row in rows:
            record_terms = terms(row["pattern"] + " " + row["cause"] + " " + row["outcome"])
            union = query_terms | record_terms
            score = len(query_terms & record_terms) / max(1, len(union))
            if score < min_similarity:
                continue
            ranked.append({"experience_id": row["experience_id"], "version": row["version"],
                           "source_kind": row["source_kind"], "trace_id": row["trace_id"],
                           "surprise": row["surprise"], "pattern": row["pattern"],
                           "outcome": row["outcome"], "cause": row["cause"],
                           "resolution": row["resolution"], "human_rule": json.loads(row["human_rule"]),
                           "similarity": round(score, 6)})
        ranked.sort(key=lambda item: (-item["similarity"], -item["experience_id"]))
        return ranked[:top_k]

    def status(self) -> dict:
        with self._connect() as db:
            version = db.execute("SELECT version,content_sha256 FROM experience_versions ORDER BY version DESC LIMIT 1").fetchone()
            count = db.execute("SELECT COUNT(*) FROM experience_records").fetchone()[0]
        return {"experience_count": int(count), "version": int(version[0]) if version else 0,
                "version_sha256": version[1] if version else None}
