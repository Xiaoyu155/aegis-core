"""Append-only SHA-256 linked audit records for local enforcement events."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Iterator


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_ZERO_HASH = "0" * 64


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _entry_digest(entry: dict) -> str:
    body = {key: value for key, value in entry.items() if key != "entry_hash"}
    return hashlib.sha256(_canonical(body)).hexdigest()


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Serialize appends across both threads and local processes."""
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class AuditLedger:
    """Store immutable event lines and detect deletion, edits, and reordering."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.head_path = self.path.with_suffix(self.path.suffix + ".head.json")

    def _read_unlocked(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    raise ValueError(f"audit ledger has an incomplete final line at {line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"audit ledger JSON error at line {line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"audit ledger row {line_number} is not an object")
                rows.append(row)
        return rows

    @staticmethod
    def _verify_rows(rows: list[dict]) -> dict:
        previous = _ZERO_HASH
        for index, row in enumerate(rows, 1):
            if row.get("sequence") != index:
                return {"valid": False, "entries": len(rows), "invalid_sequence": index}
            if row.get("prev_hash") != previous:
                return {"valid": False, "entries": len(rows), "broken_link": index}
            digest = _entry_digest(row)
            if row.get("entry_hash") != digest:
                return {"valid": False, "entries": len(rows), "tampered_entry": index}
            previous = digest
        return {"valid": True, "entries": len(rows), "head_hash": previous}

    def _verify_anchor(self, rows: list[dict], verified: dict) -> dict:
        if not rows and not self.head_path.exists():
            return verified
        try:
            anchor = json.loads(self.head_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"valid": False, "entries": len(rows), "error": "audit_head_anchor_missing_or_invalid"}
        if anchor.get("sequence") != len(rows) or anchor.get("head_hash") != verified.get("head_hash"):
            return {"valid": False, "entries": len(rows), "error": "audit_head_anchor_mismatch"}
        return {**verified, "anchored": True}

    def _write_anchor(self, row: dict) -> None:
        temporary = self.head_path.with_suffix(self.head_path.suffix + ".tmp")
        encoded = json.dumps({"sequence": row["sequence"], "head_hash": row["entry_hash"]},
                             sort_keys=True, separators=(",", ":")) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.head_path)

    def append(self, event: str, details: dict | None = None, *, timestamp: float | None = None) -> dict:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("event must be a nonempty string")
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise TypeError("details must be a JSON object")
        stamp = time.time() if timestamp is None else float(timestamp)
        if stamp < 0:
            raise ValueError("timestamp must be nonnegative")
        with _file_lock(self.lock_path):
            rows = self._read_unlocked()
            verified = self._verify_rows(rows)
            verified = self._verify_anchor(rows, verified)
            if not verified["valid"]:
                raise ValueError("audit ledger verification failed; refusing append")
            row = {"sequence": len(rows) + 1,
                   "timestamp": datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
                   "event": event.strip(), "details": details,
                   "prev_hash": verified["head_hash"]}
            row["entry_hash"] = _entry_digest(row)
            encoded = _canonical(row) + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                with os.fdopen(descriptor, "ab", closefd=True) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                raise
            self._write_anchor(row)
            return row

    def verify(self) -> dict:
        # A read-only doctor/preflight on a clean checkout must not dirty it by
        # creating an empty .lock file when no audit stream has ever existed.
        if not self.path.exists() and not self.head_path.exists():
            return {"valid": True, "entries": 0, "head_hash": _ZERO_HASH}
        with _file_lock(self.lock_path):
            try:
                rows = self._read_unlocked()
                verified = self._verify_rows(rows)
                if not verified["valid"]:
                    return verified
                return self._verify_anchor(rows, verified)
            except (OSError, ValueError) as error:
                return {"valid": False, "entries": 0, "error": str(error)}

