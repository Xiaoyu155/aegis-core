"""Dual-track Luna run records: hash-linked JSONL plus a readable delivery report."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import shutil
import os
import uuid


ZERO_HASH = "0" * 64
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _event_hash(event: dict) -> str:
    return hashlib.sha256(_canonical({key: value for key, value in event.items()
                                      if key != "event_sha256"})).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FlightRecorder:
    """Append durable, tamper-evident events and render the five-part PM report."""

    def __init__(self, repo_root: str | Path, run_id: str | None = None,
                 requirement: str = "", *, directory: str | Path | None = None) -> None:
        self.root = Path(repo_root).resolve()
        self.run_id = (run_id if run_id is not None else
                       datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                       + "-" + uuid.uuid4().hex[:8])
        if not _RUN_ID.fullmatch(self.run_id):
            raise ValueError("run_id may contain only letters, digits, dot, underscore and hyphen")
        self.directory = Path(directory).expanduser().resolve() if directory else (
            self.root / "artifacts" / "luna_logs")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.machine_path = self.directory / f"{self.run_id}.jsonl"
        self.report_path = self.directory / f"{self.run_id}_report.md"
        if self.machine_path.exists() or self.report_path.exists():
            raise FileExistsError(f"Luna run id already exists: {self.run_id}")
        descriptor = os.open(self.machine_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        self.requirement = requirement
        self.events: list[dict] = []
        self.head_hash = ZERO_HASH

    def relocate(self, directory: str | Path) -> None:
        """Move the active log to persistent storage after Git baselines are frozen."""
        destination = Path(directory).expanduser().resolve()
        if destination == self.directory:
            return
        destination.mkdir(parents=True, exist_ok=True)
        machine = destination / self.machine_path.name
        report = destination / self.report_path.name
        if machine.exists() or report.exists():
            raise FileExistsError(f"Luna run id already exists: {self.run_id}")
        shutil.move(str(self.machine_path), str(machine))
        self.directory = destination
        self.machine_path = machine
        self.report_path = report

    def record(self, step: str, *, command=None, diff_sha256: str | None = None,
               probe_elapsed_ms: float | None = None, verdict: str = "RECORDED",
               replay_handle: str | None = None, details: dict | None = None) -> dict:
        if not isinstance(step, str) or not step.strip():
            raise ValueError("step is required")
        if probe_elapsed_ms is not None and (type(probe_elapsed_ms) not in (int, float)
                                             or probe_elapsed_ms < 0):
            raise ValueError("probe_elapsed_ms must be nonnegative")
        event = {"sequence": len(self.events) + 1,
                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                 "run_id": self.run_id, "step": step,
                 "command": command, "diff_sha256": diff_sha256,
                 "probe_elapsed_ms": probe_elapsed_ms, "verdict": verdict,
                 "replay_handle": replay_handle, "details": details or {},
                 "previous_sha256": self.head_hash}
        event["event_sha256"] = _event_hash(event)
        encoded = _canonical(event) + b"\n"
        descriptor = os.open(self.machine_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            raise
        self.events.append(event)
        self.head_hash = event["event_sha256"]
        return event

    def render_report(self, *, understanding: str, plan: list[str], interceptions: list[str],
                      evidence: list[str], deliverables: list[str], final_status: str) -> Path:
        timeline = [f"- {event['timestamp']} · `{event['step']}`：{event['verdict']}"
                    + (f"（回放 `{event['replay_handle']}`）" if event.get("replay_handle") else "")
                    for event in self.events]
        body = [f"# Luna 任务战报 · {self.run_id}", "",
                f"**最终状态：{final_status}**  ",
                f"**机器日志：** `{self.machine_path.relative_to(self.root).as_posix()}`  ",
                f"**SHA-256 链头：** `{self.head_hash}`", "",
                "## 1. 需求理解", "", understanding or self.requirement or "（未提供）", "",
                "## 2. 拆解方案", "", *(f"- {item}" for item in plan), "",
                "## 3. 拦截了哪些坑 / 作弊", "", *(f"- {item}" for item in interceptions), "",
                "## 4. 真实测试证据", "", *(f"- {item}" for item in evidence), "",
                "### 完整时间线", "", *(timeline or ["- 无事件"]), "",
                "## 5. 最终交付物与合并建议", "", *(f"- {item}" for item in deliverables), "",
                f"**可复核的 JSONL SHA-256：** `{sha256_file(self.machine_path)}`", ""]
        temporary = self.report_path.with_suffix(".md.tmp")
        temporary.write_text("\n".join(body), encoding="utf-8", newline="\n")
        os.replace(temporary, self.report_path)
        return self.report_path


def verify_jsonl(path: str | Path) -> dict:
    """Verify sequence, links, event hashes and final newline of a machine log."""
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.endswith("\n"):
                    return {"valid": False, "entries": len(rows), "incomplete_line": number}
                row = json.loads(line)
                if type(row) is not dict:
                    return {"valid": False, "entries": len(rows), "invalid_entry": number}
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return {"valid": False, "entries": len(rows), "error": type(error).__name__}
    previous = ZERO_HASH
    for number, row in enumerate(rows, 1):
        if row.get("sequence") != number:
            return {"valid": False, "entries": len(rows), "invalid_sequence": number}
        if row.get("previous_sha256") != previous:
            return {"valid": False, "entries": len(rows), "broken_link": number}
        if row.get("event_sha256") != _event_hash(row):
            return {"valid": False, "entries": len(rows), "tampered_event": number}
        previous = row["event_sha256"]
    return {"valid": True, "entries": len(rows), "head_sha256": previous,
            "file_sha256": sha256_file(path)}
