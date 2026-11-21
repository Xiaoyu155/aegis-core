"""Product-facing health, acceptance and risk summaries for Luna."""
from __future__ import annotations

import json
from pathlib import Path

from aegis_core.env_doctor import EnvDoctor
from aegis_core.live_dashboard import DashboardReadModel


def status_snapshot(repo_root: str | Path = ".") -> dict:
    root = Path(repo_root).expanduser().resolve()
    doctor = EnvDoctor(root, cleanup=False)
    try:
        environment = doctor.check(strict=False)
    finally:
        doctor.cleanup_temporary_configs()
    dashboard = DashboardReadModel(root)
    overview = dashboard.overview()
    security = dashboard.security_overview().get("enforce", {})
    quarantine = dashboard.quarantine(limit=500).get("items", [])
    logs = root / "artifacts" / "luna_logs"
    reports = sorted(logs.glob("*_report.md")) if logs.is_dir() else []
    pending = []
    for machine in sorted(logs.glob("*.jsonl")) if logs.is_dir() else []:
        try:
            rows = [json.loads(line) for line in machine.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        if rows and rows[-1].get("verdict") not in {"SUCCEEDED", "SIGNATURE_VERIFIED"}:
            pending.append({"run_id": rows[-1].get("run_id"), "verdict": rows[-1].get("verdict"),
                            "step": rows[-1].get("step")})
    signing_key_ready = bool(security.get("key_ready"))
    ready = not environment.get("blocking") and signing_key_ready
    return {"status": "HEALTHY" if ready else "ATTENTION_REQUIRED", "repo": str(root),
            "project_health": environment.get("status"),
            "tests_and_tickets": overview.get("tickets", 0),
            "running": overview.get("running", 0), "risk_interceptions": len(quarantine),
            "pending_acceptance": pending, "luna_runs": len(reports),
            "signing_key": "就绪" if security.get("key_ready") else "缺少或未验证",
            "disk": environment.get("disk", {}),
            "latest_report": str(reports[-1]) if reports else None}


def format_status(snapshot: dict) -> str:
    pending = snapshot.get("pending_acceptance", [])
    disk = snapshot.get("disk", {})
    lines = [f"Luna 项目状态：{snapshot['status']}",
             f"项目健康度：{snapshot.get('project_health', 'unknown')}",
             f"工单 / 运行中：{snapshot.get('tests_and_tickets', 0)} / {snapshot.get('running', 0)}",
             f"风险拦截：{snapshot.get('risk_interceptions', 0)}",
             f"待验收：{len(pending)} 项",
             f"Ed25519 密钥：{snapshot.get('signing_key', '未知')}",
             f"磁盘可用：{disk.get('free_bytes', '未知')} bytes"]
    if snapshot.get("latest_report"):
        lines.append(f"最近战报：{snapshot['latest_report']}")
    return "\n".join(lines)
