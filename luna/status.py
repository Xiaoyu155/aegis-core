"""Product-facing health, acceptance and risk summaries for Luna."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from aegis_core.env_doctor import EnvDoctor
from aegis_core.live_dashboard import DashboardReadModel


from .model_router import QuotaLedger, config_path, load_config


def _quota(root, models):
    try:
        path = config_path(root, models)
        config = load_config(path, root) if path.is_file() else None
        return QuotaLedger(root, config).snapshot() if config else {"models": []}
    except (OSError, ValueError, sqlite3.Error) as error:
        return {"models": [], "error": str(error)}


def status_snapshot(repo_root: str | Path = ".", *, models=None) -> dict:
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
    ready = not environment.get("blocking")
    return {"status": "HEALTHY" if ready else "ATTENTION_REQUIRED", "repo": str(root),
            "project_health": environment.get("status"),
            "tests_and_tickets": overview.get("tickets", 0),
            "running": overview.get("running", 0), "risk_interceptions": len(quarantine),
            "ticket_timeline": dashboard.tickets(limit=100)["items"],
            "interceptions": [{key: value for key, value in item.items() if key != "preview"}
                              for item in quarantine],
            "model_quota": _quota(root, models),
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
    quota = snapshot.get("model_quota", {})
    if quota.get("error"):
        lines.append("配额读取失败：" + quota["error"])
    for item in quota.get("models", []):
        lines.append(f"{item['id']} 剩余配额：{item['tokens_remaining']} tokens / "
                     f"{item['requests_remaining']} 次")
    return "\n".join(lines)
