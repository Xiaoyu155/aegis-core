"""Read-only local HTTP dashboard for Aegis tickets, traces, and fleet state."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, urlsplit


_HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aegis 控制台</title>
<style>
:root{color-scheme:dark;--bg:#090f1a;--card:#121d2d;--line:#26384d;--text:#e8f0fa;--muted:#98acc3;--cyan:#54d4e8;--green:#60d394;--red:#ff707a;--amber:#ffc66d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 20% -10%,#152943,var(--bg) 55%);color:var(--text);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}
header{padding:24px max(22px,calc((100vw - 1240px)/2));border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}h1{margin:0;font-size:22px}small,.muted{color:var(--muted)}main{max-width:1240px;margin:22px auto;padding:0 20px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{background:linear-gradient(145deg,#152236,var(--card));border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 12px 30px #0003}.label{color:var(--muted);font-size:12px}.value{font-size:28px;font-weight:700;margin-top:4px}.section{margin-top:16px}.section h2{font-size:15px;margin:0 0 10px}.panels{display:grid;grid-template-columns:1fr 1fr;gap:12px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:11px;text-transform:uppercase}td{font-size:12px;max-width:380px;overflow-wrap:anywhere}.pill{padding:2px 8px;border-radius:20px;background:#26384d;color:var(--cyan)}.ok{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--amber)}.scroll{max-height:320px;overflow:auto}.bar{height:6px;background:#24364a;border-radius:8px;margin-top:10px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0}.footer{padding:18px;color:var(--muted);font-size:11px}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.panels{grid-template-columns:1fr}}@media(max-width:480px){.grid{grid-template-columns:1fr 1fr}.value{font-size:22px}}
</style><body>
<header><div><h1>◈ Aegis 确定性软件工厂</h1><small>只读实时态势 · WAL 账本 · 本机服务</small></div><div id="clock" class="muted">连接中…</div></header>
<main><section class="grid" id="overview"></section>
<section class="section panels"><div class="card"><h2>工单进度</h2><div class="scroll"><table><thead><tr><th>工单</th><th>状态</th><th>更新时间</th></tr></thead><tbody id="tickets"></tbody></table></div></div>
<div class="card"><h2>探针与拦截轨迹</h2><div class="scroll"><table><thead><tr><th>工单 / 步骤</th><th>结果</th><th>操作</th></tr></thead><tbody id="traces"></tbody></table></div></div></section>
<section class="section panels"><div class="card"><h2>签名门禁与审计</h2><div id="security" class="muted">读取密钥与账本状态…</div><div id="world" class="muted">读取世界模型校准…</div></div><div class="card"><h2>全舰队交付健康度</h2><div id="fleet" class="muted">读取舰队状态…</div><div class="bar"><i id="fleetbar"></i></div></div></section>
<section class="section card"><h2>10 × 10 红蓝对抗矩阵</h2><div class="scroll"><table><thead><tr id="matrix-head"></tr></thead><tbody id="matrix"></tbody></table></div><div id="convergence" class="muted"></div></section>
<section class="section card"><h2>Quarantine 审阅台</h2><div class="scroll"><table><thead><tr><th>时间</th><th>模式</th><th>规则</th><th>Diff 预览</th></tr></thead><tbody id="quarantine"></tbody></table></div></section>
<div class="footer">数据来自本地 SQLite 只读连接与仓库内已生成的 JSON 报告。页面每 3 秒刷新。</div></main>
<script>
const $=id=>document.getElementById(id);const esc=x=>String(x??"");
function cell(tr,value,cls){const td=document.createElement("td");td.textContent=esc(value);if(cls)td.className=cls;tr.appendChild(td)}
function row(table,values){const tr=document.createElement("tr");values.forEach(v=>cell(tr,v));table.appendChild(tr)}
async function get(path){const r=await fetch(path,{cache:"no-store"});if(!r.ok)throw Error(r.status);return(await r.json()).data}
async function refresh(){try{const o=await get("/api/overview"),t=await get("/api/tickets"),r=await get("/api/traces"),f=await get("/api/fleet"),q=await get("/api/quarantine");
$("clock").textContent="最后更新 "+new Date().toLocaleTimeString();$("overview").replaceChildren();
[["工单总数",o.tickets],["运行中",o.running],["探针轨迹",o.traces],["拦截记录",o.interceptions],["enforce 签名",o.enforce?.signature_status||"未启用"],["扫描 P95",(o.p95_ms??"—")+" ms"]].forEach(([label,value])=>{const d=document.createElement("div");d.className="card";const a=document.createElement("div");a.className="label";a.textContent=label;const b=document.createElement("div");b.className="value";b.textContent=esc(value);d.append(a,b);$("overview").appendChild(d)});
const tt=$("tickets");tt.replaceChildren();t.items.forEach(x=>row(tt,[x.ticket_id,x.status,x.updated_at]));
const tr=$("traces");tr.replaceChildren();r.items.forEach(x=>row(tr,[x.ticket_id+" / "+x.step_idx,x.actual_state_s1?.status||x.actual_state_s1?.failure_code||"recorded",x.action_a0?.op||"—"]));
const e=o.enforce||{};$("security").textContent="签名："+(e.signature_status||"缺失")+" · 指纹："+(e.token_fingerprint||"—").slice(0,16)+" · 账本："+(e.ledger_integrity||"缺失");
const m=o.world_model||{};$("world").textContent="校准："+(m.status||"无报告")+" · 样本："+(m.samples??"—")+" · enforce 转换："+(m.allowed_mode_transition?"已授权":"保持阻断");
const transforms=o.matrix_transforms||[];const mh=$("matrix-head");mh.replaceChildren();["缺陷类",...transforms].forEach(x=>{const th=document.createElement("th");th.textContent=x;mh.appendChild(th)});const mt=$("matrix");mt.replaceChildren();(o.defect_matrix||[]).forEach(x=>row(mt,[x.defect_type,...x.recall]));const rounds=o.adversarial?.rounds||[];$("convergence").textContent="逃逸率收敛："+rounds.map(x=>"R"+x.round+" "+(100*x.escape_rate).toFixed(2)+"%").join(" → ");
const qt=$("quarantine");qt.replaceChildren();q.items.forEach(x=>{const tr=document.createElement("tr");[x.created_at||x.file,x.mode,(x.findings||[]).map(f=>f.code+" → "+f.path+":"+f.line).join(", ")].forEach(v=>cell(tr,v));const td=document.createElement("td"),pre=document.createElement("pre");pre.style.margin="0";pre.style.whiteSpace="pre-wrap";(x.preview||"(无 diff)").split("\n").slice(0,50).forEach(line=>{const span=document.createElement("span");span.textContent=line+"\n";if(line.startsWith("+"))span.style.color="var(--green)";else if(line.startsWith("-"))span.style.color="var(--red)";pre.appendChild(span)});td.appendChild(pre);tr.appendChild(td);qt.appendChild(tr)});
const summary=f.summary||{};const repos=f.repositories||[];$("fleet").textContent="仓库 "+(summary.repositories_scanned??repos.length)+" · 交付分支 "+(summary.delivery_branches??"—")+" · 冲突 "+(summary.conflicts??"—");const total=summary.delivery_branches||0;const healthy=Math.max(0,total-(summary.conflicts||0));$("fleetbar").style.width=(total?Math.min(100,healthy/total*100):0)+"%";
}catch(e){$("clock").textContent="读取失败："+e.message}}refresh();setInterval(refresh,3000);
</script></body></html>'''


def _decode_json(value):
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


class DashboardReadModel:
    def __init__(self, repo_root: str | Path, db_path: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        if db_path is None:
            choices = (self.repo_root / ".aegis" / "ledger.sqlite3",
                       self.repo_root / ".aegis" / "ledger.db",
                       self.repo_root / "aegis.sqlite3")
            self.db_path = next((item for item in choices if item.exists()), choices[0])
        else:
            candidate = Path(db_path)
            self.db_path = candidate if candidate.is_absolute() else self.repo_root / candidate
            self.db_path = self.db_path.resolve()

    def _connect(self):
        if not self.db_path.is_file():
            return None
        uri = self.db_path.as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=0.2, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=200")
        return connection

    def _query(self, sql: str, params=()):
        connection = self._connect()
        if connection is None:
            return []
        try:
            return [dict(row) for row in connection.execute(sql, params)]
        except sqlite3.OperationalError as error:
            if "no such table" in str(error).lower():
                return []
            raise
        finally:
            connection.close()

    def _world_model(self) -> dict | None:
        candidates = sorted([*self.repo_root.glob("artifacts/world_model_bench*.json"),
                             *self.repo_root.glob("datasets/world_model_bench*.json")],
                            key=lambda item: item.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            gate = payload.get("gate", {}) if isinstance(payload, dict) else {}
            return {"status": gate.get("status", payload.get("status", "reported")),
                    "samples": payload.get("dataset", {}).get("total_samples"),
                    "allowed_mode_transition": bool(gate.get("allowed_mode_transition")),
                    "source": path.relative_to(self.repo_root).as_posix()}
        return None

    def _fleet(self) -> dict:
        candidates = sorted(self.repo_root.glob("artifacts/fleet_dashboard*.json"),
                            key=lambda item: item.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payload.setdefault("source", path.relative_to(self.repo_root).as_posix())
                return payload
        status = self.repo_root / "STATUS.json"
        if status.is_file():
            try:
                payload = json.loads(status.read_text(encoding="utf-8"))
                return {"repositories": [], "summary": {"repositories_scanned": 0,
                        "delivery_branches": 0, "conflicts": 0},
                        "status": payload.get("status"), "source": "STATUS.json"}
            except (OSError, json.JSONDecodeError):
                pass
        return {"repositories": [], "summary": {"repositories_scanned": 0,
                "delivery_branches": 0, "conflicts": 0}, "status": "no_report"}

    def overview(self) -> dict:
        status_rows = self._query("SELECT status, COUNT(*) AS count FROM tickets GROUP BY status")
        statuses = {item["status"]: int(item["count"]) for item in status_rows}
        trace = self._query("SELECT COUNT(*) AS count, AVG(reward) AS mean_reward FROM causal_traces")
        trace_summary = trace[0] if trace else {"count": 0, "mean_reward": None}
        interception_rows = self._query(
            "SELECT actual_state_s1 FROM causal_traces WHERE json_extract(action_a0, '$.op') "
            "IN ('repair', 'repair_attempt') ORDER BY trace_id DESC LIMIT 500")
        interceptions = 0
        for row in interception_rows:
            state = _decode_json(row.get("actual_state_s1")) or {}
            interceptions += int(bool(state.get("cheat_detected") or state.get("repeated_patch")
                                      or state.get("pre_blocked")))
        total = sum(statuses.values())
        fleet = self._fleet()
        return {"tickets": total, "statuses": statuses,
                "running": sum(count for key, count in statuses.items() if key in {"running", "queued"}),
                "traces": int(trace_summary.get("count") or 0), "mean_reward": trace_summary.get("mean_reward"),
                "interceptions": interceptions, "world_model": self._world_model(),
                "fleet": fleet.get("summary", {})}

    def tickets(self, limit: int = 100) -> dict:
        rows = self._query("SELECT ticket_id, status, created_at, updated_at, assigned_tier "
                           "FROM tickets ORDER BY updated_at DESC LIMIT ?", (limit,))
        return {"items": rows, "count": len(rows)}

    def traces(self, limit: int = 100) -> dict:
        rows = self._query("SELECT trace_id, ticket_id, step_idx, state_s0, action_a0, "
                           "predicted_next_state, actual_state_s1, reward, token_cost, timestamp "
                           "FROM causal_traces ORDER BY trace_id DESC LIMIT ?", (limit,))
        for row in rows:
            for key in ("state_s0", "action_a0", "predicted_next_state", "actual_state_s1"):
                row[key] = _decode_json(row[key])
        return {"items": rows, "count": len(rows)}

    def fleet(self) -> dict:
        return self._fleet()

    def security_state(self) -> dict:
        """Read key/token metadata with the standard library; never load private keys."""
        local = self.repo_root / ".aegis"
        keys = Path(os.environ.get("AEGIS_KEYS_DIR", Path.home() / ".aegis" / "keys")).expanduser()
        metadata = keys / "key-metadata.json"
        token_path = local / "enforce.token"
        fingerprint = None
        key_expiry = None
        key_ready = False
        try:
            item = json.loads(metadata.read_text(encoding="utf-8"))
            key_expiry = item.get("expires_at")
            fingerprint = item.get("public_key_sha256")
            key_ready = ((keys / "ed25519-public.pem").is_file()
                         and (keys / "ed25519-private.pem").is_file() and bool(fingerprint))
        except (OSError, json.JSONDecodeError):
            pass
        token_fingerprint = fingerprint
        signature_status = "missing"
        token_expires = None
        try:
            token = token_path.read_text(encoding="ascii").strip()
            body = token.split(".", 1)[0]
            import base64
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
            token_expires = payload.get("expires_at")
            signature_status = "present_unverified" if len(token.split(".")) == 2 and key_ready else "unverified"
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
            if token_path.exists():
                signature_status = "malformed"
        ledger_path = local / "audit.jsonl"
        if ledger_path.is_file():
            from .audit_ledger import AuditLedger
            ledger = AuditLedger(ledger_path).verify()
            ledger_status = "valid" if ledger["valid"] else "tampered"
            if ledger["valid"]:
                try:
                    last = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[-1])
                    if last.get("event") == "enforce_token_verified":
                        signature_status = "verified"
                    elif last.get("event") == "enforce_token_rejected":
                        signature_status = "rejected"
                    elif last.get("details", {}).get("enforce_authorized") is True:
                        signature_status = "verified_at_last_gate"
                    elif last.get("details", {}).get("enforce_reason"):
                        signature_status = "rejected"
                except (OSError, IndexError, json.JSONDecodeError):
                    pass
        else:
            ledger_status = "missing"
        return {"signature_status": signature_status, "token_fingerprint": token_fingerprint,
                "key_ready": key_ready, "key_expires_at": key_expiry,
                "token_expires_at": token_expires, "ledger_integrity": ledger_status}

    def quarantine(self, limit: int = 50) -> dict:
        directory = self.repo_root / ".aegis" / "quarantine"
        items = []
        if not directory.is_dir():
            return {"items": [], "count": 0}
        paths = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]
        for meta_path in paths:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                diff_path = meta_path.with_suffix(".diff")
                preview = diff_path.read_text(encoding="utf-8", errors="replace")[:1200] if diff_path.is_file() else ""
                items.append({**metadata, "file": meta_path.name, "preview": preview})
            except (OSError, json.JSONDecodeError):
                continue
        return {"items": items, "count": len(items)}

    def security_overview(self) -> dict:
        state = self.security_state()
        reports = {}
        for key, relative in (("adversarial", "artifacts/v20_adversarial_report.json"),
                              ("performance", "artifacts/v20_performance.json")):
            try:
                reports[key] = json.loads((self.repo_root / relative).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                reports[key] = {}
        adversarial = reports["adversarial"]
        rounds = adversarial.get("rounds", [])
        final = rounds[-1] if rounds else {}
        transform_order = final.get("transforms", [])
        matrix = []
        for name, details in final.get("per_class", {}).items():
            recalls = details.get("transform_recall", {})
            matrix.append({"defect_type": name,
                           "recall": [f"{100 * recalls.get(transform, 0.0):.1f}%" for transform in transform_order]})
        return {"enforce": state, "adversarial": adversarial,
                "p95_ms": reports["performance"].get("p95_ms"),
                "matrix_transforms": transform_order, "defect_matrix": matrix}


def create_server(repo_root: str | Path = ".", *, port: int = 8765,
                  db_path: str | Path | None = None, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    if not (0 <= port <= 65535):
        raise ValueError("port must be between 0 and 65535")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("dashboard binds to loopback only")
    model = DashboardReadModel(repo_root, db_path)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AegisDashboard/1.0"
        sys_version = ""

        def log_message(self, _format, *_args):
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self._send(200, _HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            endpoints = {"/api/overview": model.overview, "/api/tickets": model.tickets,
                         "/api/traces": model.traces, "/api/fleet": model.fleet,
                         "/api/quarantine": model.quarantine}
            handler = endpoints.get(parsed.path)
            if handler is None:
                self._json(404, {"ok": False, "error": "not_found"})
                return
            try:
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", [100])[0])
                if not 1 <= limit <= 500:
                    raise ValueError("limit must be 1..500")
                payload = handler() if parsed.path in {"/api/overview", "/api/fleet"} else handler(limit)
                if parsed.path == "/api/overview":
                    payload.update(model.security_overview())
                self._json(200, {"ok": True, "data": payload})
            except (ValueError, TypeError) as error:
                self._json(400, {"ok": False, "error": str(error)})
            except (sqlite3.Error, OSError) as error:
                self._json(503, {"ok": False, "error": "ledger_unavailable", "detail": str(error)})

        def do_POST(self):
            self._json(405, {"ok": False, "error": "read_only"})

        def do_PUT(self):
            self._json(405, {"ok": False, "error": "read_only"})

        def do_DELETE(self):
            self._json(405, {"ok": False, "error": "read_only"})

    class DashboardServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = DashboardServer((host, port), Handler)
    server.model = model
    return server


def serve(repo_root: str | Path = ".", *, port: int = 8765,
          db_path: str | Path | None = None, host: str = "127.0.0.1") -> None:
    server = create_server(repo_root, port=port, db_path=db_path, host=host)
    try:
        print(f"Aegis dashboard listening at http://{server.server_address[0]}:{server.server_address[1]}/", flush=True)
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
