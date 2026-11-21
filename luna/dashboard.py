"""Loopback-only Chinese Luna product dashboard."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlsplit

from .status import status_snapshot


_PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Luna 总控台</title><style>
body{margin:0;background:#0b1220;color:#e8f0fa;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}header{padding:24px 6%;border-bottom:1px solid #26384d}main{max-width:1050px;margin:24px auto;padding:0 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}.card{background:#142238;border:1px solid #26384d;border-radius:12px;padding:18px}.label{color:#9ab0c8;font-size:13px}.value{font-size:25px;font-weight:700;margin-top:8px}.ok{color:#63d99a}.warn{color:#ffc86f}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#abc0d6}footer{color:#7890aa;font-size:12px;padding:20px 0}</style>
<header><b>◈ Luna 产品总控台</b><div class="label">Aegis 确定性门禁 · 本机只读视图</div></header><main>
<section class="grid" id="cards"></section><section class="card"><h2>待验收清单</h2><pre id="pending">读取中…</pre></section><footer>仅绑定回环地址；每 5 秒刷新。</footer></main>
<script>async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});const s=await r.json();const fields=[['项目健康度',s.project_health],['运行中工单',s.running],['风险拦截',s.risk_interceptions],['Luna 战报',s.luna_runs],['签发密钥',s.signing_key],['可用磁盘',s.disk.free_bytes+' bytes']];const root=document.getElementById('cards');root.replaceChildren();for(const [k,v] of fields){const d=document.createElement('div');d.className='card';const a=document.createElement('div');a.className='label';a.textContent=k;const b=document.createElement('div');b.className='value '+(s.status==='HEALTHY'?'ok':'warn');b.textContent=String(v);d.append(a,b);root.append(d)}document.getElementById('pending').textContent=JSON.stringify(s.pending_acceptance,null,2)||'[]'}catch(e){document.getElementById('pending').textContent='读取失败：'+e.message}}refresh();setInterval(refresh,5000)</script></html>'''


def create_dashboard_server(repo_root=".", *, host="127.0.0.1", port=8766):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Luna dashboard binds to loopback only")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")

    class Handler(BaseHTTPRequestHandler):
        server_version = "LunaDashboard/1.0"
        sys_version = ""

        def log_message(self, _format, *_args):
            return

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/status":
                try:
                    body = json.dumps(status_snapshot(repo_root), ensure_ascii=False,
                                       allow_nan=False).encode("utf-8")
                    self._send(200, body, "application/json; charset=utf-8")
                except (OSError, ValueError) as error:
                    self._send(503, json.dumps({"error": str(error)}).encode(),
                               "application/json; charset=utf-8")
            else:
                self._send(404, b'{"error":"not_found"}', "application/json; charset=utf-8")

        def do_POST(self):
            self._send(405, b'{"error":"read_only"}', "application/json; charset=utf-8")

        do_PUT = do_POST
        do_DELETE = do_POST

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server((host, port), Handler)
    return server
