"""Loopback-only Chinese Luna product dashboard."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlsplit

from .status import status_snapshot


_PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Luna 总控台</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e8f0fa;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{padding:24px 6%;border-bottom:1px solid #26384d}h1,h2,p{margin:0}h1{font-size:24px}
h2{font-size:18px;margin-bottom:12px}main{max-width:1120px;margin:24px auto;padding:0 18px}
.grid,.panels{display:grid;gap:14px;margin-bottom:14px;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr))}
.panels{grid-template-columns:repeat(2,minmax(0,1fr))}.card{background:#142238;border:1px solid #26384d;border-radius:12px;padding:18px;min-width:0;overflow-wrap:anywhere}
.label,small{color:#abc0d6}.value{font-size:21px;font-weight:700}.warn{color:#ffc86f}
ul{margin:0;padding:0;list-style:none}li{padding:10px 0;border-top:1px solid #30455e}small{display:block}
#quota{margin-bottom:14px}@media(max-width:640px){.panels{grid-template-columns:1fr}main{margin:14px auto;padding:0 12px}.card{padding:14px}}
</style><header><h1>Luna 产品总控台</h1><p class="label">Aegis 确定性门禁 · 本机只读视图</p></header><main>
<p id="error" class="warn" role="status"></p>
<section class="grid" id="cards" aria-label="运行概览"></section>
<section class="card" id="quota"><h2>模型配额</h2><p id="reset" class="label"></p><ul id="balances"></ul></section>
<div class="panels"><section class="card"><h2>工单时间线</h2><ul id="timeline"></ul></section>
<section class="card"><h2>风险拦截记录</h2><ul id="interceptions"></ul></section></div>
<section class="card"><h2>待验收清单</h2><ul id="pending"></ul></section></main>
<script>
const byId = id => document.getElementById(id);
function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = String(text ?? '—');
  if (className) node.className = className;
  return node;
}
function list(id, rows, describe) {
  const nodes = rows.map(row => {
    const [title, detail] = describe(row);
    const item = element('li', title);
    item.append(element('small', detail));
    return item;
  });
  byId(id).replaceChildren(...(nodes.length ? nodes : [element('li', '暂无记录')]));
}
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const s = await response.json();
    const fields = [['项目健康度',s.project_health],['运行中工单',s.running],['风险拦截',s.risk_interceptions],
      ['Luna 战报',s.luna_runs],['签发密钥',s.signing_key],['可用磁盘',s.disk.free_bytes + ' bytes']];
    byId('cards').replaceChildren(...fields.map(([key, value]) => {
      const card = element('div', '', 'card');
      card.append(element('div', key, 'label'), element('div', value, 'value'));
      return card;
    }));
    const quota = s.model_quota || {};
    byId('reset').textContent = quota.error || (quota.reset_at
      ? '重置时间 ' + new Date(quota.reset_at * 1000).toLocaleString() : '');
    list('balances', quota.models || [], row => [row.id + (row.configured ? '' : ' · 未配置'),
      `已用 ${row.tokens_used} tokens / ${row.requests_used} 次 · 剩余 ${row.tokens_remaining} tokens / ${row.requests_remaining} 次`]);
    list('timeline', s.ticket_timeline || [], row => [row.ticket_id + ' · ' + row.status,
      `创建 ${row.created_at ?? '—'} · 更新 ${row.updated_at ?? '—'} · ${row.assigned_tier ?? '—'}`]);
    list('interceptions', s.interceptions || [], row => [row.ticket_id ?? row.file,
      [row.timestamp ?? row.created_at, row.reason ?? row.verdict ?? row.file].filter(Boolean).join(' · ')]);
    list('pending', s.pending_acceptance || [], row => [row.run_id, row.verdict + ' · ' + row.step]);
    byId('error').textContent = '';
  } catch (error) {
    byId('error').textContent = '读取失败：' + error.message;
  } finally {
    setTimeout(refresh, 5000);
  }
}
refresh();
</script></html>'''


def create_dashboard_server(repo_root=".", *, host="127.0.0.1", port=8766, models=None):
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
                    body = json.dumps(status_snapshot(repo_root, models=models) if models else status_snapshot(repo_root), ensure_ascii=False,
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
