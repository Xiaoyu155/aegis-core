"""Deterministic JS/TS/HTML boundary probes for the multi-language guard."""
import subprocess

import pytest

from aegis_core.invariant_guard import inspect_source, scan_changed_files


BAD = [
    ("src/net.js", "async function load(url) { return fetch(url); }", "NETWORK_FETCH_NO_ABORT_SIGNAL"),
    ("src/net.ts", "function load(url: string) { return axios.get(url); }", "NETWORK_REQUEST_NO_TIMEOUT"),
    ("src/net.jsx", "function load(url) { return axios.request({ url }); }", "NETWORK_REQUEST_NO_TIMEOUT"),
    ("src/net.tsx", "function load(url: string) { return wx.request({ url }); }", "NETWORK_REQUEST_NO_TIMEOUT"),
    ("src/net.js", "function load(url) { return tt.request({ url }); }", "NETWORK_REQUEST_NO_TIMEOUT"),
    ("src/net.js", "function load(url) { return ks.request({ url }); }", "NETWORK_REQUEST_NO_TIMEOUT"),
    ("src/net.js", "function load(url) { return my.request({ url }); }", "NETWORK_REQUEST_NO_TIMEOUT"),
    ("src/net.js", "while (true) { fetch(url); }", "NETWORK_RETRY_UNBOUNDED"),
    ("src/net.js", "for (;;) { axios.get(url); }", "NETWORK_RETRY_WITHOUT_BACKOFF"),
    ("src/money.ts", "function ratio(amount: number, quantity: number) { return amount / quantity; }", "FINANCE_UNGUARDED_DIVISION"),
    ("src/money.ts", "function quote(price: number, rate: number) { return price / rate; }", "FINANCE_UNGUARDED_DIVISION"),
    ("src/money.js", "function fee(amount, feeRate) { return amount / feeRate; }", "FINANCE_UNGUARDED_DIVISION"),
    ("src/geo.js", "function point(x, latitude) { return x / latitude; }", "FINANCE_UNGUARDED_DIVISION"),
    ("src/money.js", "function quote(amount) { return amount / 0; }", "FINANCE_UNGUARDED_DIVISION"),
    ("src/money.ts", "function pay(rawAmount: string) { const amount = Number(rawAmount); return amount * 2; }", "FINANCE_INPUT_NOT_FINITE_CHECKED"),
    ("src/money.js", "function price(value) { const price = parseFloat(value); return price + 1; }", "FINANCE_INPUT_NOT_FINITE_CHECKED"),
    ("src/money.js", "function price(value) { const price = Number.parseFloat(value); return price + 1; }", "FINANCE_INPUT_NOT_FINITE_CHECKED"),
    ("src/money.js", "function pay(inputAmount) { return parseInt(inputAmount) * 100; }", "FINANCE_INPUT_NOT_FINITE_CHECKED"),
    ("src/timer.js", "function start() { setInterval(tick, 1000); }", "RESOURCE_INTERVAL_NOT_CLEARED"),
    ("src/timer.js", "function start() { const timer = setInterval(tick, 1000); clearInterval(other); }", "RESOURCE_INTERVAL_NOT_CLEARED"),
    ("src/events.js", "function mount(node) { node.addEventListener('click', onClick); }", "RESOURCE_EVENT_LISTENER_NOT_REMOVED"),
    ("src/events.js", "function mount(node) { node.addEventListener('click', onClick); other.removeEventListener('click', onClick); }", "RESOURCE_EVENT_LISTENER_NOT_REMOVED"),
    ("src/socket.ts", "function connect(url: string) { const socket = new WebSocket(url); return socket; }", "RESOURCE_WEBSOCKET_NOT_CLOSED"),
    ("src/socket.ts", "function connect(url: string) { const socket = new WebSocket(url); other.close(); }", "RESOURCE_WEBSOCKET_NOT_CLOSED"),
    ("src/cache.js", "const events = []; function save(item) { events.push(item); }", "UNBOUNDED_GROWTH"),
    ("src/cache.ts", "const cache = new Map(); function save(key, value) { cache.set(key, value); }", "UNBOUNDED_GROWTH"),
    ("src/cache.js", "globalThis.events = []; function save(item) { globalThis.events.push(item); }", "UNBOUNDED_GROWTH"),
    ("src/parse.js", "function read(raw) { return JSON.parse(raw); }", "ASYNC_JSON_PARSE_UNGUARDED"),
    ("src/storage.js", "function read() { return localStorage.getItem('state'); }", "STORAGE_EXCEPTION_NO_FALLBACK"),
    ("src/storage.ts", "function read() { return wx.getStorageSync('state'); }", "STORAGE_EXCEPTION_NO_FALLBACK"),
    ("src/async.js", "function load() { fetch('/items').then(render); }", "ASYNC_PROMISE_UNHANDLED"),
    ("src/view.js", "function render(value) { element.innerHTML = value; }", "INJECTION_UNSAFE_HTML"),
    ("src/view.js", "function render(value) { element.insertAdjacentHTML('beforeend', value); }", "INJECTION_UNSAFE_HTML"),
    ("src/view.jsx", "const view = { dangerouslySetInnerHTML: { __html: input } };", "INJECTION_UNSAFE_HTML"),
    ("src/eval.js", "function run(code) { return eval(code); }", "INJECTION_DYNAMIC_EVAL"),
    ("src/eval.js", "function run(code) { return new Function(code)(); }", "INJECTION_DYNAMIC_FUNCTION"),
    ("src/merge.js", "function merge(target, key, value) { target.__proto__ = value; }", "INJECTION_PROTOTYPE_POLLUTION"),
    ("src/merge.ts", "function merge(key, value) { Object.prototype[key] = value; }", "INJECTION_PROTOTYPE_POLLUTION"),
]


SAFE = [
    ("src/net.js", "async function load(url) { return fetch(url, { signal: controller.signal }); }"),
    ("src/net.ts", "function load(url: string) { return axios.get(url, { timeout: 2000 }); }"),
    ("src/net.jsx", "function load(url) { return axios.request({ url, timeout: 2000 }); }"),
    ("src/net.tsx", "function load(url) { return wx.request({ url, timeout: 3000 }); }"),
    ("src/net.js", "function load(url) { return tt.request({ url, timeout: 1000 }); }"),
    ("src/net.js", "function load(url) { return ks.request({ url, timeout: 1000 }); }"),
    ("src/net.js", "function load(url) { return my.request({ url, timeout: 1000 }); }"),
    ("src/net.js", "for (let attempt = 0; attempt < 3; attempt++) { fetch(url, { signal }); await delay(100); }"),
    ("src/money.js", "function ratio(amount, quantity) { if (quantity <= 0) return 0; return amount / quantity; }"),
    ("src/money.ts", "function quote(price: number, rate: number) { if (!(rate > 0)) return 0; return price / rate; }"),
    ("src/money.js", "function pay(rawAmount) { const amount = Number(rawAmount); if (!Number.isFinite(amount)) return 0; return amount * 2; }"),
    ("src/money.ts", "function price(value: string) { const price = parseFloat(value); if (!Number.isFinite(price)) throw Error(); return price + 1; }"),
    ("src/timer.js", "function start() { const timer = setInterval(tick, 1000); clearInterval(timer); }"),
    ("src/events.js", "function mount(node) { node.addEventListener('click', onClick); node.removeEventListener('click', onClick); }"),
    ("src/socket.ts", "function connect(url: string) { const socket = new WebSocket(url); socket.close(); }"),
    ("src/cache.js", "const events = []; function save(item) { events.push(item); if (events.length > 100) events.shift(); }"),
    ("src/cache.js", "function collect(values) { const items = []; items.push(values); return items; }"),
    ("src/parse.js", "function read(raw) { try { return JSON.parse(raw); } catch (error) { return null; } }"),
    ("src/storage.js", "function read() { try { return localStorage.getItem('state'); } catch (error) { return null; } }"),
    ("src/storage.ts", "function read() { try { return wx.getStorageSync('state'); } catch (error) { return ''; } }"),
    ("src/async.js", "function load() { fetch('/items', { signal }).then(render).catch(report); }"),
    ("src/view.js", "function render(value) { element.textContent = value; }"),
    ("src/safe.js", "const description = 'eval(code); innerHTML = raw; JSON.parse(raw);'; // ignored comment"),
    ("src/merge.js", "function merge(target, key, value) { target[key] = value; }"),
    ("src/net.js", "// fetch(url); eval(source);\nfunction load(url) { return fetch(url, { signal: abort.signal }); }"),
]


@pytest.mark.parametrize("path,source,code", BAD)
def test_frontend_high_risk_probe_reports_exact_path_and_physical_line(path, source, code):
    findings = inspect_source(source, path)
    finding = next(item for item in findings if item["code"] == code)
    assert finding["path"] == path
    assert finding["line"] == 1
    assert finding["severity"] in {"high", "medium"}


@pytest.mark.parametrize("path,source", SAFE)
def test_frontend_boundary_probe_accepts_bounded_or_protected_code(path, source):
    assert inspect_source(source, path) == []


@pytest.mark.parametrize("source", [
    "<html>\n<body>\n<script>\nconst data = JSON.parse(raw);\n</script>\n</body></html>",
    "<script type='module'>\nfunction run(code) { eval(code); }\n</script>",
    "<script>\nwhile (true) { fetch(url); }\n</script>",
    "<script>\nconst items = []; function add(x) { items.push(x); }\n</script>",
    "<script>\nconst output = {}; output.innerHTML = userInput;\n</script>",
    "<script>\nfunction run(raw) { return JSON.parse(raw); }\n</script>",
    "<script>\nfunction store() { return localStorage.getItem('x'); }\n</script>",
    "<script>\nfunction start() { setInterval(tick, 5); }\n</script>",
])
def test_html_inline_scripts_are_scanned_with_physical_html_line_numbers(source):
    path = "pages/index.html"
    findings = inspect_source(source, path)
    assert findings
    assert all(item["path"] == path and item["line"] > 0 for item in findings)


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def test_changed_file_gate_scans_js_ts_and_html_and_reports_real_lines(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.ts").write_text("function safe() { return fetch(url, { signal }); }\n", encoding="utf-8")
    (root / "page.html").write_text("<script>\nconst x = 1;\n</script>\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Frontend Guard")
    _git(root, "config", "user.email", "guard@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    base = _git(root, "rev-parse", "HEAD")
    (root / "src" / "app.ts").write_text(
        "function safe() { return fetch(url, { signal }); }\nfunction bad() { return fetch(url); }\n",
        encoding="utf-8")
    (root / "page.html").write_text("<script>\nconst x = 1;\nconst data = JSON.parse(raw);\n</script>\n", encoding="utf-8")
    report = scan_changed_files(root, ["src/app.ts", "page.html"], baseline=base)
    assert report["passed"] is False
    assert set(report["scanned_files"]) == {"src/app.ts", "page.html"}
    assert {(item["path"], item["line"], item["code"]) for item in report["findings"]} == {
        ("src/app.ts", 2, "NETWORK_FETCH_NO_ABORT_SIGNAL"),
        ("page.html", 3, "ASYNC_JSON_PARSE_UNGUARDED"),
    }


@pytest.mark.parametrize("index", range(24))
def test_frontend_scanner_keeps_comment_and_string_payloads_non_executable(index):
    source = f'''// attempt {index}: eval(payload), fetch(url), JSON.parse(raw)
const text{index} = "innerHTML = payload; wx.request({{url}});";
function safe(value) {{ return value + {index}; }}
'''
    assert inspect_source(source, f"src/safe{index}.js") == []
