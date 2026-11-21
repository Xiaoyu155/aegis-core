"""High-signal regression probes for scope, lifecycle, timeout and confidence filtering."""
import subprocess

import pytest

from aegis_core.invariant_guard import inspect_source, scan_changed_files, scan_repository
from aegis_core.noise_filter import classify_asset, split_findings


NOISY_ASSETS = [
    ("vendor/lib/client.js", "vendored_dependency"),
    ("src/third_party/payments/client.ts", "third_party_dependency"),
    ("dist/assets/app.js", "generated_distribution"),
    ("public/vendor.min.js", "minified_asset"),
    ("fixtures/sample.js", "test_fixture"),
    ("test/checkout.ts", "test_source"),
    ("src/mock/api.js", "test_mock"),
    ("ui/__tests__/button.jsx", "test_fixture"),
    ("ui/__mocks__/network.js", "test_mock"),
    ("specs/payment.spec.ts", "test_source"),
    ("prototypes/checkout.html", "static_html_prototype"),
    ("原型_columns.html", "static_html_prototype"),
    ("mockups/landing.html", "static_html_prototype"),
    ("play/engine/game.js", "static_html_prototype"),
    ("eval/check-platform-config.js", "offline_evaluation_script"),
    ("backend/scripts/updateIndex.js", "offline_script"),
    ("docs/snippets/request.js", "documentation_example"),
    ("fixtures\\legacy\\page.html", "test_fixture"),
]


@pytest.mark.parametrize("path,reason", NOISY_ASSETS)
def test_nonproduction_asset_paths_are_classified_and_suppressed(path, reason):
    result = classify_asset(path, "<script>eval(input)</script>")
    assert result["suppression_reason"] == reason
    assert result["scope"] == "excluded_noise"
    source = ("<script>function run(input) { eval(input); }</script>"
              if path.lower().endswith(".html") else "function run(input) { eval(input); }")
    findings = inspect_source(source, path)
    assert findings
    assert all(item["confidence"] == "SUPPRESSED_NOISE" for item in findings)
    assert all(item["suppression_reason"] == reason for item in findings)


@pytest.mark.parametrize("path,source,expected_type", [
    ("src/main.js", "eval(input)", "js"),
    ("src/page.tsx", "eval(input)", "tsx"),
    ("src/page.jsx", "eval(input)", "jsx"),
    ("src/app.mjs", "eval(input)", "mjs"),
    ("src/app.cjs", "eval(input)", "cjs"),
    ("src/page.html", "<script>eval(input)</script>", "html"),
])
def test_first_party_asset_types_remain_actionable(path, source, expected_type):
    findings = inspect_source(source, path)
    assert findings
    assert findings[0]["scope"] == "first_party"
    assert findings[0]["asset_type"] == expected_type
    assert findings[0]["confidence"] == "CRITICAL"
    assert findings[0]["suppression_reason"] is None


@pytest.mark.parametrize("source,code,confidence", [
    ('api_key = "literal-secret"', "HARDCODED_CREDENTIAL", "CRITICAL"),
    ("def update_balance(value):\n    return value / leverage", "FINANCE_UNGUARDED_LEVERAGE", "HIGH"),
    ("def connect():\n    return requests.get('https://example.invalid')", "NETWORK_REQUEST_NO_TIMEOUT", "HIGH"),
    ("def connect():\n    return socket.connect(addr)", "RESOURCE_CONNECTION_NOT_RELEASED", "MEDIUM"),
])
def test_findings_expose_structured_confidence_and_score(source, code, confidence):
    finding = next(item for item in inspect_source(source, "src/risk.py") if item["code"] == code)
    assert finding["confidence"] == confidence
    assert finding["confidence_score"] == {"CRITICAL": 1.0, "HIGH": 0.88, "MEDIUM": 0.68}[confidence]
    assert "suppression_reason" in finding
    assert finding["suppression_reason"] is None


LIFECYCLE_CASES = [
    ("onUnload", "onLoad() { this.timer = setInterval(tick, 1000); }", "clearInterval(this.timer);"),
    ("onHide", "onLoad() { this.timer = setInterval(tick, 1000); }", "clearInterval(this.timer);"),
    ("beforeUnmount", "mounted() { this.timer = setInterval(tick, 1000); }", "clearInterval(this.timer);"),
    ("unmounted", "mounted() { this.timer = setInterval(tick, 1000); }", "clearInterval(this.timer);"),
    ("componentWillUnmount", "componentDidMount() { this.timer = setInterval(tick, 1000); }", "clearInterval(this.timer);"),
    ("onUnload", "onLoad() { this.socket = new WebSocket(url); }", "this.socket.close();"),
    ("onHide", "onLoad() { node.addEventListener('tap', handleTap); }", "node.removeEventListener('tap', handleTap);"),
    ("beforeUnmount", "mounted() { node.addEventListener('click', onClick); }", "node.removeEventListener('click', onClick);"),
]


@pytest.mark.parametrize("hook,setup,cleanup", LIFECYCLE_CASES)
def test_matching_framework_lifecycle_cleanup_suppresses_resource_finding(hook, setup, cleanup):
    source = f"class Page {{ {setup} {hook}() {{ {cleanup} }} }}"
    findings = inspect_source(source, "src/Page.js")
    resource = [item for item in findings if item["domain"] == "resource"]
    assert resource
    assert all(item["confidence"] == "SUPPRESSED_NOISE" for item in resource)
    assert all(item["suppression_reason"] == f"framework_lifecycle_cleanup:{hook}" for item in resource)


def test_react_effect_return_cleanup_is_recognized_in_its_callback_scope():
    source = "useEffect(() => { const timer = setInterval(tick, 1000); return () => clearInterval(timer); }, []);"
    assert inspect_source(source, "src/Widget.jsx") == []


@pytest.mark.parametrize("source", [
    "class Page { onLoad() { this.timer = setInterval(tick, 1000); } onUnload() { clearInterval(this.other); } }",
    "class Page { onLoad() { this.socket = new WebSocket(url); } onUnload() { this.other.close(); } }",
    "class Page { onLoad() { node.addEventListener('tap', handleTap); } onUnload() { node.removeEventListener('tap', other); } }",
    "class Page { onLoad() { this.timer = setInterval(tick, 1000); } onUnload() { this.timer = null; } }",
    "function setup() { const timer = setInterval(tick, 1000); } function onUnload() { clearInterval(other); }",
])
def test_unrelated_lifecycle_actions_do_not_suppress_real_leaks(source):
    findings = inspect_source(source, "src/Page.js")
    assert any(item["confidence"] in {"HIGH", "MEDIUM"} and item["domain"] == "resource" for item in findings)


@pytest.mark.parametrize("source", [
    "axios.defaults.timeout = 5000; function load() { return axios.get('/x'); }",
    "axios.defaults = { timeout: 5000 }; function load() { return axios.get('/x'); }",
    "const client = axios.create({ baseURL: '/api', timeout: 2500 }); function load() { return client.get('/x'); }",
    "http.defaults.timeout = 4000; function load() { return http.get('/x'); }",
    "request.defaults.timeout = 3000; function load() { return request.get('/x'); }",
])
def test_configured_http_client_timeout_suppresses_missing_local_timeout(source):
    findings = inspect_source(source, "src/api.js")
    network = [item for item in findings if item["code"] == "NETWORK_REQUEST_NO_TIMEOUT"]
    assert len(network) == 1
    assert network[0]["confidence"] == "SUPPRESSED_NOISE"
    assert network[0]["suppression_reason"].startswith("configured_timeout:")


@pytest.mark.parametrize("source", [
    "function load() { return axios.get('/x'); }",
    "function load() { return fetch('/x'); }",
    "// axios.defaults.timeout = 4000\nfunction load() { return axios.get('/x'); }",
    "const comment = 'axios.defaults.timeout = 4000'; function load() { return axios.get('/x'); }",
    "function load() { return axios.get('/x', { timeout: 0 }); }",
])
def test_timeout_without_a_valid_positive_default_remains_visible(source):
    findings = inspect_source(source, "src/api.js")
    assert any(item["code"].startswith("NETWORK_") and item["confidence"] in {"HIGH", "MEDIUM"}
               for item in findings)


@pytest.mark.parametrize("provider", ["wx", "tt", "ks", "my"])
def test_miniapp_framework_finite_request_timeout_is_not_reported_as_missing(provider):
    findings = inspect_source(f"function load() {{ return {provider}.request({{ url: '/x' }}); }}", "src/api.js")
    assert len(findings) == 1
    assert findings[0]["confidence"] == "SUPPRESSED_NOISE"
    assert findings[0]["suppression_reason"] == f"framework_default_timeout:{provider}.request"


@pytest.mark.parametrize("config,call", [
    ('{"networkTimeout":{"request":5000}}', "wx.request({url:'/x'})"),
    ('{"networkTimeout":{"request":1500}}', "tt.request({url:'/x'})"),
    ('{"networkTimeout":{"request":3500}}', "ks.request({url:'/x'})"),
    ('{"networkTimeout":{"request":2000}}', "my.request({url:'/x'})"),
])
def test_repository_miniapp_default_timeout_is_applied_to_changed_files(config, call, tmp_path):
    root = tmp_path / "miniapp"
    root.mkdir()
    (root / "app.json").write_text(config, encoding="utf-8")
    (root / "index.js").write_text("function safe() { return true; }\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Noise Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "noise@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    (root / "index.js").write_text(f"function load() {{ {call}; }}\n", encoding="utf-8")
    report = scan_changed_files(root, ["index.js"], baseline=base)
    assert report["findings"] == []
    assert len(report["suppressed_findings"]) == 1
    assert report["suppressed_findings"][0]["suppression_reason"].startswith("configured_timeout:repository_config")


def test_noise_is_separated_from_gate_blocking_findings():
    visible, suppressed = split_findings([
        {"code": "A", "confidence": "HIGH"},
        {"code": "B", "confidence": "SUPPRESSED_NOISE"},
        {"code": "C", "confidence": "CRITICAL"},
    ])
    assert [item["code"] for item in visible] == ["A", "C"]
    assert [item["code"] for item in suppressed] == ["B"]


def test_whole_repository_scan_reports_actionable_and_suppressed_sets_read_only(tmp_path):
    root = tmp_path / "scan-repo"
    root.mkdir()
    for relative, source in {
        "src/network.js": "function load(url) { return fetch(url); }\n",
        "vendor/lib.js": "function evaluate(input) { eval(input); }\n",
        "tests/test_page.js": "function evaluate(input) { eval(input); }\n",
        "原型_page.html": "<script>function evaluate(input) { eval(input); }</script>\n",
    }.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Noise Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "noise@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
    refs_before = subprocess.run(["git", "show-ref"], cwd=root, check=True, capture_output=True, text=True).stdout
    report = scan_repository(root)
    refs_after = subprocess.run(["git", "show-ref"], cwd=root, check=True, capture_output=True, text=True).stdout
    assert report["passed"] is False
    assert len(report["findings"]) == 1
    assert report["findings"][0]["path"] == "src/network.js"
    assert len(report["suppressed_findings"]) == 3
    assert report["summary"]["raw_findings"] == 4
    assert report["summary"]["noise_reduction_rate"] == 0.75
    assert refs_before == refs_after


@pytest.mark.parametrize("index", range(12))
def test_real_first_party_injection_remains_critical_across_path_depth(index):
    path = f"src/feature_{index}/nested/page.js"
    finding = inspect_source(f"function run_{index}(input) {{ eval(input); }}", path)[0]
    assert finding["confidence"] == "CRITICAL"
    assert finding["suppression_reason"] is None
    assert finding["path"] == path
