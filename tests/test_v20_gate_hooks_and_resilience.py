import json
import os
from pathlib import Path
import subprocess

import pytest

from aegis_core.git_hooks import install_hooks, remove_hooks
from aegis_core.security_gate import _safe_path, _fallback_scan, run_hook, scan_files, scan_text


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/absolute.py", "", "src/../../outside.js", "C:\\x\\bad.py"])
def test_gate_rejects_unsafe_repository_paths(path):
    with pytest.raises(ValueError):
        _safe_path(path)


@pytest.mark.parametrize("suffix", ["py", "pyw", "js", "jsx", "ts", "tsx", "mjs", "cjs"])
def test_gate_scans_supported_source_suffixes(suffix):
    source = "try:\n    pass\nexcept:\n    pass\n" if suffix in {"py", "pyw"} else "try { run(); } catch (error) {}\n"
    findings = scan_text(source, "src/module." + suffix)
    assert findings


@pytest.mark.parametrize("fallback_diff", [
    b"+except:\n+    pass\n", b"+.catch((error) => {})\n", b"+db.execute(f'SELECT {value}')\n",
    b"+if True:\n+    return True\n", b"+if (true) { return 200; }\n",
])
def test_unified_diff_fallback_detects_high_risk_added_tokens(fallback_diff):
    assert _fallback_scan(fallback_diff, "src/broken.py")


@pytest.mark.parametrize("degraded_kind", ["syntax", "binary", "oversize", "minified_js", "deep_ast"])
def test_scan_files_degrades_damaged_or_oversize_sources_without_crashing(tmp_path, degraded_kind):
    suffix = ".js" if degraded_kind == "minified_js" else ".py"
    path = tmp_path / ("broken" + suffix)
    if degraded_kind == "syntax":
        path.write_text("def unfinished(:\n    pass\n", encoding="utf-8")
    elif degraded_kind == "binary":
        path.write_bytes(b"\x00\xff\x00")
    elif degraded_kind == "oversize":
        path.write_bytes(b"#" * 4096)
    elif degraded_kind == "minified_js":
        path.write_bytes(b"const packed='" + b"A" * 4096 + b"';")
    else:
        path.write_text("x = " + "[" * 1600 + "0" + "]" * 1600, encoding="utf-8")
    diff = b"diff --git a/broken.py b/broken.py\n+++ b/broken.py\n@@ -1,0 +1,2 @@\n+except:\n+    pass\n"
    result = scan_files(tmp_path, ["broken" + suffix], diff=diff, max_file_bytes=1024)
    assert result["degraded_files"]
    assert result["findings"]


def test_clean_scan_reports_timings_and_no_findings(tmp_path):
    (tmp_path / "ok.py").write_text("def safe(value):\n    return value + 1\n", encoding="utf-8")
    report = scan_files(tmp_path, ["ok.py"])
    assert report["status"] == "clear" and report["files_scanned"] == 1
    assert report["p95_ms"] >= 0 and report["elapsed_ms"] >= report["total_ms"]


def test_precommit_gate_blocks_and_quarantines_a_staged_patch(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "app.py").write_text("try:\n    run()\nexcept:\n    pass\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    code, report = run_hook(tmp_path, mode="staged")
    assert code == 1 and report["status"] == "blocked"
    assert report["quarantine"]["stored"]
    assert (tmp_path / report["quarantine"]["diff"]).is_file()
    audit = [json.loads(line) for line in (tmp_path / ".aegis/audit.jsonl").read_text().splitlines()]
    assert audit[-1]["event"] == "patch_blocked"


def test_precommit_gate_allows_safe_staged_patch_and_audits_it(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "app.py").write_text("def safe(value):\n    return value * 2\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    code, report = run_hook(tmp_path, mode="staged")
    assert code == 0 and report["status"] == "clear"
    assert not report["quarantine"]["stored"]
    assert json.loads((tmp_path / ".aegis/audit.jsonl").read_text())["event"] == "patch_allowed"


def test_invalid_enforce_permit_fails_closed_even_for_clean_diff(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "app.py").write_text("value = 42\n", encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    (tmp_path / ".aegis").mkdir()
    (tmp_path / ".aegis/enforce.enabled").write_text("policy_version=2.0\n")
    code, report = run_hook(tmp_path, mode="staged")
    assert code == 1
    assert "ENFORCE_PERMIT_INVALID" in {item["code"] for item in report["findings"]}


def test_hook_installation_is_idempotent_and_removable(tmp_path):
    _git(tmp_path, "init", "-q")
    first = install_hooks(tmp_path)
    second = install_hooks(tmp_path)
    assert first["status"] == second["status"] == "installed"
    assert all(item["status"] == "already_installed" for item in second["hooks"])
    hooks_dir = Path(subprocess.check_output(["git", "rev-parse", "--git-path", "hooks"], cwd=tmp_path,
                                             text=True).strip())
    if not hooks_dir.is_absolute():
        hooks_dir = tmp_path / hooks_dir
    assert "aegis-autonomous-gate-v2" in (hooks_dir / "pre-commit").read_text()
    removed = remove_hooks(tmp_path)
    assert set(removed["removed"]) == {"pre-commit", "pre-push"}
    assert not (tmp_path / ".aegis/enforce.enabled").exists()


@pytest.mark.parametrize("mode", ["staged", "pre-push"])
@pytest.mark.parametrize("index", range(10))
def test_hook_report_has_stable_policy_and_file_fields(mode, index, tmp_path):
    _git(tmp_path, "init", "-q")
    source = tmp_path / f"safe_{index}.py"
    source.write_text(f"def value_{index}():\n    return {index}\n", encoding="utf-8")
    _git(tmp_path, "add", source.name)
    code, report = run_hook(tmp_path, mode=mode, stdin=b"")
    assert code == 0
    assert report["mode"] == mode and isinstance(report["files"], list)
    if mode == "staged":
        assert report["files"]

