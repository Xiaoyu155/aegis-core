"""Regression tests for the P0 outsider red-team findings."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aegis_core.audit_ledger import AuditLedger
from aegis_core.enforce import (_windows_private_key_acl_safe, create_keypair,
                                issue_token, verify_token)
from aegis_core.invariant_guard import inspect_source
from aegis_core.probe_synth import is_true_red, render_probe
from aegis_core.reviewer import audit_candidate
from aegis_core.sandbox import run_exam_gate
from luna.flight_recorder import sha256_file
from luna.runner import run_requirement
from luna.status import status_snapshot


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _clean_git(root):
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Red Team")
    _git(root, "config", "user.email", "redteam@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "outsider repository")


def test_run_in_clean_unignored_repository_does_not_lock_on_its_own_jsonl(tmp_path):
    root = tmp_path / "unfamiliar"
    root.mkdir()
    (root / "app.py").write_text("def calculate(value):\n    return value\n", encoding="utf-8")
    _clean_git(root)
    assert not (root / ".gitignore").exists()

    result = run_requirement("add explicit validation to app.calculate", root, run_id="outsider-first-run")

    assert result["status"] == "BLOCKED"
    assert "model gateway config is missing" in result["reason"]
    assert "repository baseline must be clean" not in result["reason"]
    assert Path(result["machine_log"]).is_file()
    assert (root / "artifacts/luna_logs/outsider-first-run.jsonl").is_file()


@pytest.mark.parametrize("exception", [
    "ZeroDivisionError", "ValueError", "TypeError", "KeyError", "IndexError",
])
def test_real_pytest_runtime_crash_is_a_true_red_probe(tmp_path, exception):
    root = tmp_path / exception
    root.mkdir()
    (root / "test_acceptance.py").write_text(
        "def test_acceptance():\n"
        f"    raise {exception}('red-team crash')\n", encoding="utf-8")
    receipt = run_exam_gate(root, [sys.executable, "-m", "pytest", "-q", "test_acceptance.py"])
    assert receipt["failure_code"] == "EXECUTION_ERROR"
    assert is_true_red(receipt)


@pytest.mark.parametrize("receipt", [
    {"exit_code": 1, "failure_code": "TIMEOUT_KILLED", "changed": False,
     "stdout": "Traceback (most recent call last):\nZeroDivisionError: timeout"},
    {"exit_code": 1, "failure_code": "PROBE_INTEGRITY_VIOLATION", "changed": False,
     "stdout": "Traceback (most recent call last):\nValueError: tampered"},
    {"exit_code": 1, "failure_code": "EXECUTION_ERROR", "changed": True,
     "stdout": "Traceback (most recent call last):\nZeroDivisionError: changed tree"},
    {"exit_code": 0, "failure_code": None, "changed": False,
     "stdout": "Traceback (most recent call last):\nZeroDivisionError: forged"},
])
def test_non_execution_failures_cannot_be_promoted_to_true_red(receipt):
    assert is_true_red(receipt) is False


def test_generated_node_probe_runs_without_node_24_only_flag(tmp_path):
    node = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False)
    if node.returncode:
        pytest.skip("Node.js is not installed")
    _, source, command = render_probe(
        {"kind": "numeric_bounds", "path": "app.js", "symbol": "clamp",
         "low": 0, "high": 1, "fallback": None}, [[0, 0], [2, 1]])
    assert "--test-isolation=none" not in command
    (tmp_path / "app.js").write_text("export function clamp(x) { return Math.min(1, Math.max(0, x)); }",
                                     encoding="utf-8")
    (tmp_path / "probe.test.mjs").write_text(source, encoding="utf-8")
    result = subprocess.run(["node", *command[1:-1], str(tmp_path / "probe.test.mjs")],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", [
    """def place(quantity, capital, broker):
    if capital != "usd":
        note = True
    return broker.submit_order(quantity=quantity)
""",
    """def place(quantity, broker):
    if quantity <= 0:
        print("invalid")
    return broker.submit_order(quantity=quantity)
""",
    """def place(amount, exchange):
    if amount < 0:
        raise ValueError()
    return exchange.create_order(amount=amount)
""",
    """def place(position_size, exchange):
    position_size = abs(position_size)
    return exchange.create_order(size=position_size)
""",
])
def test_unrelated_checks_print_and_negative_only_checks_do_not_guard_orders(source):
    assert any(item["code"] == "FINANCE_UNGUARDED_SIZE"
               for item in inspect_source(source, "src/orders.py"))


@pytest.mark.parametrize("denominator", [
    "cash_balance", "free_collateral", "stake_amount", "position_value",
    "reserve_funds", "account_equity", "risk_capital",
])
def test_financial_denominator_variable_families_are_checked(denominator):
    source = f"def compute(capital, {denominator}):\n    return capital / {denominator}\n"
    assert any(item["code"] == "FINANCE_UNGUARDED_LEVERAGE"
               for item in inspect_source(source, "src/risk.py"))


@pytest.mark.parametrize("source", [
    """def place(quantity, broker):
    if quantity <= 0:
        raise ValueError()
    return broker.submit_order(quantity=quantity)
""",
    """def place(size, broker):
    size = max(0, size)
    return broker.place_order(size=size)
""",
])
def test_explicit_control_flow_or_clamp_guards_order_sizes(source):
    assert not any(item["code"] == "FINANCE_UNGUARDED_SIZE"
                   for item in inspect_source(source, "src/orders.py"))


@pytest.mark.parametrize("guard,blocked", [
    ("if quantity < 0: raise ValueError()", False),
    ("if quantity == 0: raise ValueError()", False),
    ("if quantity != 0: return", False),
    ("if quantity >= 0: raise ValueError()", False),
    ("if quantity > 0: print('okay')", False),
    ("if quantity > 0: pass\n    else: raise ValueError()", True),
    ("if quantity <= 0 and capital != 'usd': raise ValueError()", False),
    ("if quantity <= 0 or capital == 'usd': raise ValueError()", True),
    ("if quantity <= 0: print('bad'); raise ValueError()", True),
    ("if 0 >= quantity: raise ValueError()", True),
    ("if 0 < quantity: pass\n    else: raise ValueError()", True),
    ("if quantity <= 0: pass", False),
    ("if not quantity > 0: raise ValueError()", True),
    ("if not quantity <= 0: raise ValueError()", False),
])
def test_order_guard_control_flow_must_reject_all_invalid_sizes(guard, blocked):
    source = f"def place(quantity, capital, broker):\n    {guard}\n    return broker.submit_order(quantity=quantity)\n"
    flagged = any(item["code"] == "FINANCE_UNGUARDED_SIZE"
                  for item in inspect_source(source, "src/orders.py"))
    assert flagged is (not blocked)


@pytest.mark.parametrize("expression,guarded", [
    ("max(quantity, 0)", True),
    ("min(max(quantity, 0), 10)", True),
    ("max(quantity, 1)", True),
    ("min(quantity, 10)", False),
    ("max(quantity, -1)", False),
    ("max(-1, quantity)", False),
    ("max(quantity, 0) - 1", False),
    ("abs(quantity)", False),
])
def test_order_clamps_must_establish_a_nonnegative_floor(expression, guarded):
    source = ("def place(quantity, broker):\n"
              f"    quantity = {expression}\n"
              "    return broker.place_order(size=quantity)\n")
    flagged = any(item["code"] == "FINANCE_UNGUARDED_SIZE"
                  for item in inspect_source(source, "src/orders.py"))
    assert flagged is (not guarded)


@pytest.mark.parametrize("guard,blocked", [
    ("if leverage == 0: raise ValueError()", True),
    ("if leverage != 0: pass\n    else: raise ValueError()", True),
    ("if leverage < 0: raise ValueError()", False),
    ("if leverage > 0: pass\n    else: raise ValueError()", True),
    ("if leverage >= 0: raise ValueError()", True),
    ("if leverage > 0: raise ValueError()", False),
    ("if leverage <= 0: raise ValueError()", True),
    ("if leverage == 0 or leverage < 0: raise ValueError()", True),
    ("if leverage == 0 and capital < 0: raise ValueError()", False),
])
def test_divisor_control_flow_must_reject_zero(guard, blocked):
    source = ("def calculate(capital, leverage):\n"
              f"    {guard}\n"
              "    return capital / leverage\n")
    flagged = any(item["code"] == "FINANCE_UNGUARDED_LEVERAGE"
                  for item in inspect_source(source, "src/risk.py"))
    assert flagged is (not blocked)


def test_positive_floor_expression_guards_a_financial_divisor():
    source = "def calculate(capital, leverage):\n    return capital / max(leverage, 1)\n"
    assert not any(item["code"] == "FINANCE_UNGUARDED_LEVERAGE"
                   for item in inspect_source(source, "src/risk.py"))


def test_unrelated_comparison_and_print_do_not_guard_financial_division():
    source = '''def calculate(capital, leverage):
    if capital != "usd":
        note = True
    return capital / leverage
'''
    assert any(item["code"] == "FINANCE_UNGUARDED_LEVERAGE"
               for item in inspect_source(source, "src/risk.py"))


@pytest.mark.parametrize("path", ["scripts/payments.py", "src/play/engine.js"])
def test_scripts_and_play_paths_are_audited_as_first_party(path):
    from aegis_core.noise_filter import classify_asset
    assert classify_asset(path)["scope"] == "first_party"


@pytest.mark.parametrize("path", [
    "scripts/", "scripts/finance/settle.py", "backend/scripts/transfer.js",
    "play/", "api/play.py", "src/player/payment.ts", "./play/order.js",
    "packages/gameplay/place_order.js",
])
def test_scripts_and_play_subtrees_never_receive_blanket_noise_suppression(path):
    from aegis_core.noise_filter import classify_asset
    result = classify_asset(path)
    assert result["scope"] == "first_party"
    assert result["suppression_reason"] is None


@pytest.mark.parametrize("provider", ["wx", "tt", "ks", "my"])
def test_miniapp_framework_defaults_cannot_suppress_missing_timeout(provider):
    from aegis_core.noise_filter import annotate_findings
    finding = {"code": "NETWORK_REQUEST_NO_TIMEOUT", "line": 1, "severity": "high"}
    source = f'{provider}.request({{ url: "/pay" }});'
    annotated = annotate_findings([finding], "src/pay.js", source,
                                  context_sources=['{"networkTimeout":{"request":5000}}'])
    assert annotated[0]["confidence"] == "HIGH"
    assert annotated[0]["suppression_reason"] is None


@pytest.mark.parametrize("provider,timeout,has_finding", [
    (provider, timeout, timeout != "5000")
    for provider in ("wx", "tt", "ks", "my")
    for timeout in ("5000", "0", "-1")
])
def test_miniapp_timeout_requires_an_explicit_positive_value(provider, timeout, has_finding):
    source = f"function load() {{ {provider}.request({{ url: '/pay', timeout: {timeout} }}); }}"
    findings = inspect_source(source, "src/pay.js")
    assert any(item["code"] == "NETWORK_REQUEST_NO_TIMEOUT" for item in findings) is has_finding


def test_l3_surfaces_preexisting_high_financial_findings_in_changed_file(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = "def place(quantity, broker):\n    return broker.submit_order(quantity=quantity)\n"
    (root / "service.py").write_text(source, encoding="utf-8")
    _clean_git(root)
    baseline = _git(root, "rev-parse", "HEAD")
    (root / "service.py").write_text(source + "# touched this line\n", encoding="utf-8")
    report = audit_candidate(root, baseline, ["service.py"])
    assert report["passed"] is False
    assert any(item["code"] == "FINANCE_UNGUARDED_SIZE" for item in report["findings"])


@pytest.mark.parametrize("source,code", [
    ("API_KEY = 'embedded-secret'\n", "HARDCODED_CREDENTIAL"),
    ("def calculate(capital, leverage):\n    return capital / leverage\n",
     "FINANCE_UNGUARDED_LEVERAGE"),
])
def test_l3_reports_existing_critical_and_high_invariants_in_touched_files(tmp_path, source, code):
    root = tmp_path / code
    root.mkdir()
    (root / "service.py").write_text(source, encoding="utf-8")
    _clean_git(root)
    baseline = _git(root, "rev-parse", "HEAD")
    (root / "service.py").write_text(source + "# unrelated touched line\n", encoding="utf-8")
    report = audit_candidate(root, baseline, ["service.py"])
    assert any(item["code"] == code for item in report["findings"])


def _signed_artifact_fixture(root, keys):
    logs = root / "artifacts/luna_logs"
    logs.mkdir(parents=True)
    log = logs / "run.jsonl"
    report = logs / "run_report.md"
    log.write_bytes(b'{"event":"final"}\n')
    report.write_bytes(b"# final report\n")
    manifest = logs / "run_manifest.json"
    value = {
        "schema_version": "luna.delivery.v1",
        "repository": str(root),
        "machine_log_path": "artifacts/luna_logs/run.jsonl",
        "machine_log_sha256": sha256_file(log),
        "report_path": "artifacts/luna_logs/run_report.md",
        "report_sha256": sha256_file(report),
    }
    manifest.write_text(json.dumps(value), encoding="utf-8")
    token = issue_token(manifest, policy_version="luna-p0", directory=keys)["token"]
    verify = lambda: verify_token(token, expected_manifest_sha256=sha256_file(manifest),
                                 expected_policy_version="luna-p0", directory=keys,
                                 manifest_path=manifest)
    return log, report, manifest, verify


def test_signed_manifest_binds_final_jsonl_and_report_bytes(tmp_path):
    keys = tmp_path / "keys"
    create_keypair(keys)
    log, report, manifest, verify = _signed_artifact_fixture(tmp_path / "repo", keys)
    assert verify()["authorized"]
    signed = json.loads(manifest.read_text(encoding="utf-8"))
    assert signed["machine_log_sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert signed["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    for artifact in (log, report):
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"# altered\n")
        assert verify()["authorized"] is False
        artifact.write_bytes(original)
        assert verify()["authorized"] is True


def test_signed_artifact_manifest_rejects_missing_manifest_path(tmp_path):
    keys = tmp_path / "keys"
    create_keypair(keys)
    _, _, manifest, verify = _signed_artifact_fixture(tmp_path / "repo", keys)
    token = json.loads(manifest.read_text(encoding="utf-8"))
    result = verify_token(
        issue_token(manifest, policy_version="luna-p0", directory=keys)["token"],
        expected_manifest_sha256=sha256_file(manifest), expected_policy_version="luna-p0",
        directory=keys)
    assert token["report_sha256"] and result["reason"] == "artifacts_unverified"


def test_missing_audit_ledger_is_not_valid_empty_ledger(tmp_path):
    result = AuditLedger(tmp_path / "audit.jsonl").verify()
    assert result["valid"] is False
    assert result["status"] == "missing"


def test_deleted_audit_log_is_missing_even_if_a_valid_anchor_remains(tmp_path):
    path = tmp_path / "audit.jsonl"
    ledger = AuditLedger(path)
    ledger.append("event")
    path.unlink()
    result = ledger.verify()
    assert result["valid"] is False
    assert result["status"] == "missing"
    assert result["entries"] == 0


def test_windows_private_key_acl_rejects_broad_read_access(monkeypatch, tmp_path):
    from aegis_core import enforce
    private = tmp_path / "private.pem"
    private.write_text("key", encoding="ascii")
    class Result:
        returncode = 0
        stdout = str(private) + " BUILTIN\\Users:(R)\n"
        stderr = ""
    monkeypatch.setattr(enforce.subprocess, "run", lambda *args, **kwargs: Result())
    assert enforce._windows_private_key_acl_safe(private) is False


@pytest.mark.parametrize("output", [
    "private.pem BUILTIN\\Users:(RX)",
    "private.pem Everyone:(F)",
    "private.pem NT AUTHORITY\\Authenticated Users:(R)",
    "private.pem *S-1-1-0:(M)",
    "private.pem *S-1-5-11:(R)",
    "private.pem *S-1-5-32-545:(RX)",
    "private.pem Everyone:(RD)",
    "private.pem BUILTIN\\Users:(REA)",
    "private.pem NT AUTHORITY\\Authenticated Users:(RC)",
    "private.pem *S-1-1-0:(M)",
    "private.pem *S-1-5-11:(F)",
    "private.pem *S-1-5-32-545:(R)",
])
def test_windows_private_key_acl_blocks_all_broad_builtin_read_groups(monkeypatch, tmp_path, output):
    from aegis_core import enforce
    private = tmp_path / "private.pem"
    class Result:
        returncode = 0
        stdout = output
        stderr = ""
    monkeypatch.setattr(enforce.subprocess, "run", lambda *args, **kwargs: Result())
    assert enforce._windows_private_key_acl_safe(private) is False


@pytest.mark.parametrize("returncode,output", [(0, "private.pem agent:(F)"),
                                                (0, "private.pem BUILTIN\\Users:(W)"),
                                                (1, "error")])
def test_windows_private_key_acl_only_accepts_proven_narrow_read_permissions(
        monkeypatch, tmp_path, returncode, output):
    from aegis_core import enforce
    private = tmp_path / "private.pem"
    class Result:
        pass
    result = Result()
    result.returncode, result.stdout, result.stderr = returncode, output, ""
    monkeypatch.setattr(enforce.subprocess, "run", lambda *args, **kwargs: result)
    assert enforce._windows_private_key_acl_safe(private) is (returncode == 0)


def test_windows_private_key_acl_fails_closed_when_icacls_is_unavailable(monkeypatch, tmp_path):
    from aegis_core import enforce
    private = tmp_path / "private.pem"
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("icacls")
    monkeypatch.setattr(enforce.subprocess, "run", unavailable)
    assert enforce._windows_private_key_acl_safe(private) is False


def test_dashboard_health_cannot_be_healthy_without_key_or_with_blockers(monkeypatch, tmp_path):
    from luna import status
    class Doctor:
        def __init__(self, *_args, **_kwargs): pass
        def check(self, **_kwargs): return {"blocking": ["missing_dependency"], "status": "BLOCKED"}
        def cleanup_temporary_configs(self): pass
    class Dashboard:
        def __init__(self, *_args, **_kwargs): pass
        def overview(self): return {"tickets": 0, "running": 0}
        def security_overview(self): return {"enforce": {"key_ready": False}}
        def quarantine(self, **_kwargs): return {"items": []}
    monkeypatch.setattr(status, "EnvDoctor", Doctor)
    monkeypatch.setattr(status, "DashboardReadModel", Dashboard)
    assert status_snapshot(tmp_path)["status"] != "HEALTHY"

def test_unfamiliar_unignored_cli_run_crash_probe_and_final_file_signatures(tmp_path, monkeypatch, capsys):
    from luna import runner
    from luna.cli import main as luna_main

    root = tmp_path / "foreign"
    root.mkdir()
    target = "service.py"
    vulnerable = ("def clamp_amount(value):\n"
                  "    if value == 0:\n"
                  "        return 1 / value\n"
                  "    if type(value) is not int:\n"
                  "        return None\n"
                  "    return min(10, max(0, value))\n")
    repaired = ("def clamp_amount(value):\n"
                "    if type(value) is not int:\n"
                "        return None\n"
                "    return min(10, max(0, value))\n")
    (root / target).write_text(vulnerable, encoding="utf-8")
    _clean_git(root)
    assert not (root / ".gitignore").exists()
    keys = tmp_path / "outside-keys"
    create_keypair(keys)
    candidate = {"candidate_id": "fixed_division",
                 "strategy": "validate then clamp without dividing by user input",
                 "patch": runner._patch_for(root, target, repaired), "token_cost": 3}

    class LocalMockRouter:
        chain = ["redteam-local"]

        def call(self, slot, messages, **kwargs):
            return {"status": "admitted", "output": json.dumps({"candidates": [candidate]}),
                    "tier": "coder", "slot_id": "local-redteam", "provider": "local-mock"}

    monkeypatch.setattr(runner, "_router", lambda _path: LocalMockRouter())
    requirement = f"{target}::clamp_amount clamps integers to [0, 10] and returns None for invalid inputs."
    code = luna_main(["run", requirement, "--repo", str(root), "--keys-dir", str(keys),
                      "--run-id", "foreign-redteam", "--timeout", "10"])
    output = capsys.readouterr().out
    assert code == 0, output
    assert "Luna 运行：SUCCEEDED" in output
    logs = root / "artifacts/luna_logs"
    machine = logs / "foreign-redteam.jsonl"
    report = logs / "foreign-redteam_report.md"
    manifest = logs / "foreign-redteam_manifest.json"
    signature_path = logs / "foreign-redteam_signature.json"
    rows = [json.loads(line) for line in machine.read_text(encoding="utf-8").splitlines()]
    red = next(row for row in rows if row["step"] == "隔离红绿探针与 L3 审查")
    assert red["details"]["baseline_true_red"] == "EXECUTION_ERROR"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert sha256_file(machine) == data["machine_log_sha256"]
    assert sha256_file(report) == data["report_sha256"]
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    checked = verify_token(
        signature["token"], expected_manifest_sha256=sha256_file(manifest),
        expected_policy_version="luna-v1", expected_policy_sha256=data["policy_sha256"],
        directory=keys, manifest_path=manifest)
    assert checked["authorized"]
    assert not (root / ".gitignore").exists()
    assert "repository baseline must be clean" not in output
    monkeypatch.setenv("AEGIS_KEYS_DIR", str(tmp_path / "missing-dashboard-keys"))
    assert status_snapshot(root)["status"] != "HEALTHY"
