"""Permanent regressions for the full red-team exploit chain in PR #24."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from urllib.request import urlopen

import pytest

from aegis_core.ast_mutator import _isolated_tree, generate_mutants, run_mutation_campaign
from aegis_core.audit_ledger import AuditLedger
from aegis_core.enforce import _windows_private_key_acl_safe, create_keypair, inspect_keys
from aegis_core.invariant_guard import inspect_source, scan_changed_files
from aegis_core.probe_synth import (acceptance_rule, expected_value, render_probe,
                                    static_cases)
from aegis_core.v20_samples import (ast_structure_skeleton, generate_v21_diverse_release)
from luna import status as luna_status


DIVISOR_NAMES = (
    "principal", "balance", "nav", "notional", "denom", "divisor", "equity", "account_principal",
    "cash_balance", "free_collateral", "stake_amount", "position_value", "reserve_funds", "account_equity",
    "risk_capital", "available_balance", "portfolio_nav", "trade_notional", "loan_principal",
    "current_balance", "net_asset_value", "division_denom", "settlement_divisor", "total_equity",
    "initial_principal", "closing_balance", "gross_notional", "margin_balance", "base_equity", "denominator",
)

CLAMP_CASES = (
    ("max(capital, 0)", True), ("max(0, capital)", True),
    ("min(max(capital, 0), 1000)", True), ("max(capital, 0.0)", True),
    ("max(capital, -1)", True), ("min(capital, 1000)", True),
    ("abs(capital)", True), ("capital or 1", True),
    ("max(capital, 1)", False), ("max(1, capital)", False),
    ("min(max(capital, 1), 1000)", False), ("max(capital, 0.01)", False),
    ("max(capital, True)", True), ("min(1000, capital)", True),
    ("max(capital, 0) - 1", True), ("max(capital, -0.1)", True),
    ("max(capital, 2)", False), ("min(max(capital, 2), 500)", False),
    ("max(capital, 0) + 1", True), ("max(capital, 0) * 2", True),
    ("max(0, capital - 1)", True), ("max(capital - 1, 0)", True),
    ("max(capital, 0) if capital else 0", True), ("max(capital, 0) if enabled else 1", True),
)

BROAD_PRINCIPALS = (
    "Everyone", "BUILTIN\\Users", "NT AUTHORITY\\Authenticated Users",
    "*S-1-1-0", "*S-1-5-11", "*S-1-5-32-545", "所有人", "BUILTIN\\用户",
)
READ_RIGHTS = ("F", "M", "RX", "R", "RD", "RA", "REA", "RC", "WDAC")


def _codes(source: str, code: str) -> list[dict]:
    return [item for item in inspect_source(source, "src/redteam.py") if item["code"] == code]


@pytest.mark.parametrize("name", DIVISOR_NAMES)
def test_static_guard_recognizes_financial_divisor_names(name):
    source = f"def calculate(amount, {name}):\n    return 1000 / {name}\n"
    assert _codes(source, "FINANCE_UNGUARDED_LEVERAGE")


@pytest.mark.parametrize("expression,should_flag", CLAMP_CASES)
def test_zero_floor_clamps_never_guard_division_but_positive_floors_do(expression, should_flag):
    source = ("def calculate(capital):\n"
              f"    capital = {expression}\n"
              "    return 1000 // capital\n")
    assert bool(_codes(source, "FINANCE_UNGUARDED_LEVERAGE")) is should_flag


@pytest.mark.parametrize("source,should_flag", [
    ("def get_position(position):\n    return position\n", True),
    ("def get_position(self):\n    return self.position\n", True),
    ("def get_position(position):\n    return {'position': position}\n", True),
    ("def get_position(self):\n    return max(self.position, 0)\n", False),
    ("def get_position(position):\n    if position < 0:\n        raise ValueError()\n    return position\n", False),
    ("def get_position(position):\n    if position <= 0:\n        return 0\n    return position\n", False),
    ("def get_position(self):\n    if self.position < 0:\n        return 0\n    return self.position\n", False),
    ("def get_position(position):\n    if position == 0:\n        raise ValueError()\n    return position\n", True),
    ("def get_position(position):\n    position = max(position, 0)\n    return position\n", False),
    ("def get_position(position):\n    if position < 0:\n        print('bad')\n    return position\n", True),
    ("def get_position(position):\n    return (position, 1)\n", True),
    ("def get_position(position):\n    if 0 > position:\n        return 0\n    return position\n", False),
])
def test_negative_position_returns_require_a_real_rejecting_guard(source, should_flag):
    assert bool(_codes(source, "FINANCE_UNGUARDED_POSITION_RETURN")) is should_flag


@pytest.mark.parametrize("principal", BROAD_PRINCIPALS)
@pytest.mark.parametrize("rights", READ_RIGHTS)
def test_windows_acl_rejects_readable_aces_for_all_broad_principal_spellings(
        monkeypatch, tmp_path, principal, rights):
    from aegis_core import enforce

    class Result:
        returncode = 0
        stdout = f"private.pem {principal}:({rights})\n"
        stderr = ""

    monkeypatch.setattr(enforce.subprocess, "run", lambda *args, **kwargs: Result())
    assert _windows_private_key_acl_safe(tmp_path / "private.pem") is False


@pytest.mark.parametrize("stdout", [
    "", "   \n", "icacls returned no ACEs", "private.pem owner:(UNKNOWN)",
    "private.pem owner:(F;R)", "Successfully processed 1 files; Failed processing 0 files",
    "private.pem owner:", "private.pem :(F)", "private.pem owner:(I)",
    "private.pem owner:(ABC)",
])
def test_windows_acl_parser_fails_closed_on_empty_or_unparsed_output(monkeypatch, tmp_path, stdout):
    from aegis_core import enforce

    class Result:
        returncode = 0
        stderr = ""

    Result.stdout = stdout
    monkeypatch.setattr(enforce.subprocess, "run", lambda *args, **kwargs: Result())
    assert _windows_private_key_acl_safe(tmp_path / "private.pem") is False


@pytest.mark.parametrize("principal,should_be_safe", [
    ("BUILTIN\\Administrators", True), ("NT AUTHORITY\\SYSTEM", True),
    ("BUILTIN\\Users", False), ("Everyone", False),
])
def test_windows_acl_parser_strips_the_queried_path_before_principal_check(
        monkeypatch, principal, should_be_safe):
    from aegis_core import enforce

    key_path = Path("C:/Users/runneradmin/AppData/Local/Temp/keys/private.pem")

    class Result:
        returncode = 0
        stdout = f"{key_path} {principal}:(F)\n"
        stderr = ""

    monkeypatch.setattr(enforce.subprocess, "run", lambda *args, **kwargs: Result())
    assert _windows_private_key_acl_safe(key_path) is should_be_safe


def test_ast_mutation_package_mirror_copies_when_symlinks_are_unavailable(monkeypatch, tmp_path):
    root = tmp_path / "source"
    package = root / "aegis_core"
    (package / "nested").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helpers.py").write_text("VALUE = 7\n", encoding="utf-8")
    (package / "nested" / "extra.py").write_text("OTHER = 8\n", encoding="utf-8")
    target = package / "capital.py"
    target.write_text("def exposure(value): return value\n", encoding="utf-8")
    probe = root / "tests" / "probe.py"
    probe.parent.mkdir()
    probe.write_text("def test_probe(): pass\n", encoding="utf-8")
    destination = tmp_path / "isolated"

    def no_symlink(*_args, **_kwargs):
        raise OSError("symbolic links are unavailable")

    monkeypatch.setattr(Path, "symlink_to", no_symlink)
    _isolated_tree(root, target, probe, "def exposure(value): return value + 1\n", destination)
    assert (destination / "aegis_core" / "helpers.py").read_text(encoding="utf-8") == "VALUE = 7\n"
    assert (destination / "aegis_core" / "nested" / "extra.py").read_text(encoding="utf-8") == "OTHER = 8\n"
    assert (destination / "aegis_core" / "capital.py").read_text(encoding="utf-8") == "def exposure(value): return value + 1\n"
    assert (destination / "tests" / "probe.py").read_text(encoding="utf-8") == "def test_probe(): pass\n"


@pytest.mark.parametrize("path_index", range(12))
def test_zero_byte_audit_ledgers_are_explicitly_invalid_empty_files(tmp_path, path_index):
    ledger_path = tmp_path / f"empty-{path_index}" / "audit.jsonl"
    ledger_path.parent.mkdir()
    ledger_path.touch()
    report = AuditLedger(ledger_path).verify()
    assert report["valid"] is False
    assert report["status"] == "empty"
    assert report["entries"] == 0
    assert not ledger_path.with_suffix(".jsonl.lock").exists()


def test_l3_changed_file_scan_rejects_a_zero_clamp_that_was_previously_allowed(tmp_path):
    root = tmp_path / "l3-repo"
    root.mkdir()
    target = root / "risk.py"
    target.write_text("def calculate(capital):\n    capital = max(capital, 1)\n    return 1000 // capital\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Red Team"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "redteam@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "risk.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "safe positive floor"], check=True)
    base = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    target.write_text("def calculate(capital):\n    capital = max(capital, 0)\n    return 1000 // capital\n", encoding="utf-8")
    report = scan_changed_files(root, ["risk.py"], baseline=base)
    assert report["passed"] is False
    assert "FINANCE_UNGUARDED_LEVERAGE" in {item["code"] for item in report["findings"]}


def test_l3_changed_file_scan_reports_unguarded_principal_division(tmp_path):
    root = tmp_path / "principal-l3-repo"
    root.mkdir()
    target = root / "risk.py"
    target.write_text("def calculate(amount, principal):\n    if principal == 0:\n        raise ValueError()\n    return 1000 / principal\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Red Team"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "redteam@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "risk.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "zero-only principal guard"], check=True)
    base = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    target.write_text("def calculate(amount, principal):\n    return 1000 / principal\n", encoding="utf-8")
    report = scan_changed_files(root, ["risk.py"], baseline=base)
    assert report["passed"] is False
    assert "FINANCE_UNGUARDED_LEVERAGE" in {item["code"] for item in report["findings"]}


class _ReadyDoctor:
    def __init__(self, *_args, **_kwargs):
        pass

    def check(self, **_kwargs):
        return {"status": "READY", "blocking": [], "disk": {"free_bytes": 1234}}

    def cleanup_temporary_configs(self):
        pass


class _Dashboard:
    def __init__(self, *_args, **_kwargs):
        pass

    def overview(self):
        return {"tickets": 7, "running": 2}

    def security_overview(self):
        return {"enforce": {"key_ready": True}}

    def quarantine(self, **_kwargs):
        return {"items": []}


@pytest.mark.parametrize("valid", [False, None, 0, "true", {}, {"pair_valid": True}, {"valid": False}])
def test_dashboard_key_presence_cannot_override_failed_real_key_inspection(monkeypatch, tmp_path, valid):
    monkeypatch.setattr(luna_status, "EnvDoctor", _ReadyDoctor)
    monkeypatch.setattr(luna_status, "DashboardReadModel", _Dashboard)
    monkeypatch.setattr(luna_status, "inspect_keys", lambda _directory: {"valid": valid})
    report = luna_status.status_snapshot(tmp_path)
    assert report["status"] == "ATTENTION_REQUIRED"
    assert report["tests_and_tickets"] == 7


@pytest.mark.parametrize("verdict", [
    "BLOCKED", "FAILED", "ERROR", "NEEDS_REVIEW", "PENDING_ACCEPTANCE", "TIMEOUT",
    "EXECUTION_ERROR", "BLOCKED_MISSING_CONSTRAINT", "BLOCKED_DIRTY_BASELINE", "CANDIDATE_FAILED",
    "UNVERIFIED", "UNKNOWN", "", None,
])
def test_any_pending_acceptance_run_prevents_healthy_status(monkeypatch, tmp_path, verdict):
    monkeypatch.setattr(luna_status, "EnvDoctor", _ReadyDoctor)
    monkeypatch.setattr(luna_status, "DashboardReadModel", _Dashboard)
    monkeypatch.setattr(luna_status, "inspect_keys", lambda _directory: {"valid": True})
    logs = tmp_path / "artifacts" / "luna_logs"
    logs.mkdir(parents=True)
    row = {"run_id": "pending-e2e", "verdict": verdict}
    (logs / "pending-e2e.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = luna_status.status_snapshot(tmp_path)
    assert report["pending_acceptance"]
    assert report["status"] == "ATTENTION_REQUIRED"


def test_valid_key_material_and_no_pending_acceptance_can_report_healthy(monkeypatch, tmp_path):
    monkeypatch.setattr(luna_status, "EnvDoctor", _ReadyDoctor)
    monkeypatch.setattr(luna_status, "DashboardReadModel", _Dashboard)
    monkeypatch.setattr(luna_status, "inspect_keys", lambda _directory: {"valid": True, "pair_valid": True})
    report = luna_status.status_snapshot(tmp_path)
    assert report["status"] == "HEALTHY"
    assert report["signing_key"] == "就绪"
    assert report["pending_acceptance"] == []


def test_real_ed25519_key_inspection_detects_modified_key_pair(tmp_path):
    keys = tmp_path / "keys"
    create_keypair(keys)
    assert inspect_keys(keys)["valid"] is True
    (keys / "ed25519-public.pem").write_text("not the public key", encoding="ascii")
    report = inspect_keys(keys)
    assert report["valid"] is False
    assert report["pair_valid"] is False


def _formula_fixture(root: Path, source: str):
    package = root / "aegis_core"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = package / "capital.py"
    target.write_text(source, encoding="utf-8")
    rule = acceptance_rule(
        "aegis_core/capital.py::exposure raises ValueError when capital <= 0 "
        "and returns 1000 // capital for positive integers.")
    assert rule is not None and rule["kind"] == "positive_formula"
    cases = static_cases(rule)
    filename, probe_source, command = render_probe(rule, cases)
    probe_path = root / filename
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(probe_source, encoding="utf-8")
    return target, probe_path, command, rule, cases


def _pytest_probe(root: Path, filename: str, command: list[str]):
    argv = [sys.executable if item == "{python}" else item for item in command]
    return subprocess.run(argv, cwd=root, capture_output=True, text=True, check=False)


def test_formula_probe_preserves_golden_outputs_and_accepts_value_error_repair(tmp_path):
    fixed = ("def exposure(capital):\n"
             "    if capital <= 0:\n"
             "        raise ValueError('capital must be positive')\n"
             "    return 1000 // capital\n")
    _target, probe, command, rule, cases = _formula_fixture(tmp_path, fixed)
    assert expected_value(rule, 0) == {"raises": "ValueError"}
    assert dict((value, expected) for value, expected in cases)[2] == 500
    result = _pytest_probe(tmp_path, probe.name, command)
    assert result.returncode == 0, result.stdout + result.stderr


def test_formula_probe_kills_clamp_cheat_that_discards_the_business_formula(tmp_path):
    cheated = ("def exposure(capital):\n"
               "    if capital <= 0:\n"
               "        raise ValueError('capital must be positive')\n"
               "    return min(1000, max(0, capital))\n")
    _target, probe, command, _rule, _cases = _formula_fixture(tmp_path, cheated)
    result = _pytest_probe(tmp_path, probe.name, command)
    assert result.returncode != 0
    assert "AEGIS_ACCEPTANCE" in result.stdout + result.stderr


def test_ast_mutation_campaign_executes_isolated_mutants_and_reports_real_kill_rate(tmp_path):
    fixed = ("def exposure(capital):\n"
             "    if capital <= 0:\n"
             "        raise ValueError('capital must be positive')\n"
             "    return 1000 // capital\n")
    target, probe, _command, _rule, _cases = _formula_fixture(tmp_path, fixed)
    generated = generate_mutants(fixed, max_mutants=64)
    kinds = {item["kind"] for item in generated}
    assert {"reverse_comparison", "change_numeric_boundary", "raise_to_pass", "remove_guard"} <= kinds
    report = run_mutation_campaign(tmp_path, target.relative_to(tmp_path), probe.relative_to(tmp_path),
                                  max_mutants=64, timeout=5)
    assert report["status"] == "complete", report
    assert report["generated"] >= 4
    assert report["killed"] == report["generated"]
    assert report["mutation_kill_rate"] == 1.0
    assert report["passed"] is True


def test_ast_mutation_campaign_rechecks_a_failed_direct_baseline_with_pytest(monkeypatch, tmp_path):
    fixed = ("def exposure(capital):\n"
             "    if capital <= 0:\n"
             "        raise ValueError('capital must be positive')\n"
             "    return 1000 // capital\n")
    target, probe, _command, _rule, _cases = _formula_fixture(tmp_path, fixed)
    from aegis_core import sandbox

    receipts = []

    def gate(_root, command, **_kwargs):
        if isinstance(command, str):
            command = command.split()
        command = list(command)
        receipts.append(command)
        if len(receipts) == 1:
            passed = False
        elif "pytest" in command:
            passed = True
        else:
            passed = False
        return {"passed": passed, "exit_code": 0 if passed else 1, "stdout": "", "stderr": "",
                "timed_out": False}

    monkeypatch.setattr(sandbox, "run_exam_gate", gate)
    report = run_mutation_campaign(tmp_path, target.relative_to(tmp_path), probe.relative_to(tmp_path),
                                  max_mutants=1, timeout=1)
    assert report["baseline"]["passed"] is True
    assert report["baseline"]["runner"] == "pytest"
    assert report["baseline"]["direct_runner_probe"]["passed"] is False
    assert report["killed"] == report["generated"] == 1


def test_ast_mutation_score_is_zero_when_the_candidate_does_not_pass_its_frozen_probe(tmp_path):
    cheated = ("def exposure(capital):\n"
               "    if capital <= 0:\n"
               "        raise ValueError('capital must be positive')\n"
               "    return max(0, min(1000, capital))\n")
    target, probe, _command, _rule, _cases = _formula_fixture(tmp_path, cheated)
    report = run_mutation_campaign(tmp_path, target.relative_to(tmp_path), probe.relative_to(tmp_path),
                                  max_mutants=64, timeout=5)
    assert report["status"] == "baseline_failed"
    assert report["mutation_kill_rate"] == 0.0
    assert report["passed"] is False


def test_v21_release_has_240_redlight_verified_samples_and_200_ast_skeletons(tmp_path):
    result = generate_v21_diverse_release(tmp_path)
    output = Path(result["output"])
    payload = (output / "samples.jsonl").read_bytes()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in payload.splitlines()]
    assert len(rows) == 240
    assert manifest["distinct_ast_skeleton_count"] >= 200
    assert len(set(row["ast_skeleton_sha256"] for row in rows)) >= 200
    assert hashlib.sha256(payload).hexdigest() == manifest["samples_sha256"]
    assert all(row["expected_finding"] == "UNGUARDED_DIVISOR" for row in rows)
    assert all(hashlib.sha256(row["source"].encode()).hexdigest() == row["source_sha256"] for row in rows)
    assert len({ast_structure_skeleton(row["source"]) for row in rows}) >= 200
    minimum = generate_v21_diverse_release(tmp_path, output_dir="minimum-release", sample_count=200)
    minimum_manifest = json.loads((Path(minimum["output"]) / "manifest.json").read_text(encoding="utf-8"))
    assert minimum_manifest["sample_count"] == 200
    assert minimum_manifest["distinct_ast_skeleton_count"] == 200


def test_luna_and_aegis_dashboards_share_ticket_count_and_health_source(monkeypatch, tmp_path):
    monkeypatch.setattr(luna_status, "EnvDoctor", _ReadyDoctor)
    monkeypatch.setattr(luna_status, "inspect_keys", lambda _directory: {"valid": True, "pair_valid": True})
    keys = tmp_path / "keys"
    create_keypair(keys)
    monkeypatch.setenv("AEGIS_KEYS_DIR", str(keys))
    db = tmp_path / ".aegis" / "ledger.sqlite3"
    db.parent.mkdir()
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE tickets(ticket_id TEXT, status TEXT, created_at TEXT, updated_at TEXT, assigned_tier TEXT)")
    connection.executemany("INSERT INTO tickets VALUES (?, ?, ?, ?, ?)", [
        ("a", "running", "1", "1", "coder"), ("b", "queued", "1", "1", "coder"),
        ("c", "succeeded", "1", "1", "coder")])
    connection.commit()
    connection.close()
    from aegis_core.live_dashboard import create_server
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/overview"
        with urlopen(url, timeout=5) as response:
            overview = json.loads(response.read())["data"]
        luna = luna_status.status_snapshot(tmp_path)
        assert overview["tickets"] == luna["tests_and_tickets"] == 3
        assert overview["status"] == luna["status"] == "HEALTHY"
        assert overview["signing_key"] == luna["signing_key"] == "就绪"
    finally:
        server.shutdown()
        server.server_close()
