from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import pytest

from aegis_core.audit_ledger import AuditLedger
from aegis_core.enforce import (create_keypair, inspect_keys, issue_token, verify_token)


@pytest.mark.parametrize("index", range(40))
def test_audit_ledger_appends_canonical_hash_linked_entries(tmp_path, index):
    ledger = AuditLedger(tmp_path / "audit.jsonl")
    entry = ledger.append(f"event-{index}", {"index": index, "label": "审计"}, timestamp=1_800_000_000 + index)
    report = ledger.verify()
    assert report["valid"] and report["entries"] == 1
    assert entry["prev_hash"] == "0" * 64
    assert entry["entry_hash"] == report["head_hash"]


@pytest.mark.parametrize("tamper", ["details", "prev_hash", "sequence", "entry_hash", "reorder", "truncate"])
@pytest.mark.parametrize("index", range(10))
def test_audit_ledger_detects_record_edits_deletions_and_reordering(tmp_path, tamper, index):
    path = tmp_path / f"audit-{tamper}-{index}.jsonl"
    ledger = AuditLedger(path)
    ledger.append("first", {"index": index})
    ledger.append("second", {"index": index + 1})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if tamper == "details":
        rows[0]["details"]["index"] += 100
    elif tamper == "prev_hash":
        rows[1]["prev_hash"] = "f" * 64
    elif tamper == "sequence":
        rows[0]["sequence"] = 9
    elif tamper == "entry_hash":
        rows[0]["entry_hash"] = "e" * 64
    elif tamper == "reorder":
        rows.reverse()
    else:
        rows.pop()
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8")
    assert not ledger.verify()["valid"]


@pytest.mark.parametrize("event", ["", " ", "\n", "\t"])
def test_audit_ledger_rejects_empty_event_names(tmp_path, event):
    with pytest.raises(ValueError):
        AuditLedger(tmp_path / "audit.jsonl").append(event)


def test_audit_ledger_serializes_concurrent_thread_writers(tmp_path):
    ledger = AuditLedger(tmp_path / "audit.jsonl")
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda number: ledger.append("parallel", {"n": number}), range(60)))
    report = ledger.verify()
    assert report["valid"] and report["entries"] == 60


def test_audit_ledger_refuses_to_append_after_tampering(tmp_path):
    ledger = AuditLedger(tmp_path / "audit.jsonl")
    ledger.append("first")
    path = ledger.path
    path.write_text(path.read_text().replace("first", "other"), encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        ledger.append("second")


def test_audit_verification_is_read_only_for_a_pristine_checkout(tmp_path):
    path = tmp_path / ".aegis" / "audit.jsonl"
    report = AuditLedger(path).verify()
    assert report["valid"] and report["entries"] == 0
    assert not path.parent.exists()
    assert not path.with_suffix(path.suffix + ".lock").exists()


def _signed_token(tmp_path, *, now=None):
    keys = tmp_path / "isolated-keys"
    create_keypair(keys, now=now)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"samples":2}\n', encoding="utf-8")
    issued = issue_token(manifest, policy_version="2.0", expires_in_seconds=300,
                         directory=keys, policy_digest="a" * 64, now=now)
    return keys, manifest, issued


def test_ed25519_keypair_is_nonoverwriting_and_permission_checked(tmp_path):
    keys, _, _ = _signed_token(tmp_path)
    report = inspect_keys(keys)
    assert report["valid"] and report["algorithm"] == "Ed25519"
    assert report["private_key_mode"] == "0600" if os.name != "nt" else report["permissions_valid"]
    with pytest.raises(FileExistsError):
        create_keypair(keys)


def test_token_signature_binds_manifest_policy_and_expiry(tmp_path):
    keys, manifest, issued = _signed_token(tmp_path)
    token = issued["token"]
    valid = verify_token(token, expected_manifest_sha256=issued["payload"]["manifest_sha256"],
                         expected_policy_version="2.0", expected_policy_sha256="a" * 64, directory=keys)
    assert valid["authorized"]
    mismatch = verify_token(token, expected_manifest_sha256="0" * 64,
                            expected_policy_version="2.0", expected_policy_sha256="a" * 64, directory=keys)
    assert mismatch["reason"] == "manifest_mismatch"
    assert verify_token(token, expected_manifest_sha256=issued["payload"]["manifest_sha256"],
                        expected_policy_version="1.9", expected_policy_sha256="a" * 64,
                        directory=keys)["reason"] == "policy_mismatch"
    assert verify_token(token, expected_manifest_sha256=issued["payload"]["manifest_sha256"],
                        expected_policy_version="2.0", expected_policy_sha256="b" * 64,
                        directory=keys)["reason"] == "policy_digest_mismatch"
    manifest.write_text('{"samples":3}\n', encoding="utf-8")
    assert issued["payload"]["manifest_sha256"] != __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()


def test_token_expiration_signature_tamper_and_malformed_payload_are_rejected(tmp_path):
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    keys, _, issued = _signed_token(tmp_path, now=fixed)
    expired = verify_token(issued["token"], expected_manifest_sha256=issued["payload"]["manifest_sha256"],
                           expected_policy_version="2.0", expected_policy_sha256="a" * 64,
                           directory=keys, now=fixed + timedelta(seconds=301))
    assert expired["reason"] == "token_expired"
    left, right = issued["token"].split(".")
    changed = ("A" if right[:1] != "A" else "B") + right[1:]
    tampered = verify_token(left + "." + changed,
                            expected_manifest_sha256=issued["payload"]["manifest_sha256"],
                            expected_policy_version="2.0", expected_policy_sha256="a" * 64, directory=keys, now=fixed)
    assert tampered["reason"] in {"invalid_signature", "malformed_token"}
    assert verify_token("not-a-token", expected_manifest_sha256="x", expected_policy_version="2.0",
                        expected_policy_sha256="a" * 64, directory=keys, now=fixed)["reason"] == "malformed_token"


@pytest.mark.parametrize("seconds", [0, -1, 31_536_001, 1.5, True])
def test_token_issuer_rejects_invalid_lifetimes(tmp_path, seconds):
    keys, manifest, _ = _signed_token(tmp_path)
    with pytest.raises(ValueError):
        issue_token(manifest, policy_version="2.0", expires_in_seconds=seconds,
                    directory=keys, policy_digest="a" * 64)
