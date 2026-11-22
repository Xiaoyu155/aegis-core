"""Local Ed25519 signing and verification for fail-closed enforcement permits."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess


class EnforceError(RuntimeError):
    """A permit could not be safely issued or verified."""


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as error:
        raise EnforceError("Ed25519 support requires the cryptography package") from error
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def key_directory(home: str | Path | None = None) -> Path:
    configured = os.environ.get("AEGIS_KEYS_DIR")
    if configured and home is None:
        return Path(configured).expanduser().resolve()
    base = Path(home).expanduser() if home is not None else Path.home()
    return base / ".aegis" / "keys"


def _paths(directory: str | Path | None = None) -> tuple[Path, Path, Path]:
    base = Path(directory).expanduser().resolve() if directory else key_directory().resolve()
    private = Path(os.environ.get("AEGIS_SIGNING_KEY_PATH", base / "ed25519-private.pem")).expanduser()
    public = Path(os.environ.get("AEGIS_VERIFYING_KEY_PATH", base / "ed25519-public.pem")).expanduser()
    metadata = base / "key-metadata.json"
    return private.resolve(), public.resolve(), metadata.resolve()


def _write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def create_keypair(directory: str | Path | None = None, *, expires_days: int = 90,
                   now: datetime | None = None) -> dict:
    if type(expires_days) is not int or not 1 <= expires_days <= 3650:
        raise ValueError("expires_days must be an integer between 1 and 3650")
    serialization, private_type, _ = _crypto()
    base = Path(directory).expanduser().resolve() if directory else key_directory().resolve()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(base, 0o700)
    except OSError:
        pass
    private_path = base / "ed25519-private.pem"
    public_path = base / "ed25519-public.pem"
    metadata_path = base / "key-metadata.json"
    if any(path.exists() for path in (private_path, public_path, metadata_path)):
        raise FileExistsError(f"refusing to overwrite existing Aegis key material in {base}")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    expires = current.astimezone(timezone.utc) + timedelta(days=expires_days)
    private_key = private_type.generate()
    private_bytes = private_key.private_bytes(serialization.Encoding.PEM,
                                              serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption())
    public_bytes = private_key.public_key().public_bytes(serialization.Encoding.PEM,
                                                        serialization.PublicFormat.SubjectPublicKeyInfo)
    metadata = {"schema_version": "aegis.ed25519_key.v1",
                "algorithm": "Ed25519", "created_at": current.astimezone(timezone.utc).isoformat(),
                "expires_at": expires.isoformat(), "public_key_sha256": hashlib.sha256(public_bytes).hexdigest()}
    _write_exclusive(private_path, private_bytes, 0o600)
    try:
        _write_exclusive(public_path, public_bytes, 0o644)
        _write_exclusive(metadata_path, (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode(), 0o600)
    except Exception:
        private_path.unlink(missing_ok=True)
        public_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise
    return {"status": "created", "algorithm": "Ed25519", "private_key": str(private_path),
            "public_key": str(public_path), "expires_at": expires.isoformat(),
            "private_key_permissions": "0600"}


def _windows_private_key_acl_safe(private_path: Path) -> bool:
    """Fail closed if icacls cannot prove that broad built-in groups lack read access."""
    try:
        result = subprocess.run(["icacls", str(private_path)], capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode or not result.stdout.strip():
        return False
    broad_names = {"everyone", "users", "authenticated users", "所有人", "用户"}
    broad_sids = {"s-1-1-0", "s-1-5-11", "s-1-5-32-545"}
    ace_flags = {"I", "OI", "CI", "IO", "NP", "NX"}
    known_rights = {"F", "M", "RX", "R", "W", "RD", "WD", "AD", "REA", "WEA",
                    "RA", "WA", "AS", "RC", "WDAC", "WO", "S", "DE", "DC"}
    readable_rights = {"F", "M", "RX", "R", "RD", "RA", "REA", "RC", "WDAC"}
    ace_pattern = re.compile(r"^\s*(?P<principal>.+?):(?P<rights>(?:\([A-Za-z]+\)\s*)+)$")
    parsed_aces = 0
    for line in result.stdout.splitlines():
        match = ace_pattern.fullmatch(line)
        if match is None:
            continue
        principal = match.group("principal").strip().casefold().replace("/", "\\")
        if principal == Path(private_path).name.casefold():
            return False
        rights = [token.upper() for token in re.findall(r"\(([A-Za-z]+)\)", match.group("rights"))]
        if (not rights or not any(token in known_rights for token in rights)
                or any(token not in ace_flags | known_rights for token in rights)):
            return False
        parsed_aces += 1
        principal_parts = {part.strip() for part in re.split(r"[\\:\s]+", principal) if part.strip()}
        is_broad = bool(principal_parts & broad_names or principal_parts & broad_sids
                        or re.search(r"\bs-1-1-0\b|\bs-1-5-11\b|\bs-1-5-32-545\b", principal))
        if is_broad and any(token in readable_rights for token in rights):
            return False
    return parsed_aces > 0


def inspect_keys(directory: str | Path | None = None, *, now: datetime | None = None) -> dict:
    private_path, public_path, metadata_path = _paths(directory)
    state = {"algorithm": "Ed25519", "private_key_path": str(private_path),
             "public_key_path": str(public_path), "private_key_present": private_path.is_file(),
             "public_key_present": public_path.is_file(), "metadata_present": metadata_path.is_file(),
             "permissions_valid": False, "pair_valid": False, "expires_at": None,
             "expired": None, "valid": False, "errors": []}
    if not private_path.is_file() or not public_path.is_file() or not metadata_path.is_file():
        state["errors"].append("key_material_missing")
        return state
    if os.name != "nt":
        mode = stat.S_IMODE(private_path.stat().st_mode)
        state["private_key_mode"] = f"{mode:04o}"
        state["permissions_valid"] = mode == 0o600
        if not state["permissions_valid"]:
            state["errors"].append("private_key_permissions_must_be_0600")
    else:
        state["permissions_valid"] = _windows_private_key_acl_safe(private_path)
        if not state["permissions_valid"]:
            state["errors"].append("private_key_permissions_grant_broad_read_access")
    try:
        serialization, _, _ = _crypto()
        private_key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
        public_key = serialization.load_pem_public_key(public_path.read_bytes())
        public_bytes = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        private_public = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        state["pair_valid"] = hmac_compare(public_bytes, private_public)
        if not state["pair_valid"]:
            state["errors"].append("key_pair_mismatch")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(metadata["expires_at"])
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        state["expires_at"] = expires.astimezone(timezone.utc).isoformat()
        state["expired"] = expires <= current
        if state["expired"]:
            state["errors"].append("key_expired")
        expected_fingerprint = metadata.get("public_key_sha256")
        state["fingerprint"] = hashlib.sha256(public_path.read_bytes()).hexdigest()
        if expected_fingerprint != state["fingerprint"]:
            state["errors"].append("public_key_fingerprint_mismatch")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        state["errors"].append("invalid_key_material:" + type(error).__name__)
    state["valid"] = (state["permissions_valid"] and state["pair_valid"]
                       and state["expired"] is False and not state["errors"])
    return state


def hmac_compare(left: bytes, right: bytes) -> bool:
    import hmac
    return hmac.compare_digest(left, right)


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_sha256(repo_root: str | Path = ".") -> str:
    """Fingerprint gate code and the release manifest as one signed policy input."""
    root = Path(repo_root).resolve()
    files = (root / "aegis_core" / "security_rules.py",
             root / "aegis_core" / "reviewer.py",
             root / "aegis_core" / "security_gate.py",
             root / "aegis_core" / "enforce.py",
             root / "aegis_core" / "audit_ledger.py",
             root / "aegis_core" / "git_hooks.py",
             root / "datasets" / "release_v20" / "manifest_v2.json")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        if not path.is_file():
            digest.update(b"MISSING\0")
            continue
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _manifest_artifacts(manifest_path: str | Path, manifest: dict) -> dict[str, str]:
    names = {"machine_log_path": "machine_log_sha256", "report_path": "report_sha256"}
    present = any(name in manifest for name in (*names, *names.values()))
    if not present:
        return {}
    if any(not isinstance(manifest.get(path_field), str) or
           not isinstance(manifest.get(hash_field), str) or
           not re.fullmatch(r"[a-f0-9]{64}", manifest[hash_field])
           for path_field, hash_field in names.items()):
        raise ValueError("manifest must bind both the final machine log and report hashes")
    repository = Path(manifest.get("repository", Path(manifest_path).resolve().parent.parent.parent))
    if not repository.is_absolute():
        repository = Path(manifest_path).resolve().parent / repository
    repository = repository.resolve()
    hashes = {}
    for path_field, hash_field in names.items():
        relative = Path(manifest[path_field])
        artifact = (repository / relative).resolve()
        if relative.is_absolute() or not artifact.is_relative_to(repository):
            raise ValueError("manifest artifact path escapes repository")
        actual = manifest_sha256(artifact)
        if actual != manifest[hash_field]:
            raise ValueError(f"manifest artifact hash mismatch: {path_field}")
        hashes[relative.as_posix()] = actual
    return hashes


def issue_token(manifest_path: str | Path, *, policy_version: str,
                expires_in_seconds: int = 3600, directory: str | Path | None = None,
                policy_digest: str | None = None,
                now: datetime | None = None) -> dict:
    if not isinstance(policy_version, str) or not policy_version.strip():
        raise ValueError("policy_version must be nonempty")
    if type(expires_in_seconds) is not int or not 1 <= expires_in_seconds <= 31_536_000:
        raise ValueError("expires_in_seconds must be between 1 and one year")
    state = inspect_keys(directory, now=now)
    if not state["valid"]:
        raise EnforceError("invalid signing key: " + ", ".join(state["errors"]))
    private_path, _public_path, _metadata_path = _paths(directory)
    serialization, _, _ = _crypto()
    private_key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    issued = current.astimezone(timezone.utc)
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest_digest = manifest_sha256(manifest_file)
    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        artifacts = _manifest_artifacts(manifest_file, manifest_data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise EnforceError("invalid delivery manifest artifacts: " + str(error)) from error
    payload = {"schema_version": "aegis.enforce_token.v1", "algorithm": "Ed25519",
               "manifest_sha256": manifest_digest, "policy_version": policy_version,
               "policy_sha256": policy_digest,
               "artifacts_sha256": artifacts or None,
               "issued_at": issued.isoformat(),
               "expires_at": (issued + timedelta(seconds=expires_in_seconds)).isoformat()}
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    signature = private_key.sign(body)
    token = _b64(body) + "." + _b64(signature)
    return {"status": "issued", "token": token, "payload": payload,
            "fingerprint": state.get("fingerprint")}


def verify_token(token: str, *, expected_manifest_sha256: str,
                 expected_policy_version: str, expected_policy_sha256: str | None = None,
                 directory: str | Path | None = None, manifest_path: str | Path | None = None,
                 now: datetime | None = None) -> dict:
    try:
        payload_part, signature_part = token.split(".", 1)
        body, signature = _unb64(payload_part), _unb64(signature_part)
        payload = json.loads(body.decode("utf-8"))
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode("utf-8")
        if body != canonical:
            raise ValueError("noncanonical token payload")
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        return {"authorized": False, "reason": "malformed_token", "detail": str(error)}
    state = inspect_keys(directory, now=now)
    if not state["valid"]:
        return {"authorized": False, "reason": "invalid_verifying_key", "key_errors": state["errors"]}
    _private, public_path, _metadata = _paths(directory)
    serialization, _, _ = _crypto()
    try:
        serialization.load_pem_public_key(public_path.read_bytes()).verify(signature, body)
    except Exception:
        return {"authorized": False, "reason": "invalid_signature"}
    try:
        expires = datetime.fromisoformat(payload["expires_at"])
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return {"authorized": False, "reason": "invalid_expiry"}
    if payload.get("schema_version") != "aegis.enforce_token.v1" or payload.get("algorithm") != "Ed25519":
        return {"authorized": False, "reason": "unsupported_token"}
    if expires <= current:
        return {"authorized": False, "reason": "token_expired", "payload": payload}
    if payload.get("manifest_sha256") != expected_manifest_sha256:
        return {"authorized": False, "reason": "manifest_mismatch", "payload": payload}
    artifact_hashes = payload.get("artifacts_sha256")
    if artifact_hashes:
        if manifest_path is None:
            return {"authorized": False, "reason": "artifacts_unverified", "payload": payload}
        try:
            path = Path(manifest_path).expanduser().resolve()
            if manifest_sha256(path) != expected_manifest_sha256:
                return {"authorized": False, "reason": "manifest_mismatch", "payload": payload}
            manifest_data = json.loads(path.read_text(encoding="utf-8"))
            if _manifest_artifacts(path, manifest_data) != artifact_hashes:
                return {"authorized": False, "reason": "artifact_mismatch", "payload": payload}
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            return {"authorized": False, "reason": "artifact_mismatch", "detail": str(error),
                    "payload": payload}
    if payload.get("policy_version") != expected_policy_version:
        return {"authorized": False, "reason": "policy_mismatch", "payload": payload}
    if payload.get("policy_sha256") != expected_policy_sha256:
        return {"authorized": False, "reason": "policy_digest_mismatch", "payload": payload}
    return {"authorized": True, "reason": "verified", "payload": payload,
            "fingerprint": state.get("fingerprint")}

