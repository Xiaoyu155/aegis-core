"""Fast changed-file security scan and quarantine support for local Git hooks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time

from .audit_ledger import AuditLedger
from .reviewer import _scan_js, _scan_python


CODE_SUFFIXES = {".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
_FALLBACK_PATTERNS = (
    re.compile(rb"\bexcept\s*:\s*\n\s*pass\b"),
    re.compile(rb"\.catch\s*\(\s*(?:\(\s*\w*\s*\)|\w+)?\s*=>\s*\{\s*\}\s*\)"),
    re.compile(rb"\b(?:execute|query)\s*\(\s*(?:f['\"]|`)[^\n]*(?:\{|\$\{)"),
    re.compile(rb"\bif\s*\(?\s*True\s*\)?\s*:"),
    re.compile(rb"\bif\s*\(\s*true\s*\)"),
)


def _git(root: Path, *args: str, input: bytes | None = None, timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, input=input, capture_output=True,
                          timeout=timeout, check=False)


def _safe_path(value: str) -> str:
    if re.match(r"^[A-Za-z]:", value) or value.startswith(("\\\\", "//")):
        raise ValueError(f"unsafe repository path: {value}")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe repository path: {value}")
    return path.as_posix()


def scan_text(source: str, relative_path: str) -> list[dict]:
    suffix = Path(relative_path).suffix.lower()
    if suffix == ".py" or suffix == ".pyw":
        findings, _ = _scan_python(source, relative_path)
    elif suffix in CODE_SUFFIXES:
        findings, _ = _scan_js(source, relative_path)
    else:
        findings = []
    return [{"code": item.code, "path": item.path, "symbol": item.symbol,
             "line": item.line, "detail": item.detail} for item in findings]


def _fallback_scan(diff: bytes, relative: str) -> list[dict]:
    additions = []
    line = 0
    for raw in diff.splitlines():
        if raw.startswith(b"@@"):
            match = re.search(rb"\+(\d+)", raw)
            if match:
                line = int(match.group(1)) - 1
        elif raw.startswith(b"+") and not raw.startswith(b"+++"):
            line += 1
            additions.append((line, raw[1:]))
        elif not raw.startswith(b"-"):
            line += 1
    combined = b"\n".join(item for _line, item in additions)
    findings = []
    for pattern in _FALLBACK_PATTERNS:
        for match in pattern.finditer(combined):
            evidence_line = 1 + combined[:match.start()].count(b"\n")
            source_line = additions[min(evidence_line - 1, len(additions) - 1)][0] if additions else 1
            findings.append({"code": "UNIFIED_DIFF_REVIEW", "path": relative, "symbol": "<diff>",
                             "line": source_line, "detail": "Suspicious high-risk token found in diff fallback"})
    return findings


def _load_file(root: Path, relative: str, *, revision: str | None = None,
               max_bytes: int = 2 * 1024 * 1024) -> tuple[bytes, str | None]:
    safe = _safe_path(relative)
    if revision is not None:
        size = _git(root, "cat-file", "-s", f"{revision}:{safe}")
        if size.returncode == 0 and size.stdout.strip().isdigit() and int(size.stdout.strip()) > max_bytes:
            return b"\0", None
        result = _git(root, "show", f"{revision}:{safe}")
        if result.returncode:
            return b"", "revision_source_unavailable"
        return result.stdout, None
    path = root / safe
    if path.is_symlink():
        return b"", "symlink_not_scanned"
    try:
        if path.stat().st_size > max_bytes:
            return b"\0", None
        with path.open("rb") as handle:
            return handle.read(max_bytes + 1), None
    except OSError:
        return b"", "source_unreadable"


def scan_files(root: str | Path, files: list[str], *, revision: str | None = None,
               diff: bytes = b"", max_file_bytes: int = 2 * 1024 * 1024) -> dict:
    repo = Path(root).resolve()
    total_started = time.perf_counter()
    findings, degraded = [], []
    timings = []
    for raw_path in sorted(set(files)):
        relative = _safe_path(raw_path)
        suffix = Path(relative).suffix.lower()
        if suffix not in CODE_SUFFIXES:
            continue
        source_bytes, read_error = _load_file(repo, relative, revision=revision,
                                              max_bytes=max_file_bytes)
        started = time.perf_counter()
        if read_error:
            degraded.append({"path": relative, "reason": read_error})
            continue
        if b"\0" in source_bytes or len(source_bytes) > max_file_bytes:
            degraded.append({"path": relative, "reason": "binary_or_oversize_unified_diff_fallback"})
            findings.extend(_fallback_scan(diff, relative))
            timings.append((time.perf_counter() - started) * 1000)
            continue
        try:
            source = source_bytes.decode("utf-8-sig")
            current = scan_text(source, relative)
            findings.extend(current)
        except (UnicodeDecodeError, SyntaxError, RecursionError, ValueError, MemoryError) as error:
            degraded.append({"path": relative, "reason": type(error).__name__ + "_unified_diff_fallback"})
            findings.extend(_fallback_scan(diff, relative))
        timings.append((time.perf_counter() - started) * 1000)
    elapsed = sorted(timings)
    p95_index = max(0, min(len(elapsed) - 1, int((len(elapsed) * 0.95 + 0.999999)) - 1)) if elapsed else 0
    return {"status": "blocked" if findings else "clear", "files_scanned": len(timings),
            "findings": findings, "degraded_files": degraded,
            "p95_ms": elapsed[p95_index] if elapsed else 0.0,
            "total_ms": sum(timings), "elapsed_ms": (time.perf_counter() - total_started) * 1000,
            "revision": revision}


def _quarantine(root: Path, diff: bytes, report: dict, mode: str) -> dict:
    if not diff:
        return {"stored": False, "reason": "empty_diff"}
    directory = root / ".aegis" / "quarantine"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(diff).hexdigest()
    name = f"{time.time_ns()}-{digest[:16]}"
    diff_path = directory / f"{name}.diff"
    metadata_path = directory / f"{name}.json"
    temporary = diff_path.with_suffix(".tmp")
    temporary.write_bytes(diff)
    os.replace(temporary, diff_path)
    metadata = {"schema_version": "aegis.quarantine.v1", "mode": mode,
                "diff_sha256": digest, "findings": report["findings"],
                "degraded_files": report["degraded_files"], "created_at_ns": time.time_ns()}
    temp_meta = metadata_path.with_suffix(".tmp")
    temp_meta.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_meta, metadata_path)
    return {"stored": True, "diff": diff_path.relative_to(root).as_posix(),
            "metadata": metadata_path.relative_to(root).as_posix(), "diff_sha256": digest}


def _changed_names(root: Path, mode: str, stdin: bytes = b"") -> tuple[list[str], str | None, bytes, dict]:
    if mode == "staged":
        names = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").stdout
        diff = _git(root, "diff", "--cached", "--binary").stdout
        files = [item.decode("utf-8", errors="replace") for item in names.split(b"\0") if item]
        # Read staged blobs, not potentially different worktree buffers.
        files_data = {}
        for relative in files:
            result = _git(root, "show", f":{_safe_path(relative)}")
            if result.returncode == 0:
                files_data[relative] = result.stdout
        return files, None, diff, files_data
    if mode == "pre-push":
        files, diffs, revisions = set(), [], []
        for raw in stdin.decode("utf-8", errors="replace").splitlines():
            fields = raw.split()
            if len(fields) != 4:
                continue
            _local_ref, local_sha, _remote_ref, remote_sha = fields
            if not local_sha or set(local_sha) == {"0"}:
                continue
            base = remote_sha if remote_sha and set(remote_sha) != {"0"} else f"{local_sha}^"
            names = _git(root, "diff", "--name-only", "--diff-filter=ACMR", f"{base}..{local_sha}", "--").stdout
            files.update(item.decode("utf-8", errors="replace") for item in names.splitlines() if item)
            diffs.append(_git(root, "diff", "--binary", f"{base}..{local_sha}", "--").stdout)
            revisions.append(local_sha)
        return sorted(files), revisions[-1] if revisions else None, b"\n".join(diffs), {"revisions": revisions}
    raise ValueError("mode must be staged or pre-push")


def run_hook(root: str | Path = ".", *, mode: str, stdin: bytes = b"") -> tuple[int, dict]:
    repo = Path(root).resolve()
    files, revision, diff, extra = _changed_names(repo, mode, stdin)
    staged_blobs = extra if mode == "staged" else {}
    if staged_blobs:
        findings, degraded, timings = [], [], []
        for relative, blob in sorted(staged_blobs.items()):
            if Path(relative).suffix.lower() not in CODE_SUFFIXES:
                continue
            started = time.perf_counter()
            if len(blob) > 2 * 1024 * 1024 or b"\0" in blob:
                degraded.append({"path": relative, "reason": "binary_or_oversize_unified_diff_fallback"})
                findings.extend(_fallback_scan(diff, relative))
            else:
                try:
                    findings.extend(scan_text(blob.decode("utf-8-sig"), relative))
                except (UnicodeDecodeError, SyntaxError, RecursionError, ValueError, MemoryError) as error:
                    degraded.append({"path": relative, "reason": type(error).__name__ + "_unified_diff_fallback"})
                    findings.extend(_fallback_scan(diff, relative))
            timings.append((time.perf_counter() - started) * 1000)
        report = {"status": "blocked" if findings else "clear", "files_scanned": len(timings),
                  "findings": findings, "degraded_files": degraded,
                  "p95_ms": max(timings, default=0.0), "total_ms": sum(timings)}
    else:
        report = scan_files(repo, files, revision=revision, diff=diff)
    report.update({"mode": mode, "files": files})
    enforcement_marker = repo / ".aegis" / "enforce.enabled"
    if enforcement_marker.is_file():
        from .enforce import manifest_sha256, policy_sha256, verify_token
        manifest_path = repo / "datasets" / "release_v20" / "manifest_v2.json"
        token_path = repo / ".aegis" / "enforce.token"
        permit = None
        if manifest_path.is_file() and token_path.is_file():
            try:
                token = token_path.read_text(encoding="ascii").strip()
                permit = verify_token(token, expected_manifest_sha256=manifest_sha256(manifest_path),
                                      expected_policy_version="2.0",
                                      expected_policy_sha256=policy_sha256(repo))
            except (OSError, ValueError):
                permit = None
        if not permit or not permit.get("authorized"):
            reason = (permit or {}).get("reason", "enforce_permit_missing_or_unreadable")
            report["findings"].append({"code": "ENFORCE_PERMIT_INVALID", "path": ".aegis/enforce.token",
                                       "symbol": "<policy>", "line": 1,
                                       "detail": "Enforcement permit is absent, expired, or mismatched: " + reason})
            report["status"] = "blocked"
        report["enforce_permit"] = permit or {"authorized": False, "reason": "manifest_or_token_missing"}
    quarantine = _quarantine(repo, diff, report, mode) if report["findings"] else {"stored": False}
    report["quarantine"] = quarantine
    audit_path = repo / ".aegis" / "audit.jsonl"
    try:
        AuditLedger(audit_path).append("patch_blocked" if report["findings"] else "patch_allowed",
                                      {"mode": mode, "files": files,
                                       "finding_codes": sorted({item["code"] for item in report["findings"]}),
                                       "enforce_authorized": bool(report.get("enforce_permit", {}).get("authorized")),
                                       "enforce_reason": report.get("enforce_permit", {}).get("reason"),
                                       "quarantine": quarantine, "p95_ms": report["p95_ms"]})
    except (OSError, ValueError) as error:
        report["audit_error"] = str(error)
        return 2, report
    return (1 if report["findings"] else 0), report

