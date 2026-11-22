"""Noise classification and confidence grading for invariant findings.

The module is deliberately source-only. It uses repository paths and nearby
configuration text; it never imports code from the scanned repository.
"""
from __future__ import annotations

from pathlib import PurePosixPath
import json
import re
from typing import Iterable


_NOISE_DIRECTORIES = {
    "vendor": "vendored_dependency",
    "vendors": "vendored_dependency",
    "third_party": "third_party_dependency",
    "third-party": "third_party_dependency",
    "dist": "generated_distribution",
    "fixtures": "test_fixture",
    "fixture": "test_fixture",
    "mock": "test_mock",
    "mocks": "test_mock",
    "__tests__": "test_fixture",
    "__mocks__": "test_mock",
    "tests": "test_source",
    "test": "test_source",
    "spec": "test_source",
    "specs": "test_source",
    "__fixtures__": "test_fixture",
    "test-fixtures": "test_fixture",
    "eval": "offline_evaluation_script",
    "tools": "offline_tooling",
    "probe": "offline_probe",
    "probes": "offline_probe",
    "docs": "documentation_example",
    "doc": "documentation_example",
}

_HOOK = re.compile(
    r"\b(onUnload|onHide|beforeUnmount|onBeforeUnmount|unmounted|onUnmounted|"
    r"componentWillUnmount|destroyed)\s*\([^)]*\)\s*\{"
)
_POSITIVE_TIMEOUT = r"(?:[1-9]\d{0,8}(?:\.\d+)?|\d*\.\d*[1-9]\d*)"


def _positive_timeout_value(source: str, value: str) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return bool(re.search(
            rf"\b(?:const|let|var)\s+{re.escape(value)}\s*=\s*{_POSITIVE_TIMEOUT}\b", source, re.I))


def classify_asset(path: str, source: str = "") -> dict:
    """Return deterministic scope/type metadata for a source asset."""
    normalized = str(path).replace("\\", "/").lstrip("./")
    parsed = PurePosixPath(normalized)
    lower = normalized.lower()
    segments = [part.lower() for part in parsed.parts]
    reason = None
    asset_type = parsed.suffix.lower().lstrip(".") or "unknown"

    if parsed.name.lower().endswith(".min.js") or parsed.name.lower().endswith(".min.css"):
        reason = "minified_asset"
    else:
        reason = next((_NOISE_DIRECTORIES[item] for item in segments if item in _NOISE_DIRECTORIES), None)
    basename = parsed.name.lower()
    if reason is None and (basename == "test.js" or basename.startswith("test_")
                           or re.search(r"(?:^|[._-])(?:test|spec)(?:[._-]|$)", basename)):
        reason = "test_source"
    if reason is None and "lib" in segments and re.search(
            r"three(?:\.module)?\.js|gltfloader|copyright|license|three\.js", source[:12000], re.I):
        reason = "vendored_dependency"
    if reason is None and any(marker in lower for marker in ("原型", "prototype", "mockup", "static-prototype")):
        reason = "static_html_prototype"
    if reason is None and parsed.suffix.lower() == ".html":
        if any(token in segments for token in ("prototype", "prototypes", "mockup", "mockups", "static-prototype")):
            reason = "static_html_prototype"
        elif re.search(r"\bdata-aegis-(?:prototype|mockup)\b", source, re.I):
            reason = "static_html_prototype"
    if reason is None and re.search(r"(?:^|/)(?:dist|vendor|third_party)/", lower):
        reason = "generated_or_vendored_path"

    return {
        "path": normalized,
        "scope": "excluded_noise" if reason else "first_party",
        "asset_type": asset_type,
        "suppression_reason": reason,
    }


def _mask(source: str) -> str:
    """Mask comments and strings while retaining line and character offsets."""
    chars = list(source)
    state = "code"
    quote = ""
    index = 0
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in "'\"`":
                quote = char
                state = "string"
                chars[index] = " "
            elif char == "/" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 1
                state = "line"
            elif char == "/" and nxt == "*":
                chars[index] = chars[index + 1] = " "
                index += 1
                state = "block"
        elif state == "string":
            if char != "\n":
                chars[index] = " "
            if char == "\\":
                if index + 1 < len(source) and source[index + 1] != "\n":
                    chars[index + 1] = " "
                    index += 1
            elif char == quote:
                state = "code"
        elif state == "line":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block":
            if char == "*" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 1
                state = "code"
            elif char != "\n":
                chars[index] = " "
        index += 1
    return "".join(chars)


def _matching_brace(masked: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(masked)


def _hook_bodies(masked: str):
    for match in _HOOK.finditer(masked):
        opening = masked.find("{", match.start(), match.end())
        yield match.group(1), masked[opening + 1:_matching_brace(masked, opening)]


def _call_arguments(masked: str, start: int) -> list[str]:
    opening = masked.find("(", start)
    if opening < 0:
        return []
    stack = []
    ranges = []
    begin = opening + 1
    for index in range(begin, len(masked)):
        char = masked[index]
        if char in "([{":
            stack.append(char)
        elif char in ")]}" and stack:
            stack.pop()
        elif char == ")" and not stack:
            if begin < index:
                ranges.append((begin, index))
            break
        elif char == "," and not stack:
            ranges.append((begin, index))
            begin = index + 1
    return [re.sub(r"\s+", "", masked[left:right]) for left, right in ranges]


def _resource_cleanup_reason(finding: dict, source: str) -> str | None:
    code = finding.get("code")
    if code not in {"RESOURCE_INTERVAL_NOT_CLEARED", "RESOURCE_EVENT_LISTENER_NOT_REMOVED",
                    "RESOURCE_WEBSOCKET_NOT_CLOSED"}:
        return None
    masked = _mask(source)
    hooks = list(_hook_bodies(masked))
    if not hooks:
        return None
    line = int(finding.get("line", 1))
    source_lines = source.splitlines()
    code_lines = masked.splitlines()
    if not (1 <= line <= len(source_lines) and line <= len(code_lines)):
        return None
    offset = sum(len(item) + 1 for item in code_lines[:line - 1])
    source_line = source_lines[line - 1]
    masked_line = code_lines[line - 1]

    if code == "RESOURCE_INTERVAL_NOT_CLEARED":
        call = re.search(r"\bsetInterval\s*\(", masked_line)
        if not call:
            return None
        prefix = masked_line[:call.start()]
        assigned = re.search(r"(?:\b(?:const|let|var)\s+)?([\w$.]+)\s*=\s*$", prefix)
        resource = assigned.group(1) if assigned else None
        for name, body in hooks:
            for clear in re.finditer(r"\bclearInterval\s*\(", body):
                args = _call_arguments(body, clear.start())
                if args and ((resource and args[0] == re.sub(r"\s+", "", resource))
                             or (resource is None and len(re.findall(r"\bsetInterval\s*\(", masked)) == 1)):
                    return f"framework_lifecycle_cleanup:{name}"

    elif code == "RESOURCE_WEBSOCKET_NOT_CLOSED":
        call = re.search(r"\bnew\s+WebSocket\s*\(", masked_line)
        if not call:
            return None
        assigned = re.search(r"(?:\b(?:const|let|var)\s+)?([\w$.]+)\s*=\s*$", masked_line[:call.start()])
        resource = assigned.group(1) if assigned else None
        if not resource:
            return None
        target = re.escape(resource)
        for name, body in hooks:
            if re.search(rf"\b{target}\s*\.\s*close\s*\(", body):
                return f"framework_lifecycle_cleanup:{name}"

    else:
        call = re.search(r"\baddEventListener\s*\(", masked_line)
        if not call:
            return None
        target_match = re.search(r"([\w$.]+)\s*\.\s*$", masked_line[:call.start()])
        if not target_match:
            return None
        target = re.sub(r"\s+", "", target_match.group(1))
        arguments = _call_arguments(masked_line, call.start())
        if len(arguments) < 2:
            return None
        for name, body in hooks:
            for remove in re.finditer(r"\bremoveEventListener\s*\(", body):
                remove_target = re.search(r"([\w$.]+)\s*\.\s*$", body[max(0, remove.start() - 120):remove.start()])
                remove_arguments = _call_arguments(body, remove.start())
                if (remove_target and re.sub(r"\s+", "", remove_target.group(1)) == target
                        and remove_arguments[:2] == arguments[:2]):
                    return f"framework_lifecycle_cleanup:{name}"
    return None


def _configured_timeout_reason(finding: dict, source: str, context_sources: Iterable[str]) -> str | None:
    if finding.get("code") not in {"NETWORK_REQUEST_NO_TIMEOUT", "NETWORK_FETCH_NO_ABORT_SIGNAL"}:
        return None
    masked_source = _mask(source)
    current_line = masked_source.splitlines()[max(0, int(finding.get("line", 1)) - 1)] if masked_source.splitlines() else ""
    candidates = [("same_file", source), *(("repository_config", item) for item in context_sources)]
    for origin, text in candidates:
        masked_text = _mask(text)
        configured_clients = set()
        for assignment in re.finditer(r"\b([\w$]+)\.defaults\.timeout\s*=\s*([\w$]+)", masked_text, re.I):
            if _positive_timeout_value(masked_text, assignment.group(2)):
                configured_clients.add(assignment.group(1))
        for assignment in re.finditer(
                r"\b([\w$]+)\.defaults\s*=\s*\{[^}]*\btimeout\s*:\s*([\w$]+)",
                masked_text, re.I | re.S):
            if _positive_timeout_value(masked_text, assignment.group(2)):
                configured_clients.add(assignment.group(1))
        for client in configured_clients:
            if re.search(rf"\b{re.escape(client)}\s*\.\s*(?:get|post|put|patch|delete|request)\s*\(", current_line):
                return f"configured_timeout:{origin}:{client}.defaults"
        aliases = re.finditer(
            r"(?:const|let|var)\s+([\w$]+)\s*=\s*(?:axios|http|client)\.create\s*\(\s*\{[^}]*\btimeout\s*:\s*([\w$]+)",
            masked_text, re.I | re.S)
        for alias_match in aliases:
            alias, timeout_value = alias_match.groups()
            if not _positive_timeout_value(masked_text, timeout_value):
                continue
            if re.search(rf"\b{re.escape(alias)}\s*\.\s*(?:get|post|put|patch|delete|request)\s*\(", current_line):
                return f"configured_timeout:{origin}:{alias}.create"
    return None


def _confidence(finding: dict) -> tuple[str, float]:
    code = finding.get("code", "")
    if code in {"HARDCODED_CREDENTIAL", "INJECTION_DYNAMIC_EVAL", "INJECTION_DYNAMIC_FUNCTION",
                "INJECTION_PROTOTYPE_POLLUTION"}:
        return "CRITICAL", 1.0
    if str(finding.get("severity", "")).lower() == "high":
        return "HIGH", 0.88
    return "MEDIUM", 0.68


def annotate_findings(findings: Iterable[dict], path: str, source: str, *,
                      context_sources: Iterable[str] = ()) -> list[dict]:
    """Add scope, confidence, and justified suppression fields to findings."""
    asset = classify_asset(path, source)
    annotated = []
    for original in findings:
        finding = dict(original)
        finding.update({"scope": asset["scope"], "asset_type": asset["asset_type"],
                        "suppression_reason": asset["suppression_reason"]})
        if asset["suppression_reason"]:
            finding.update(confidence="SUPPRESSED_NOISE", confidence_score=0.0,
                           suppression_reason=asset["suppression_reason"])
        else:
            reason = _resource_cleanup_reason(finding, source)
            reason = reason or _configured_timeout_reason(finding, source, context_sources)
            if reason:
                finding.update(scope="framework_managed", confidence="SUPPRESSED_NOISE",
                               confidence_score=0.0, suppression_reason=reason)
            else:
                confidence, score = _confidence(finding)
                finding.update(confidence=confidence, confidence_score=score)
        annotated.append(finding)
    return sorted(annotated, key=lambda item: (item.get("path", ""), item.get("line", 0), item.get("code", "")))


def split_findings(findings: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    visible, suppressed = [], []
    for finding in findings:
        (suppressed if finding.get("confidence") == "SUPPRESSED_NOISE" else visible).append(finding)
    return visible, suppressed
