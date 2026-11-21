"""Conservative language-aware checks for high-risk patch regressions.

The scanner deliberately reports only syntax with a direct dangerous shape.
It is a local gate, not a substitute for runtime tests or security review.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class RuleFinding:
    code: str
    line: int
    symbol: str
    detail: str


_CONTROL_NAME = re.compile(r"(?:auth|authoriz|permission|access|role|validat|guard|policy|permit)", re.I)
_IDEM = re.compile(r"idempotency[_A-Za-z0-9]*key", re.I)
_HTTP_CALLS = {"get", "post", "put", "patch", "delete", "request", "urlopen"}
_SQL_METHODS = {"execute", "executemany", "query", "raw"}


def _literal_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _literal_200(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value == 200:
        return True
    if isinstance(node, ast.Dict):
        return any(isinstance(key, ast.Constant) and key.value in {"status", "status_code", "statusCode"}
                   and _literal_200(value) for key, value in zip(node.keys, node.values))
    return False


def _expression_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return ""


def scan_python(source: str) -> list[RuleFinding]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    findings: list[RuleFinding] = []

    def report(code: str, node: ast.AST, symbol: str, detail: str) -> None:
        findings.append(RuleFinding(code, getattr(node, "lineno", 1), symbol, detail))

    module_names = {ast.unparse(target) for node in tree.body
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target])}
    has_idempotency_context = bool(re.search(r"idempotency[\s_-]*key", source, re.I))

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.LtE):
            left = _expression_text(node.left).lower()
            right = _expression_text(node.comparators[0]).lower()
            if right.startswith("len(") or right.endswith((".length", ".size")):
                report("OFF_BY_ONE_BOUNDARY", node, "<module>",
                       "Index comparison includes the collection length")
        if isinstance(node, ast.ExceptHandler):
            broad = node.type is None or _expression_text(node.type) in {"Exception", "BaseException"}
            silent = (not node.body or all(isinstance(item, (ast.Pass, ast.Continue, ast.Break))
                      or isinstance(item, ast.Return) and (item.value is None or isinstance(item.value, ast.Constant))
                      for item in node.body))
            if broad and silent:
                report("EXCEPTION_SWALLOWED", node, "<module>", "Broad exception handler discards the failure")
            elif node.body and all(isinstance(item, ast.Pass) for item in node.body):
                report("EXCEPTION_SWALLOWED", node, "<module>", "Exception handler silently discards the failure")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr.lower()
            owner = _expression_text(node.func.value).lower()
            if method in _HTTP_CALLS and (owner.startswith(("requests", "httpx", "urllib.request"))):
                if not any(keyword.arg == "timeout" for keyword in node.keywords):
                    report("MISSING_TIMEOUT", node, "<module>", f"{owner}.{method} call has no explicit timeout")
            if method in _SQL_METHODS and node.args:
                query = node.args[0]
                if isinstance(query, (ast.JoinedStr, ast.BinOp)):
                    report("SQL_COMMAND_INJECTION", node, "<module>", "SQL or command text is interpolated before execution")

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = getattr(node, "value", None)
            target_text = " ".join(_expression_text(target) for target in targets)
            if (_IDEM.search(target_text) or has_idempotency_context) and value is not None:
                rendered = _expression_text(value).lower()
                if re.search(r"(?:uuid|random|randint|token_hex|time\.time|time_ns|datetime\.now|utcnow)", rendered):
                    report("NONDETERMINISTIC_IDEMPOTENCY_KEY", node, "<module>",
                           "Idempotency key is derived from random or wall-clock data")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _CONTROL_NAME.search(node.name):
                if any(isinstance(item, ast.If) and _literal_true(item.test) for item in ast.walk(node)):
                    report("AUTHORIZATION_BYPASS", node, node.name,
                           "Policy-sensitive function contains an unconditional true branch")
            handlers = [item for item in ast.walk(node) if isinstance(item, ast.ExceptHandler)]
            if any(isinstance(item, ast.Return) and _literal_200(item.value)
                   for handler in handlers for item in ast.walk(handler)):
                report("FAILURE_AS_SUCCESS", node, node.name,
                       "Exception path returns a hard-coded HTTP 200 result")

            if isinstance(node, ast.AsyncFunctionDef):
                has_lock = any(isinstance(item, (ast.AsyncWith, ast.With)) and
                               any("lock" in _expression_text(context).lower() for context in item.items)
                               for item in ast.walk(node))
                if not has_lock:
                    for item in ast.walk(node):
                        shared_target = None
                        if isinstance(item, (ast.AugAssign, ast.AnnAssign)):
                            shared_target = item.target
                        elif isinstance(item, ast.Assign):
                            shared_target = item.targets[0] if item.targets else None
                        if isinstance(shared_target, ast.Attribute):
                            receiver = _expression_text(shared_target.value)
                            if (receiver in {"self", "cls", "type(self)"} and
                                    any(isinstance(child, ast.Await) for child in ast.walk(node))):
                                report("SHARED_STATE_WITHOUT_LOCK", item, node.name,
                                       "Async mutation of shared state has no visible lock context")
                                break
                        elif isinstance(shared_target, ast.Name) and shared_target.id in module_names:
                            report("SHARED_STATE_WITHOUT_LOCK", item, node.name,
                                   "Async mutation of module state has no visible lock context")
                            break

        if isinstance(node, ast.If) and _CONTROL_NAME.search(" ".join(_expression_text(item) for item in ast.walk(node))):
            if _literal_true(node.test):
                report("AUTHORIZATION_BYPASS", node, "<module>", "Unconditional true branch bypasses a policy check")

    unique = {(item.code, item.line, item.symbol, item.detail): item for item in findings}
    return sorted(unique.values(), key=lambda item: (item.line, item.code, item.symbol))


def _mask_js_comments(source: str) -> str:
    """Blank comments while retaining strings and offsets for conservative rules."""
    chars = list(source)
    index = 0
    quote = None
    escaped = False
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"`":
            quote = char
            index += 1
            continue
        if char == "/" and nxt == "/":
            end = source.find("\n", index)
            end = len(source) if end < 0 else end
            for position in range(index, end):
                chars[position] = " "
            index = end
            continue
        if char == "/" and nxt == "*":
            end = source.find("*/", index + 2)
            end = len(source) if end < 0 else end + 2
            for position in range(index, end):
                if chars[position] != "\n":
                    chars[position] = " "
            index = end
            continue
        index += 1
    return "".join(chars)


def _line(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


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


def scan_javascript(source: str) -> list[RuleFinding]:
    masked = _mask_js_comments(source)
    findings: list[RuleFinding] = []

    def add(code: str, offset: int, detail: str, symbol: str = "<module>") -> None:
        findings.append(RuleFinding(code, _line(source, offset), symbol, detail))

    for match in re.finditer(r"\bcatch\s*\(\s*([\w$]+)\s*\)\s*\{(?P<body>[^{}]*)\}", masked):
        body = match.group("body")
        if not body.strip() or re.fullmatch(r"\s*(?:return\s*(?:undefined|null)?\s*;?|[\w$]+\s*;?)\s*", body):
            add("EXCEPTION_SWALLOWED", match.start(), "Catch block discards its exception")
    for match in re.finditer(r"\.catch\s*\(\s*(?:\([^)]*\)|[\w$]+)?\s*=>\s*(?:\{\s*\}|undefined|null|void\s+0)?\s*\)", masked):
        add("EXCEPTION_SWALLOWED", match.start(), "Promise rejection is handled by a no-op callback")

    for match in re.finditer(r"\bfetch\s*\((?P<args>[^;)]*(?:\([^)]*\)[^;)]*)?)\)", masked):
        args = match.group("args")
        if "signal" not in args:
            add("MISSING_TIMEOUT", match.start(), "fetch call has no AbortController signal")
    for match in re.finditer(r"\b[\w$]+\s*<=\s*[\w$.]+\.(?:length|size)\b", masked):
        add("OFF_BY_ONE_BOUNDARY", match.start(), "Index comparison includes the collection length")
    bounded_cache_classes = set()
    for declaration in re.finditer(r"\bclass\s+([\w$]+)\s*\{", masked):
        opening = masked.find("{", declaration.start(), declaration.end())
        body = masked[opening + 1:_matching_brace(masked, opening)]
        evicts_by_capacity = (re.search(r"\bwhile\s*\([^)]*\.size\s*>\s*this\.[\w$]+", body)
                              and re.search(r"\.delete\s*\(", body))
        if evicts_by_capacity:
            bounded_cache_classes.add(declaration.group(1))
    for class_name in bounded_cache_classes:
        for match in re.finditer(rf"\bnew\s+{re.escape(class_name)}\s*\(\s*\{{(?P<args>[^{{}}]*)\}}\s*\)", masked):
            if not re.search(r"\b(?:max|maxSize|maxEntries)\s*:", match.group("args")):
                add("UNBOUNDED_GROWTH", match.start(), "LRU cache has no explicit capacity bound", class_name)
        for match in re.finditer(rf"\bnew\s+{re.escape(class_name)}\s*\(\s*\)", masked):
            add("UNBOUNDED_GROWTH", match.start(), "LRU cache is constructed without a capacity bound", class_name)
    for match in re.finditer(r"\b(?:axios|http|client)\s*\.\s*(?:get|post|put|patch|delete)\s*\(([^;]*)\)", masked):
        if not re.search(r"\btimeout\s*:", match.group(1)):
            add("MISSING_TIMEOUT", match.start(), "HTTP client call has no explicit timeout")

    has_idempotency_context = bool(re.search(r"idempotency[\s_-]*key", source, re.I))
    assignment = r"\b([\w$]*idempotency[\w$]*key|[\w$]+)\s*[:=]\s*([^,;\n}]+)"
    for match in re.finditer(assignment, masked, re.I):
        value_has_randomness = re.search(r"(?:randomUUID|Date\.now|Math\.random|crypto\.randomUUID|uuid)", match.group(2), re.I)
        if value_has_randomness and (_IDEM.search(match.group(1)) or has_idempotency_context):
            add("NONDETERMINISTIC_IDEMPOTENCY_KEY", match.start(),
                "Idempotency key is derived from random or wall-clock data", match.group(1))

    # Shared module bindings mutated inside async functions without a lock.
    shared = {match.group(1) for match in re.finditer(r"\b(?:let|var)\s+([\w$]+)\s*=", masked)}
    for match in re.finditer(r"\basync\s+function\s+([\w$]+)\s*\([^)]*\)\s*\{", masked):
        end = masked.find("}", match.end())
        body = masked[match.end():end if end >= 0 else len(masked)]
        if re.search(r"\b(?:lock|mutex)\b", body, re.I):
            continue
        for name in shared:
            if re.search(rf"\b{re.escape(name)}\s*(?:\+\+|--|[+\-*/]?=)", body):
                add("SHARED_STATE_WITHOUT_LOCK", match.start(),
                    f"Async mutation of module state {name} has no visible lock", match.group(1))
                break

    for match in re.finditer(r"\b(?:authorize|isAuthorized|hasPermission|canAccess|validate\w*)\s*\([^)]*\)\s*\{(?P<body>[^{}]*)\}", masked, re.I):
        if re.search(r"\bif\s*\(\s*true\s*\)", match.group("body")) or re.search(r"\breturn\s+true\s*;", match.group("body")):
            add("AUTHORIZATION_BYPASS", match.start(), "Authorization function has an unconditional true branch", match.group(0).split("(", 1)[0])
    for match in re.finditer(r"\bif\s*\(\s*true\s*\)", masked):
        context = masked[max(0, match.start() - 120):match.start()]
        if re.search(r"auth|permission|role|access|validat|guard|policy", context, re.I):
            add("AUTHORIZATION_BYPASS", match.start(), "Unconditional true branch bypasses policy validation")

    for match in re.finditer(r"\bcatch\s*\([^)]*\)\s*\{", masked):
        opening = masked.find("{", match.start(), match.end())
        closing = _matching_brace(masked, opening)
        body = masked[opening + 1:closing]
        if re.search(r"(?:status(?:Code)?\s*[:=]\s*200|return\s*\[\s*\])", body):
            add("FAILURE_AS_SUCCESS", match.start(), "Exception path returns an empty or hard-coded success result")

    for match in re.finditer(r"\b(?:execute|query|raw|prepare)\s*\(\s*(`[^`]*`|[^,)]*\+[^,)]*)", source, re.I):
        if "${" in match.group(1) or "+" in match.group(1):
            add("SQL_COMMAND_INJECTION", match.start(), "SQL or command text is interpolated before execution")

    return sorted({(item.code, item.line, item.symbol, item.detail): item for item in findings}.values(),
                  key=lambda item: (item.line, item.code, item.symbol))


def scan_source(source: str, path: str | Path) -> list[RuleFinding]:
    suffix = str(path).lower().rsplit(".", 1)[-1]
    if suffix in {"py", "pyw"}:
        return scan_python(source)
    if suffix in {"js", "jsx", "ts", "tsx", "mjs", "cjs"}:
        return scan_javascript(source)
    return []


def inspect_javascript_test_assertion_delta(baseline: str, candidate: str, *, path: str = "tests") -> dict:
    before = list(re.finditer(r"\bassert\s*\.\s*(?:equal|strictEqual|deepEqual|ok|throws|rejects)\s*\(",
                              _mask_js_comments(baseline)))
    after = list(re.finditer(r"\bassert\s*\.\s*(?:equal|strictEqual|deepEqual|ok|throws|rejects)\s*\(",
                             _mask_js_comments(candidate)))
    findings = []
    masked_candidate = _mask_js_comments(candidate)
    if len(after) < len(before):
        findings.append({"code": "TEST_ASSERTION_REMOVED", "path": path,
                         "detail": f"assertion count decreased from {len(before)} to {len(after)}"})
    for match in re.finditer(r"\bassert\s*\.\s*ok\s*\(\s*true\s*\)", masked_candidate):
        findings.append({"code": "TEST_ASSERTION_ALWAYS_TRUE", "path": path,
                         "line": _line(candidate, match.start()),
                         "detail": "assert.ok(true) cannot detect a regression"})
    return {"passed": not findings, "findings": findings,
            "baseline_assertions": len(before), "candidate_assertions": len(after)}

