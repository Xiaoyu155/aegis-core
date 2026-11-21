"""High-confidence static guards for finance, persistence, network and resources.

The guard is intentionally conservative: it reports source patterns that can be
explained with a concrete risky operation, and the reviewer compares findings
against the immutable candidate baseline so legacy findings do not block work.
It parses source only and never imports candidate modules.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import re
from pathlib import Path

from .noise_filter import annotate_findings, classify_asset, split_findings


@dataclass(frozen=True)
class InvariantFinding:
    code: str
    domain: str
    path: str
    symbol: str
    line: int
    severity: str
    detail: str


_SQL_WRITE = re.compile(r"\b(?:INSERT\s+INTO|UPDATE\s+[\w.]+|DELETE\s+FROM)\b", re.I)
_SQL_SELECT = re.compile(r"\bSELECT\s+\*\s+FROM\s+([\w.]+)(.*)", re.I | re.S)
_SENSITIVE = re.compile(r"(?:^|_)(?:api_?key|access_?token|secret(?:_?key)?|password|private_?key)(?:_|$)", re.I)
_MONEY = re.compile(r"price|balance|notional|amount|quantity|equity|pnl|profit|fee", re.I)
_NETWORK_MODULES = {"requests", "httpx", "urllib3"}
_RESOURCE_CALLS = {"connect", "acquire", "open", "Popen", "create_connection"}
_JS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".html"}
_JS_CALL = re.compile(r"\b(?:fetch|axios\.(?:get|post|put|patch|delete|request)|(?:wx|tt|ks|my)\.request)\s*\(")
_JS_NUMBER = re.compile(r"price|amount|balance|fee|rate|ratio|leverage|quantity|money|cost|profit|pnl|equity|latitude|longitude|coordinate|\b(?:x|y|z)\b", re.I)


def _mask_js(source: str) -> str:
    """Blank comments and string contents while retaining offsets and line breaks."""
    chars = list(source)
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in "'\"`":
                state, quote = "string", char
                chars[index] = " "
            elif char == "/" and nxt == "/":
                state = "line"
                chars[index] = chars[index + 1] = " "
                index += 1
            elif char == "/" and nxt == "*":
                state = "block"
                chars[index] = chars[index + 1] = " "
                index += 1
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


def _js_matching(masked: str, start: int, opening="(", closing=")") -> int:
    depth = 0
    for index in range(start, len(masked)):
        if masked[index] == opening:
            depth += 1
        elif masked[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return len(masked)


def _js_arguments(source: str, masked: str, match) -> list[str]:
    opening = masked.find("(", match.start(), match.end())
    closing = _js_matching(masked, opening)
    ranges = []
    start = opening + 1
    stack = []
    for index in range(start, closing):
        char = masked[index]
        if char in "([{":
            stack.append(char)
        elif char in ")]}" and stack:
            stack.pop()
        elif char == "," and not stack:
            ranges.append((start, index))
            start = index + 1
    if start < closing:
        ranges.append((start, closing))
    return [source[left:right].strip() for left, right in ranges]


def _js_scopes(masked: str):
    stack = []
    scopes = []
    for index, char in enumerate(masked):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            start = stack.pop()
            scopes.append((start, index + 1))
    return sorted(scopes, key=lambda span: (span[0], -span[1]))


def _js_scope_bounds(masked: str, scopes, position: int) -> tuple[int, int]:
    containing = [(start, end) for start, end in scopes if start < position < end]
    if not containing:
        return 0, len(masked)
    containing.sort(key=lambda span: span[1] - span[0])
    for start, end in containing:
        prefix = masked[max(0, start - 240):start]
        if re.search(r"\bfunction\b[^{}]*\)\s*$|=>\s*$", prefix):
            return start, end
    return containing[0]


def _js_scope(masked: str, scopes, position: int) -> str:
    start, end = _js_scope_bounds(masked, scopes, position)
    return masked[start:end]


def _js_has_try(masked: str, position: int, scopes) -> bool:
    # A storage/parse operation is protected only when it is inside a try body.
    for match in re.finditer(r"\btry\s*\{", masked[:position]):
        opening = masked.find("{", match.start(), match.end())
        if _js_matching(masked, opening, "{", "}") >= position:
            return True
    return False


def _inspect_javascript(source: str, path: str, *, line_offset: int = 0) -> list[dict]:
    masked = _mask_js(source)
    scopes = _js_scopes(masked)
    findings: list[InvariantFinding] = []

    def add(code, domain, position, detail, severity="high"):
        line = line_offset + masked.count("\n", 0, position) + 1
        prefix = masked[:position]
        functions = list(re.finditer(r"\bfunction\s+([\w$]+)\s*\(|\b(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[\w$]+)\s*=>", prefix))
        symbol = next((item.group(1) or item.group(2) for item in reversed(functions)), "<module>")
        findings.append(InvariantFinding(code, domain, path, symbol, line, severity, detail))

    # axios.create clients inherit the timeout configured on their instance.
    aliases = set(re.findall(r"\b(?:const|let|var)\s+([\w$]+)\s*=\s*axios\.create\s*\(", masked))
    configured_clients = set(re.findall(
        r"\b([\w$]+)\.defaults\.timeout\s*=\s*(?:[1-9]\d*(?:\.\d+)?|\d*\.\d*[1-9]\d*)", masked))
    configured_clients.update(match.group(1) for match in re.finditer(
        r"\b([\w$]+)\.defaults\s*=\s*\{[^}]*\btimeout\s*:\s*(?:[1-9]\d*(?:\.\d+)?|\d*\.\d*[1-9]\d*)",
        masked, re.S))
    network_clients = aliases | configured_clients
    network_matches = list(_JS_CALL.finditer(masked))
    for alias in network_clients:
        network_matches.extend(re.finditer(
            rf"\b{re.escape(alias)}\s*\.\s*(?:get|post|put|patch|delete|request)\s*\(", masked))
    network_matches = sorted({(item.start(), item.end()): item for item in network_matches}.values(),
                             key=lambda item: item.start())

    # Network calls: option text is read from the original source after locating
    # each call in the comment/string-masked representation.
    for match in network_matches:
        opening = masked.find("(", match.start(), match.end())
        end = _js_matching(masked, opening)
        call = source[match.start():min(end + 1, len(source))]
        masked_call = masked[match.start():min(end + 1, len(masked))]
        call_name = match.group(0).split("(", 1)[0].strip()
        if call_name == "fetch":
            bounded = bool(re.search(r"\bsignal\s*(?::|[,}])", call))
            code = "NETWORK_FETCH_NO_ABORT_SIGNAL"
        else:
            bounded = bool(re.search(r"\bsignal\s*:", masked_call))
            timeout_value = re.search(r"\btimeout\s*:\s*([^,}]+)", masked_call)
            if timeout_value:
                expression = timeout_value.group(1).strip()
                try:
                    bounded = bounded or float(expression) > 0
                except ValueError:
                    identifiers = re.findall(r"\b[A-Za-z_$][\w$]*\b", expression)
                    bounded = bounded or any(re.search(
                        rf"\b(?:const|let|var)\s+{re.escape(value)}\s*=\s*(?:[1-9]\d*(?:\.\d+)?|\d*\.\d*[1-9]\d*)\b", masked)
                        for value in identifiers)
                    for identifier in identifiers:
                        declaration = re.search(
                            rf"\b(?:const|let|var)\s+{re.escape(identifier)}\s*=\s*([^;]+)", masked)
                        if declaration and "?" in declaration.group(1):
                            defaults = re.findall(r"\b[A-Z_$][\w$]*TIMEOUT[\w$]*\b", declaration.group(1))
                            if any(re.search(
                                    rf"\b(?:const|let|var)\s+{re.escape(default)}\s*=\s*(?:[1-9]\d*(?:\.\d+)?|\d*\.\d*[1-9]\d*)\b", masked)
                                   for default in defaults):
                                bounded = True
            code = "NETWORK_REQUEST_NO_TIMEOUT"
        if not bounded:
            add(code, "external_api", match.start(), "Request has no explicit timeout or abort signal")

    loop_pattern = re.compile(r"\b(?:while\s*\(([^)]*)\)|for\s*\(([^)]*)\))\s*\{")
    for match in loop_pattern.finditer(masked):
        opening = masked.find("{", match.start(), match.end())
        end = _js_matching(masked, opening, "{", "}")
        body = masked[opening:end + 1]
        if not _JS_CALL.search(body) and not any(
                re.search(rf"\b{re.escape(alias)}\s*\.\s*(?:get|post|put|patch|delete|request)\s*\(", body)
                for alias in network_clients):
            continue
        header = (match.group(1) or match.group(2) or "").strip()
        backoff = bool(re.search(r"\b(?:setTimeout|sleep|delay|backoff|retryAfter|exponential)\b", body, re.I))
        if not backoff:
            add("NETWORK_RETRY_WITHOUT_BACKOFF", "external_api", match.start(),
                "Retry loop issues network requests without a delay or backoff")
        unbounded = (match.group(1) or "").strip() in {"true", "1"} or header in {"", ";;"}
        if unbounded:
            add("NETWORK_RETRY_UNBOUNDED", "external_api", match.start(),
                "Retry loop has no visible attempt bound")

    # Risky division involving money/rates/coordinates. A visible positive
    # guard in the same lexical block suppresses a finding.
    for match in re.finditer(r"([\w.$\[\]]+)\s*/\s*([\w.$\[\]]+|\d+(?:\.\d+)?)", masked):
        numerator, denominator = match.group(1), match.group(2)
        if not (_JS_NUMBER.search(numerator) or _JS_NUMBER.search(denominator)):
            continue
        scope = _js_scope(masked, scopes, match.start())
        guard = denominator not in {"0", "0.0"} and re.search(
            rf"\b{re.escape(denominator)}\b\s*(?:<=|<|===?|==)\s*0"
            rf"|\b{re.escape(denominator)}\b\s*>\s*0|0\s*<\s*\b{re.escape(denominator)}\b"
            rf"|\b{re.escape(denominator)}\b\s*&&|Number\.isFinite\s*\(\s*{re.escape(denominator)}\b",
            scope)
        if not guard:
            add("FINANCE_UNGUARDED_DIVISION", "finance", match.start(),
                f"Core numeric division by {denominator} has no visible finite positive-value guard")

    # User controlled numeric inputs should be checked before reaching financial
    # arithmetic or order APIs. Scope checks avoid a guard in a sibling function.
    for match in re.finditer(r"\b(?:Number|parseFloat|parseInt)\s*(?:\.\s*(?:parseFloat|parseInt))?\s*\(\s*([\w.$]+)", masked):
        input_name = match.group(1).split(".")[-1]
        prefix = masked[max(0, match.start() - 120):match.start()]
        assignment = re.search(r"(?:(?:const|let|var)\s+)?(\w+)\s*=\s*$", prefix)
        checked_name = assignment.group(1) if assignment else input_name
        scope = _js_scope(masked, scopes, match.start())
        if not (_JS_NUMBER.search(input_name) or re.search(r"(?:input|value|amount|price|rate|fee|coord|quantity)", input_name, re.I)):
            continue
        if not re.search(r"\bNumber\.isFinite\s*\(\s*" + re.escape(checked_name) + r"\b", scope):
            # Report only where a computed value is subsequently used in a
            # relevant arithmetic expression or order/payment operation.
            if re.search(r"(?:[+*/-]|submitOrder|placeOrder|createOrder|payment|amount|price|fee)", scope, re.I):
                add("FINANCE_INPUT_NOT_FINITE_CHECKED", "finance", match.start(),
                    f"Numeric input {input_name} reaches core calculation without Number.isFinite")

    # Event and timer lifetimes must be paired in their lexical scope.
    for match in re.finditer(r"\bsetInterval\s*\(", masked):
        scope = _js_scope(masked, scopes, match.start())
        prefix = masked[max(0, match.start() - 100):match.start()]
        assigned = re.search(r"\b(?:const|let|var)\s+(\w+)\s*=\s*$", prefix)
        timer = assigned.group(1) if assigned else None
        cleanups = re.findall(r"\bclearInterval\s*\(\s*([^)]*)\)", scope)
        if not cleanups or (timer and not any(re.match(rf"\s*{re.escape(timer)}\b", item) for item in cleanups)):
            add("RESOURCE_INTERVAL_NOT_CLEARED", "resource", match.start(),
                "Interval has no matching clearInterval cleanup")
    for match in re.finditer(r"\baddEventListener\s*\(", masked):
        scope_start, scope_end = _js_scope_bounds(masked, scopes, match.start())
        scope = masked[scope_start:scope_end]
        args = _js_arguments(source, masked, match)
        target_match = re.search(r"([\w$.]+)\s*\.\s*$", masked[max(0, match.start() - 160):match.start()])
        target = target_match.group(1) if target_match else ""
        if len(args) < 2 or not target:
            add("RESOURCE_EVENT_LISTENER_NOT_REMOVED", "resource", match.start(),
                "Event listener has no matching removeEventListener cleanup")
            continue
        matched_remove = False
        for removal in re.compile(r"\bremoveEventListener\s*\(").finditer(masked, scope_start, scope_end):
            remove_args = _js_arguments(source, masked, removal)
            remove_target_match = re.search(r"([\w$.]+)\s*\.\s*$",
                                            masked[max(0, removal.start() - 160):removal.start()])
            if (remove_target_match and remove_target_match.group(1) == target and len(remove_args) >= 2
                    and re.sub(r"\s+", "", remove_args[0]) == re.sub(r"\s+", "", args[0])
                    and re.sub(r"\s+", "", remove_args[1]) == re.sub(r"\s+", "", args[1])):
                matched_remove = True
                break
        if not matched_remove:
            add("RESOURCE_EVENT_LISTENER_NOT_REMOVED", "resource", match.start(),
                "Event listener has no matching removeEventListener cleanup")
    for match in re.finditer(r"\bnew\s+WebSocket\s*\(", masked):
        scope = _js_scope(masked, scopes, match.start())
        prefix = masked[max(0, match.start() - 100):match.start()]
        assigned = re.search(r"\b(?:const|let|var)\s+(\w+)\s*=\s*$", prefix)
        socket = assigned.group(1) if assigned else None
        close_pattern = rf"\b{re.escape(socket)}\s*\.\s*close\s*\(" if socket else r"\.\s*close\s*\("
        if not re.search(close_pattern, scope):
            add("RESOURCE_WEBSOCKET_NOT_CLOSED", "resource", match.start(),
                "WebSocket has no matching close cleanup")

    collection_pattern = r"\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\[\s*\]|new\s+Map\s*\(\s*\))|\b(?:globalThis|window)\.(\w+)\s*=\s*(?:\[\s*\]|new\s+Map\s*\(\s*\))"
    for match in re.finditer(collection_pattern, masked):
        name = match.group(1) or match.group(2)
        scope = _js_scope(masked, scopes, match.start())
        bounds = _js_scope_bounds(masked, scopes, match.start())
        is_global_collection = bounds == (0, len(masked)) or match.group(2) is not None
        writes = re.search(rf"\b{re.escape(name)}\s*(?:\.push\s*\(|\.set\s*\()", scope)
        if (is_global_collection and writes
                and not re.search(rf"\b{re.escape(name)}\.(?:shift|splice|delete)\s*\(", scope)):
            if not re.search(rf"\b{re.escape(name)}\.length\s*[><=]+\s*\d+", scope):
                add("UNBOUNDED_GROWTH", "resource", match.start(),
                    f"Collection {name} grows without a visible size bound or eviction")

    for match in re.finditer(r"\bJSON\s*\.\s*parse\s*\(", masked):
        if not _js_has_try(masked, match.start(), scopes):
            add("ASYNC_JSON_PARSE_UNGUARDED", "storage", match.start(), "JSON.parse is not protected by a try/catch")

    for match in re.finditer(r"\b(?:localStorage|sessionStorage)\s*\.\s*(?:getItem|setItem|removeItem|clear)\s*\(|\b(?:wx|tt|ks|my)\s*\.\s*(?:get|set|remove|clear)StorageSync\s*\(", masked):
        if not _js_has_try(masked, match.start(), scopes):
            # A visible fallback on the same statement is also sufficient.
            line_end = masked.find("\n", match.start())
            line = masked[match.start():line_end if line_end >= 0 else len(masked)]
            if not re.search(r"\?\?|\|\|\s*(?:\[|\{|['\"`]|\w+)", line):
                add("STORAGE_EXCEPTION_NO_FALLBACK", "storage", match.start(),
                    "Storage access has no exception guard or fallback")

    for match in re.finditer(r"\.then\s*\(", masked):
        end = _js_matching(masked, masked.find("(", match.start(), match.end()))
        tail = masked[end + 1:end + 240]
        if not re.match(r"\s*\.\s*(?:catch|finally)\s*\(", tail):
            add("ASYNC_PROMISE_UNHANDLED", "async", match.start(), "Promise chain has no catch or finally handler")

    for pattern, code, detail in (
        (r"\b(?:\w+\s*\.\s*)?innerHTML\s*=", "INJECTION_UNSAFE_HTML", "Dynamic content is assigned to innerHTML"),
        (r"\b(?:\w+\s*\.\s*)?insertAdjacentHTML\s*\(", "INJECTION_UNSAFE_HTML", "Dynamic content is inserted as raw HTML"),
        (r"\bdangerouslySetInnerHTML\s*:", "INJECTION_UNSAFE_HTML", "React markup bypasses HTML escaping"),
        (r"\beval\s*\(", "INJECTION_DYNAMIC_EVAL", "eval executes dynamically supplied code"),
        (r"\bnew\s+Function\s*\(", "INJECTION_DYNAMIC_FUNCTION", "Function constructor compiles dynamic code"),
        (r"(?:\b__proto__\s*=|\[\s*['\"]__proto__['\"]\s*\]\s*=|Object\s*\.\s*prototype\s*\[|constructor\s*\.\s*prototype\s*\[)", "INJECTION_PROTOTYPE_POLLUTION", "Assignment can mutate an object prototype"),
    ):
        for match in re.finditer(pattern, masked):
            add(code, "injection", match.start(), detail)

    return [asdict(item) for item in sorted(findings, key=lambda item: (item.path, item.line, item.code))]


def _sensitive_field(name: str) -> bool:
    """Match credential field names without treating environment references as secrets."""
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    if normalized.endswith(("_env", "_name", "_field", "_ref", "_path", "_var")):
        return False
    return bool(_SENSITIVE.search(normalized))


def _text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _calls(node: ast.AST):
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            yield item


def _qualified(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_sql(node: ast.AST) -> str:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return ""
    return value if isinstance(value, str) else ""


class _Guard:
    def __init__(self, source: str, path: str):
        self.source = source
        self.path = path
        self.tree = ast.parse(source)
        self.lines = source.splitlines()
        self.parents = {child: parent for parent in ast.walk(self.tree) for child in ast.iter_child_nodes(parent)}
        self.findings: list[InvariantFinding] = []

    def add(self, code: str, domain: str, node: ast.AST, detail: str, severity="high", symbol="<module>"):
        self.findings.append(InvariantFinding(code, domain, self.path,
                                               symbol, getattr(node, "lineno", 1), severity, detail))

    def run(self):
        self._module_secrets()
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(node)
        return sorted(self.findings, key=lambda item: (item.path, item.line, item.code, item.symbol))

    def _module_secrets(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = _text(target).rsplit(".", 1)[-1]
                    if _sensitive_field(name) and isinstance(node.value, ast.Constant) and node.value.value:
                        self.add("HARDCODED_CREDENTIAL", "external_api", node,
                                 "Credential-like field is assigned a literal value")
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and isinstance(key.value, str) and _sensitive_field(key.value):
                        if isinstance(value, ast.Constant) and value.value:
                            self.add("HARDCODED_CREDENTIAL", "external_api", key,
                                     "Credential-like mapping field contains a literal value")

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        symbol = node.name
        segment = ast.get_source_segment(self.source, node) or ""
        calls = list(_calls(node))
        self._finance(node, calls, segment, symbol)
        self._database(node, calls, segment, symbol)
        self._network(node, calls, segment, symbol)
        self._resources(node, calls, segment, symbol)

    def _finance(self, node, calls, segment, symbol):
        financial = bool(_MONEY.search(symbol))
        normalized_segment = re.sub(r"\s+", " ", segment)
        for call in calls:
            name = _qualified(call.func)
            if name.split(".")[-1].lower() in {"submit_order", "place_order", "create_order", "send_order"}:
                financial = True
                qty_names = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                             if re.search(r"quantity|amount|size|volume|notional", arg.arg, re.I)}
                source = segment.lower()
                guards = any(re.search(rf"\b{re.escape(name)}\b\s*(?:<=|<|==)\s*0|0\s*(?:>=|>)\s*\b{re.escape(name)}\b", source)
                             for name in qty_names)
                if qty_names and not guards:
                    self.add("FINANCE_UNGUARDED_SIZE", "finance", call,
                             "Order submission has quantity-like inputs without a visible nonpositive-size guard", symbol=symbol)
            if name in {"float", "builtins.float"} and call.args:
                value_name = _text(call.args[0])
                if _MONEY.search(value_name):
                    self.add("FINANCE_BINARY_FLOAT_MONEY", "finance", call,
                             "Monetary value is explicitly converted to binary float", severity="medium", symbol=symbol)
        for item in ast.walk(node):
            if isinstance(item, ast.BinOp) and isinstance(item.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                denominator = _text(item.right)
                if re.search(r"lever|margin|capital|equity", denominator, re.I):
                    checked = re.search(
                        rf"(?:if|assert)[^\n]*\b{re.escape(denominator)}\b[^\n]*(?:<=|>=|==|!=|<|>|not)"
                        rf"|(?:if|assert)[^\n]*(?:<=|>=|==|!=|<|>)\s*\b{re.escape(denominator)}\b",
                        normalized_segment,
                    )
                    if not checked:
                        self.add("FINANCE_UNGUARDED_LEVERAGE", "finance", item,
                                 f"Division by {denominator} has no visible zero or positive-value guard", symbol=symbol)

    def _database(self, node, calls, segment, symbol):
        writes = []
        for call in calls:
            name = _qualified(call.func).lower()
            if name.split(".")[-1] not in {"query", "execute", "run", "exec"} or not call.args:
                continue
            sql = _literal_sql(call.args[0])
            if sql and _SQL_WRITE.search(sql):
                writes.append((call, sql))
            select = _SQL_SELECT.search(sql)
            if select and not re.search(r"\b(?:WHERE|LIMIT)\b", select.group(2), re.I):
                if re.search(r"users?|orders?|trades?|events?|records?|jobs?|tasks?|sessions?", select.group(1), re.I):
                    self.add("DATABASE_UNBOUNDED_SCAN", "database", call,
                             "Full-row SELECT has no WHERE or LIMIT bound", symbol=symbol)
        transaction_visible = bool(re.search(r"\b(?:transaction|begin|commit|rollback|with_transaction)\b", segment, re.I))
        if len(writes) > 1 and not transaction_visible:
            self.add("DATABASE_MULTI_WRITE_WITHOUT_TRANSACTION", "database", writes[0][0],
                     f"Function performs {len(writes)} SQL writes without a visible transaction boundary", symbol=symbol)
        for call, sql in writes:
            if re.search(r"\bUPDATE\b[^;]*\bSET\b[^;]*\bstatus\b", sql, re.I):
                where = re.split(r"\bWHERE\b", sql, flags=re.I)
                if len(where) > 1 and re.search(r"\b(?:id|task_id|order_id)\b", where[1], re.I) and not re.search(r"\bstatus\b|\bversion\b", where[1], re.I):
                    self.add("STATE_TRANSITION_NOT_COMPARE_AND_SET", "database", call,
                             "Status update checks row identity but not the prior status or version", symbol=symbol)

    def _network(self, node, calls, segment, symbol):
        for call in calls:
            qualified = _qualified(call.func)
            parts = qualified.split(".")
            method = parts[-1].lower()
            is_python_http = (len(parts) >= 2 and parts[-2].lower() in _NETWORK_MODULES
                              and method in {"get", "post", "put", "delete", "patch", "request"})
            is_fetch = method == "fetch"
            if is_python_http and not any(keyword.arg == "timeout" for keyword in call.keywords):
                self.add("NETWORK_REQUEST_NO_TIMEOUT", "external_api", call,
                         "HTTP client call has no explicit timeout", symbol=symbol)
            elif is_fetch and not any(keyword.arg == "signal" for keyword in call.keywords):
                self.add("NETWORK_FETCH_NO_ABORT_SIGNAL", "external_api", call,
                         "fetch call has no local AbortSignal timeout", severity="medium", symbol=symbol)
        for loop in (item for item in ast.walk(node) if isinstance(item, (ast.For, ast.AsyncFor, ast.While))):
            loop_source = ast.get_source_segment(self.source, loop) or ""
            has_http = bool(re.search(r"\b(?:requests|httpx|urllib3)\.(?:get|post|request)\s*\(", loop_source))
            has_backoff = bool(re.search(r"\b(?:sleep|backoff|retry_after|exponential)\b", loop_source, re.I))
            if has_http and not has_backoff:
                self.add("NETWORK_RETRY_WITHOUT_BACKOFF", "external_api", loop,
                         "Retry loop repeats an HTTP request without a delay or backoff", symbol=symbol)

    def _resources(self, node, calls, segment, symbol):
        for call in calls:
            name = _qualified(call.func)
            final = name.split(".")[-1]
            if final == "open" and isinstance(call.func, ast.Name):
                inside_with = any(isinstance(parent, (ast.With, ast.AsyncWith)) for parent in self._ancestors(call))
                if not inside_with and ".close(" not in segment:
                    self.add("RESOURCE_OPEN_NOT_CLOSED", "resource", call,
                             "File is opened outside a context manager and no close call is visible", symbol=symbol)
            if final == "acquire":
                if ".release(" not in segment and not any(isinstance(parent, (ast.With, ast.AsyncWith))
                                                           for parent in self._ancestors(call)):
                    self.add("RESOURCE_LOCK_NOT_RELEASED", "resource", call,
                             "Lock is acquired without a release or context-manager boundary", symbol=symbol)
            if final in {"connect", "create_connection"} and not re.search(r"\.(?:close|release)\s*\(", segment):
                self.add("RESOURCE_CONNECTION_NOT_RELEASED", "resource", call,
                         "Connection is opened without a visible close or release", severity="medium", symbol=symbol)
            if final == "add" and "worktree" in name.lower() and "worktree remove" not in segment.lower():
                self.add("RESOURCE_WORKTREE_NOT_REMOVED", "resource", call,
                         "Git worktree is created without a matching cleanup command", symbol=symbol)

    def _ancestors(self, node):
        current = node
        while current in self.parents:
            current = self.parents[current]
            yield current


def inspect_source(source: str, path: str = "<source>", *, context_sources=()) -> list[dict]:
    """Return deterministic findings for Python or supported JS/TS/HTML source."""
    if type(source) is not str or type(path) is not str:
        raise TypeError("source and path must be text")
    normalized = str(path).replace("\\", "/")
    suffix = Path(path).suffix.lower()
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        findings = _inspect_javascript(source, normalized)
    elif suffix == ".html":
        findings = []
        for match in re.finditer(r"<script\b[^>]*>(.*?)</script\s*>", source, re.I | re.S):
            if not match.group(1).strip():
                continue
            line_offset = source.count("\n", 0, match.start(1))
            findings.extend(_inspect_javascript(match.group(1), normalized, line_offset=line_offset))
    else:
        try:
            findings = [asdict(item) for item in _Guard(source, normalized).run()]
        except SyntaxError as error:
            findings = [asdict(InvariantFinding("SOURCE_SYNTAX_ERROR", "source", normalized,
                                                "<module>", error.lineno or 1, "high", "Source cannot be parsed"))]
    return annotate_findings(findings, normalized, source, context_sources=context_sources)


def _repository_timeout_context(root: Path) -> list[str]:
    """Read only likely shared config/client files for repo-wide timeout defaults."""
    from .sandbox import run_git

    result = run_git(root, "ls-files", "-z", check=False)
    if result.returncode:
        return []
    context = []
    accepted = {"app.json", "project.config.json", "manifest.json"}
    for raw in result.stdout.decode("utf-8", errors="replace").split("\0"):
        if not raw:
            continue
        relative = Path(raw).as_posix()
        name = Path(relative).name.lower()
        if name not in accepted and not re.search(r"(?:request|http|axios|api.client|client.config)\.(?:js|jsx|ts|tsx|mjs|cjs|json)$", name):
            continue
        path = root / relative
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            context.append(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError):
            continue
    return context


def scan_changed_files(root: str | Path, changed_files, *, baseline: str | None = None) -> dict:
    """Find new invariant violations in Python, JS/TS and inline HTML files."""
    root = Path(root).resolve()
    findings = []
    suppressed = []
    scanned = []
    from .sandbox import run_git

    timeout_context = _repository_timeout_context(root)

    for relative in sorted(set(changed_files)):
        path = (root / relative).resolve()
        if (not path.is_relative_to(root) or not path.is_file() or path.is_symlink()
                or path.suffix.lower() not in _JS_EXTENSIONS | {".py"}):
            continue
        try:
            after = path.read_text(encoding="utf-8-sig")
            earlier = ""
            if baseline:
                result = run_git(root, "cat-file", "blob", f"{baseline}:{Path(relative).as_posix()}", check=False)
                if result.returncode == 0:
                    earlier = result.stdout.decode("utf-8-sig", errors="replace")
            before = inspect_source(earlier, relative, context_sources=timeout_context)
            after_findings = inspect_source(after, relative, context_sources=timeout_context)
        except (OSError, ValueError):
            continue
        previous = {(item["code"], item["symbol"], item["detail"]) for item in before}
        new_findings = [item for item in after_findings
                        if (item["code"], item["symbol"], item["detail"]) not in previous]
        visible, filtered = split_findings(new_findings)
        findings.extend(visible)
        suppressed.extend(filtered)
        scanned.append(Path(relative).as_posix())
    findings.sort(key=lambda item: (item["path"], item["line"], item["code"]))
    suppressed.sort(key=lambda item: (item["path"], item["line"], item["code"]))
    raw_count = len(findings) + len(suppressed)
    return {"passed": not findings, "scanned_files": scanned, "findings": findings,
            "suppressed_findings": suppressed,
            "summary": {"raw_findings": raw_count, "actionable_findings": len(findings),
                        "suppressed_findings": len(suppressed),
                        "noise_reduction_rate": (len(suppressed) / raw_count if raw_count else 0.0)}}


def scan_repository(root: str | Path, *, baseline: str | None = None) -> dict:
    """Read-only scan of tracked Python, JS/TS, and inline-HTML source files."""
    root = Path(root).resolve()
    from .sandbox import run_git

    result = run_git(root, "ls-files", "-z", check=False)
    if result.returncode:
        raise ValueError(f"not a git repository: {root}")
    tracked = result.stdout.decode("utf-8", errors="replace").split("\0")
    return scan_changed_files(root, tracked, baseline=baseline)


def inspect_test_assertion_delta(baseline: str, candidate: str, *, path: str = "tests") -> dict:
    """Fail when a candidate removes or trivially weakens Python assertions.

    This narrow comparison is used for generated mutation evidence and review
    of test edits. It parses both versions without importing either module.
    """
    try:
        before_tree = ast.parse(baseline, filename=f"{path}:baseline")
        after_tree = ast.parse(candidate, filename=f"{path}:candidate")
    except SyntaxError as error:
        return {"passed": False, "findings": [{"code": "TEST_SOURCE_SYNTAX_ERROR",
                "path": path, "detail": str(error)}]}
    before = [node for node in ast.walk(before_tree) if isinstance(node, ast.Assert)]
    after = [node for node in ast.walk(after_tree) if isinstance(node, ast.Assert)]
    findings = []
    if len(after) < len(before):
        findings.append({"code": "TEST_ASSERTION_REMOVED", "path": path,
                         "detail": f"assertion count decreased from {len(before)} to {len(after)}"})

    def contains_true(node):
        return any(isinstance(item, ast.Constant) and item.value is True for item in ast.walk(node))

    for node in after:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            findings.append({"code": "TEST_ASSERTION_ALWAYS_TRUE", "path": path,
                             "line": node.lineno, "detail": "assert True cannot detect a regression"})
        elif isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or) and contains_true(node.test):
            findings.append({"code": "TEST_ASSERTION_ALWAYS_TRUE", "path": path,
                             "line": node.lineno, "detail": "assertion is made unconditional by `or True`"})

    def complexity(node):
        return sum(1 for _ in ast.walk(node))

    if len(after) >= len(before) and sum(complexity(item.test) for item in after) < sum(
            complexity(item.test) for item in before):
        findings.append({"code": "TEST_ASSERTION_SIMPLIFIED", "path": path,
                         "detail": "candidate assertions contain fewer predicate nodes"})
    return {"passed": not findings, "findings": findings,
            "baseline_assertions": len(before), "candidate_assertions": len(after)}

