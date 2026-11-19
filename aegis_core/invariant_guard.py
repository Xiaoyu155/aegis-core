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


def inspect_source(source: str, path: str = "<source>") -> list[dict]:
    """Return deterministic domain findings for one Python source file."""
    if type(source) is not str or type(path) is not str:
        raise TypeError("source and path must be text")
    try:
        findings = _Guard(source, Path(path).as_posix()).run()
    except SyntaxError as error:
        return [asdict(InvariantFinding("SOURCE_SYNTAX_ERROR", "source", Path(path).as_posix(),
                                        "<module>", error.lineno or 1, "high", "Source cannot be parsed"))]
    return [asdict(item) for item in findings]


def scan_changed_files(root: str | Path, changed_files, *, baseline: str | None = None) -> dict:
    """Find new invariant violations in changed Python files, relative to baseline."""
    root = Path(root).resolve()
    findings = []
    scanned = []
    from .sandbox import run_git

    for relative in sorted(set(changed_files)):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink() or path.suffix != ".py":
            continue
        try:
            after = path.read_text(encoding="utf-8-sig")
            earlier = ""
            if baseline:
                result = run_git(root, "cat-file", "blob", f"{baseline}:{Path(relative).as_posix()}", check=False)
                if result.returncode == 0:
                    earlier = result.stdout.decode("utf-8-sig", errors="replace")
            before = inspect_source(earlier, relative)
            after_findings = inspect_source(after, relative)
        except (OSError, ValueError):
            continue
        previous = {(item["code"], item["symbol"], item["detail"]) for item in before}
        findings.extend(item for item in after_findings
                        if (item["code"], item["symbol"], item["detail"]) not in previous)
        scanned.append(Path(relative).as_posix())
    findings.sort(key=lambda item: (item["path"], item["line"], item["code"]))
    return {"passed": not findings, "scanned_files": scanned, "findings": findings}

