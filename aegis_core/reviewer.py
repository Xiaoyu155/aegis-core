"""Independent, differential static and semantic checks for L3 candidates."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .ast_slicer import JS_SUFFIXES, _mask_js_comments_and_strings, decode
from .sandbox import run_git

_SECRET = re.compile(r"(?:^|_)(?:api_?key|access_?token|secret(?:_?key)?|password|private_?key)(?:_|$)", re.I)
_CONTROL = re.compile(r"(?:risk|guard|limit|authoriz|validate|circuit|permission|quota|check)", re.I)
_GROW = {"append", "extend", "insert", "add", "update", "setdefault"}
_SHRINK = {"pop", "popleft", "remove", "discard", "clear", "popitem"}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    symbol: str
    line: int
    detail: str

    def fingerprint(self):
        return (self.code, self.symbol, self.detail)


def _bindings(node):
    if isinstance(node, ast.Assign):
        return node.targets, node.value
    if isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
        return [node.target], node.value
    return [], None


def _exit_suite(nodes):
    return bool(nodes) and isinstance(nodes[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))


def _nonzero_test(test):
    """Return the expression verified by a simple zero check and which branch is safe."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return ast.unparse(test.operand), False
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, right = test.left, test.comparators[0]
        if isinstance(left, ast.Constant) and left.value == 0:
            left, right = right, left
        if isinstance(right, ast.Constant) and right.value == 0:
            if isinstance(test.ops[0], (ast.NotEq, ast.Gt, ast.Lt)):
                return ast.unparse(left), True
            if isinstance(test.ops[0], ast.Eq):
                return ast.unparse(left), False
    if isinstance(test, (ast.Name, ast.Attribute, ast.Call)):
        return ast.unparse(test), True
    return None, False


def _walk_scope(node):
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, _SCOPES):
            yield child
            yield from _walk_scope(child)


def _bounded_deque(value):
    return (isinstance(value, ast.Call) and ast.unparse(value.func).split(".")[-1] == "deque"
            and any(keyword.arg == "maxlen" and isinstance(keyword.value, ast.Constant)
                    and type(keyword.value.value) is int and keyword.value.value > 0 for keyword in value.keywords))


def _fixed_truth(expression):
    if isinstance(expression, ast.Constant):
        return bool(expression.value)
    if isinstance(expression, ast.BoolOp):
        values = [_fixed_truth(value) for value in expression.values]
        if isinstance(expression.op, ast.And) and False in values:
            return False
        if isinstance(expression.op, ast.Or) and True in values:
            return True
    return None


class _PythonAudit(ast.NodeVisitor):
    def __init__(self, path, module):
        self.path = path
        self.findings = []
        self.controls = {}
        self.scope = []
        self.known_bounded = set()
        self.module_storage = {ast.unparse(target) for node in module.body for target in _bindings(node)[0]}
        self.module_bounded = {ast.unparse(target) for node in module.body for targets, value in [_bindings(node)]
                               for target in targets if _bounded_deque(value)}
        self.storage = set(self.module_storage)
        self.shrink = set(self.module_bounded)
        self.nonzero = set()
        self.catches_zero = False

    def report(self, code, node, detail):
        self.findings.append(Finding(code, self.path, ".".join(self.scope) or "<module>", node.lineno, detail))

    def visit_ClassDef(self, node):
        self.scope.append(node.name)
        previous = self.known_bounded
        constructors = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"]
        self.known_bounded = {ast.unparse(target) for constructor in constructors for item in constructor.body
                              for targets, value in [_bindings(item)] for target in targets if _bounded_deque(value)}
        self.generic_visit(node)
        self.known_bounded = previous
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        previous = self.storage, self.shrink, self.nonzero, self.catches_zero
        items = list(_walk_scope(node))
        bindings = [(ast.unparse(target), value) for item in items
                    for targets, value in [_bindings(item)] for target in targets]
        local = {name for name, _ in bindings}
        globals_declared = {name for item in items if isinstance(item, (ast.Global, ast.Nonlocal)) for name in item.names}
        bounded = {name for name, value in bindings if _bounded_deque(value)}
        self.storage = (self.module_storage - local) | globals_declared
        self.shrink = ((self.known_bounded | self.module_bounded) - local) | bounded
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                function = statement.value.func
                if isinstance(function, ast.Attribute) and function.attr in {"clear"}:
                    self.shrink.add(ast.unparse(function.value))
            if not isinstance(statement, ast.If) or not isinstance(statement.test, ast.Compare):
                continue
            test = statement.test
            if (isinstance(test.left, ast.Call) and ast.unparse(test.left.func) == "len" and len(test.left.args) == 1
                    and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Gt, ast.GtE))):
                receiver = ast.unparse(test.left.args[0])
                removals = {ast.unparse(item.value.func.value) for item in statement.body
                            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                            and isinstance(item.value.func, ast.Attribute) and item.value.func.attr in _SHRINK}
                if receiver in removals:
                    self.shrink.add(receiver)
        self.nonzero, self.catches_zero = set(), False
        self.visit_suite(node.body)
        if _CONTROL.search(node.name):
            conditions = set()
            for statement in node.body:
                candidates = [statement, *_walk_scope(statement)]
                for item in candidates:
                    if isinstance(item, (ast.If, ast.IfExp, ast.Assert)) and _fixed_truth(item.test) is None:
                        conditions.add(ast.unparse(item.test))
                    if isinstance(item, ast.Return) and isinstance(item.value, (ast.Compare, ast.BoolOp)):
                        if _fixed_truth(item.value) is None:
                            conditions.add(ast.unparse(item.value))
                if isinstance(statement, (ast.Return, ast.Raise)):
                    break
            self.controls[".".join(self.scope)] = (node.lineno, conditions)
        self.storage, self.shrink, self.nonzero, self.catches_zero = previous
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_suite(self, statements):
        previous = self.nonzero.copy()
        for node in statements:
            self.visit(node)
            if isinstance(node, (ast.If, ast.Assert)):
                expression, positive = _nonzero_test(node.test)
                if expression and (isinstance(node, ast.Assert) and positive or
                                   isinstance(node, ast.If) and
                                   (_exit_suite(node.body) and not positive or _exit_suite(node.orelse) and positive)):
                    self.nonzero.add(expression)
        self.nonzero = previous

    def visit_If(self, node):
        self.visit(node.test)
        before = self.nonzero.copy()
        expression, positive = _nonzero_test(node.test)
        for safe, statements in ((positive, node.body), (not positive, node.orelse)):
            self.nonzero = before.copy()
            if safe and expression:
                self.nonzero.add(expression)
            self.visit_suite(statements)
        self.nonzero = before

    def visit_Try(self, node):
        before = self.catches_zero
        self.catches_zero = before or any(handler.type is not None and
            any(isinstance(item, ast.Name) and item.id in {"ZeroDivisionError", "ArithmeticError"}
                for item in ast.walk(handler.type)) for handler in node.handlers)
        self.visit_suite(node.body)
        self.catches_zero = before
        for handler in node.handlers:
            self.visit(handler)
        self.visit_suite(node.orelse)
        self.visit_suite(node.finalbody)

    visit_TryStar = visit_Try

    def visit_ExceptHandler(self, node):
        broad = node.type is None or ast.unparse(node.type) in {"Exception", "BaseException"}
        silent = all(isinstance(item, (ast.Pass, ast.Continue, ast.Break)) or
                     isinstance(item, ast.Return) and (item.value is None or isinstance(item.value, ast.Constant))
                     for item in node.body)
        if silent and (broad or all(isinstance(item, ast.Pass) for item in node.body)):
            self.report("EXCEPTION_SWALLOWED", node, "Exception handler silently discards failures")
        self.visit_suite(node.body)

    def visit_BinOp(self, node):
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            denominator = ast.unparse(node.right)
            try:
                value = ast.literal_eval(node.right)
            except (ValueError, TypeError):
                value = None
            if type(value) in {int, float, complex} and value == 0:
                self.report("DIVISION_BY_ZERO", node, "Literal zero divisor")
            could_be_zero = isinstance(node.right, (ast.Name, ast.Attribute)) or (
                isinstance(node.right, ast.Call) and ast.unparse(node.right.func) in {"len", "sum"})
            if (not self.catches_zero and denominator not in self.nonzero
                    and could_be_zero and not denominator.isupper()):
                self.report("UNGUARDED_DIVISOR", node, "Check zero before dividing by " + denominator)
        self.generic_visit(node)

    def visit_Call(self, node):
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in _GROW:
            receiver = ast.unparse(function.value)
            if (receiver in self.storage or receiver.startswith(("self.", "cls."))) and receiver not in self.shrink:
                self.report("UNBOUNDED_GROWTH", node, "Persistent collection " + receiver + " grows without a visible bound")
        for keyword in node.keywords:
            self._check_secret(keyword.arg or "", keyword.value, keyword)
        self.generic_visit(node)

    def _check_secret(self, name, value, node):
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        if (isinstance(value, ast.Constant) and isinstance(value.value, (str, bytes)) and value.value
                and _SECRET.search(name) and not name.endswith(("_env", "_name", "_field", "_ref", "_path", "_var"))):
            self.report("HARDCODED_SECRET", node, "Credential field contains a literal value")

    def visit_Assign(self, node):
        targets, value = _bindings(node)
        for target in targets:
            self._check_secret(ast.unparse(target).rsplit(".", 1)[-1], value, node)
        self.generic_visit(node)

    visit_AnnAssign = visit_Assign
    visit_NamedExpr = visit_Assign

    def visit_Dict(self, node):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                self._check_secret(key.value, value, key)
        self.generic_visit(node)


def _scan_python(source, path):
    tree = ast.parse(source)
    audit = _PythonAudit(path, tree)
    audit.visit_suite(tree.body)
    return audit.findings, audit.controls


def _scan_js(source, path):
    masked = _mask_js_comments_and_strings(source)
    findings = []
    patterns = (
        ("DIVISION_BY_ZERO", r"(?:/|%)\s*0\b(?!\.)", "Literal zero divisor"),
        ("EXCEPTION_SWALLOWED", r"\bcatch\s*(?:\([^)]*\))?\s*\{\s*(?:return\s*;)?\s*\}", "Empty catch discards failures"),
        ("UNBOUNDED_GROWTH", r"\bthis\.([\w$]+)\.(?:push|add|set)\s*\(", "Persistent receiver may grow without a bound"),
    )
    for code, pattern, detail in patterns:
        for match in re.finditer(pattern, masked):
            if code == "UNBOUNDED_GROWTH" and re.search(
                    rf"\bthis\.{re.escape(match[1])}\.(?:pop|shift|splice|delete|clear)\s*\(", masked):
                continue
            symbol = match[1] if match.lastindex else "<module>"
            findings.append(Finding(code, path, symbol, masked.count("\n", 0, match.start()) + 1, detail))
    for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*[:=]\s*(['\"])([^'\"]+)\2", source):
        if masked[match.start():match.start() + len(match[1])].strip() and _SECRET.search(match[1]):
            findings.append(Finding("HARDCODED_SECRET", path, "<module>", source.count("\n", 0, match.start()) + 1,
                                    "Credential field contains a literal value"))
    return findings, {}


def audit_candidate(root: Path, baseline: str, changed_files):
    """Review candidate files with baseline source only. Never consume author rationale or execute the patch."""
    findings = []
    audited = []
    for relative in sorted(set(changed_files)):
        path = root / relative
        suffix = path.suffix.lower()
        scanner = _scan_python if suffix == ".py" else _scan_js if suffix in JS_SUFFIXES else None
        if scanner is None:
            findings.append(Finding("AUDIT_UNSUPPORTED_LANGUAGE", relative, "<module>", 1,
                                    "No independent static reviewer is configured for this file type"))
            continue
        audited.append(relative)
        try:
            prior = run_git(root, "cat-file", "blob", f"{baseline}:{relative}", check=False)
            source_before = ""
            if prior.returncode == 0:
                try:
                    source_before = prior.stdout.decode("utf-8-sig")
                except UnicodeDecodeError:
                    source_before = prior.stdout.decode("gbk")
            source_after = decode(path)[0] if path.is_file() and not path.is_symlink() else ""
            earlier, controls_before = scanner(source_before, relative)
            later, controls_after = scanner(source_after, relative)
        except (OSError, ValueError, SyntaxError) as error:
            findings.append(Finding("AUDIT_SOURCE_UNREADABLE", relative, "<module>", 1, type(error).__name__))
            continue
        previous = Counter(item.fingerprint() for item in earlier)
        for item in later:
            fingerprint = item.fingerprint()
            if previous[fingerprint]:
                previous[fingerprint] -= 1
            else:
                findings.append(item)
        for symbol, (line, predicates) in controls_before.items():
            if predicates and not controls_after.get(symbol, (line, set()))[1]:
                findings.append(Finding("CONTROL_GUARD_REMOVED", relative, symbol, line,
                                        "All conditional checks disappeared from a policy-related function"))
    ordered = sorted(findings, key=lambda finding: (finding.path, finding.line, finding.code))
    return {"reviewer": "independent_static_semantic", "passed": not ordered,
            "audited_files": audited, "findings": [asdict(item) for item in ordered]}
