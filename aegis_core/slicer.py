"""Static, deterministic, same-directory causal slicing. Never executes source."""
import ast
import copy
from pathlib import Path

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _stub(node):
    node = copy.deepcopy(node)
    doc = ast.get_docstring(node, clean=False)
    body = [ast.Expr(value=ast.Constant(doc))] if doc is not None else []
    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, (*_FUNCTIONS, ast.ClassDef)):
                body.append(ast.parse(_stub(child)).body[0])
            elif isinstance(child, ast.AnnAssign):
                child.value = None
                body.append(child)
    node.body = body + [ast.Expr(value=ast.Constant(Ellipsis))]
    return ast.unparse(ast.fix_missing_locations(node))


def _calls(node):
    """Walk executable scope, excluding nested definition bodies."""
    class Calls(ast.NodeVisitor):
        def __init__(self):
            self.calls = []
        def visit_Call(self, item):
            self.calls.append(item)
            self.generic_visit(item)
        def visit_FunctionDef(self, item):
            if item is node:
                for stmt in item.body:
                    self.visit(stmt)
        visit_AsyncFunctionDef = visit_FunctionDef
        def visit_ClassDef(self, item):
            pass
    visitor = Calls()
    visitor.visit(node)
    return visitor.calls


def extract_causal_slice(repo_root: Path, target_file: str, target_symbol: str) -> dict:
    root = Path(repo_root).resolve()
    target_path = (root / target_file).resolve()
    if not target_path.is_relative_to(root):
        raise ValueError("target escapes repository")
    modules = {}
    for path in sorted(target_path.parent.glob("*.py")):
        if not path.resolve().is_relative_to(root):
            raise ValueError("module symlink escapes repository")
        source = path.read_text(encoding="utf-8")
        modules[path.stem] = (path, source, ast.parse(source, filename=str(path)))
    module = target_path.stem
    if module not in modules:
        raise FileNotFoundError(target_path)
    symbols, bindings, scopes = {}, {}, {}
    for mod, (_, _, tree) in modules.items():
        bindings[mod] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.level <= 1 and node.module in modules:
                    for alias in node.names:
                        bindings[mod][alias.asname or alias.name] = (node.module, alias.name)
                elif node.level == 1 and node.module is None:
                    for alias in node.names:
                        if alias.name in modules:
                            bindings[mod][alias.asname or alias.name] = (alias.name, "")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        bindings[mod][alias.asname or alias.name] = (alias.name, "")
            elif isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
                symbols[(mod, node.name)] = node
                if isinstance(node, _FUNCTIONS):
                    scopes[(mod, node.name)] = node
                else:
                    for child in node.body:
                        if isinstance(child, _FUNCTIONS):
                            symbols[(mod, node.name + "." + child.name)] = child
                            scopes[(mod, node.name + "." + child.name)] = child
        scopes[(mod, "<module>")] = tree
    target_key = (module, target_symbol)
    if target_key not in symbols:
        raise KeyError(target_symbol)

    def shadowed(mod, scope, name):
        node = scopes.get((mod, scope))
        if not isinstance(node, _FUNCTIONS):
            return False
        arguments = node.args
        bound = {arg.arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
        if arguments.vararg:
            bound.add(arguments.vararg.arg)
        if arguments.kwarg:
            bound.add(arguments.kwarg.arg)
        # Conservatively treat a function-local binding as unresolved, including
        # assignments below a call (Python's lexical local-name semantics).
        class Bindings(ast.NodeVisitor):
            def visit_Name(self, item):
                if isinstance(item.ctx, ast.Store):
                    bound.add(item.id)
            def visit_FunctionDef(self, item):
                bound.add(item.name)
            visit_AsyncFunctionDef = visit_FunctionDef
            def visit_ClassDef(self, item):
                bound.add(item.name)
            def visit_Import(self, item):
                bound.update(alias.asname or alias.name.split('.')[0] for alias in item.names)
            def visit_ImportFrom(self, item):
                bound.update(alias.asname or alias.name for alias in item.names)
            def visit_ExceptHandler(self, item):
                if item.name:
                    bound.add(item.name)
                self.generic_visit(item)
        visitor = Bindings()
        for statement in node.body:
            visitor.visit(statement)
        return name in bound

    def resolve(mod, scope, expr):
        if isinstance(expr, ast.Name):
            if shadowed(mod, scope, expr.id):
                return None
            key = bindings[mod].get(expr.id, (mod, expr.id))
            return key if key in symbols else None
        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value, ast.Name):
                name = expr.value.id
                if name in {"self", "cls"} and "." in scope:
                    key = (mod, scope.rsplit(".", 1)[0] + "." + expr.attr)
                elif shadowed(mod, scope, name):
                    return None
                elif name in bindings[mod]:
                    dest_mod, prefix = bindings[mod][name]
                    key = (dest_mod, prefix + "." + expr.attr if prefix else expr.attr)
                else:
                    key = (mod, name + "." + expr.attr)
                return key if key in symbols else None
            if isinstance(expr.value, ast.Call):
                cls = resolve(mod, scope, expr.value.func)
                if cls:
                    key = (cls[0], cls[1] + "." + expr.attr)
                    return key if key in symbols else None
        return None

    target = symbols[target_key]
    dependencies, unresolved = {}, set()
    for call in _calls(target):
        key = resolve(module, target_symbol, call.func)
        if key is None:
            unresolved.add(ast.unparse(call.func))
        elif key != target_key:
            path = modules[key[0]][0].relative_to(root).as_posix()
            dependencies[key] = {"file": path, "symbol": key[1], "stub": _stub(symbols[key])}
    upstream = []
    for (mod, scope), node in sorted(scopes.items()):
        if (mod, scope) == target_key:
            continue
        for call in _calls(node):
            if resolve(mod, scope, call.func) == target_key:
                upstream.append({"file": modules[mod][0].relative_to(root).as_posix(),
                                 "caller": scope, "line": call.lineno,
                                 "call": ast.unparse(call)})
    _, source, _ = modules[module]
    start = min([target.lineno] + [d.lineno for d in target.decorator_list])
    full_source = "\n".join(source.splitlines()[start - 1:target.end_lineno])
    downstream = [dependencies[k] for k in sorted(dependencies)]
    total = sum(len(s.encode()) for _, s, _ in modules.values())
    retained = len(full_source.encode()) + sum(len(d["stub"].encode()) for d in downstream)
    return {"target": {"file": target_path.relative_to(root).as_posix(),
                       "symbol": target_symbol, "source": full_source},
            "downstream": downstream, "upstream": upstream,
            "unresolved_calls": sorted(unresolved),
            "compression_ratio": 1 - retained / total if total else 0.0}
