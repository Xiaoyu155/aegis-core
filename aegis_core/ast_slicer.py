"""Static repository index, mixed-encoding diagnostics and bounded causal slices."""
import ast
import copy
import json
import os
from pathlib import Path

SKIP = {'.git', '.venv', '.worktrees', '__pycache__', 'node_modules', 'dist', 'build', 'vendor'}


def decode(path: Path) -> tuple[str, str]:
    raw = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'gbk'):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError('source is neither valid UTF-8 nor GBK')


def signature(node):
    node = copy.deepcopy(node)
    node.decorator_list = []
    doc = ast.get_docstring(node, clean=False)
    body = [ast.Expr(ast.Constant(doc))] if doc else []
    if isinstance(node, ast.ClassDef):
        node.bases, node.keywords = [], []
        body += [ast.parse(signature(child)).body[0] for child in node.body
                 if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    node.body = body + [ast.Expr(ast.Constant(Ellipsis))]
    return ast.unparse(ast.fix_missing_locations(node))


class ImportGraph:
    def __init__(self, repo_root: Path):
        self.root = Path(repo_root).resolve()
        self.modules, self.paths, self.symbols, self.bindings, self.diagnostics = {}, {}, {}, {}, []
        for directory, names, files in os.walk(self.root, followlinks=False):
            names[:] = sorted(n for n in names if n not in SKIP and not Path(directory, n).is_symlink())
            for name in sorted(files):
                path = Path(directory, name)
                if path.suffix != '.py' or path.is_symlink():
                    continue
                relative = path.relative_to(self.root).as_posix()
                module = relative[:-3].replace('/', '.')
                if module.endswith('.__init__'):
                    module = module[:-9]
                try:
                    text, encoding = decode(path)
                    tree = ast.parse(text, filename=relative)
                except (ValueError, SyntaxError, OSError) as error:
                    self.diagnostics.append({'file': relative, 'error': type(error).__name__})
                    continue
                self.modules[module] = (relative, text, tree)
                self.paths[relative] = module
                self.bindings[module] = {}
                self._index(module, tree.body)
                package = module if name == '__init__.py' else module.rpartition('.')[0]
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.bindings[module][alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
                    elif isinstance(node, ast.ImportFrom):
                        parent = package.split('.') if package else []
                        if node.level:
                            if node.level > len(parent) + 1:
                                continue
                            parent = parent[:len(parent) - node.level + 1]
                            base = '.'.join(parent + ([node.module] if node.module else []))
                        else:
                            base = node.module or ''
                        for alias in node.names:
                            self.bindings[module][alias.asname or alias.name] = '.'.join(filter(None, (base, alias.name)))
        self.calls = {key: set() for key in self.symbols}
        self.types = {key: set() for key in self.symbols}
        for key, node in self.symbols.items():
            module = self.symbol_modules[key]
            local = {}
            args = getattr(node, 'args', None)
            if args:
                for arg in args.posonlyargs + args.args + args.kwonlyargs:
                    local[arg.arg] = ast.unparse(arg.annotation) if arg.annotation else None
            # Executable statements of this scope, excluding nested definitions.
            def walk(item):
                yield item
                for child in ast.iter_child_nodes(item):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    yield from walk(child)
            for child in walk(node):
                if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            local[target.id] = ast.unparse(child.value.func)
                if isinstance(child, ast.Call):
                    resolved = self._resolve(module, key, ast.unparse(child.func), local)
                    if resolved:
                        self.calls[key].add(resolved)
                elif isinstance(child, ast.Name):
                    resolved = self._resolve(module, key, child.id, {})
                    if resolved and isinstance(self.symbols[resolved], ast.ClassDef):
                        self.types[key].add(resolved)
        self.callers = {key: set() for key in self.symbols}
        for caller, callees in self.calls.items():
            for callee in callees:
                self.callers[callee].add(caller)

    def _index(self, module, nodes, prefix=''):
        if not hasattr(self, 'symbol_modules'):
            self.symbol_modules = {}
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                key = module + ':' + name
                self.symbols[key] = node
                self.symbol_modules[key] = module
                self._index(module, node.body, name + '.')

    def _resolve(self, module, scope, name, local, seen=None):
        seen = set() if seen is None else seen
        if (module, name) in seen:
            return None
        seen.add((module, name))
        parts = name.split('.')
        if parts[0] in ('self', 'cls'):
            roten = scope.split(':')[1].rsplit('.', 1)[0]
            name = roten + '.' + '.'.join(parts[1:])
        elif parts[0] in local:
            if not local[parts[0]]:
                return None
            name = '.'.join([local[parts[0]], *parts[1:]])
        candidates = [module + ':' + name]
        bound = self.bindings.get(module, {}).get(name.split('.')[0])
        if bound:
            expanded = '.'.join([bound, *name.split('.')[1:]])
            for mod in sorted(self.modules, key=len, reverse=True):
                if expanded.startswith(mod + '.'):
                    symbol = expanded[len(mod) + 1:]
                    candidates.append(mod + ':' + symbol)
                    if '.' not in symbol and symbol in self.bindings[mod]:
                        resolved = self._resolve(mod, scope, symbol, {}, seen)
                        if resolved:
                            return resolved
        return next((key for key in candidates if key in self.symbols), None)

    def slice(self, target_file, target_symbol=None, *, line=None, token_budget=4000):
        if type(token_budget) is not int or token_budget < 256:
            raise ValueError('budget must be >=256')
        path = (self.root / target_file).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('target outside repository')
        relative = path.relative_to(self.root).as_posix()
        module = self.paths.get(relative)
        keys = [key for key in self.symbols if self.symbol_modules[key] == module]
        if target_symbol:
            key = module + ':' + target_symbol if module else ''
        else:
            matches = [k for k in keys if self.symbols[k].lineno <= (line or 0) <= self.symbols[k].end_lineno]
            key = min(matches, key=lambda k: self.symbols[k].end_lineno - self.symbols[k].lineno) if matches else ''
        if key not in self.symbols:
            raise KeyError('target symbol/line not indexed')
        order, queue, seen = [], [(key, 'target', 0)], set()
        while queue:
            current, relation, distance = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            order.append((current, relation, distance))
            if distance < 3:
                queue += [(k, label, distance + 1) for label, neighbors in
                          [('callee', self.calls[current]), ('caller', self.callers[current]), ('type', self.types[current])]
                          for k in sorted(neighbors)]
        package = {'target': key, 'items': [], 'truncated': False, 'diagnostics': self.diagnostics}
        for current, relation, distance in order:
            mod = self.symbol_modules[current]
            filename, text, _ = self.modules[mod]
            node = self.symbols[current]
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            full = '\n'.join(text.splitlines()[start - 1:node.end_lineno])
            item = {'symbol': current, 'file': filename, 'start_line': start, 'end_line': node.end_lineno,
                    'relation': relation, 'distance': distance, 'source': full if relation == 'target' else signature(node)}
            package['items'].append(item)
            if len(json.dumps(package, ensure_ascii=False).encode()) + 128 > token_budget:
                package['items'].pop()
                item['source'] = signature(node)
                package['items'].append(item)
                if len(json.dumps(package, ensure_ascii=False).encode()) + 128 > token_budget:
                    package['items'].pop()
                package['truncated'] = True
        # Diagnostics are bounded too.
        while len(json.dumps(package).encode()) + 80 > token_budget and package['diagnostics']:
            package['diagnostics'].pop()
        package['budget_unit'] = 'utf8_bytes_upper_bound'
        package['token_upper_bound'] = len(json.dumps(package).encode()) + 32
        return package
