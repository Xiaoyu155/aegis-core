"""Static repository index, mixed-encoding diagnostics and bounded causal slices."""
import ast
import copy
import json
import os
from pathlib import Path
import posixpath
import re


JS_SUFFIXES = {'.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts'}


def is_protected_test_path(path: str) -> bool:
    """Return whether a repository-relative path is a test or gate file."""
    normalized = str(path).replace('\\', '/').lower()
    parts = normalized.split('/')
    name = parts[-1]
    return (any(part in {'tests', 'test', '__tests__'} for part in parts)
            or name.startswith(('test_', 'check_', 'verify_', 'validate_'))
            or name.endswith(tuple(f'_test.{extension}' for extension in
                                    ('py', 'js', 'jsx', 'mjs', 'cjs', 'ts', 'tsx', 'mts', 'cts')))
            or name.endswith(('.test.py', '.spec.py', '.test.js', '.test.jsx', '.test.mjs',
                              '.test.cjs', '.test.ts', '.test.tsx', '.test.mts', '.test.cts',
                              '.spec.js', '.spec.jsx', '.spec.ts', '.spec.tsx', '.spec.mjs',
                              '.spec.cjs', '.spec.mts', '.spec.cts')))


def _mask_js_comments_and_strings(source: str) -> str:
    """Mask comments and literals while preserving offsets and newlines."""
    out = list(source)
    i, state = 0, 'code'
    quote = ''
    while i < len(source):
        char = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ''
        if state == 'code':
            if char == '/' and nxt == '/':
                state = 'line'; out[i] = out[i + 1] = ' '; i += 2; continue
            if char == '/' and nxt == '*':
                state = 'block'; out[i] = out[i + 1] = ' '; i += 2; continue
            if char in ('"', "'", '`'):
                quote = char; state = 'string'; out[i] = ' '
        elif state == 'line':
            if char == '\n': state = 'code'
            else: out[i] = ' '
        elif state == 'block':
            if char == '*' and nxt == '/':
                out[i] = out[i + 1] = ' '; i += 2; state = 'code'; continue
            if char != '\n': out[i] = ' '
        else:
            if char == '\\':
                if char != '\n': out[i] = ' '
                if i + 1 < len(source) and source[i + 1] != '\n': out[i + 1] = ' '
                i += 2; continue
            if char == quote: state = 'code'
            if char != '\n': out[i] = ' '
        i += 1
    return ''.join(out)


def _brace_end(masked: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == '{': depth += 1
        elif masked[index] == '}':
            depth -= 1
            if depth == 0: return index + 1
    return len(masked)


def _js_symbols(source: str):
    """Extract named JS/TS functions, classes and class methods without executing code."""
    masked = _mask_js_comments_and_strings(source)
    patterns = [
        re.compile(r'\b(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)[^{}]*\{'),
        re.compile(r'\b(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)[^{}]*\{'),
        re.compile(r'\b(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:<[^>]+>\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::\s*[A-Za-z_$][\w$<>,.\[\] ]*)?=>\s*\{'),
    ]
    found = {}
    for pattern in patterns:
        for match in pattern.finditer(masked):
            name = match.group(1)
            opening = masked.find('{', match.start(), match.end())
            end = _brace_end(masked, opening)
            found.setdefault(name, (match.start(), opening, end))
    for class_match in re.finditer(r'\bclass\s+([A-Za-z_$][\w$]*)[^{}]*\{', masked):
        class_name = class_match.group(1)
        opening = masked.find('{', class_match.start(), class_match.end())
        end = _brace_end(masked, opening)
        body = masked[opening + 1:end - 1]
        offset = opening + 1
        for method in re.finditer(r'(?m)(?:^|[;{}\n])\s*(?:static\s+|async\s+|get\s+|set\s+)*([A-Za-z_$][\w$]*)(?:<[^>{}]+>)?\s*\([^;{}]*\)\s*(?::\s*[^;{}]+)?\{', body):
            method_name = method.group(1)
            method_open = offset + body.find('{', method.start(), method.end())
            leading = body[method.start():method.start(1)]
            modifiers_start = offset + method.start() + len(leading) - len(leading.lstrip(';{} \t\n'))
            found.setdefault(class_name + '.' + method_name,
                             (modifiers_start, method_open, _brace_end(masked, method_open)))
    lines = source.splitlines()
    result = {}
    for name, (start, opening, end) in sorted(found.items(), key=lambda item: item[1][0]):
        start_line = source.count('\n', 0, start) + 1
        end_line = source.count('\n', 0, end) + 1
        result[name] = {'name': name, 'start_line': start_line, 'end_line': end_line,
                        'source': '\n'.join(lines[start_line - 1:end_line]),
                        'signature': source[start:opening].strip()}
    return result

SKIP = {'.git', '.venv', '.worktrees', '__pycache__', 'node_modules', 'dist', 'build', 'vendor',
        '.next', 'coverage', 'out', 'target'}
MAX_INDEX_BYTES = 512 * 1024


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
        self.js_modules, self.js_symbols, self.file_dependencies = {}, {}, {}
        for directory, names, files in os.walk(self.root, followlinks=False):
            names[:] = sorted(n for n in names if n not in SKIP and not Path(directory, n).is_symlink())
            for name in sorted(files):
                path = Path(directory, name)
                if path.suffix != '.py' or path.is_symlink():
                    continue
                relative = path.relative_to(self.root).as_posix()
                if path.stat().st_size > MAX_INDEX_BYTES:
                    self.diagnostics.append({'file': relative, 'error': 'SourceTooLarge'})
                    continue
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
        for directory, names, files in os.walk(self.root, followlinks=False):
            names[:] = sorted(n for n in names if n not in SKIP and not Path(directory, n).is_symlink())
            for name in sorted(files):
                path = Path(directory, name)
                if path.is_symlink():
                    continue
                relative = path.relative_to(self.root).as_posix()
                if path.suffix.lower() in JS_SUFFIXES:
                    if path.stat().st_size > MAX_INDEX_BYTES:
                        self.diagnostics.append({'file': relative, 'error': 'SourceTooLarge'})
                        continue
                    try:
                        text, _ = decode(path)
                    except (ValueError, OSError) as error:
                        self.diagnostics.append({'file': relative, 'error': type(error).__name__})
                        continue
                    symbols = _js_symbols(text)
                    self.js_modules[relative] = (text, symbols)
                    self.js_symbols.update({relative + ':' + key: {"file": relative, **value}
                                            for key, value in symbols.items()})
        # Resolve relative module specifiers after every JS file is indexed.
        js_paths = set(self.js_modules)
        for relative, (source, _) in self.js_modules.items():
            dependencies = set()
            masked = _mask_js_comments_and_strings(source)
            for match in re.finditer(r'\b(?:from\s*|import\s*\(?\s*|require\s*\()\s*[\'\"]([^\'\"]+)', source):
                specifier = match[1]
                if masked[match.start()].strip() and specifier.startswith('.'):
                    base = posixpath.normpath(posixpath.join(posixpath.dirname(relative), specifier))
                    candidates = [base, *(base + suffix for suffix in JS_SUFFIXES),
                                  *(base + '/index' + suffix for suffix in JS_SUFFIXES)]
                    dependencies.update(path for path in candidates if path in js_paths and path != relative)
            self.file_dependencies[relative] = sorted(dependencies)
        for module, (relative, _, _) in self.modules.items():
            dependencies = set()
            for bound in self.bindings.get(module, {}).values():
                requested_module = bound if bound in self.modules else bound.rsplit('.', 1)[0] if '.' in bound else bound
                matches = []
                for candidate in self.modules:
                    if (requested_module == candidate or requested_module.startswith(candidate + '.')
                            or candidate.endswith('.' + requested_module)
                            or candidate.rsplit('.', 1)[-1] == requested_module):
                        matches.append(candidate)
                if matches:
                    origin = Path(relative).parent
                    candidate = min(matches, key=lambda name: (
                        not (requested_module == name or requested_module.startswith(name + '.')),
                        -len(name) if requested_module.startswith(name + '.') else 0,
                        len(set((self.modules[name][0].split('/')[:-1])) ^ set(origin.parts)),
                        len(self.modules[name][0]), name))
                    if candidate != module:
                        dependencies.add(self.modules[candidate][0])
            self.file_dependencies[relative] = sorted(dependencies)
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
        if relative in self.js_modules:
            text, symbols = self.js_modules[relative]
            if target_symbol:
                name = target_symbol
                if name not in symbols:
                    raise KeyError('target symbol/line not indexed')
            else:
                matches = [name for name, item in symbols.items()
                           if item['start_line'] <= (line or 0) <= item['end_line']]
                if not matches:
                    raise KeyError('target symbol/line not indexed')
                name = min(matches, key=lambda key: symbols[key]['end_line'] - symbols[key]['start_line'])
            key = relative + ':' + name
            item = symbols[name]
            package = {'target': key, 'items': [{'symbol': key, 'file': relative,
                        'start_line': item['start_line'], 'end_line': item['end_line'],
                        'relation': 'target', 'distance': 0, 'source': item['source']}],
                       'truncated': False, 'diagnostics': self.diagnostics}
            if len(json.dumps(package, ensure_ascii=False).encode()) + 32 > token_budget:
                source = item['source']
                package['items'][0]['source'] = source[:max(80, token_budget - 256)]
                package['truncated'] = True
            package['budget_unit'] = 'utf8_bytes_upper_bound'
            package['token_upper_bound'] = len(json.dumps(package, ensure_ascii=False).encode()) + 32
            return package
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
