"""Symbol-level impact graphs and deterministic context compression.

The index is static: source is parsed or lexed, never imported or executed.  Python
edges reuse the repository's AST resolver; JavaScript, TypeScript and inline HTML
scripts use bounded lexical resolution through local imports and script references.
"""
from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .ast_slicer import ImportGraph, _mask_js_comments_and_strings
from .frontend import HTML_SUFFIXES


_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CALL = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)(?:\s*\.\s*([A-Za-z_$][\w$]*))?\s*\(")
_DYNAMIC_CALL = re.compile(r"\[\s*([^\]\n]+?)\s*\]\s*\(")
_IMPORT_FROM = re.compile(r"\bimport\s+(?:type\s+)?([^;\n]+?)\s+from\s*['\"]([^'\"]+)['\"]")
_REQUIRE = re.compile(r"\b(?:const|let|var)\s+([^=;]+?)\s*=\s*require\(\s*['\"]([^'\"]+)['\"]\s*\)")
_RESERVED_CALLS = {"if", "for", "while", "switch", "catch", "function", "with", "return",
                   "typeof", "new", "super", "import", "require", "assert", "test", "describe"}


def context_token_count(text: str) -> int:
    """Count stable lexical tokens used by the budget contract."""
    if type(text) is not str:
        raise TypeError("context must be text")
    return len(_TOKEN.findall(text))


class ContextBudgetError(ValueError):
    def __init__(self, required_tokens: int, budget_tokens: int):
        self.required_tokens = required_tokens
        self.budget_tokens = budget_tokens
        super().__init__(f"complete changed-file context needs {required_tokens} tokens; budget is {budget_tokens}")


@dataclass(frozen=True)
class _Symbol:
    key: str
    file: str
    name: str
    source: str
    signature: str
    summary: str
    language: str


class ImpactAnalyzer:
    """Build a bidirectional symbol graph for a repository snapshot."""

    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root).resolve()
        self.index = ImportGraph(self.root)
        self.symbols: dict[str, _Symbol] = {}
        self.callees: dict[str, set[str]] = {}
        self.callers: dict[str, set[str]] = {}
        self.references: dict[str, set[str]] = {}
        self.unresolved: dict[str, set[str]] = {}
        self._js_names: dict[str, set[str]] = {}
        self._js_imports: dict[str, dict[str, tuple[str, str]]] = {}
        self._add_python()
        self._add_javascript()
        self._finish_edges()

    def _add_python(self) -> None:
        for key, node in self.index.symbols.items():
            module = self.index.symbol_modules[key]
            filename, text, _ = self.index.modules[module]
            start = min([node.lineno] + [item.lineno for item in node.decorator_list])
            source = "\n".join(text.splitlines()[start - 1:node.end_lineno])
            name = key.split(":", 1)[-1]
            self.symbols[key] = _Symbol(key, filename, name, source, ast.unparse(_signature_node(node)),
                                        _python_summary(node), "python")
            self.callees[key] = set(self.index.calls.get(key, ()))
            self.references[key] = set(self.index.types.get(key, ())) | self.callees[key]
            self.unresolved[key] = _python_dynamic_references(node)

    def _add_javascript(self) -> None:
        for filename, (full_source, items) in self.index.js_modules.items():
            self._js_names[filename] = set()
            self._js_imports[filename] = self._imports_for(filename)
            for local_name, item in items.items():
                key = filename + ":" + local_name
                name = local_name
                source = item["source"]
                signature = item.get("signature", name)
                summary = _js_summary(name, signature, source)
                self.symbols[key] = _Symbol(key, filename, name, source, signature, summary, "javascript")
                self._js_names[filename].add(local_name)
                self.callees.setdefault(key, set())
                self.references.setdefault(key, set())
        for filename, (_, items) in self.index.js_modules.items():
            for local_name, item in items.items():
                key = filename + ":" + local_name
                name = local_name
                source = item["source"]
                unresolved = set()
                masked = _mask_js_comments_and_strings(source)
                for dynamic in _DYNAMIC_CALL.finditer(masked):
                    unresolved.add("[" + dynamic.group(1).strip() + "](...)")
                for match in _CALL.finditer(masked):
                    first, second = match.group(1), match.group(2)
                    callee_text = f"{first}.{second}" if second else first
                    if first in _RESERVED_CALLS:
                        continue
                    prefix = masked[max(0, match.start() - 32):match.start()].rstrip()
                    if re.search(r"(?:\bfunction|\bclass|\bnew)\s*$", prefix):
                        continue
                    resolved = self._resolve_js(filename, name, first, second)
                    if resolved:
                        self.callees[key].add(resolved)
                        self.references[key].add(resolved)
                    elif not second or first in {"this", "self"}:
                        # Built-ins are not unresolved edges; unresolved application names are.
                        if first not in {"setTimeout", "setInterval", "Promise", "Error", "Map", "Set",
                                         "Array", "Object", "String", "Number", "Boolean", "JSON",
                                         "Math", "Date", "RegExp", "console", "fetch", "URL", "WeakMap"}:
                            unresolved.add(callee_text + "(...)")
                imports = self._js_imports[filename]
                for local, (_, original) in imports.items():
                    if re.search(rf"(?<![\w$]){re.escape(local)}(?![\w$])", masked):
                        target = self._resolve_js(filename, name, local, None)
                        if target:
                            self.references[key].add(target)
                self.unresolved[key] = unresolved

    def _imports_for(self, filename: str) -> dict[str, tuple[str, str]]:
        source, _ = self.index.js_modules[filename]
        scripts = [source]
        if filename in self.index.frontend_documents:
            scripts = [block.source for block in self.index.frontend_documents[filename][0]
                       if block.language == "javascript"]
        bindings: dict[str, tuple[str, str]] = {}
        for script in scripts:
            masked = _mask_js_comments_and_strings(script)
            for match in _IMPORT_FROM.finditer(script):
                specifier = match.group(2)
                if not specifier.startswith((".", "/")):
                    continue
                destination = _resolve_local_file(filename, specifier, self.index.file_dependencies)
                if destination is None:
                    continue
                clause = match.group(1).strip()
                default = re.match(r"([A-Za-z_$][\w$]*)", clause)
                if default and default.group(1) not in {"type", "as"}:
                    bindings[default.group(1)] = (destination, "default")
                named = re.search(r"\{([^}]+)\}", clause)
                if named:
                    for part in named.group(1).split(","):
                        found = re.fullmatch(r"\s*(?:type\s+)?([\w$]+)(?:\s+as\s+([\w$]+))?\s*", part)
                        if found:
                            bindings[found.group(2) or found.group(1)] = (destination, found.group(1))
                namespace = re.search(r"\*\s+as\s+([\w$]+)", clause)
                if namespace:
                    bindings[namespace.group(1)] = (destination, "*")
            for match in _REQUIRE.finditer(script):
                destination = _resolve_local_file(filename, match.group(2), self.index.file_dependencies)
                if destination is None:
                    continue
                variable = match.group(1).strip()
                destructure = re.fullmatch(r"\{([^}]+)\}", variable)
                if destructure:
                    for part in destructure.group(1).split(","):
                        found = re.fullmatch(r"\s*([\w$]+)(?:\s*:\s*([\w$]+))?\s*", part)
                        if found:
                            bindings[found.group(2) or found.group(1)] = (destination, found.group(1))
                elif re.fullmatch(r"[\w$]+", variable):
                    bindings[variable] = (destination, "*")
        return bindings

    def _resolve_js(self, filename: str, owner: str, first: str, second: str | None) -> str | None:
        items = self.index.js_modules.get(filename, ("", {}))[1]
        local_name = f"{first}.{second}" if second else first
        if first in {"this", "self"} and second:
            class_name = owner.rsplit(".", 1)[0] if "." in owner else ""
            local_name = f"{class_name}.{second}" if class_name else second
        elif second and first in items and items[first].get("name") == first:
            local_name = f"{first}.{second}"
        if local_name in items:
            return filename + ":" + local_name
        binding = self._js_imports.get(filename, {}).get(first)
        if binding:
            destination, exported = binding
            target_name = exported if exported != "*" else second
            if exported == "*" and second:
                target_name = second
            if second and exported != "*":
                target_name = exported + "." + second
            if target_name:
                candidate = destination + ":" + target_name
                if candidate in self.symbols:
                    return candidate
                # A default import aliases the module default export; if only a named
                # function is indexed, do not guess which export it represented.
            return None
        neighbor_files = self.index.file_dependencies.get(filename, ())
        if not second:
            matches = [path + ":" + symbol for path in neighbor_files
                       for symbol in self._js_names.get(path, ())
                       if symbol.split(".")[-1] == first]
            if len(matches) == 1:
                return matches[0]
        return None

    def _finish_edges(self) -> None:
        self.callers = {key: set() for key in self.symbols}
        for caller, callees in self.callees.items():
            for callee in callees:
                if callee in self.callers:
                    self.callers[callee].add(caller)
        for key in self.symbols:
            self.references.setdefault(key, set())
            self.references[key].update(self.callees.get(key, ()))

    @property
    def graph_metrics(self) -> dict:
        return {"symbols": len(self.symbols),
                "call_edges": sum(map(len, self.callees.values())),
                "reference_edges": sum(map(len, self.references.values())),
                "unresolved_references": sum(map(len, self.unresolved.values())),
                "files": len(set(item.file for item in self.symbols.values()))}

    def _target_keys(self, targets) -> set[str]:
        if isinstance(targets, (str, dict)):
            targets = [targets]
        result: set[str] = set()
        for target in targets:
            if isinstance(target, dict):
                file = str(target.get("file", ""))
                symbol = target.get("symbol")
                if symbol is not None:
                    target = f"{file}::{symbol}"
                else:
                    target = file
            if type(target) is not str or not target:
                raise ValueError("each target must be a symbol selector or repository-relative file")
            if target in self.symbols:
                result.add(target)
                continue
            if "::" in target:
                file, symbol = target.split("::", 1)
                candidate = file + ":" + symbol
                if candidate in self.symbols:
                    result.add(candidate)
                    continue
                matches = {key for key, item in self.symbols.items()
                           if item.file == Path(file).as_posix() and item.name == symbol}
                if matches:
                    result.update(matches)
                    continue
            if ":" in target and target in self.symbols:
                result.add(target)
                continue
            normalized = Path(target).as_posix()
            if Path(normalized).is_absolute() or ".." in Path(normalized).parts:
                raise ValueError("impact target must stay inside the repository")
            matching = {key for key, item in self.symbols.items() if item.file == normalized}
            if not matching:
                module = normalized[:-3].replace("/", ".") if normalized.endswith(".py") else normalized
                matching = {key for key, item in self.symbols.items()
                            if item.language == "python" and key.startswith(module + ":")}
            if not matching:
                raise KeyError(target)
            result.update(matching)
        if not result:
            raise ValueError("at least one indexed symbol is required")
        return result

    def analyze(self, targets, *, max_depth: int = 2) -> dict:
        if type(max_depth) is not int or not 0 <= max_depth <= 32:
            raise ValueError("max_depth must be an integer in 0..32")
        seeds = self._target_keys(targets)
        reached: dict[str, dict] = {}
        for direction, adjacency, label in (("upstream", self.callers, "caller"),
                                             ("downstream", self.callees, "callee")):
            queue = deque((key, 0) for key in sorted(seeds))
            visited = {key: 0 for key in seeds}
            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for neighbor in sorted(adjacency.get(current, ())):
                    if neighbor not in visited:
                        visited[neighbor] = depth + 1
                        queue.append((neighbor, depth + 1))
            for key, depth in visited.items():
                if key in seeds:
                    continue
                old = reached.get(key)
                if old is None or depth < old["distance"]:
                    reached[key] = {"direction": direction, "relation": label, "distance": depth}
                elif old["direction"] != direction:
                    old["direction"] = "both"
                    old["relation"] = "caller_and_callee"
        records = []
        for key in sorted(seeds):
            item = self.symbols[key]
            records.append(_symbol_record(item, relation="target", direction="target", distance=0,
                                          references=sorted(self.references[key]), unresolved=sorted(self.unresolved[key])))
        for key, info in sorted(reached.items(), key=lambda pair: (pair[1]["distance"], pair[1]["direction"], pair[0])):
            item = self.symbols[key]
            records.append(_symbol_record(item, **info,
                                          references=sorted(self.references[key]), unresolved=sorted(self.unresolved[key])))
        upstream = [entry for entry in records if entry["direction"] in {"upstream", "both"}]
        downstream = [entry for entry in records if entry["direction"] in {"downstream", "both"}]
        return {"targets": sorted(seeds), "max_depth": max_depth, "symbols": records,
                "upstream_callers": upstream, "downstream_callees": downstream,
                "unresolved": {key: sorted(values) for key, values in sorted(self.unresolved.items()) if values},
                "graph_metrics": self.graph_metrics}

    def compress_context(self, targets, *, max_context_tokens: int, max_depth: int = 3) -> dict:
        """Keep changed files whole, nearby contracts precise and distant nodes summarized."""
        if type(max_context_tokens) is not int or max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be a positive integer")
        if isinstance(targets, (str, dict)):
            raw_targets = [targets]
        else:
            raw_targets = list(targets)
        seeds = self._target_keys(raw_targets)
        changed_files = {self.symbols[key].file for key in seeds}
        impact = self.analyze(seeds, max_depth=max_depth)
        direct_items = []
        source_by_file = {}
        for filename in sorted(changed_files):
            source = self._file_source(filename)
            source_by_file[filename] = source
            direct_items.append({"symbol": f"{filename}:<file>", "file": filename, "distance": 0,
                                 "direction": "target", "representation": "full_source", "text": source})
        direct_tokens = context_token_count(_render_context(direct_items))
        if direct_tokens > max_context_tokens:
            raise ContextBudgetError(direct_tokens, max_context_tokens)
        candidates = []
        all_affected = {entry["symbol"] for entry in impact["symbols"]}
        naive_files = set(changed_files)
        for entry in impact["symbols"]:
            naive_files.add(entry["file"])
            if entry["file"] in changed_files:
                continue
            distance = entry["distance"]
            item = self.symbols[entry["symbol"]]
            if distance == 1:
                representation, text = "signature", item.signature
            else:
                representation, text = "abstract_summary", item.summary
            candidates.append({"symbol": entry["symbol"], "file": entry["file"], "distance": distance,
                               "direction": entry["direction"], "representation": representation, "text": text})
        candidates.sort(key=lambda row: (row["distance"], row["direction"], row["symbol"]))
        included = list(direct_items)
        included_keys = {key for key in seeds}
        tokens = direct_tokens
        for candidate in candidates:
            trial = included + [candidate]
            trial_tokens = context_token_count(_render_context(trial))
            if trial_tokens <= max_context_tokens:
                included.append(candidate)
                included_keys.add(candidate["symbol"])
                tokens = trial_tokens
        # The uncompressed comparison is the complete indexed source tree. This
        # makes the metric meaningful for a large repository even when only a
        # small subset is reachable from the changed symbols.
        naive_files = set(self.index.paths) | set(self.index.js_modules)
        full_sources = {filename: self._file_source(filename) for filename in naive_files}
        naive_tokens = sum(context_token_count(source) for source in full_sources.values())
        # Symbols in complete changed files count as recalled even without duplicate snippets.
        recalled = {key for key in all_affected if self.symbols[key].file in changed_files} | included_keys
        ratio = 0.0 if naive_tokens <= 0 else max(0.0, min(1.0, 1.0 - tokens / naive_tokens))
        return {"context": _render_context(included), "items": included,
                "changed_files": sorted(changed_files), "token_count": tokens,
                "max_context_tokens": max_context_tokens, "naive_token_count": naive_tokens,
                "compression_ratio": round(ratio, 6),
                "critical_symbol_recall": round(len(recalled & all_affected) / max(1, len(all_affected)), 6),
                "critical_symbol_count": len(all_affected), "recalled_symbol_count": len(recalled & all_affected),
                "omitted_symbols": sorted(all_affected - recalled),
                "graph_metrics": impact["graph_metrics"], "budget_unit": "lexical_tokens_v1"}

    def _file_source(self, filename: str) -> str:
        module = self.index.paths.get(filename)
        if module and module in self.index.modules:
            return self.index.modules[module][1]
        if filename in self.index.js_modules:
            return self.index.js_modules[filename][0]
        path = (self.root / filename).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise KeyError(filename)
        return path.read_text(encoding="utf-8")


def _signature_node(node: ast.AST) -> ast.AST:
    import copy
    value = copy.deepcopy(node)
    value.decorator_list = []
    doc = ast.get_docstring(value, clean=False)
    body = [ast.Expr(ast.Constant(doc))] if doc else []
    if isinstance(value, ast.ClassDef):
        value.bases, value.keywords = [], []
        for child in value.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body.append(_signature_node(child))
        value.body = body + [ast.Expr(ast.Constant(Ellipsis))]
    else:
        value.body = body + [ast.Expr(ast.Constant(Ellipsis))]
    return ast.fix_missing_locations(value)


def _python_summary(node: ast.AST) -> str:
    name = getattr(node, "name", "<symbol>")
    doc = ast.get_docstring(node, clean=True) or ""
    first = next((line.strip() for line in doc.splitlines() if line.strip()), "")
    if not first and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        returns = ast.unparse(node.returns) if node.returns else "unspecified return"
        first = f"Callable with {len(node.args.args)} positional parameters; {returns}."
    elif not first:
        first = "Structured declaration; inspect the signature and references for its role."
    return f"{name}: {first[:240]}"


def _js_summary(name: str, signature: str, source: str) -> str:
    comment = re.search(r"(?:/\*\*?\s*(.*?)\s*\*/|//\s*(.*))", source, re.S)
    hint = " ".join((comment.group(1) or comment.group(2) or "").split()) if comment else ""
    hint = re.sub(r"\s*\*\s*", " ", hint)
    return f"{name}: {(hint or signature or 'JavaScript/TypeScript symbol')[:240]}"


def _python_dynamic_references(node: ast.AST) -> set[str]:
    found = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "getattr" and len(item.args) > 1:
            if not isinstance(item.args[1], ast.Constant) or not isinstance(item.args[1].value, str):
                found.add("getattr(" + ast.unparse(item.args[0]) + ", <dynamic>)")
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and isinstance(item.func.value, ast.Subscript):
            found.add(ast.unparse(item.func) + "(<dynamic>)")
    return found


def _resolve_local_file(filename: str, specifier: str, dependencies: dict[str, list[str]]) -> str | None:
    # ImportGraph already resolved extensions, index files and HTML script paths.
    # Prefer those concrete repository-relative paths and never return a synthetic
    # extensionless base (which would silently drop a cross-file call edge).
    candidates = set(dependencies.get(filename, ()))
    path = Path(filename)
    base = (path.parent / specifier).as_posix()
    for candidate in sorted(candidates):
        if candidate == base or candidate.startswith(base + ".") or candidate.startswith(base + "/"):
            if candidate != filename:
                return candidate
    return None


def _symbol_record(item: _Symbol, *, relation: str, direction: str, distance: int,
                   references: list[str], unresolved: list[str]) -> dict:
    return {"symbol": item.key, "file": item.file, "name": item.name, "language": item.language,
            "relation": relation, "direction": direction, "distance": distance,
            "signature": item.signature, "source": item.source, "summary": item.summary,
            "references": references, "unresolved": unresolved}


def _render_context(items: list[dict]) -> str:
    chunks = []
    for item in items:
        chunks.append(f"[{item['representation']} {item['symbol']} {item['file']} distance={item['distance']} {item['direction']}]\n"
                      + item["text"])
    return "\n\n".join(chunks)
