"""Freeze statically visible cross-module contracts before a repair layer fans out."""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from .ast_slicer import (ImportGraph, JS_SUFFIXES, _brace_end, _js_symbols, _mask_js_comments_and_strings,
                         decode, is_protected_test_path)
from .frontend import HTML_SUFFIXES, parse_html

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_TYPE_ALIAS = getattr(ast, "TypeAlias", ())


def _literal_shape(node):
    if isinstance(node, ast.Dict) and all(isinstance(key, ast.Constant)
                                         and isinstance(key.value, str) for key in node.keys):
        return tuple(sorted((key.value, _literal_shape(value))
                            for key, value in zip(node.keys, node.values)))
    return ()


def _python_surface(source):
    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, _FUNCTION):
                name = prefix + node.name
                kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                decorators = " ".join("@" + ast.unparse(item) for item in node.decorator_list)
                type_params = getattr(node, "type_params", [])
                generic = "[" + ", ".join(map(ast.unparse, type_params)) + "]" if type_params else ""
                yield name, "signature", f"{decorators} {kind} {name}{generic}({ast.unparse(node.args)}){returns}".strip()
                # Only inspect returns in this function's own lexical scope.
                pending = list(node.body)
                shapes = set()
                while pending:
                    statement = pending.pop()
                    if isinstance(statement, (*_FUNCTION, ast.ClassDef, ast.Lambda)):
                        continue
                    if isinstance(statement, ast.Return):
                        shape = _literal_shape(statement.value)
                        if shape:
                            shapes.add(shape)
                    pending.extend(ast.iter_child_nodes(statement))
                if shapes:
                    yield name, "return_keys", json.dumps(sorted(shapes), ensure_ascii=False)
            elif isinstance(node, ast.ClassDef):
                name = prefix + node.name
                bases = [*node.bases, *node.keywords]
                type_params = getattr(node, "type_params", [])
                decorators = " ".join("@" + ast.unparse(item) for item in node.decorator_list)
                header = f"class {name}({', '.join(map(ast.unparse, bases))})"
                yield name, "class", f"{decorators} {header} {', '.join(map(ast.unparse, type_params))}".strip()
                if any(ast.unparse(base).split(".")[-1] == "TypedDict" for base in node.bases):
                    keys = sorted(item.target.id for item in node.body
                                  if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name))
                    yield name, "state_keys", json.dumps(keys)
                if any(ast.unparse(item.func if isinstance(item, ast.Call) else item).split(".")[-1] == "dataclass"
                       for item in node.decorator_list):
                    fields = [ast.unparse(item) for item in node.body
                              if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)]
                    yield name, "dataclass_fields", "\n".join(fields)
                yield from visit(node.body, name + ".")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = prefix + target.id
                    if isinstance(node, ast.AnnAssign):
                        yield name, "annotation", ast.unparse(node.annotation)
                        if ast.unparse(node.annotation).split(".")[-1] == "TypeAlias" and node.value is not None:
                            yield name, "type_alias", ast.unparse(node.value)
                    shape = _literal_shape(node.value)
                    if shape:
                        yield name, "state_keys", json.dumps(shape, ensure_ascii=False)
                    if name == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        for item in node.value.elts:
                            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                yield item.value, "export", item.value
            elif isinstance(node, _TYPE_ALIAS):
                yield prefix + node.name.id, "type_alias", ast.unparse(node)

    tree = ast.parse(source)
    entries = defaultdict(list)
    for name, kind, value in visit(tree.body):
        entries[name, kind].append(value)
    shared_mappings = {target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                       and isinstance(node.value, ast.Dict)
                       for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                       if isinstance(target, ast.Name)}
    written_keys = defaultdict(set)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del))
                and isinstance(node.value, ast.Name) and node.value.id in shared_mappings
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            written_keys[node.value.id].add(node.slice.value)
    for name, keys in written_keys.items():
        entries[name, "written_state_keys"].append(json.dumps(sorted(keys), ensure_ascii=False))
    return {key: "\n".join(values) for key, values in entries.items()}


def _js_properties(source, masked, start, end):
    """Read statically named keys at a braced object's top level."""
    depth, part = 0, start
    sections = []
    for index in range(start, end):
        character = masked[index]
        if character in "({[":
            depth += 1
        elif character in ")}]":
            depth -= 1
        elif depth == 0 and character in ",;":
            sections.append(source[part:index])
            part = index + 1
    sections.append(source[part:end])
    keys = []
    for section in sections:
        match = re.match(r'''\s*(?:readonly\s+)?(?:([\w$]+)|['"]([^'"]+)['"])\s*(?:\?|:|\(|$)''', section)
        if match:
            keys.append(match[1] or match[2])
    return keys


def _js_surface(source):
    masked = _mask_js_comments_and_strings(source)
    result = {(name, "signature"): re.sub(r"\s+", " ", item["signature"])
              for name, item in _js_symbols(source).items()}
    declarations = r"\b(?:const|let|var)\s+([\w$]+)\s*=\s*\{|\b(?:interface|type)\s+([\w$]+)[^;{}]*\{"
    for match in re.finditer(declarations, masked):
        opening = match.end() - 1
        end = _brace_end(masked, opening) - 1
        name = match[1] or match[2]
        keys = _js_properties(source, masked, opening + 1, end)
        result[name, "state_keys"] = json.dumps(sorted(keys))
        if match[2]:
            result[name, "annotation"] = re.sub(r"\s+", " ", source[opening + 1:end].strip())
    for match in re.finditer(r"\b(module\.exports\s*=|export)\s*\{", masked):
        opening = match.end() - 1
        end = _brace_end(masked, opening) - 1
        fields = re.finditer(r"(?:^|,)\s*([\w$]+)(?:\s*(?:as|:)\s*([\w$]+))?", masked[opening + 1:end])
        for field in fields:
            name, target = field[1], field[2] or field[1]
            if match[1] == "export":
                name, target = target, name
            result[name, "export"] = target
    for match in re.finditer(r"\b(?:module\.)?exports\.([\w$]+)\s*=", masked):
        result[match[1], "export"] = match[1]
    return result


def _surface(path, relative):
    if not path.is_file() or path.is_symlink():
        return {}
    source, _ = decode(path)
    if relative.endswith(".py"):
        return _python_surface(source)
    if path.suffix.lower() in JS_SUFFIXES:
        return _js_surface(source)
    if path.suffix.lower() in HTML_SUFFIXES:
        blocks, _ = parse_html(source)
        browser_api = {}
        for block in blocks:
            if block.language != "javascript":
                continue
            surface = _js_surface(block.source)
            browser_api.update({key: value for key, value in surface.items()
                                if key[1] == "export" or (key[1] == "signature" and value.startswith("export "))})
            masked = _mask_js_comments_and_strings(block.source)
            for match in re.finditer(r"\b(?:window|globalThis)\.([\w$]+)\s*=(?!=)\s*([\w$]+)?", masked):
                name, target = match[1], match[2]
                browser_api[name, "export"] = name
                if (target, "signature") in surface:
                    browser_api[name, "signature"] = surface[target, "signature"]
                if (target, "state_keys") in surface:
                    browser_api[name, "state_keys"] = surface[target, "state_keys"]
        return browser_api
    return {}


@dataclass(frozen=True, order=True)
class InterfaceEntry:
    path: str
    symbol: str
    kind: str
    expected: str


@dataclass(frozen=True)
class InterfaceContractLock:
    """Immutable baseline, fingerprints and exact before/after signature evidence."""

    baseline: str
    entries: tuple[InterfaceEntry, ...]

    @classmethod
    def capture(cls, graph: ImportGraph, baseline: str, targets) -> InterfaceContractLock:
        targets = set(targets)
        consumers = {relative for relative in targets if Path(relative).suffix.lower() in HTML_SUFFIXES}
        for caller, dependencies in graph.file_dependencies.items():
            if not is_protected_test_path(caller):
                consumers.update(targets.intersection(dependencies))
        for caller, references in graph.calls.items():
            caller_file = graph.modules[graph.symbol_modules[caller]][0]
            if not is_protected_test_path(caller_file):
                consumers.update(graph.modules[graph.symbol_modules[key]][0] for key in references
                                 if graph.modules[graph.symbol_modules[key]][0] != caller_file
                                 and graph.modules[graph.symbol_modules[key]][0] in targets)
        exposed = defaultdict(set)
        for consumer, imports in graph.bindings.items():
            if is_protected_test_path(graph.modules[consumer][0]):
                continue
            for bound in imports.values():
                for relative in consumers:
                    provider = graph.paths.get(relative)
                    if provider is not None and provider != consumer and bound.startswith(provider + "."):
                        exposed[relative].add(bound[len(provider) + 1:].split(".")[0])
        entries = []
        for relative in sorted(consumers):
            surface = _surface(graph.root / relative, relative)
            for (name, kind), value in surface.items():
                parts = name.split(".")
                public = all(not part.startswith("_") or part.startswith("__") and part.endswith("__") for part in parts)
                exported_private = parts[0].startswith("_") and (parts[0] in exposed[relative]
                                                                  or (parts[0], "export") in surface)
                if not relative.endswith(".py") or public or exported_private:
                    entries.append(InterfaceEntry(relative, name, kind, value))
        return cls(baseline, tuple(sorted(entries)))

    @classmethod
    def from_dict(cls, document):
        lock = cls(document["baseline"], tuple(InterfaceEntry(**entry) for entry in document["entries"]))
        if lock.to_dict() != document:
            raise ValueError("invalid interface lock digest")
        return lock

    def to_dict(self):
        content = {"baseline": self.baseline, "entries": [asdict(entry) for entry in self.entries]}
        encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        return {**content, "sha256": hashlib.sha256(encoded).hexdigest()}

    def violations(self, root: Path, changed_files):
        relevant = set(changed_files)
        surfaces = {}
        failures = []
        for entry in self.entries:
            if entry.path not in relevant:
                continue
            if entry.path not in surfaces:
                try:
                    surfaces[entry.path] = _surface(root / entry.path, entry.path)
                except (OSError, ValueError, SyntaxError) as error:
                    surfaces[entry.path] = {}
                    failures.append({"path": entry.path, "kind": "unreadable_interface",
                                     "observed": type(error).__name__})
            observed = surfaces[entry.path].get((entry.symbol, entry.kind))
            if observed != entry.expected:
                failures.append({**asdict(entry), "observed": observed})
        return failures
