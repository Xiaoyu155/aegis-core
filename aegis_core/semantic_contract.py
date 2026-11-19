"""Static cross-module inference for implicit object and API contracts.

The analyzer only parses source.  It never imports project code, so it is safe to
run against an untrusted checkout.  Inference is deliberately confidence-scored:
ambiguous dynamic dispatch remains visible instead of being guessed.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .ast_slicer import ImportGraph, _js_symbols, _mask_js_comments_and_strings
from .impact_analyzer import ImpactAnalyzer

_TYPE_NAMES = {"None": "null", "NoneType": "null", "str": "string", "int": "integer",
               "float": "number", "bool": "boolean", "dict": "object", "list": "array",
               "tuple": "array", "set": "array", "Any": "unknown", "object": "unknown"}
_FIELD = re.compile(r"(?m)([A-Za-z_$][\w$]*)\s*\??\s*:\s*([A-Za-z_$][\w$]*(?:\s*<[^;{}]+>)?(?:\[\])?(?:\s*\|\s*null)?)")
_OBJECT = re.compile(r"\breturn\s*\{([^{}]{0,12000})\}", re.S)


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    file: str
    line: int
    symbol: str
    source: str
    confidence: float = 1.0

    def as_dict(self):
        return {"name": self.name, "type": self.type, "file": self.file, "line": self.line,
                "symbol": self.symbol, "source": self.source, "confidence": self.confidence}


def _type_name(node: ast.AST | None) -> str:
    if node is None:
        return "unknown"
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None: return "null"
        if type(value) is bool: return "boolean"
        if type(value) is int: return "integer"
        if type(value) is float: return "number"
        if type(value) is str: return "string"
    if isinstance(node, (ast.List, ast.ListComp, ast.Tuple, ast.Set, ast.SetComp)):
        values = node.elts if hasattr(node, "elts") else []
        inner = sorted({_type_name(item) for item in values})
        return "array[" + "|".join(inner or ["unknown"]) + "]"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "object"
    if isinstance(node, ast.Name):
        return _TYPE_NAMES.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return _TYPE_NAMES.get(node.attr, node.attr)
    if isinstance(node, ast.Subscript):
        base = _type_name(node.value)
        arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        arguments = [item for item in arguments if not (isinstance(item, ast.Constant) and item.value is Ellipsis)]
        arg = ",".join(_type_name(item) for item in arguments) or "unknown"
        return f"{base}[{arg}]"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return "|".join(sorted({_type_name(node.left), _type_name(node.right)}))
    return "unknown"


def _literal_fields(node: ast.Dict, *, file: str, symbol: str, source: str):
    fields = []
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, (str, int)):
            fields.append(Field(str(key.value), _type_name(value), file, getattr(key, "lineno", 0),
                                symbol, source, 0.98))
    return fields


def _enclosing_class(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.ClassDef | None:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef):
            return parent
        parent = parents.get(parent)
    return None


def _annotation_class(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        base = _annotation_class(node.value)
        if base in {"Optional", "Union", "Annotated"}:
            values = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return next((_annotation_class(item) for item in values
                         if _annotation_class(item) not in {None, "None"}), None)
        return base
    return None


class _FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.reads = defaultdict(list)
        self.calls = set()
        self.source_calls = defaultdict(set)
        self.dynamic = set()
        self.registrations = set()

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            call = _call_name(node.value.func)
            if call:
                for target in node.targets:
                    if isinstance(target, ast.Name): self.source_calls[target.id].add(call)
        else:
            source = _root_name(node.value)
            if source:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.source_calls[target.id].update(self.source_calls.get(source, ()))
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.target, ast.Name):
            call = _call_name(node.value.func)
            if call: self.source_calls[node.target.id].add(call)
        elif isinstance(node.target, ast.Name):
            source = _root_name(node.value) if node.value is not None else None
            if source:
                self.source_calls[node.target.id].update(self.source_calls.get(source, ()))
        self.generic_visit(node)

    def visit_For(self, node):
        source = _root_name(node.iter)
        origins = self.source_calls.get(source, ()) if source else ()
        targets = list(_binding_names(node.target))
        for target in targets:
            self.source_calls[target].update(origins)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        source = _root_name(node.iter)
        origins = self.source_calls.get(source, ()) if source else ()
        for target in _binding_names(node.target):
            self.source_calls[target].update(origins)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            root = _root_name(node.value)
            if root:
                self.reads[root].append((node.slice.value, node.lineno, "subscript"))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and node.value.id not in {"self", "cls"}:
            self.reads[node.value.id].append((node.attr, node.lineno, "attribute"))
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _call_name(node.func)
        if name:
            self.calls.add(name)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) > 1:
            attr_node = node.args[1]
            literal_name = (attr_node.value if isinstance(attr_node, ast.Constant)
                            and isinstance(attr_node.value, str) else None)
            receiver = node.args[0].id if isinstance(node.args[0], ast.Name) else None
            self.dynamic.add((literal_name if literal_name is not None else "*", receiver, node.lineno,
                              ast.unparse(attr_node), literal_name is None))
        if name and (name.lower().endswith((".register", ".add", ".connect", ".route", ".get", ".post",
                                           ".put", ".delete", ".add_url_rule", ".add_listener",
                                           ".addlistener", ".subscribe", ".on"))):
            for arg in node.args:
                if isinstance(arg, ast.Name): self.registrations.add(arg.id)
        self.generic_visit(node)


def _root_name(node):
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.value if isinstance(node, (ast.Attribute, ast.Subscript)) else node.func
    return node.id if isinstance(node, ast.Name) else None


def _binding_names(node):
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from _binding_names(item)


def _call_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _decorator_name(node):
    return _call_name(node.func) if isinstance(node, ast.Call) else _call_name(node)


class SemanticContractAnalyzer:
    """Infer typed and implicit schemas and connect producers to consumers."""

    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root).resolve()
        self.graph = ImportGraph(self.root)
        self.impact = ImpactAnalyzer(self.root)
        self.fields: dict[str, list[Field]] = defaultdict(list)
        self.consumers: list[dict] = []
        self.producers: dict[str, dict] = {}
        self.decorators: list[dict] = []
        self.callbacks: list[dict] = []
        self.dynamic_references: list[dict] = []
        self._python()
        self._javascript()
        self._connect()

    def _python(self):
        for module, (filename, text, tree) in self.graph.modules.items():
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            module_functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                    options = [arg.value for arg in node.args if isinstance(arg, ast.Constant)
                               and isinstance(arg.value, str) and arg.value.startswith("-")]
                    explicit_dest = next((item.value.value for item in node.keywords
                                           if item.arg == "dest" and isinstance(item.value, ast.Constant)
                                           and isinstance(item.value.value, str)), None)
                    if options:
                        option = next((item for item in reversed(options) if item.startswith("--")), options[-1])
                        field_name = explicit_dest or option.lstrip("-").replace("-", "_")
                        self.fields[filename + "::argparse.Namespace"].append(
                            Field(field_name, "unknown", filename, node.lineno,
                                  module + ":argparse.Namespace", "argparse_option", .9))
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    is_schema = any((_call_name(base) or getattr(base, "id", "")).endswith("TypedDict")
                                    for base in node.bases)
                    is_data = any(_decorator_name(item) == "dataclass" or _decorator_name(item).endswith(".dataclass")
                                  for item in node.decorator_list)
                    for member in node.body:
                        if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name) and (is_schema or is_data):
                            self.fields[filename + "::" + class_name].append(
                                Field(member.target.id, _type_name(member.annotation), filename,
                                      member.lineno, module + ":" + class_name, "typed_class", 1.0))
                    for member in ast.walk(node):
                        if isinstance(member, (ast.Assign, ast.AnnAssign)):
                            value = member.value
                            targets = member.targets if isinstance(member, ast.Assign) else [member.target]
                            for target in targets:
                                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                                        and target.value.id in {"self", "cls"}):
                                    annotation = member.annotation if isinstance(member, ast.AnnAssign) else None
                                    self.fields[filename + "::" + class_name].append(
                                        Field(target.attr, _type_name(annotation) if annotation else _type_name(value),
                                              filename, target.lineno, module + ":" + class_name,
                                              "class_attribute", .92))
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                owner_class = _enclosing_class(node, parents)
                local_name = (owner_class.name + "." if owner_class else "") + node.name
                qname = module + ":" + local_name
                visitor = _FunctionVisitor(); visitor.visit(node)
                params = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]]
                local_dicts = {}
                for child in ast.walk(node):
                    if isinstance(child, (ast.Assign, ast.AnnAssign)) and isinstance(child.value, ast.Dict):
                        for target in (child.targets if isinstance(child, ast.Assign) else [child.target]):
                            if isinstance(target, ast.Name):
                                local_dicts[target.id] = child.value
                                self.fields[filename + "::" + local_name].extend(
                                    _literal_fields(child.value, file=filename, symbol=qname, source="dict_literal"))
                    if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                        self.fields[filename + "::" + local_name].extend(
                            _literal_fields(child.value, file=filename, symbol=qname, source="return_dict"))
                    if isinstance(child, ast.Return) and isinstance(child.value, ast.Name) and child.value.id in local_dicts:
                        self.fields[filename + "::" + local_name].extend(
                            _literal_fields(local_dicts[child.value.id], file=filename, symbol=qname, source="returned_dict"))
                    if isinstance(child, ast.Return) and isinstance(child.value, (ast.List, ast.Tuple)):
                        for item in child.value.elts:
                            if isinstance(item, ast.Dict):
                                self.fields[filename + "::" + local_name].extend(
                                    _literal_fields(item, file=filename, symbol=qname, source="returned_record_list"))
                returned_types = {_type_name(item.annotation) for item in node.args.args if item.annotation}
                self.producers[qname] = {"file": filename, "name": node.name, "owner_class": owner_class.name if owner_class else None,
                                         "params": params,
                                         "fields": sorted({f.name for values in [self.fields[filename + "::" + local_name]] for f in values}),
                                         "calls": sorted(visitor.calls), "returns": sorted(returned_types),
                                         "line": node.lineno, "language": "python"}
                for decorator in node.decorator_list:
                    name = _decorator_name(decorator) or ""
                    recognized = name in {"property", "staticmethod", "classmethod"} or any(
                        token in name.lower() for token in ("route", "get", "post", "put", "delete", "patch"))
                    if recognized:
                        self.decorators.append({"file": filename, "symbol": qname, "decorator": name,
                                                "line": decorator.lineno, "kind": "descriptor" if name in {
                                                    "property", "staticmethod", "classmethod"} else "route"})
                for callback in sorted(visitor.registrations):
                    self.callbacks.append({"file": filename, "owner": qname, "callback": callback,
                                           "line": node.lineno, "kind": "registration"})
                parameter_types = {arg.arg: _annotation_class(arg.annotation)
                                   for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]}
                if owner_class:
                    for receiver in ("self", "cls"):
                        if receiver in parameter_types:
                            parameter_types[receiver] = owner_class.name
                for dynamic, receiver, line, name_expression, dynamic_name in sorted(
                        visitor.dynamic, key=lambda item: (item[0], item[1] or "", item[2], item[3])):
                    receiver_type = parameter_types.get(receiver) if receiver else None
                    self.dynamic_references.append({"file": filename, "symbol": qname, "name": dynamic,
                                                    "receiver": receiver, "receiver_type": receiver_type,
                                                    "module": module,
                                                    "source_calls": sorted(visitor.source_calls.get(receiver, ())) if receiver else [],
                                                    "name_expression": name_expression,
                                                    "dynamic_name": dynamic_name,
                                                    "resolved_to": [], "line": line,
                                                    "resolved": False})
                data_roots = set(params) | set(visitor.source_calls) | set(local_dicts)
                for param in sorted(data_roots):
                    reads = visitor.reads.get(param, ())
                    for field, line, kind in sorted(set(reads)):
                        self.consumers.append({"file": filename, "symbol": qname, "parameter": param,
                                               "field": field, "line": line, "access": kind,
                                               "language": "python", "calls": sorted(visitor.calls),
                                               "source_calls": sorted(visitor.source_calls.get(param, ()))})

    def _javascript(self):
        for filename, (text, items) in self.graph.js_modules.items():
            for interface in re.finditer(r"\b(?:interface|type)\s+([A-Za-z_$][\w$]*)\s*(?:=\s*)?\{([^{}]*)\}", text, re.S):
                schema = filename + "::" + interface.group(1)
                start = text.count("\n", 0, interface.start(2)) + 1
                for field in _FIELD.finditer(interface.group(2)):
                    line = start + interface.group(2).count("\n", 0, field.start())
                    self.fields[schema].append(Field(field.group(1), _js_type(field.group(2)), filename,
                                                     line, filename + ":" + interface.group(1),
                                                     "typescript_contract", 1.0))
            for name, item in items.items():
                source = item["source"]
                qname = filename + ":" + name
                fields = []
                for match in _OBJECT.finditer(source):
                    body = match.group(1)
                    for key in re.finditer(r"(?:^|[,\n])\s*(['\"]?)([\w$-]+)\1\s*:\s*([^,\n}]+)", body):
                        fields.append((key.group(2), _js_value_type(key.group(3)), source.count("\n", 0, match.start(1) + key.start(2)) + item.get("start_line", 1)))
                for field, kind, line in fields:
                    self.fields[filename + "::" + name].append(Field(field, kind, filename, line, qname, "return_object", .94))
                # Object destructuring and response.foo / response["foo"] cover both
                # JSON API consumers and ordinary TypeScript object flows.
                masked = _mask_js_comments_and_strings(source)
                destructured = re.finditer(r"(?:const|let|var)\s*\{([^}]+)\}\s*=\s*([A-Za-z_$][\w$]*)", masked)
                consumers = []
                source_calls = defaultdict(set)
                for assignment in re.finditer(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:await\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(", masked):
                    source_calls[assignment.group(1)].add(assignment.group(2).split(".")[-1])
                for match in destructured:
                    for part in match.group(1).split(","):
                        key = part.strip().split(":", 1)[0].strip()
                        if re.fullmatch(r"[\w$]+", key): consumers.append((match.group(2), key, "destructure"))
                for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*(?:\.\s*([A-Za-z_$][\w$]*)|\[\s*['\"]([^'\"]+)['\"]\s*\])", masked):
                    if match.group(2) or match.group(3):
                        consumers.append((match.group(1), match.group(2) or match.group(3), "property"))
                calls = set()
                for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*\(", masked):
                    calls.add(match.group(1))
                for variable, field, access in sorted(set(consumers)):
                    # Ignore property uses that are method names and built-in namespaces.
                    self.consumers.append({"file": filename, "symbol": qname, "parameter": variable,
                                           "field": field, "line": source.count("\n", 0, masked.find(field)) + item.get("start_line", 1),
                                           "access": access, "language": "javascript", "calls": sorted(calls),
                                           "source_calls": sorted(source_calls.get(variable, ()))})
                self.producers[qname] = {"file": filename, "name": name,
                                         "fields": sorted({field for field, _, _ in fields}),
                                         "calls": sorted(calls), "line": item.get("start_line", 1),
                                         "language": "javascript"}

    def _connect(self):
        call_targets = defaultdict(set)
        short = defaultdict(set)
        for key, record in self.producers.items():
            short[record["name"]].add(key)
        for caller, edges in self.impact.callees.items():
            for callee in edges:
                call_targets[caller].add(callee)
        dynamic_names = defaultdict(set)
        for key, record in self.producers.items():
            dynamic_names[record["name"]].add(key)
        for reference in self.dynamic_references:
            if reference.get("dynamic_name"):
                continue
            candidates = dynamic_names.get(reference["name"], set())
            receiver_type = reference.get("receiver_type")
            typed = {key for key in candidates if receiver_type and
                     self.producers.get(key, {}).get("owner_class") == receiver_type}
            if len(typed) == 1:
                reference["resolved_to"] = sorted(typed)
                reference["resolution_kind"] = "typed_method"
                reference["resolved"] = True
            elif len(candidates) == 1:
                reference["resolved_to"] = sorted(candidates)
                reference["resolution_kind"] = "unique_callable"
                reference["resolved"] = True
            elif receiver_type:
                field_schemas = [schema for schema, values in self.fields.items()
                                 if schema.endswith("::" + receiver_type)
                                 and any(field.name == reference["name"] for field in values)]
                if len(field_schemas) == 1:
                    reference["resolved_to"] = [field_schemas[0] + "::" + reference["name"]]
                    reference["resolution_kind"] = "typed_field"
                    reference["resolved"] = True
            if not reference["resolved"]:
                source_targets = set()
                for source_call in reference.get("source_calls", ()):
                    source_name = source_call.rsplit(".", 1)[-1]
                    call_candidates = set(dynamic_names.get(source_name, ()))
                    binding = self.graph.bindings.get(reference["module"], {}).get(source_name, "")
                    if "." in binding:
                        imported_module, imported_name = binding.rsplit(".", 1)
                        imported_key = imported_module + ":" + imported_name
                        if imported_key in self.producers:
                            call_candidates = {imported_key}
                    call_candidates = {key for key in call_candidates
                                       if reference["name"] in self.producers.get(key, {}).get("fields", ())}
                    if receiver_type and receiver_type not in {"Any", "object"}:
                        typed_calls = {key for key in call_candidates
                                       if self.producers.get(key, {}).get("owner_class") == receiver_type}
                        if typed_calls:
                            call_candidates = typed_calls
                    source_targets.update(call_candidates)
                if len(source_targets) == 1:
                    producer = next(iter(source_targets))
                    reference["resolved_to"] = [producer + "::" + reference["name"]]
                    reference["resolution_kind"] = "producer_field_flow"
                    reference["resolved"] = True
            if not reference["resolved"] and reference.get("receiver") in {"args", "namespace", "opts"}:
                argument_schemas = [schema for schema, values in self.fields.items()
                                    if schema.endswith("::argparse.Namespace")
                                    and any(field.name == reference["name"] for field in values)]
                if len(argument_schemas) == 1:
                    reference["resolved_to"] = [argument_schemas[0] + "::" + reference["name"]]
                    reference["resolution_kind"] = "argparse_option"
                    reference["resolved"] = True
        self.flows = []
        self.mismatches = []
        for consumer in self.consumers:
            names = set(consumer.get("source_calls", ()))
            owner = consumer["symbol"]
            edges = set(call_targets.get(owner, ()))
            targets = set(edges)
            resolved_aliases = set()
            for name in names:
                if name in self.producers:
                    targets.add(name)
                targets.update(short.get(name, ()))
                module = self.graph.paths.get(consumer["file"])
                binding = self.graph.bindings.get(module, {}).get(name, "")
                if "." in binding:
                    imported_module, imported_name = binding.rsplit(".", 1)
                    imported_key = imported_module + ":" + imported_name
                    if imported_key in self.producers:
                        targets.add(imported_key)
                        resolved_aliases.add(imported_key)
            targets = {target for target in targets if target in resolved_aliases
                       or self.producers.get(target, {}).get("name") in names
                       or target.rsplit(":", 1)[-1] in names}
            # For renamed imports, the static call graph has already proven the
            # edge and the assigned local is tied to exactly one call expression.
            if not targets and len(names) == 1 and len(edges) == 1:
                targets = edges
            for target in sorted(targets):
                record = self.producers.get(target)
                if not record:
                    continue
                self.flows.append({"producer": target, "consumer": owner, "field": consumer["field"],
                                   "consumer_file": consumer["file"], "line": consumer["line"],
                                   "confidence": .86 if target in call_targets.get(owner, ()) else .64})
                if record.get("fields") and consumer["field"] not in record["fields"]:
                    self.mismatches.append({"kind": "missing_producer_field", "producer": target,
                                            "consumer": owner, "file": consumer["file"],
                                            "line": consumer["line"], "field": consumer["field"],
                                            "available_fields": record["fields"]})

    def analyze(self, runtime_profile: dict | None = None) -> dict:
        typed_schemas = [{"schema": name, "fields": [field.as_dict() for field in sorted(values, key=lambda f: (f.name, f.line))]}
                         for name, values in sorted(self.fields.items())]
        unresolved = sum(len(items) for items in self.impact.unresolved.values())
        dynamic_references = [dict(item) for item in self.dynamic_references]
        static_resolved = sum(item["resolved"] for item in dynamic_references)
        total_dynamic = len(dynamic_references)
        runtime_overlay = {"observed_call_edges": 0, "resolved_dynamic_sites": [], "dynamic_sites_resolved": 0,
                           "runtime_profiled_dynamic_sites": 0,
                           "runtime_profiled_dynamic_sites_resolved": 0,
                           "runtime_profiled_dynamic_resolution_rate": None}
        if runtime_profile is not None:
            from .runtime_profiler import merge_into_semantic
            runtime_overlay = merge_into_semantic(self, runtime_profile)
            by_site = {(item["file"], item["line"], item["symbol"], item["name"]): item
                       for item in runtime_overlay["resolved_dynamic_sites"]}
            for item in dynamic_references:
                observed = by_site.get((item["file"], item["line"], item["symbol"], item["name"]))
                if observed:
                    targets = observed.get("resolved_to", [])
                    if isinstance(targets, str): targets = [targets]
                    item.update({"resolved": True, "resolved_to": targets,
                                 "runtime_observed_name": observed.get("observed_name", []),
                                 "runtime_result_schemas": observed.get("result_schemas", []),
                                 "resolution_kind": "runtime_observed"})
        resolved_dynamic = sum(item["resolved"] for item in dynamic_references)
        unresolved_dynamic = total_dynamic - resolved_dynamic
        resolution_kinds = defaultdict(int)
        for item in dynamic_references:
            if item["resolved"]:
                resolution_kinds[item["resolution_kind"]] += 1
        result = {"schema_version": "aegis.semantic_contract.v1", "schemas": typed_schemas,
                "producers": dict(sorted(self.producers.items())), "consumers": self.consumers,
                "flows": self.flows, "mismatches": self.mismatches,
                "decorators": self.decorators, "callbacks": self.callbacks,
                "dynamic_references": dynamic_references,
                "metrics": {"schema_count": len(typed_schemas), "fields": sum(len(row["fields"]) for row in typed_schemas),
                            "flows": len(self.flows), "contract_mismatches": len(self.mismatches),
                            "dynamic_sites": total_dynamic, "resolved_dynamic_sites": resolved_dynamic,
                            "unresolved_dynamic_sites": unresolved_dynamic,
                            "legacy_dynamic_unresolved_sites": total_dynamic,
                            "dynamic_unresolved_reduction_rate": static_resolved / total_dynamic if total_dynamic else 1.0,
                            "legacy_unresolved_references": unresolved,
                            "dynamic_resolution_rate": static_resolved / total_dynamic if total_dynamic else 1.0,
                            "dynamic_resolution_kinds": dict(sorted(resolution_kinds.items())),
                            "decorators_recognized": len(self.decorators), "callbacks_recognized": len(self.callbacks)}}
        if runtime_profile is not None:
            result["runtime_overlay"] = runtime_overlay
            runtime_schemas = []
            for reference in dynamic_references:
                if not reference.get("runtime_result_schemas"):
                    continue
                shapes = []
                for encoded in reference["runtime_result_schemas"]:
                    try:
                        shapes.append(json.loads(encoded))
                    except (TypeError, ValueError):
                        continue
                if shapes:
                    runtime_schemas.append({"file": reference["file"], "symbol": reference["symbol"],
                        "line": reference["line"], "attribute": reference.get("runtime_observed_name", []),
                        "observed_shapes": shapes, "source": "runtime_profile"})
            result["runtime_schemas"] = runtime_schemas
            result["metrics"].update({
                "runtime_observed_call_edges": runtime_overlay["observed_call_edges"],
                "runtime_resolved_dynamic_sites": runtime_overlay["dynamic_sites_resolved"],
                "runtime_profiled_dynamic_sites": runtime_overlay["runtime_profiled_dynamic_sites"],
                "runtime_profiled_dynamic_sites_resolved": runtime_overlay["runtime_profiled_dynamic_sites_resolved"],
                "runtime_profiled_dynamic_resolution_rate": runtime_overlay["runtime_profiled_dynamic_resolution_rate"],
                "runtime_augmented_dynamic_resolution_rate": resolved_dynamic / total_dynamic if total_dynamic else 1.0,
                "runtime_augmented_unresolved_reduction_rate": resolved_dynamic / total_dynamic if total_dynamic else 1.0,
            })
            result["impact_graph_metrics"] = self.impact.graph_metrics
        return result

    def compare(self, baseline: "SemanticContractAnalyzer | dict") -> dict:
        """Compare current inferred schemas to a baseline and report field/type drift."""
        before = baseline.analyze() if isinstance(baseline, SemanticContractAnalyzer) else baseline
        after = self.analyze()
        def flatten(document):
            return {(schema["schema"], field["name"]): field["type"]
                    for schema in document.get("schemas", ()) for field in schema.get("fields", ())}
        old, new = flatten(before), flatten(after)
        added = [{"schema": key[0], "field": key[1], "type": new[key]} for key in sorted(new.keys() - old.keys())]
        removed = [{"schema": key[0], "field": key[1], "type": old[key]} for key in sorted(old.keys() - new.keys())]
        changed = [{"schema": key[0], "field": key[1], "before": old[key], "after": new[key]}
                   for key in sorted(old.keys() & new.keys()) if old[key] != new[key]]
        return {"schema_version": "aegis.semantic_diff.v1", "added_fields": added,
                "removed_fields": removed, "type_changes": changed,
                "breaking_changes": removed + changed, "after_mismatches": after["mismatches"]}


def _js_value_type(value):
    value = value.strip()
    if value.startswith(("'", '"', "`")): return "string"
    if value in {"true", "false"}: return "boolean"
    if re.match(r"^-?\d+\s*$", value): return "integer"
    if re.match(r"^-?\d+\.\d+", value): return "number"
    if value.startswith("["): return "array"
    if value.startswith("{"): return "object"
    if value in {"null", "undefined"}: return "null"
    return "unknown"


def _js_type(annotation):
    value = re.sub(r"\s+", "", annotation)
    aliases = {"string": "string", "number": "number", "boolean": "boolean", "void": "null",
               "null": "null", "any": "unknown", "unknown": "unknown"}
    if value.endswith("[]"): return "array[" + _js_type(value[:-2]) + "]"
    if value in aliases: return aliases[value]
    if value.startswith("Array<") and value.endswith(">"):
        return "array[" + _js_type(value[6:-1]) + "]"
    return value
