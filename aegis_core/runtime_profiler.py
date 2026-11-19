"""Bounded runtime profiling for Python test commands in an isolated worktree.

The profile contains names, types and shape metadata only.  It never serializes
argument values, source lines, environment variables or command arguments.
Python 3.12 uses ``sys.monitoring`` for line/return/raise events and a profile
hook for argument schemas; Python 3.11 uses the trace API as a compatibility
backend.  The target always runs in a child process so hooks cannot leak into
the caller or interfere with its test runner.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
from typing import Iterable


_BOOTSTRAP = r'''import atexit, builtins, dis, json, os, re, sys, types

output = os.environ.get("AEGIS_PROFILE_OUTPUT")
root = os.path.realpath(os.environ.get("AEGIS_PROFILE_ROOT", os.getcwd()))
limit = int(os.environ.get("AEGIS_PROFILE_MAX_EVENTS", "50000"))
secret = re.compile(r"password|secret|token|api.?key|credential|private.?key", re.I)
functions, edges, exceptions, heat, dynamic_attributes = {}, {}, {}, {}, {}
event_count = 0
tool_id = None
events = None
backend = "trace"

def relpath(filename):
    if filename.startswith("<") and filename.endswith(">"):
        return None
    real = os.path.realpath(filename)
    try:
        if os.path.commonpath((root, real)) != root: return None
    except ValueError:
        return None
    return os.path.relpath(real, root).replace(os.sep, "/")

def key(frame):
    path = relpath(frame.f_code.co_filename)
    if path is None: return None
    return path + "::" + getattr(frame.f_code, "co_qualname", frame.f_code.co_name)

def code_key(code):
    path = relpath(code.co_filename)
    if path is None: return None
    return path + "::" + getattr(code, "co_qualname", code.co_name)

def shape(value, depth=0):
    if value is None: return "null"
    if type(value) is bool: return "boolean"
    if type(value) is int: return "integer"
    if type(value) is float: return "number"
    if type(value) is str: return "string"
    if type(value) is dict:
        if depth >= 2: return "object"
        fields = {}
        for name, item in sorted(dict.items(value), key=lambda pair: str(pair[0]))[:24]:
            name = str(name)
            if len(name) <= 80 and name.isidentifier() and not secret.search(name):
                fields[name] = shape(item, depth + 1)
        return {"type": "object", "fields": fields, "truncated": len(value) > 24}
    if type(value) in (list, tuple, set, frozenset):
        values = list(value)[:16]
        kinds = set()
        for item in values:
            item_shape = shape(item, depth + 1)
            kinds.add(item_shape if isinstance(item_shape, str) else "object")
        return {"type": "array", "items": sorted(kinds) or ["unknown"], "truncated": len(value) > 16}
    return type(value).__name__[:80]

def record_call(frame):
    global event_count
    if event_count >= limit: return
    caller = frame.f_back
    target_key = key(frame)
    if target_key is None: return
    event_count += 1
    item = functions.setdefault(target_key, {"file": relpath(frame.f_code.co_filename),
        "qualname": getattr(frame.f_code, "co_qualname", frame.f_code.co_name), "calls": 0,
        "arguments": {}, "returns": {}, "hot_lines": {}})
    item["calls"] += 1
    arg_count = frame.f_code.co_argcount + frame.f_code.co_kwonlyargcount
    if frame.f_code.co_flags & 0x04: arg_count += 1
    if frame.f_code.co_flags & 0x08: arg_count += 1
    for name in frame.f_code.co_varnames[:arg_count]:
        if secret.search(name): continue
        value = frame.f_locals.get(name)
        schema = json.dumps(shape(value), sort_keys=True, separators=(",", ":"))
        kinds = item["arguments"].setdefault(name, {})
        kinds[schema] = min(limit, kinds.get(schema, 0) + 1)
    if caller is not None:
        caller_key = key(caller)
        if caller_key is not None:
            line = caller.f_lineno
            edge_key = (caller_key, target_key, line)
            edges[edge_key] = min(limit, edges.get(edge_key, 0) + 1)

def record_return(frame, value):
    item = functions.get(key(frame))
    if item is None: return
    schema = json.dumps(shape(value), sort_keys=True, separators=(",", ":"))
    item["returns"][schema] = min(limit, item["returns"].get(schema, 0) + 1)

def record_line(frame, line):
    function_key = key(frame)
    if function_key is None or event_count >= limit: return
    item = functions.setdefault(function_key, {"file": relpath(frame.f_code.co_filename),
        "qualname": getattr(frame.f_code, "co_qualname", frame.f_code.co_name), "calls": 0,
        "arguments": {}, "returns": {}, "hot_lines": {}})
    item["hot_lines"][str(line)] = min(limit, item["hot_lines"].get(str(line), 0) + 1)

def trace(frame, event, arg):
    global event_count
    if event_count >= limit: return None
    if event == "call":
        if relpath(frame.f_code.co_filename) is None: return None
        record_call(frame)
        return trace
    if relpath(frame.f_code.co_filename) is None: return None
    event_count += 1
    if event == "line": record_line(frame, frame.f_lineno)
    elif event == "return": record_return(frame, arg)
    elif event == "exception":
        exc_type, exc, _ = arg
        location = (key(frame), frame.f_lineno, getattr(exc_type, "__name__", "Exception"))
        exceptions[location] = min(limit, exceptions.get(location, 0) + 1)
    return trace

def profile(frame, event, arg):
    if event != "call" or event_count >= limit or relpath(frame.f_code.co_filename) is None:
        return
    if tool_id is not None and events is not None:
        try:
            monitoring.set_local_events(tool_id, frame.f_code,
                                        events.LINE | events.PY_RETURN)
        except (ValueError, RuntimeError):
            pass
    record_call(frame)

def line_event(code, line):
    global event_count
    if event_count >= limit: return
    filename = relpath(code.co_filename)
    if filename is None: return
    event_count += 1
    qname = getattr(code, "co_qualname", code.co_name)
    k = filename + "::" + qname
    item = functions.setdefault(k, {"file": filename, "qualname": qname, "calls": 0,
        "arguments": {}, "returns": {}, "hot_lines": {}})
    item["hot_lines"][str(line)] = min(limit, item["hot_lines"].get(str(line), 0) + 1)

def return_event(code, offset, value):
    global event_count
    if event_count >= limit: return
    filename = relpath(code.co_filename)
    if filename is None: return
    event_count += 1
    qname = getattr(code, "co_qualname", code.co_name)
    item = functions.setdefault(filename + "::" + qname, {"file": filename, "qualname": qname,
        "calls": 0, "arguments": {}, "returns": {}, "hot_lines": {}})
    schema = json.dumps(shape(value), sort_keys=True, separators=(",", ":"))
    item["returns"][schema] = min(limit, item["returns"].get(schema, 0) + 1)

def raise_event(code, offset, exc):
    global event_count
    if event_count >= limit: return
    filename = relpath(code.co_filename)
    if filename is None: return
    event_count += 1
    line = code.co_firstlineno
    for instruction, source_line in dis.findlinestarts(code):
        if instruction > offset: break
        line = source_line
    location = (filename + "::" + getattr(code, "co_qualname", code.co_name), line,
                type(exc).__name__[:80])
    exceptions[location] = min(limit, exceptions.get(location, 0) + 1)

_original_getattr = builtins.getattr
def observed_getattr(obj, name, *default):
    global event_count
    result = _original_getattr(obj, name, *default)
    if event_count >= limit or type(name) is not str or secret.search(name):
        return result
    caller = sys._getframe(1)
    caller_key = key(caller)
    if caller_key is None: return result
    target_obj = result.__func__ if type(result) is types.MethodType else result
    target_code = target_obj.__code__ if type(target_obj) is types.FunctionType else None
    target = code_key(target_code) if target_code is not None else None
    if target is None:
        cls = type(result)
        module = type.__getattribute__(cls, "__module__")
        qualname = type.__getattribute__(cls, "__qualname__")
        target = "type:" + str(module)[:80] + "." + str(qualname)[:80]
    schema = json.dumps(shape(result), sort_keys=True, separators=(",", ":"))
    event_key = (caller_key, caller.f_lineno, name[:80], target, schema)
    dynamic_attributes[event_key] = min(limit, dynamic_attributes.get(event_key, 0) + 1)
    event_count += 1
    return result

builtins.getattr = observed_getattr

def finish():
    if not output: return
    edge_rows = [{"caller": caller, "caller_file": caller.split("::", 1)[0],
        "caller_qualname": caller.split("::", 1)[1], "caller_line": line,
        "callee": callee, "callee_file": callee.split("::", 1)[0],
        "callee_qualname": callee.split("::", 1)[1], "count": count}
        for (caller, callee, line), count in sorted(edges.items())]
    exception_rows = [{"symbol": symbol, "line": line, "type": name, "count": count}
        for (symbol, line, name), count in sorted(exceptions.items())]
    dynamic_rows = [{"caller": caller, "caller_file": caller.split("::", 1)[0],
        "caller_qualname": caller.split("::", 1)[1], "line": line, "name": name,
        "target": target, "result_schema": schema, "count": count}
        for (caller, line, name, target, schema), count in sorted(dynamic_attributes.items())]
    doc = {"schema_version": "aegis.runtime_profile.v1", "backend": backend,
        "functions": dict(sorted(functions.items())), "call_edges": edge_rows,
        "exceptions": exception_rows, "dynamic_attributes": dynamic_rows,
        "event_count": min(event_count, limit),
        "event_limit_reached": event_count >= limit}
    try:
        with open(output + ".tmp", "w", encoding="utf-8") as stream:
            json.dump(doc, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        os.replace(output + ".tmp", output)
    except OSError:
        pass

if hasattr(sys, "monitoring") and hasattr(sys.monitoring, "events"):
    monitoring = sys.monitoring
    for candidate in range(6):
        try:
            if monitoring.get_tool(candidate) is None:
                monitoring.use_tool_id(candidate, "aegis-runtime-profiler")
                tool_id = candidate
                break
        except (ValueError, RuntimeError):
            continue
    if tool_id is not None:
        try:
            events = monitoring.events
            monitoring.register_callback(tool_id, events.LINE, line_event)
            monitoring.register_callback(tool_id, events.PY_RETURN, return_event)
            monitoring.register_callback(tool_id, events.RAISE, raise_event)
            # Keep line and return callbacks local to repository code.
            # CPython does not permit RAISE as a local event, so exceptions
            # remain global and are filtered by repository path in raise_event.
            monitoring.set_events(tool_id, events.RAISE)
            sys.setprofile(profile)
            backend = "sys.monitoring+profile"
        except Exception:
            try: monitoring.set_events(tool_id, 0)
            except Exception: pass
            try: monitoring.free_tool_id(tool_id)
            except Exception: pass
            tool_id = None
if tool_id is None:
    backend = "sys.settrace"
    sys.settrace(trace)
atexit.register(finish)
'''


class RuntimeProfiler:
    """Execute a real Python command and return a bounded, value-free profile."""

    def __init__(self, repo_root: str | Path, *, timeout_sec: float = 120,
                 max_events: int = 50_000):
        self.root = Path(repo_root).resolve()
        if not self.root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        if type(timeout_sec) not in (int, float) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be positive")
        self.timeout_sec = float(timeout_sec)
        self.max_events = max_events

    def run(self, command: Iterable[str], *, cwd: str | Path | None = None,
            env: dict[str, str] | None = None) -> dict:
        if isinstance(command, (str, bytes)):
            raise ValueError("command must be an argument array")
        args = list(command)
        if not args or any(type(arg) is not str or not arg for arg in args):
            raise ValueError("command must contain nonempty string arguments")
        directory = (Path(cwd).resolve() if cwd is not None else self.root)
        if not directory.is_relative_to(self.root) or not directory.is_dir():
            raise ValueError("profile cwd must be a directory inside repo_root")
        with tempfile.TemporaryDirectory(prefix="aegis-runtime-") as scratch:
            scratch_path = Path(scratch)
            profile_path = scratch_path / "profile.json"
            bootstrap = scratch_path / "sitecustomize.py"
            bootstrap.write_text(textwrap.dedent(_BOOTSTRAP), encoding="utf-8")
            process_env = os.environ.copy()
            process_env.update(env or {})
            old_path = process_env.get("PYTHONPATH", "")
            process_env["PYTHONPATH"] = str(scratch_path) + (os.pathsep + old_path if old_path else "")
            process_env["AEGIS_PROFILE_OUTPUT"] = str(profile_path)
            process_env["AEGIS_PROFILE_ROOT"] = str(self.root)
            process_env["AEGIS_PROFILE_MAX_EVENTS"] = str(self.max_events)
            try:
                completed = subprocess.run(args, cwd=directory, env=process_env,
                                           timeout=self.timeout_sec, check=False)
                exit_code = completed.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                exit_code = None
                timed_out = True
            if profile_path.is_file():
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            else:
                profile = {"schema_version": "aegis.runtime_profile.v1", "backend": "unavailable",
                           "functions": {}, "call_edges": [], "exceptions": [], "dynamic_attributes": [], "event_count": 0,
                           "event_limit_reached": False}
            profile.update({"exit_code": exit_code, "timed_out": timed_out,
                            "completed": not timed_out and exit_code == 0})
            return profile


def merge_into_impact(analyzer, profile: dict) -> dict:
    """Merge observed calls and resolved dynamic attribute references into impact."""
    resolved = 0
    added = 0
    dynamic_resolved = 0
    dynamic_added = 0

    def match(file: str, qualname: str):
        candidates = [key for key, symbol in analyzer.symbols.items()
                      if symbol.language == "python" and symbol.file == file
                      and (symbol.name == qualname or symbol.name.endswith("." + qualname))]
        return candidates[0] if len(candidates) == 1 else None

    for edge in profile.get("call_edges", ()):
        caller = match(edge.get("caller_file", ""), edge.get("caller_qualname", ""))
        callee = match(edge.get("callee_file", ""), edge.get("callee_qualname", ""))
        if not caller or not callee:
            continue
        resolved += 1
        analyzer.callees.setdefault(caller, set())
        analyzer.callers.setdefault(callee, set())
        analyzer.references.setdefault(caller, set())
        if callee not in analyzer.callees[caller]:
            added += 1
        analyzer.callees[caller].add(callee)
        analyzer.callers[callee].add(caller)
        analyzer.references[caller].add(callee)
    for event in profile.get("dynamic_attributes", ()):
        caller = match(event.get("caller_file", ""), event.get("caller_qualname", ""))
        target = event.get("target", "")
        if not caller or "::" not in target:
            continue
        target_file, target_name = target.split("::", 1)
        callee = match(target_file, target_name)
        if not callee:
            continue
        dynamic_resolved += 1
        analyzer.references.setdefault(caller, set())
        if callee not in analyzer.references[caller]:
            dynamic_added += 1
        analyzer.references[caller].add(callee)
    return {"observed_edges": len(profile.get("call_edges", ())),
            "merged_edges": resolved, "new_edges": added,
            "observed_dynamic_attributes": len(profile.get("dynamic_attributes", ())),
            "merged_dynamic_references": dynamic_resolved, "new_dynamic_references": dynamic_added}


def merge_into_semantic(analyzer, profile: dict) -> dict:
    """Connect runtime-observed ``getattr`` targets and calls to unresolved sites."""
    observed = {}
    observed_by_caller = {}
    for edge in profile.get("call_edges", ()):
        key = (edge.get("caller_file"), edge.get("caller_line"))
        observed.setdefault(key, set()).add(edge.get("callee"))
        owner = (edge.get("caller_file"), edge.get("caller_qualname"))
        observed_by_caller.setdefault(owner, set()).add(edge.get("callee"))
    unresolved_per_symbol_name = {}
    for reference in analyzer.dynamic_references:
        owner = reference.get("symbol", "").rsplit(":", 1)
        if len(owner) == 2:
            key = (reference.get("file"), owner[1], reference.get("name"))
            unresolved_per_symbol_name[key] = unresolved_per_symbol_name.get(key, 0) + 1
    resolved = []
    profiled_sites = set()
    resolved_profiled_sites = set()
    for reference in analyzer.dynamic_references:
        owner = reference.get("symbol", "").rsplit(":", 1)
        if len(owner) != 2:
            continue
        site = (reference.get("file"), owner[1], reference.get("line"))
        events = [item for item in profile.get("dynamic_attributes", ())
                  if item.get("caller_file") == site[0] and item.get("caller_qualname") == site[1]
                  and item.get("line") == site[2]
                  and (reference.get("dynamic_name") or item.get("name") == reference.get("name"))]
        if events:
            profiled_sites.add(site)
            if any(item.get("target") for item in events):
                resolved_profiled_sites.add(site)
    for reference in analyzer.dynamic_references:
        if reference.get("resolved"):
            continue
        owner = reference.get("symbol", "").rsplit(":", 1)
        if len(owner) != 2:
            continue
        candidates = [item for item in profile.get("dynamic_attributes", ())
                      if item.get("caller_file") == reference.get("file")
                      and item.get("caller_qualname") == owner[1]
                      and item.get("line") == reference.get("line")
                      and (reference.get("dynamic_name") or item.get("name") == reference.get("name"))]
        targets = sorted({item.get("target") for item in candidates if item.get("target")})
        if targets:
            observed_names = sorted({item.get("name") for item in candidates if item.get("name")})
            resolved.append({"file": reference["file"], "line": reference["line"],
                             "symbol": reference["symbol"], "name": reference["name"],
                             "observed_name": observed_names, "resolved_to": targets,
                             "result_schemas": sorted({item.get("result_schema", "") for item in candidates})})
    for reference in analyzer.dynamic_references:
        if reference.get("resolved") or any(item["file"] == reference.get("file")
                and item["line"] == reference.get("line") and item["symbol"] == reference.get("symbol")
                for item in resolved):
            continue
        if reference.get("dynamic_name"):
            continue
        candidates = observed.get((reference.get("file"), reference.get("line")), set())
        matching = sorted(candidate for candidate in candidates
                          if candidate and candidate.rsplit("::", 1)[-1].split(".")[-1] == reference.get("name"))
        if not matching:
            owner = reference.get("symbol", "").rsplit(":", 1)
            if len(owner) == 2 and unresolved_per_symbol_name.get(
                    (reference.get("file"), owner[1], reference.get("name"))) == 1:
                owner_edges = observed_by_caller.get((reference.get("file"), owner[1]), set())
                matching = sorted(candidate for candidate in owner_edges
                                  if candidate and candidate.rsplit("::", 1)[-1].split(".")[-1]
                                  == reference.get("name"))
        if len(matching) == 1:
            resolved.append({"file": reference["file"], "line": reference["line"],
                             "symbol": reference["symbol"], "name": reference["name"],
                             "resolved_to": matching[0]})
    return {"observed_call_edges": len(profile.get("call_edges", ())),
            "resolved_dynamic_sites": resolved, "dynamic_sites_resolved": len(resolved),
            "runtime_profiled_dynamic_sites": len(profiled_sites),
            "runtime_profiled_dynamic_sites_resolved": len(resolved_profiled_sites),
            "runtime_profiled_dynamic_resolution_rate": (
                len(resolved_profiled_sites) / len(profiled_sites) if profiled_sites else None)}

