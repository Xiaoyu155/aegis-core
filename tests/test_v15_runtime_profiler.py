"""Physical child-process profiling and runtime graph enrichment cases."""
import json
from pathlib import Path
import sys

import pytest

from aegis_core.impact_analyzer import ImpactAnalyzer
from aegis_core.runtime_profiler import RuntimeProfiler
from aegis_core.semantic_contract import SemanticContractAnalyzer


@pytest.mark.parametrize("case", range(40))
def test_runtime_profiler_captures_real_dynamic_dispatch_and_return_schema(case, tmp_path):
    name = f"save_{case}"
    source = f'''class Handler:
    def {name}(self, payload):
        return {{"accepted": True, "record_id": payload["record_id"], "attempt": {case}}}

def dispatch(handler, action, payload):
    selected = action
    callback = getattr(handler, selected)
    return callback(payload)

def make_callback(handler):
    return lambda payload: getattr(handler, "{name}")(payload)

handler = Handler()
callback = make_callback(handler)
result = dispatch(handler, "{name}", {{"record_id": {case}, "password": "must-not-leak"}})
assert result["accepted"] is True
assert callback({{"record_id": {case}}})["record_id"] == {case}
'''
    path = tmp_path / "profile_case.py"
    path.write_text(source, encoding="utf-8")
    report = RuntimeProfiler(tmp_path, timeout_sec=10, max_events=2000).run(
        [sys.executable, str(path)])

    assert report["completed"] is True
    assert report["backend"] in {"sys.monitoring+profile", "sys.settrace"}
    assert report["event_count"] > 0
    assert report["functions"][f"profile_case.py::Handler.{name}"]["calls"] == 2
    assert any(edge["callee"] == f"profile_case.py::Handler.{name}" for edge in report["call_edges"])
    returned = report["functions"][f"profile_case.py::Handler.{name}"]["returns"]
    shapes = [json.loads(shape) for shape in returned]
    assert any(shape.get("fields", {}).get("accepted") == "boolean"
               and shape.get("fields", {}).get("record_id") == "integer" for shape in shapes)
    assert "must-not-leak" not in json.dumps(report)
    assert "password" not in json.dumps(report)


def test_runtime_edges_resolve_ambiguous_getattr_and_merge_into_impact_graph(tmp_path):
    source = '''class Fast:
    def run(self, item):
        return {"kind": "fast", "count": item}

class Safe:
    def run(self, item):
        return {"kind": "safe", "count": item}

def dispatch(handler, item):
    return getattr(handler, "run")(item)

dispatch(Safe(), 4)
'''
    (tmp_path / "routes.py").write_text(source, encoding="utf-8")
    profile = RuntimeProfiler(tmp_path).run([sys.executable, str(tmp_path / "routes.py")])
    contracts = SemanticContractAnalyzer(tmp_path)
    static = contracts.analyze()
    assert static["metrics"]["unresolved_dynamic_sites"] == 1
    merged = contracts.analyze(profile)
    assert merged["metrics"]["runtime_resolved_dynamic_sites"] == 1
    assert merged["metrics"]["runtime_augmented_dynamic_resolution_rate"] == 1.0
    assert merged["metrics"]["runtime_profiled_dynamic_sites"] == 1
    assert merged["metrics"]["runtime_profiled_dynamic_resolution_rate"] == 1.0
    reference = merged["dynamic_references"][0]
    assert reference["resolution_kind"] == "runtime_observed"
    assert reference["resolved_to"] == ["routes.py::Safe.run"]
    assert any(item["caller_qualname"] == "dispatch" and item["name"] == "run"
               for item in profile["dynamic_attributes"])
    assert merged["runtime_schemas"]

    impact = ImpactAnalyzer(tmp_path)
    merge = impact.merge_runtime(profile)
    assert merge["new_edges"] >= 1
    result = impact.analyze({"file": "routes.py", "symbol": "Safe.run"}, max_depth=1)
    assert any(item["symbol"].endswith(":dispatch") and item["relation"] == "caller"
               for item in result["symbols"])


def test_runtime_edges_join_getattr_assignment_to_later_invocation_line(tmp_path):
    source = '''class Fast:
    def run(self):
        return {"kind": "fast"}

class Safe:
    def run(self):
        return {"kind": "safe"}

def dispatch(handler):
    callback = getattr(handler, "run")
    return callback()

dispatch(Fast())
'''
    (tmp_path / "routes.py").write_text(source, encoding="utf-8")
    profile = RuntimeProfiler(tmp_path).run([sys.executable, str(tmp_path / "routes.py")])
    analyzer = SemanticContractAnalyzer(tmp_path)
    static = analyzer.analyze()
    assert static["metrics"]["unresolved_dynamic_sites"] == 1
    merged = analyzer.analyze(profile)
    assert merged["metrics"]["runtime_resolved_dynamic_sites"] == 1
    assert merged["dynamic_references"][0]["resolved_to"] == ["routes.py::Fast.run"]


def test_runtime_merge_leaves_multiple_unexecuted_same_name_sites_unresolved(tmp_path):
    source = '''class Fast:
    def run(self):
        return 1

class Safe:
    def run(self):
        return 2

def dispatch(handler, use_second):
    if use_second:
        callback = getattr(handler, "run")
    else:
        callback = getattr(handler, "run")
    return callback()

dispatch(Fast(), False)
'''
    (tmp_path / "routes.py").write_text(source, encoding="utf-8")
    profile = RuntimeProfiler(tmp_path).run([sys.executable, str(tmp_path / "routes.py")])
    merged = SemanticContractAnalyzer(tmp_path).analyze(profile)
    assert merged["metrics"]["dynamic_sites"] == 2
    assert merged["metrics"]["runtime_resolved_dynamic_sites"] == 1
    assert merged["metrics"]["unresolved_dynamic_sites"] == 1


def test_runtime_profile_captures_variable_name_and_property_response_schema(tmp_path):
    source = '''class Profile:
    @property
    def response(self):
        return {"accepted": True, "count": 3}

def lookup(value, attribute):
    return getattr(value, attribute)

result = lookup(Profile(), "response")
assert result["accepted"] is True
'''
    (tmp_path / "profile.py").write_text(source, encoding="utf-8")
    profile = RuntimeProfiler(tmp_path).run([sys.executable, str(tmp_path / "profile.py")])
    contract = SemanticContractAnalyzer(tmp_path).analyze(profile)

    reference = contract["dynamic_references"][0]
    assert reference["dynamic_name"] is True
    assert reference["runtime_observed_name"] == ["response"]
    assert reference["resolved_to"] == ["type:builtins.dict"]
    assert contract["metrics"]["runtime_resolved_dynamic_sites"] == 1
    assert contract["runtime_schemas"][0]["observed_shapes"][0]["fields"] == {
        "accepted": "boolean", "count": "integer"}


def test_runtime_profile_is_bounded_and_marks_event_limit(tmp_path):
    (tmp_path / "hot.py").write_text(
        "def tick(value):\n    return value + 1\nfor index in range(300): tick(index)\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path, max_events=20).run([sys.executable, str(tmp_path / "hot.py")])
    assert report["exit_code"] == 0
    assert report["event_count"] <= 20
    assert report["event_limit_reached"] is True


def test_external_python_returns_do_not_exhaust_the_repository_event_budget(tmp_path):
    source = '''import json
def encode(value):
    return json.dumps(value)
for index in range(1):
    encode({"index": index})
def target():
    return {"observed": True}
target()
'''
    (tmp_path / "entry.py").write_text(source, encoding="utf-8")
    report = RuntimeProfiler(tmp_path, max_events=100).run(
        [sys.executable, str(tmp_path / "entry.py")])

    assert report["exit_code"] == 0
    assert report["event_count"] < 100
    assert report["functions"]["entry.py::target"]["calls"] == 1
    assert report["functions"]["entry.py::encode"]["calls"] == 1


def test_runtime_profile_records_exception_site_without_exception_message(tmp_path):
    (tmp_path / "error_path.py").write_text(
        "def fail():\n    raise RuntimeError('sensitive-payload-value')\n"
        "try:\n    fail()\nexcept RuntimeError:\n    pass\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path).run([sys.executable, str(tmp_path / "error_path.py")])
    assert any(item["type"] == "RuntimeError" and item["line"] == 2 for item in report["exceptions"])
    assert "sensitive-payload-value" not in json.dumps(report)


def test_runtime_profile_times_out_real_child_and_returns_no_false_success(tmp_path):
    (tmp_path / "waiter.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path, timeout_sec=0.1).run([sys.executable, str(tmp_path / "waiter.py")])
    assert report["timed_out"] is True
    assert report["completed"] is False
    assert report["exit_code"] is None


def test_profiler_rejects_string_commands_and_paths_outside_repository(tmp_path):
    profiler = RuntimeProfiler(tmp_path)
    with pytest.raises(ValueError, match="argument array"):
        profiler.run("python app.py")
    with pytest.raises(ValueError, match="inside repo_root"):
        profiler.run([sys.executable, "-V"], cwd=tmp_path.parent)


@pytest.mark.parametrize("bad_command", [[], [""], [sys.executable, ""]])
def test_profiler_rejects_empty_command_parts(tmp_path, bad_command):
    with pytest.raises(ValueError, match="nonempty string"):
        RuntimeProfiler(tmp_path).run(bad_command)


@pytest.mark.parametrize("timeout,max_events", [(0, 10), (-1, 10), (1, 0), (1, -1)])
def test_profiler_rejects_invalid_budgets(tmp_path, timeout, max_events):
    with pytest.raises(ValueError):
        RuntimeProfiler(tmp_path, timeout_sec=timeout, max_events=max_events)


def test_runtime_profile_report_has_no_command_environment_or_values(tmp_path):
    (tmp_path / "private.py").write_text(
        "import os\nTOKEN='PRIVATE_SENTINEL'\ndef run(password): return {'token': os.environ.get('PRIVATE_KEY')}\n"
        "run('VALUE_SENTINEL')\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path).run([sys.executable, str(tmp_path / "private.py")],
                                           env={"PRIVATE_KEY": "ENV_SENTINEL"})
    serialized = json.dumps(report)
    for secret in ("PRIVATE_SENTINEL", "VALUE_SENTINEL", "ENV_SENTINEL"):
        assert secret not in serialized


@pytest.mark.parametrize("function_count", [1, 2, 3, 5, 8])
def test_profiler_preserves_distinct_function_shapes_without_source_values(tmp_path, function_count):
    lines = []
    for index in range(function_count):
        lines.extend((f"def f{index}(payload):", f"    return {{'slot_{index}': payload}}", f"f{index}({index})"))
    path = tmp_path / "many.py"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path).run([sys.executable, str(path)])
    assert report["exit_code"] == 0
    assert sum(key.startswith("many.py::f") for key in report["functions"]) == function_count


@pytest.mark.parametrize("exit_status", [0, 1, 2, 17])
def test_runtime_profile_preserves_real_child_exit_status(tmp_path, exit_status):
    path = tmp_path / "status.py"
    path.write_text(f"raise SystemExit({exit_status})\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path).run([sys.executable, str(path)])
    assert report["exit_code"] == exit_status
    assert report["completed"] is (exit_status == 0)


@pytest.mark.parametrize("size", [0, 1, 24, 25])
def test_runtime_dict_shapes_bound_field_count_and_mark_truncation(tmp_path, size):
    fields = ", ".join(f"'field_{index:02}': {index}" for index in range(size))
    path = tmp_path / "shape.py"
    path.write_text(f"def inspect(payload):\n    return payload\ninspect({{{fields}}})\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path).run([sys.executable, str(path)])
    arguments = report["functions"]["shape.py::inspect"]["arguments"]["payload"]
    shape = json.loads(next(iter(arguments)))
    assert len(shape["fields"]) == min(size, 24)
    assert shape["truncated"] is (size > 24)


@pytest.mark.parametrize("size", [0, 16, 17])
def test_runtime_array_shapes_bound_element_type_samples(tmp_path, size):
    values = ", ".join(str(index) for index in range(size))
    path = tmp_path / "arrays.py"
    path.write_text(f"def inspect(values):\n    return len(values)\ninspect([{values}])\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path).run([sys.executable, str(path)])
    arguments = report["functions"]["arrays.py::inspect"]["arguments"]["values"]
    shape = json.loads(next(iter(arguments)))
    assert shape["type"] == "array"
    assert shape["items"] == (["integer"] if size else ["unknown"])
    assert shape["truncated"] is (size > 16)

