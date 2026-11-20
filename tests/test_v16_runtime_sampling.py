"""Call-site reservoir sampling under repeated real child-process execution."""
import json
import re
import sys

import pytest

from aegis_core.runtime_profiler import RuntimeProfiler


@pytest.mark.parametrize("case", range(20))
def test_repeated_call_site_aggregates_argument_schema_without_spending_slots(case, tmp_path):
    calls = 200 + case
    source = f'''def consume(payload):
    return payload

for index in range({calls}):
    consume({{"field_{case}": [index]}})
'''
    path = tmp_path / "hot.py"
    path.write_text(source, encoding="utf-8")
    report = RuntimeProfiler(tmp_path, timeout_sec=15, max_events=128).run([sys.executable, str(path)])
    edge = next(item for item in report["call_edges"] if item["callee_qualname"] == "consume")
    sample = next(item for item in report["reservoir_sample"]
                  if item["kind"] == "call_edge" and item["site"][1].endswith("::consume"))
    assert report["completed"] is True
    assert report["functions"]["hot.py::consume"]["calls"] == calls
    assert edge["count"] == calls
    assert len(re.fullmatch(r"[0-9a-f]{64}", edge["arg_schema_hash"]).group(0)) == 64
    assert sample["count"] == calls
    assert report["reservoir_size"] <= 128
    assert report["unique_event_count"] < calls


@pytest.mark.parametrize("case", range(20))
def test_dynamic_getattr_hot_site_is_counted_once_and_keeps_target_schema(case, tmp_path):
    calls = 150 + case
    source = f'''class Handler:
    def method_{case}(self, payload):
        return {{"case": {case}, "value": payload}}

def dispatch(handler, name, payload):
    return getattr(handler, name)(payload)

handler = Handler()
for index in range({calls}):
    dispatch(handler, "method_{case}", index)
'''
    path = tmp_path / "dispatch.py"
    path.write_text(source, encoding="utf-8")
    report = RuntimeProfiler(tmp_path, timeout_sec=15, max_events=128).run([sys.executable, str(path)])
    event = next(item for item in report["dynamic_attributes"] if item["name"] == f"method_{case}")
    sample = next(item for item in report["reservoir_sample"]
                  if item["kind"] == "dynamic_getattr" and item["site"][2] == f"method_{case}")
    assert report["completed"] is True
    assert event["count"] == calls
    assert event["target"] == f"dispatch.py::Handler.method_{case}"
    assert json.loads(event["result_schema"]) == "method"
    assert sample["count"] == calls
    assert report["reservoir_size"] <= 128
    assert report["unique_event_count"] < calls


def test_hundred_thousand_hot_calls_leave_room_for_tail_cold_path(tmp_path):
    source = '''def tick(value):
    return value + 1

def cold_branch():
    return {"cold": True}

for index in range(100_000):
    tick(index)
result = cold_branch()
assert result["cold"] is True
'''
    path = tmp_path / "load.py"
    path.write_text(source, encoding="utf-8")
    report = RuntimeProfiler(tmp_path, timeout_sec=45, max_events=128).run([sys.executable, str(path)])
    tick_edges = [item for item in report["call_edges"] if item["callee_qualname"] == "tick"]
    cold_edges = [item for item in report["call_edges"] if item["callee_qualname"] == "cold_branch"]
    assert tick_edges and cold_edges, json.dumps({
        "backend": report.get("backend"),
        "function_keys": sorted(report.get("functions", {})),
        "edge_callees": [item.get("callee_qualname") for item in report.get("call_edges", [])],
        "reservoir_kinds": sorted({item.get("kind") for item in report.get("reservoir_sample", [])}),
    }, sort_keys=True)
    tick = tick_edges[0]
    cold = cold_edges[0]
    assert report["completed"] is True
    assert report["functions"]["load.py::tick"]["calls"] == 100_000
    assert tick["count"] == 100_000
    assert cold["count"] == 1
    assert report["call_edge_unique_count"] < 8
    assert report["call_edge_repeat_hits"] >= 99_999
    assert report["reservoir_size"] <= 128
    assert report["unique_event_count"] < 128
    assert report["event_count"] == 128
    assert report["event_limit_reached"] is True
    assert report["reservoir_replacements"] == 0
    assert any(item["callee_qualname"] == "cold_branch" for item in report["call_edges"])


def test_call_edge_signature_aggregates_distinct_lines(tmp_path):
    path = tmp_path / "lines.py"
    path.write_text('''def submit():
    return True

submit()
submit()
''', encoding="utf-8")
    report = RuntimeProfiler(tmp_path, max_events=32).run([sys.executable, str(path)])
    edges = [item for item in report["call_edges"] if item["callee_qualname"] == "submit"]
    samples = [item for item in report["reservoir_sample"]
               if item["kind"] == "call_edge" and item["site"][1].endswith("::submit")]
    assert len(edges) == 1
    assert edges[0]["count"] == 2
    assert len(samples) == 1
    assert samples[0]["count"] == 2


def test_priority_call_paths_survive_secondary_event_churn(tmp_path):
    noise = "\n".join(f"    marker_{index} = {index}" for index in range(80))
    source = f'''def noise():
{noise}

def pulse(value):
    return value

def cold_branch():
    return True

noise()
for index in range(2_000):
    pulse(index)
cold_branch()
'''
    path = tmp_path / "priority.py"
    path.write_text(source, encoding="utf-8")
    report = RuntimeProfiler(tmp_path, timeout_sec=20, max_events=16).run([sys.executable, str(path)])
    edges = {item["callee_qualname"]: item for item in report["call_edges"]}
    assert report["completed"] is True
    assert edges["pulse"]["count"] == 2_000
    assert edges["cold_branch"]["count"] == 1
    assert report["reservoir_size"] <= 16


@pytest.mark.parametrize("limit", [1, 2, 4, 8, 16, 32])
def test_reservoir_stays_within_budget_and_records_replacements(limit, tmp_path):
    functions = "\n".join(f"def branch_{index}(): return {index}" for index in range(24))
    calls = "\n".join(f"branch_{index}()" for index in range(24))
    path = tmp_path / "many.py"
    path.write_text(functions + "\n" + calls + "\n", encoding="utf-8")
    report = RuntimeProfiler(tmp_path, timeout_sec=15, max_events=limit).run([sys.executable, str(path)])
    assert report["completed"] is True
    assert report["reservoir_size"] <= limit
    assert report["unique_event_count"] >= report["reservoir_size"]
    assert report["reservoir_replacements"] <= report["unique_event_count"]
    assert len(report["reservoir_sample"]) == report["reservoir_size"]


def test_profile_records_call_edge_argument_schema_hash_without_values(tmp_path):
    path = tmp_path / "schema.py"
    path.write_text('''def submit_order(amount, password_token):
    return amount > 0

submit_order(12.5, "never-serialize-this")
''', encoding="utf-8")
    report = RuntimeProfiler(tmp_path, max_events=64).run([sys.executable, str(path)])
    edge = next(item for item in report["call_edges"] if item["callee_qualname"] == "submit_order")
    assert re.fullmatch(r"[0-9a-f]{64}", edge["arg_schema_hash"])
    assert "never-serialize-this" not in json.dumps(report)
    assert "password_token" not in json.dumps(report)
