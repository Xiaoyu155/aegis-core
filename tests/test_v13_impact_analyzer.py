"""Boundary and budget tests for the mixed-language Phase 13 impact graph."""
from pathlib import Path

import pytest

from aegis_core.impact_analyzer import ContextBudgetError, ImpactAnalyzer, context_token_count


def _write(root: Path, files: dict[str, str]) -> Path:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def _by_name(report, suffix):
    return next(item for item in report["symbols"] if item["symbol"].endswith(suffix))


def test_python_caller_and_callee_depths_are_directional(tmp_path):
    root = _write(tmp_path, {"app.py": "def top():\n    middle()\ndef middle():\n    leaf()\ndef leaf():\n    return 1\n"})
    report = ImpactAnalyzer(root).analyze("app.py::leaf", max_depth=2)
    assert [(item["name"], item["distance"]) for item in report["upstream_callers"]] == [("middle", 1), ("top", 2)]
    assert report["downstream_callees"] == []


def test_python_two_way_neighbor_walk_is_cycle_safe(tmp_path):
    root = _write(tmp_path, {"cycle.py": "def a():\n    b()\ndef b():\n    a()\ndef c():\n    a()\n"})
    report = ImpactAnalyzer(root).analyze("cycle.py::a", max_depth=5)
    assert len({item["symbol"] for item in report["symbols"]}) == len(report["symbols"])
    assert _by_name(report, ":b")["distance"] == 1
    assert _by_name(report, ":c")["direction"] == "upstream"


def test_python_cross_file_renamed_import_resolves_to_original_symbol(tmp_path):
    root = _write(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/worker.py": "def execute_job(value: int) -> int:\n    return value + 1\n",
        "pkg/api.py": "from .worker import execute_job as run\ndef endpoint(value: int) -> int:\n    return run(value)\n",
    })
    report = ImpactAnalyzer(root).analyze("pkg/worker.py::execute_job")
    assert any(item["symbol"] == "pkg.api:endpoint" for item in report["upstream_callers"])


def test_python_type_reference_graph_tracks_class_annotations(tmp_path):
    root = _write(tmp_path, {
        "types.py": "class Account:\n    pass\n",
        "service.py": "from types import Account\ndef load() -> Account:\n    return Account()\n",
    })
    analyzer = ImpactAnalyzer(root)
    report = analyzer.analyze("types.py::Account", max_depth=1)
    assert any(item["symbol"] == "service:load" for item in report["upstream_callers"])
    assert analyzer.references["service:load"]


def test_javascript_local_function_call_is_indexed(tmp_path):
    root = _write(tmp_path, {"src/app.js": "function run(){ return helper(); }\nfunction helper(){ return 1; }\n"})
    report = ImpactAnalyzer(root).analyze("src/app.js::helper")
    assert [item["name"] for item in report["upstream_callers"]] == ["run"]


def test_typescript_signature_preserves_parameter_and_return_types(tmp_path):
    root = _write(tmp_path, {"src/math.ts": "export function add(left: number, right: number): number { return left + right; }\n"})
    report = ImpactAnalyzer(root).analyze("src/math.ts::add")
    assert "left: number" in report["symbols"][0]["signature"]
    assert "): number" in report["symbols"][0]["signature"]


def test_renamed_javascript_import_resolves_reference_and_call(tmp_path):
    root = _write(tmp_path, {
        "src/math.js": "export function addOne(value){ return value + 1; }\n",
        "src/app.js": "import { addOne as increment } from './math.js';\nexport function run(){ return increment(2); }\n",
    })
    report = ImpactAnalyzer(root).analyze("src/math.js::addOne")
    assert any(item["symbol"] == "src/app.js:run" for item in report["upstream_callers"])


def test_html_inline_script_resolves_a_loaded_script_call(tmp_path):
    root = _write(tmp_path, {
        "web/helpers.js": "function paint(){ return 'ok'; }\n",
        "web/index.html": "<script src='./helpers.js'></script>\n<script>\nfunction start(){ return paint(); }\n</script>\n",
    })
    report = ImpactAnalyzer(root).analyze("web/helpers.js::paint")
    assert any(item["symbol"] == "web/index.html:start" for item in report["upstream_callers"])


def test_html_inline_scripts_share_document_global_scope(tmp_path):
    root = _write(tmp_path, {"page.html": "<script>function first(){ return second(); }</script>\n<script>function second(){ return 2; }</script>"})
    report = ImpactAnalyzer(root).analyze("page.html::second")
    assert any(item["symbol"].startswith("page.html:first") for item in report["upstream_callers"])


def test_dynamic_javascript_property_call_is_unresolved_not_guessed(tmp_path):
    root = _write(tmp_path, {"app.js": "function save(){ return 1; }\nfunction dispatch(obj, method){ return obj[method](); }\n"})
    analyzer = ImpactAnalyzer(root)
    dispatch = "app.js:dispatch"
    assert analyzer.callees[dispatch].isdisjoint({"app.js:save"})
    assert any("<dynamic>" not in item and "method" in item for item in analyzer.unresolved[dispatch])


def test_python_dynamic_getattr_is_reported_as_unresolved(tmp_path):
    root = _write(tmp_path, {"app.py": "def call(obj, name):\n    return getattr(obj, name)()\n"})
    analyzer = ImpactAnalyzer(root)
    assert "getattr(obj, <dynamic>)" in analyzer.unresolved["app:call"]


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 8])
def test_requested_depth_bounds_upstream_walk(tmp_path, depth):
    root = _write(tmp_path, {"m.py": "def a(): b()\ndef b(): c()\ndef c(): d()\ndef d(): pass\n"})
    report = ImpactAnalyzer(root).analyze("m.py::d", max_depth=depth)
    assert max((item["distance"] for item in report["symbols"]), default=0) <= depth


@pytest.mark.parametrize("bad", [-1, 33, 1.0, True])
def test_depth_rejects_invalid_values(tmp_path, bad):
    root = _write(tmp_path, {"m.py": "def f(): pass\n"})
    with pytest.raises(ValueError):
        ImpactAnalyzer(root).analyze("m.py::f", max_depth=bad)


@pytest.mark.parametrize("selector", ["../escape.py::f", "/tmp/escape.py::f"])
def test_target_path_cannot_escape_repository(tmp_path, selector):
    root = _write(tmp_path, {"m.py": "def f(): pass\n"})
    with pytest.raises((ValueError, KeyError)):
        ImpactAnalyzer(root).analyze(selector)


def test_unindexed_symbol_is_an_error(tmp_path):
    root = _write(tmp_path, {"m.py": "def actual(): pass\n"})
    with pytest.raises(KeyError):
        ImpactAnalyzer(root).analyze("m.py::missing")


def test_direct_file_is_kept_complete_in_compressed_context(tmp_path):
    source = "def target(value: int) -> int:\n    # preserve every line\n    return value + 1\n\nCONSTANT = 3\n"
    root = _write(tmp_path, {"app.py": source})
    result = ImpactAnalyzer(root).compress_context("app.py::target", max_context_tokens=80)
    assert source in result["context"]
    assert result["items"][0]["representation"] == "full_source"


def test_direct_file_budget_overflow_is_explicit(tmp_path):
    root = _write(tmp_path, {"large.py": "def target():\n" + "    value = 1\n" * 100 + "    return value\n"})
    with pytest.raises(ContextBudgetError) as error:
        ImpactAnalyzer(root).compress_context("large.py::target", max_context_tokens=12)
    assert error.value.required_tokens > error.value.budget_tokens


def test_one_hop_context_uses_signature_and_keeps_annotations(tmp_path):
    root = _write(tmp_path, {
        "api.py": "from helper import normalize\ndef changed(value: str) -> str:\n    return normalize(value)\n",
        "helper.py": "def normalize(value: str) -> str:\n    return value.strip()\n",
    })
    result = ImpactAnalyzer(root).compress_context("api.py::changed", max_context_tokens=80)
    helper = next(item for item in result["items"] if item["symbol"] == "helper:normalize")
    assert helper["representation"] == "signature"
    assert "value: str" in helper["text"] and "-> str" in helper["text"]


def test_two_hop_context_uses_abstract_summary(tmp_path):
    root = _write(tmp_path, {
        "a.py": "from b import second\ndef first(): return second()\n",
        "b.py": "from c import third\ndef second(): return third()\n",
        "c.py": "def third() -> int:\n    return 42\n",
    })
    result = ImpactAnalyzer(root).compress_context("a.py::first", max_context_tokens=80)
    third = next(item for item in result["items"] if item["symbol"] == "c:third")
    assert third["representation"] == "abstract_summary"
    assert "third" in third["text"]


def test_context_stays_inside_budget_and_reports_exact_lexical_count(tmp_path):
    root = _write(tmp_path, {"m.py": "def a(): return b()\ndef b(): return c()\ndef c(): return 1\n"})
    result = ImpactAnalyzer(root).compress_context("m.py::a", max_context_tokens=60)
    assert result["token_count"] == context_token_count(result["context"])
    assert result["token_count"] <= 60


def test_budget_drops_lower_priority_symbols_and_reports_recall(tmp_path):
    root = _write(tmp_path, {
        "a.py": "from b import direct\ndef target(): return direct()\n",
        "b.py": "from c import distant\ndef direct(): return distant()\n",
        "c.py": "def distant(): return 1\n",
    })
    result = ImpactAnalyzer(root).compress_context("a.py::target", max_context_tokens=30)
    assert result["critical_symbol_count"] >= result["recalled_symbol_count"]
    assert 0 <= result["critical_symbol_recall"] <= 1
    assert result["omitted_symbols"] or result["critical_symbol_recall"] == 1


def test_compression_ratio_is_bounded_and_source_based(tmp_path):
    root = _write(tmp_path, {"m.py": "def target(): return helper()\ndef helper():\n    return 1\n" + "# note\n" * 20})
    result = ImpactAnalyzer(root).compress_context("m.py::target", max_context_tokens=80)
    assert 0 <= result["compression_ratio"] <= 1
    assert result["naive_token_count"] > 0


def test_multiple_changed_files_are_all_complete(tmp_path):
    first, second = "def one(): return 1\n", "def two(): return 2\n"
    root = _write(tmp_path, {"one.py": first, "two.js": "function two(){ return 2; }\n"})
    result = ImpactAnalyzer(root).compress_context(["one.py::one", "two.js::two"], max_context_tokens=80)
    assert first in result["context"]
    assert "function two(){ return 2; }" in result["context"]
    assert set(result["changed_files"]) == {"one.py", "two.js"}


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_context_budget_requires_positive_integer(tmp_path, budget):
    root = _write(tmp_path, {"m.py": "def f(): pass\n"})
    with pytest.raises(ValueError):
        ImpactAnalyzer(root).compress_context("m.py::f", max_context_tokens=budget)


def test_context_lexical_counter_is_unicode_stable():
    assert context_token_count("修复 API: clamp(最大值)") == 7


def test_index_does_not_execute_python_module(tmp_path):
    marker = tmp_path / "executed"
    root = _write(tmp_path, {"m.py": f"from pathlib import Path\nPath({str(marker)!r}).write_text('no')\ndef f(): pass\n"})
    ImpactAnalyzer(root).analyze("m.py::f")
    assert not marker.exists()


def test_graph_metrics_include_language_and_edge_counts(tmp_path):
    root = _write(tmp_path, {"m.py": "def a(): return b()\ndef b(): return 1\n"})
    metrics = ImpactAnalyzer(root).graph_metrics
    assert metrics["symbols"] == 2
    assert metrics["call_edges"] == 1
    assert metrics["files"] == 1


def test_js_ambiguous_dependency_name_is_not_guessed(tmp_path):
    root = _write(tmp_path, {
        "a.js": "export function common(){ return 1; }\n",
        "b.js": "export function common(){ return 2; }\n",
        "main.js": "import './a.js'; import './b.js'; function run(){ return common(); }\n",
    })
    analyzer = ImpactAnalyzer(root)
    assert not analyzer.callees["main.js:run"]


def test_class_method_this_call_resolves_within_class(tmp_path):
    root = _write(tmp_path, {"service.js": "class Worker { run(){ return this.execute(); } execute(){ return 1; } }\n"})
    report = ImpactAnalyzer(root).analyze("service.js::Worker.execute")
    assert any(item["symbol"] == "service.js:Worker.run" for item in report["upstream_callers"])


def test_reference_records_imported_symbols_without_call(tmp_path):
    root = _write(tmp_path, {
        "value.js": "export function hydrate(){ return {}; }\n",
        "types.ts": "import { hydrate as makeState } from './value.js';\nexport const create = () => makeState;\n",
    })
    analyzer = ImpactAnalyzer(root)
    assert "value.js:hydrate" in analyzer.references["types.ts:create"]
