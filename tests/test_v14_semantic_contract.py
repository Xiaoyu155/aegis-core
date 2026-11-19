"""Cross-file schema inference, dynamic dispatch and semantic drift boundaries."""
from pathlib import Path
import re

import pytest

from aegis_core.semantic_contract import SemanticContractAnalyzer


def _repo(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


@pytest.mark.parametrize("field,annotation,expected", [
    ("user_id", "int", "integer"), ("display_name", "str", "string"),
    ("enabled", "bool", "boolean"), ("ratio", "float", "number"),
    ("payload", "dict", "object"), ("tags", "list[str]", "array[string]"), ("nothing", "None", "null"),
    ("custom", "AccountId", "AccountId"), ("maybe", "str | None", "null|string"),
    ("scores", "tuple[int, ...]", "array[integer]"), ("lookup", "Mapping[str, str]", "Mapping[string,string]"),
    ("union", "int | str", "integer|string"), ("items", "set[int]", "array[integer]"),
])
def test_typed_dict_field_types_are_inferred(field, annotation, expected, tmp_path):
    root = _repo(tmp_path, {"models.py": f"from typing import TypedDict\nclass Record(TypedDict):\n    {field}: {annotation}\n"})
    report = SemanticContractAnalyzer(root).analyze()
    record = next(row for row in report["schemas"] if row["schema"] == "models.py::Record")
    assert next(item["type"] for item in record["fields"] if item["name"] == field) == expected


@pytest.mark.parametrize("field,annotation,expected", [
    ("count", "int", "integer"), ("title", "str", "string"), ("active", "bool", "boolean"),
    ("cost", "float", "number"), ("values", "list[int]", "array[integer]"), ("resource", "object", "unknown"),
])
def test_dataclass_field_annotations_form_a_schema(field, annotation, expected, tmp_path):
    root = _repo(tmp_path, {"model.py": f"from dataclasses import dataclass\n@dataclass\nclass Row:\n    {field}: {annotation}\n"})
    report = SemanticContractAnalyzer(root).analyze()
    record = next(row for row in report["schemas"] if row["schema"] == "model.py::Row")
    assert record["fields"][0]["type"] == expected


@pytest.mark.parametrize("literal,expected", [
    ("'x'", "string"), ("3", "integer"), ("3.5", "number"), ("True", "boolean"),
    ("None", "null"), ("[]", "array[unknown]"), ("[1, 2]", "array[integer]"),
    ("{'nested': 1}", "object"), ("(1, 'a')", "array[integer|string]"),
    ("{1, 2}", "array[integer]"),
])
def test_dict_literal_values_get_stable_types(literal, expected, tmp_path):
    root = _repo(tmp_path, {"service.py": f"def load():\n    return {{'field': {literal}}}\n"})
    report = SemanticContractAnalyzer(root).analyze()
    schema = next(row for row in report["schemas"] if row["schema"] == "service.py::load")
    assert schema["fields"][0]["type"] == expected


@pytest.mark.parametrize("before,after,kind", [
    ("id: int", "id: str", "type"), ("id: int", "key: int", "remove"),
    ("id: int", "id: int\n    name: str", "add"), ("id: str", "id: bool", "type"),
    ("value: float", "value: int", "type"), ("old: bool", "new: bool", "remove"),
])
def test_schema_snapshot_comparison_reports_semantic_drift(before, after, kind, tmp_path):
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    _repo(old_root, {"types.py": f"from typing import TypedDict\nclass Contract(TypedDict):\n    {before}\n"})
    _repo(new_root, {"types.py": f"from typing import TypedDict\nclass Contract(TypedDict):\n    {after}\n"})
    diff = SemanticContractAnalyzer(new_root).compare(SemanticContractAnalyzer(old_root))
    key = "type_changes" if kind == "type" else "removed_fields" if kind == "remove" else "added_fields"
    assert diff[key]
    if kind == "add":
        assert diff["added_fields"] and not diff["breaking_changes"]
    else:
        assert diff["breaking_changes"]


@pytest.mark.parametrize("field", ["user_id", "email", "display_name", "created_at", "roles", "active"])
def test_renamed_python_import_propagates_return_schema_to_consumer(field, tmp_path):
    root = _repo(tmp_path, {
        "producer.py": f"def fetch_record():\n    return {{{field!r}: 'value'}}\n",
        "consumer.py": f"from producer import fetch_record as load\ndef use():\n    data = load()\n    return data[{field!r}]\n",
    })
    report = SemanticContractAnalyzer(root).analyze()
    assert any(edge["producer"] == "producer:fetch_record" and edge["field"] == field for edge in report["flows"])
    assert not report["mismatches"]


@pytest.mark.parametrize("field", ["legacy_id", "missing_name", "wrong_key", "unavailable", "deleted", "stale"])
def test_cross_module_implicit_field_deletion_is_a_contract_violation(field, tmp_path):
    root = _repo(tmp_path, {
        "producer.py": "def fetch_record():\n    return {'current_id': 1}\n",
        "consumer.py": f"from producer import fetch_record as load\ndef use():\n    data = load()\n    return data[{field!r}]\n",
    })
    report = SemanticContractAnalyzer(root).analyze()
    assert any(item["field"] == field and item["consumer"] == "consumer:use" for item in report["mismatches"])
    assert all(item["line"] > 0 for item in report["mismatches"])


@pytest.mark.parametrize("decorator,kind", [
    ("property", "descriptor"), ("staticmethod", "descriptor"), ("classmethod", "descriptor"),
    ("api.get('/users')", "route"), ("api.post('/users')", "route"),
    ("router.put('/users')", "route"), ("router.delete('/users')", "route"),
    ("blueprint.route('/users', methods=['GET'])", "route"),
])
def test_property_method_and_route_decorators_are_recognized(decorator, kind, tmp_path):
    signature = "@" + decorator + "\ndef endpoint(cls):\n    return None\n"
    if decorator in {"staticmethod", "classmethod"}:
        signature = "class View:\n    " + signature.replace("\n", "\n    ")
    root = _repo(tmp_path, {"routes.py": signature})
    report = SemanticContractAnalyzer(root).analyze()
    assert any(row["decorator"].endswith(decorator.split("(")[0].split(".")[-1]) for row in report["decorators"])


@pytest.mark.parametrize("registration", [
    "router.add(handler)", "events.connect(handler)", "app.get('/x', handler)",
    "router.post('/x', handler)", "signal.register(handler)", "bus.add_listener(handler)",
])
def test_callback_registration_is_linked_to_handler_name(registration, tmp_path):
    root = _repo(tmp_path, {"callbacks.py": f"def handler():\n    return 1\ndef install():\n    {registration}\n"})
    report = SemanticContractAnalyzer(root).analyze()
    assert any(row["callback"] == "handler" for row in report["callbacks"])


@pytest.mark.parametrize("field", ["userId", "account_id", "displayName", "createdAt", "active", "roles"])
def test_typescript_interface_schema_fields_are_indexed(field, tmp_path):
    root = _repo(tmp_path, {"src/types.ts": f"export interface User {{\n  {field}: string;\n}}\n"})
    report = SemanticContractAnalyzer(root).analyze()
    assert any(row["schema"] == "src/types.ts::User" and row["fields"][0]["name"] == field
               for row in report["schemas"])


@pytest.mark.parametrize("field", ["userId", "email", "displayName", "active", "createdAt", "role"])
def test_javascript_api_response_property_flow_is_linked_across_renamed_import(field, tmp_path):
    root = _repo(tmp_path, {
        "src/api.ts": f"export function loadUser() {{ return {{{field}: 'x'}}; }}\n",
        "src/view.ts": f"import {{ loadUser as fetchUser }} from './api';\nexport function render() {{ const response = fetchUser(); return response.{field}; }}\n",
    })
    report = SemanticContractAnalyzer(root).analyze()
    assert any(item["field"] == field and item["consumer"] == "src/view.ts:render" for item in report["flows"])


@pytest.mark.parametrize("markup", [
    "<script src='./api.js'></script><script>function render(){const data=load(); return data.userId;}</script>",
    "<script src='./api.js'></script><script>function render(){const data=load(); return data.email;}</script>",
    "<script src='./api.js'></script><script>function render(){const data=load(); return data.active;}</script>",
    "<script src='./api.js'></script><script>function render(){const data=load(); return data.role;}</script>",
])
def test_html_inline_script_tracks_external_script_response_fields(markup, tmp_path):
    field = re.search(r"data\.(\w+)", markup).group(1)
    root = _repo(tmp_path, {"web/api.js": f"function load() {{ return {{{field}: 1}}; }}\n",
                            "web/index.html": markup})
    report = SemanticContractAnalyzer(root).analyze()
    assert any(item["consumer"].startswith("web/index.html:") and item["field"] == field
               for item in report["flows"])


@pytest.mark.parametrize("target", ["handle", "dispatch", "resolve", "render", "save", "lookup"])
def test_dynamic_python_getattr_is_resolved_only_for_unique_symbols(target, tmp_path):
    root = _repo(tmp_path, {"dispatch.py": f"def {target}():\n    return 1\ndef call(obj):\n    return getattr(obj, {target!r})()\n"})
    report = SemanticContractAnalyzer(root).analyze()
    assert any(row["name"] == target and row["resolved"] for row in report["dynamic_references"])


def test_ambiguous_dynamic_python_dispatch_is_retained_as_unresolved(tmp_path):
    root = _repo(tmp_path, {"a.py": "def run(): return 1\n", "b.py": "def run(): return 2\n",
                            "dispatch.py": "def call(obj): return getattr(obj, 'run')()\n"})
    report = SemanticContractAnalyzer(root).analyze()
    assert not next(row for row in report["dynamic_references"] if row["name"] == "run")["resolved"]


def test_typed_receiver_disambiguates_same_named_methods_across_classes(tmp_path):
    root = _repo(tmp_path, {"dispatch.py": """class Primary:
    def execute(self): return 'primary'
    def invoke(self): return getattr(self, 'execute')()
class Fallback:
    def execute(self): return 'fallback'
"""})
    report = SemanticContractAnalyzer(root).analyze()
    dynamic = next(row for row in report["dynamic_references"] if row["name"] == "execute")
    assert dynamic["resolved"] is True
    assert dynamic["receiver_type"] == "Primary"
    assert dynamic["resolved_to"] == ["dispatch:Primary.execute"]
    assert dynamic["resolution_kind"] == "typed_method"


def test_typed_receiver_resolves_class_attribute_schema_without_guessing(tmp_path):
    root = _repo(tmp_path, {"models.py": """from dataclasses import dataclass
@dataclass
class Record:
    record_id: int
    def read(self): return getattr(self, 'record_id')
"""})
    dynamic = next(row for row in SemanticContractAnalyzer(root).analyze()["dynamic_references"]
                   if row["name"] == "record_id")
    assert dynamic["resolved"] is True
    assert dynamic["resolution_kind"] == "typed_field"
    assert dynamic["resolved_to"] == ["models.py::Record::record_id"]


def test_dynamic_json_field_follows_aliased_cross_module_import(tmp_path):
    root = _repo(tmp_path, {
        "producer.py": "def fetch_record():\n    return {'record_id': 17}\n",
        "consumer.py": "from producer import fetch_record as load\n"
                      "def read_record():\n    response = load()\n    return getattr(response, 'record_id')\n",
    })
    dynamic = next(row for row in SemanticContractAnalyzer(root).analyze()["dynamic_references"]
                   if row["name"] == "record_id")
    assert dynamic["resolved"] is True
    assert dynamic["resolution_kind"] == "producer_field_flow"
    assert dynamic["resolved_to"] == ["producer:fetch_record::record_id"]


def test_dynamic_field_flows_through_local_alias_and_iterated_record_list(tmp_path):
    root = _repo(tmp_path, {"steps.py": "def load_steps():\n    return [{'action': 'open'}]\n"
                                     "def first_action():\n    steps = load_steps()\n    for step in steps:\n"
                                     "        alias = step\n        return getattr(alias, 'action')\n"})
    dynamic = next(row for row in SemanticContractAnalyzer(root).analyze()["dynamic_references"]
                   if row["name"] == "action")
    assert dynamic["resolved"] is True
    assert dynamic["resolution_kind"] == "producer_field_flow"
    assert dynamic["resolved_to"] == ["steps:load_steps::action"]


def test_cli_dynamic_namespace_field_resolves_to_unique_declared_argument(tmp_path):
    root = _repo(tmp_path, {"cli.py": """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--retry-count')
def get_retry_count(args):
    return getattr(args, 'retry_count')
"""})
    report = SemanticContractAnalyzer(root).analyze()
    dynamic = next(row for row in report["dynamic_references"] if row["name"] == "retry_count")
    assert dynamic["resolved"] is True
    assert dynamic["resolution_kind"] == "argparse_option"
    assert dynamic["resolved_to"] == ["cli.py::argparse.Namespace::retry_count"]


def test_cli_explicit_dest_overrides_option_spelling_in_schema(tmp_path):
    root = _repo(tmp_path, {"cli.py": """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--retry-count', dest='attempts')
def get_attempts(args):
    return getattr(args, 'attempts')
"""})
    dynamic = next(row for row in SemanticContractAnalyzer(root).analyze()["dynamic_references"]
                   if row["name"] == "attempts")
    assert dynamic["resolved"] is True
    assert dynamic["resolved_to"] == ["cli.py::argparse.Namespace::attempts"]


def test_analyzer_does_not_execute_repository_source(tmp_path):
    marker = tmp_path / "executed"
    root = _repo(tmp_path / "repo", {"app.py": f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef run(): return 1\n"})
    SemanticContractAnalyzer(root).analyze()
    assert not marker.exists()


def test_json_subscript_consumer_and_typed_dict_contract_are_cross_linked(tmp_path):
    root = _repo(tmp_path, {"schema.py": "from typing import TypedDict\nclass Payload(TypedDict):\n    record_id: int\ndef fetch(): return {'record_id': 1}\n",
                            "handler.py": "from schema import fetch\ndef respond():\n    body = fetch()\n    return body['record_id']\n"})
    report = SemanticContractAnalyzer(root).analyze()
    assert any(item["field"] == "record_id" and item["consumer"] == "handler:respond" for item in report["flows"])
    assert report["metrics"]["fields"] >= 2


def test_metrics_report_dynamic_resolution_and_unresolved_baseline_counts(tmp_path):
    root = _repo(tmp_path, {"app.py": "def save(): return 1\ndef run(x): return getattr(x, 'save')()\n"})
    metrics = SemanticContractAnalyzer(root).analyze()["metrics"]
    assert metrics["dynamic_sites"] == 1
    assert metrics["resolved_dynamic_sites"] == 1
    assert metrics["unresolved_dynamic_sites"] == 0
    assert metrics["dynamic_unresolved_reduction_rate"] == 1.0
    assert 0 <= metrics["dynamic_resolution_rate"] <= 1
