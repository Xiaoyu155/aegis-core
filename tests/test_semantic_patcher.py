"""Symbol-anchored edits and diff recovery over real Python/JS source text."""
import difflib
import os
from pathlib import Path

import pytest

from aegis_core.patch_applier import PatchApplicationError, apply_patch
from aegis_core.semantic_patcher import (
    SemanticPatchError,
    apply_file_patch,
    apply_structured_patch,
    javascript_symbols,
    python_symbols,
    rollback_file_patch,
)


@pytest.mark.parametrize("prefix", ["", "# header\n", "# one\n# two\n", "\n\n# shifted\n", "# a\n# b\n# c\n"])
@pytest.mark.parametrize("spacing", ["", " ", "  "])
def test_python_function_replace_uses_ast_symbol_after_line_and_whitespace_drift(prefix, spacing):
    source = prefix + "def clamp(value):\n    if value <= 10:\n        return value\n    return 10\n"
    source = source.replace("if value <=", "if " + spacing + "value  <=")
    result = apply_structured_patch(source, symbol_name="clamp",
                                    code="def clamp(value):\n    return min(10, max(0, value))")
    assert result.startswith(prefix)
    assert "def clamp(value):\n    return min(10, max(0, value))" in result
    compile(result, "<patched>", "exec")


@pytest.mark.parametrize("symbol,expected", [
    ("Outer.Inner.normalize", "return value + 1"),
    ("Inner.normalize", "return value + 2"),
    ("normalize", "return value + 3"),
])
def test_nested_python_class_method_can_be_resolved_at_three_symbol_depths(symbol, expected):
    source = "class Outer:\n    class Inner:\n        def normalize(self, value):\n            return value\n"
    result = apply_structured_patch(source, symbol_name=symbol,
                                    code=f"def normalize(self, value):\n    {expected}")
    assert expected in result
    compile(result, "<nested>", "exec")


@pytest.mark.parametrize("operation,needle", [
    ("before", "def audit"),
    ("after", "def audit"),
    ("replace", "def audit"),
])
def test_python_before_after_and_replace_keep_the_module_parseable(operation, needle):
    source = "def target(value):\n    return value\n"
    result = apply_structured_patch(source, symbol_name="target", operation=operation,
                                    code="def audit(value):\n    return value + 1")
    assert needle in result
    assert result.count("def target") == (0 if operation == "replace" else 1)
    compile(result, "<injection>", "exec")


def test_decorated_python_function_replacement_includes_decorator_span():
    source = "def trace(fn):\n    return fn\n\n@trace\ndef run(value):\n    return value\n"
    result = apply_structured_patch(source, symbol_name="run",
                                    code="@trace\ndef run(value):\n    return value + 1")
    assert result.count("@trace") == 1
    compile(result, "<decorated>", "exec")


def test_same_simple_name_in_two_python_classes_requires_qualified_symbol():
    source = "class A:\n    def work(self):\n        return 1\nclass B:\n    def work(self):\n        return 2\n"
    with pytest.raises(SemanticPatchError, match="ambiguous"):
        apply_structured_patch(source, symbol_name="work", code="def work(self):\n    return 3")
    assert "A.work" in str(python_symbols(source)[1].qualified_name) or len(python_symbols(source)) == 5


@pytest.mark.parametrize("operation", ["replace", "before", "after"])
def test_invalid_python_candidate_is_rejected_before_return(operation):
    source = "def target():\n    return 1\n"
    with pytest.raises(SemanticPatchError, match="invalid Python"):
        apply_structured_patch(source, symbol_name="target", operation=operation,
                               code="def broken(:\n    pass")


@pytest.mark.parametrize("source", [
    "function load(url) { return fetch(url); }\n",
    "async function load(url) { return await fetch(url); }\n",
    "const load = (url) => { return fetch(url); };\n",
    "let load = async (url) => { return await fetch(url); };\n",
])
def test_javascript_function_and_arrow_symbols_are_replaced_without_line_numbers(source):
    name = "load"
    result = apply_structured_patch(source, symbol_name=name, language="javascript",
                                    code="function load(url) { return url + '/safe'; }")
    assert "'/safe'" in result
    javascript_symbols(result)


@pytest.mark.parametrize("method", ["run", "load", "onReady", "getValue", "dispatch"])
def test_javascript_class_methods_resolve_by_qualified_name_and_keep_braces_balanced(method):
    source = f"class Handler {{\n  {method}(value) {{ return value; }}\n}}\n"
    result = apply_structured_patch(source, symbol_name=f"Handler.{method}", language="ts",
                                    code=f"{method}(value) {{ return value + 1; }}")
    assert "return value + 1" in result
    javascript_symbols(result)


def test_javascript_comments_strings_and_template_braces_do_not_corrupt_symbol_ranges():
    source = "// fake() { }}\nclass Handler {\n  run(value) { const s = `{{${value}}}`; return value; }\n}\n"
    result = apply_structured_patch(source, symbol_name="Handler.run", language="js",
                                    code="run(value) { const s = `{{${value}}}`; return value + 1; }")
    assert result.startswith("// fake() { }}\n")
    javascript_symbols(result)


@pytest.mark.parametrize("prefix", ["# offset\n", "# offset\n\n", "# a\n# b\n", "# x\n# y\n# z\n", "\n# offset\n"])
@pytest.mark.parametrize("spaces", ["  ", "   "])
def test_unified_diff_falls_back_to_python_ast_context_after_offset_and_spacing_changes(tmp_path, prefix, spaces):
    original = "def clamp(value):\n    if value <= 10:\n        return value\n    return 10\n"
    revised = original.replace("value <= 10", "value < 10")
    patch = "".join(difflib.unified_diff(original.splitlines(True), revised.splitlines(True),
                                        fromfile="a/module.py", tofile="b/module.py"))
    root = tmp_path / "repo"
    root.mkdir()
    changed = prefix + original.replace("if value", "if" + spaces + "value")
    (root / "module.py").write_text(changed, encoding="utf-8")
    result = apply_patch(root, patch)
    assert result["mode"] == "semantic_fallback"
    output = (root / "module.py").read_text(encoding="utf-8")
    assert "if value < 10:" in output
    compile(output, "<fallback>", "exec")


def test_unified_diff_fallback_uses_nested_python_method_and_validates_full_tree(tmp_path):
    original = "class Outer:\n    class Inner:\n        def normalize(self, value):\n            return value\n"
    revised = original.replace("return value", "return value + 1")
    patch = "".join(difflib.unified_diff(original.splitlines(True), revised.splitlines(True),
                                        fromfile="a/nested.py", tofile="b/nested.py"))
    root = tmp_path / "repo"
    root.mkdir()
    (root / "nested.py").write_text("# moved\n" + original.replace("return value", "return   value"), encoding="utf-8")
    report = apply_patch(root, patch)
    assert report["mode"] == "semantic_fallback"
    assert "return value + 1" in (root / "nested.py").read_text(encoding="utf-8")


def test_unified_diff_fallback_supports_javascript_block_scopes(tmp_path):
    original = "class Counter {\n  add(value) { return value; }\n}\n"
    revised = original.replace("return value", "return value + 1")
    patch = "".join(difflib.unified_diff(original.splitlines(True), revised.splitlines(True),
                                        fromfile="a/counter.js", tofile="b/counter.js"))
    root = tmp_path / "repo"
    root.mkdir()
    (root / "counter.js").write_text("// shift\n" + original.replace("return value", "return   value"), encoding="utf-8")
    report = apply_patch(root, patch)
    assert report["mode"] == "semantic_fallback"
    assert "return value + 1" in (root / "counter.js").read_text(encoding="utf-8")


@pytest.mark.parametrize("case", range(3))
def test_file_patch_writes_atomically_and_rolls_back_only_unchanged_result(tmp_path, case):
    path = tmp_path / f"module_{case}.py"
    original = f"def value_{case}():\n    return {case}\n"
    path.write_text(original, encoding="utf-8")
    evidence = apply_file_patch(path, symbol_name=f"value_{case}",
                                code=f"def value_{case}():\n    return {case + 1}")
    assert path.read_text(encoding="utf-8") == evidence["updated"]
    rollback_file_patch(evidence)
    assert path.read_text(encoding="utf-8") == original


def test_file_rollback_does_not_overwrite_a_later_independent_edit(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("def value():\n    return 1\n", encoding="utf-8")
    evidence = apply_file_patch(path, symbol_name="value", code="def value():\n    return 2")
    path.write_text("# later edit\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SemanticPatchError, match="rollback refused"):
        rollback_file_patch(evidence)
    assert path.read_text(encoding="utf-8").startswith("# later edit")


def test_patch_application_rolls_back_all_files_when_any_syntax_check_fails(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "one.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (root / "two.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    before = (root / "one.py").read_text(encoding="utf-8")
    patch = "--- a/one.py\n+++ b/one.py\n@@ -1,2 +1,2 @@ def value\n def value():\n-    return 1\n+    return 2\n"
    patch += "--- a/two.py\n+++ b/two.py\n@@ -1,2 +1,2 @@ def value\n def value():\n-    return 1\n+    return (\n"
    with pytest.raises(PatchApplicationError):
        apply_patch(root, patch)
    assert (root / "one.py").read_text(encoding="utf-8") == before


def test_python_symbol_index_exposes_nested_class_method_signature():
    symbols = python_symbols("class A:\n    class B:\n        async def run(self):\n            return True\n")
    assert "A.B.run" in {item.qualified_name for item in symbols}
