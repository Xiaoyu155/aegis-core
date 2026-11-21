"""Source-aware, deterministic edits for Python and brace-delimited JS/TS.

This module deliberately supports a small set of explicit operations. It does
not guess from a line number: callers name a function, class, or method and the
engine refuses ambiguous matches or syntax-invalid results.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re
import textwrap


class SemanticPatchError(ValueError):
    """A structured edit could not be applied safely."""


@dataclass(frozen=True)
class Symbol:
    name: str
    qualified_name: str
    start: int
    end: int


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", source):
        offsets.append(match.end())
    return offsets


def _python_char_column(line: str, byte_column: int) -> int:
    raw = line.encode("utf-8")[:byte_column]
    return len(raw.decode("utf-8", errors="ignore"))


def python_symbols(source: str) -> list[Symbol]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SemanticPatchError(f"invalid Python source: {error}") from error
    lines = source.splitlines(keepends=True)
    offsets = _line_offsets(source)
    result: list[Symbol] = []

    def visit(body, classes: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                start_line = max(0, first - 1)
                end_line = node.end_lineno or node.lineno
                first_column = min([node.col_offset, *(item.col_offset for item in node.decorator_list)])
                start = offsets[start_line] + _python_char_column(lines[start_line], first_column)
                end = offsets[end_line] if end_line < len(offsets) else len(source)
                result.append(Symbol(node.name, ".".join((*classes, node.name)), start, end))
                visit(node.body, (*classes, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
                start_line = max(0, first - 1)
                end_line = node.end_lineno or node.lineno
                start = offsets[start_line] if start_line < len(offsets) else len(source)
                if start_line < len(lines):
                    # AST columns count UTF-8 bytes; use the decorator/function
                    # column to preserve nested-class indentation exactly.
                    byte_column = min([node.col_offset, *(item.col_offset for item in node.decorator_list)])
                    start += _python_char_column(lines[start_line], byte_column)
                end = offsets[end_line] if end_line < len(offsets) else len(source)
                result.append(Symbol(node.name, ".".join((*classes, node.name)), start, end))
                # Nested definitions are still addressable through their full
                # class-qualified name when they are methods of a nested class.
                if not classes:
                    visit(node.body, classes)

    visit(tree.body)
    return sorted(result, key=lambda item: (item.start, -(item.end - item.start), item.qualified_name))


def _mask_js(source: str) -> str:
    chars = list(source)
    state = "code"
    quote = ""
    index = 0
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in "'\"`":
                quote, state = char, "string"
                chars[index] = " "
            elif char == "/" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 1
                state = "line"
            elif char == "/" and nxt == "*":
                chars[index] = chars[index + 1] = " "
                index += 1
                state = "block"
        elif state == "string":
            if char != "\n":
                chars[index] = " "
            if char == "\\":
                if index + 1 < len(source) and source[index + 1] != "\n":
                    chars[index + 1] = " "
                    index += 1
            elif char == quote:
                state = "code"
        elif state == "line":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block":
            if char == "*" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                index += 1
                state = "code"
            elif char != "\n":
                chars[index] = " "
        index += 1
    return "".join(chars)


def _brace_end(masked: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    raise SemanticPatchError("unclosed JavaScript/TypeScript block")


def _find_opening_brace(masked: str, start: int) -> int | None:
    parens = brackets = 0
    for index in range(start, len(masked)):
        char = masked[index]
        if char == "(":
            parens += 1
        elif char == ")":
            parens = max(0, parens - 1)
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets = max(0, brackets - 1)
        elif char == "{" and parens == 0 and brackets == 0:
            return index
        elif char == ";" and parens == 0 and brackets == 0:
            return None
    return None


def _js_symbols(source: str) -> list[Symbol]:
    masked = _mask_js(source)
    candidates: list[Symbol] = []
    seen: set[tuple[int, int, str]] = set()

    def add(name: str, start: int, opening: int) -> None:
        end = _brace_end(masked, opening)
        key = (start, end, name)
        if key not in seen:
            seen.add(key)
            candidates.append(Symbol(name, name, start, end))

    for match in re.finditer(r"\bclass\s+([A-Za-z_$][\w$]*)[^{};]*\{", masked):
        opening = masked.find("{", match.start(), match.end())
        add(match.group(1), match.start(), opening)

    for match in re.finditer(r"\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", masked):
        opening = _find_opening_brace(masked, match.end())
        if opening is not None:
            add(match.group(1), match.start(), opening)

    arrow = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^;{}]*\)|[A-Za-z_$][\w$]*)\s*=>\s*\{")
    for match in arrow.finditer(masked):
        add(match.group(1), match.start(), masked.find("{", match.start(), match.end()))

    method = re.compile(
        r"(?:^|[;{}\n])\s*(?:(?:public|private|protected|static|async|readonly|override)\s+)*"
        r"([A-Za-z_$][\w$]*)\s*\([^;{}]*\)\s*(?::\s*[^{};=]+)?\s*\{"
    )
    for match in method.finditer(masked):
        start = match.start(1)
        opening = masked.find("{", match.start(1), match.end())
        # A method must be inside a class block; object literal callbacks are
        # intentionally excluded from signature lookup.
        parents = [item for item in candidates if item.name and source[item.start:item.start + 20].lstrip().startswith("class")
                   and item.start < start < item.end]
        if parents:
            add(match.group(1), start, opening)

    classes = [item for item in candidates if source[item.start:item.start + 30].lstrip().startswith("class")]
    qualified: list[Symbol] = []
    for item in candidates:
        owners = [parent for parent in classes if parent.start < item.start < item.end < parent.end]
        owners.sort(key=lambda parent: parent.start)
        if owners and item not in owners:
            path = [owner.name for owner in owners] + [item.name]
            qualified.append(Symbol(item.name, ".".join(path), item.start, item.end))
        else:
            qualified.append(item)
    return sorted(qualified, key=lambda item: (item.start, -(item.end - item.start), item.qualified_name))


def javascript_symbols(source: str) -> list[Symbol]:
    masked = _mask_js(source)
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(masked):
        if char in "([{":
            stack.append((char, index))
        elif char in ")]}":
            if not stack or stack[-1][0] != pairs[char]:
                raise SemanticPatchError(f"unbalanced JavaScript/TypeScript delimiter at offset {index}")
            stack.pop()
    if stack:
        raise SemanticPatchError(f"unclosed JavaScript/TypeScript delimiter at offset {stack[-1][1]}")
    return _js_symbols(source)


def _select_symbol(symbols: list[Symbol], requested: str) -> Symbol:
    matches = [item for item in symbols if item.qualified_name == requested]
    if not matches:
        matches = [item for item in symbols if item.name == requested]
    if not matches:
        matches = [item for item in symbols if item.qualified_name.endswith("." + requested)]
    if not matches:
        raise SemanticPatchError(f"symbol not found: {requested}")
    if len(matches) != 1:
        names = ", ".join(sorted(item.qualified_name for item in matches))
        raise SemanticPatchError(f"ambiguous symbol {requested!r}; use one of: {names}")
    return matches[0]


def _newline(source: str) -> str:
    return "\r\n" if "\r\n" in source else "\n"


def _indent_block(code: str, indent: str, newline: str) -> str:
    dedented = textwrap.dedent(code).strip("\r\n")
    if not dedented:
        raise SemanticPatchError("patch code must not be empty")
    return newline.join((indent + line if line else "") for line in dedented.splitlines()) + newline


def _validate_js_delimiters(source: str) -> None:
    javascript_symbols(source)


def apply_structured_patch(source: str, *, symbol_name: str, code: str,
                           operation: str = "replace", language: str | None = None) -> str:
    """Replace a symbol or inject code immediately before/after its block.

    Python validation uses the native AST parser. JavaScript and TypeScript use
    a comment/string-aware delimiter and block scan, so malformed block edits
    fail before any caller writes the result.
    """
    if not isinstance(source, str) or not isinstance(code, str):
        raise TypeError("source and code must be strings")
    if operation not in {"replace", "before", "after"}:
        raise ValueError("operation must be replace, before, or after")
    kind = (language or "python").lower()
    if kind in {"py", "python"}:
        symbol = _select_symbol(python_symbols(source), symbol_name)
        lines = source.splitlines(keepends=True)
        line_start = source.rfind("\n", 0, symbol.start) + 1
        indent = source[line_start:symbol.start]
    elif kind in {"js", "javascript", "ts", "typescript", "jsx", "tsx"}:
        symbol = _select_symbol(javascript_symbols(source), symbol_name)
        line_start = source.rfind("\n", 0, symbol.start) + 1
        indent = source[line_start:symbol.start]
    else:
        raise ValueError(f"unsupported language: {language}")
    newline = _newline(source)
    inserted = _indent_block(code, indent, newline)
    replace_start = line_start
    if operation == "replace":
        updated = source[:replace_start] + inserted + source[symbol.end:]
    elif operation == "before":
        updated = source[:replace_start] + inserted + source[replace_start:]
    else:
        boundary = symbol.end
        if kind not in {"py", "python"}:
            if source.startswith("\r\n", boundary):
                boundary += 2
            elif source.startswith("\n", boundary):
                boundary += 1
            elif boundary < len(source):
                inserted = newline + inserted
        updated = source[:boundary] + inserted + source[boundary:]
    if kind in {"py", "python"}:
        try:
            ast.parse(updated)
        except SyntaxError as error:
            raise SemanticPatchError(f"patch creates invalid Python syntax: {error}") from error
    else:
        _validate_js_delimiters(updated)
    return updated


def rollback_structured_patch(updated: str, original: str, *, expected_updated: str | None = None) -> str:
    """Return the original text only while the edited value still matches.

    The optional expected value makes rollback compare-and-swap safe for files
    that may have been edited again since the patch was applied.
    """
    if expected_updated is not None and updated != expected_updated:
        raise SemanticPatchError("rollback refused: source changed after structured patch")
    return original


def apply_file_patch(path, *, symbol_name: str, code: str, operation: str = "replace",
                     language: str | None = None) -> dict:
    """Atomically edit one source file and return rollback evidence."""
    from pathlib import Path
    import os
    import tempfile
    target = Path(path)
    original_bytes = target.read_bytes()
    try:
        original = original_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SemanticPatchError("source file is not UTF-8") from error
    updated = apply_structured_patch(original, symbol_name=symbol_name, code=code,
                                    operation=operation, language=language or target.suffix.lstrip("."))
    mode = target.stat().st_mode
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {"path": str(target), "original": original, "updated": updated,
            "operation": operation, "symbol_name": symbol_name}


def rollback_file_patch(evidence: dict) -> None:
    """Restore the saved bytes when the file still has the expected result."""
    from pathlib import Path
    target = Path(evidence["path"])
    try:
        current = target.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise SemanticPatchError("source file is no longer UTF-8") from error
    rollback_structured_patch(current, evidence["original"], expected_updated=evidence["updated"])
    target.write_text(evidence["original"], encoding="utf-8", newline="")
