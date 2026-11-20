"""Unified-diff application with an AST/block-scoped recovery path."""
from __future__ import annotations

import ast
import difflib
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .semantic_patcher import SemanticPatchError, javascript_symbols, python_symbols


class PatchApplicationError(ValueError):
    """A patch could not be applied or validated without ambiguity."""


def _patch_paths(patch: str) -> list[str]:
    paths = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:])
    return list(dict.fromkeys(paths))


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise PatchApplicationError(f"patch path escapes repository: {relative}") from error
    return target


def _validate(path: Path, source: str) -> None:
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as error:
            raise PatchApplicationError(f"Python AST validation failed for {path}: {error}") from error
    elif suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}:
        try:
            javascript_symbols(source)
        except SemanticPatchError as error:
            raise PatchApplicationError(f"JS/TS structural validation failed for {path}: {error}") from error


def _parse_file_hunks(patch: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    current = None
    hunk = None
    for line in patch.splitlines(keepends=True):
        if line.startswith("+++ b/"):
            current = line[6:].rstrip("\r\n")
            result.setdefault(current, [])
            hunk = None
            continue
        if line.startswith("@@ ") and current is not None:
            header = line.rstrip("\r\n")
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@(?:\s*(.*))?$", header)
            if not match:
                raise PatchApplicationError(f"unsupported unified-diff hunk header: {header}")
            hunk = {"old_start": int(match.group(1)), "context": match.group(2) or "",
                    "old": [], "new": []}
            result[current].append(hunk)
            continue
        if hunk is None or not line:
            continue
        marker, body = line[0], line[1:]
        if marker == " ":
            hunk["old"].append(body)
            hunk["new"].append(body)
        elif marker == "-":
            hunk["old"].append(body)
        elif marker == "+":
            hunk["new"].append(body)
        elif marker == "\\":
            continue
        else:
            hunk = None
    return result


def _norm(line: str) -> str:
    return re.sub(r"\s+", "", line)


def _line_index(offsets: list[int], position: int) -> int:
    low, high = 0, len(offsets)
    while low < high:
        mid = (low + high) // 2
        if offsets[mid] <= position:
            low = mid + 1
        else:
            high = mid
    return max(0, low - 1)


def _symbol_ranges(source: str, path: Path) -> list[tuple[str, int, int]]:
    symbols = python_symbols(source) if path.suffix.lower() == ".py" else javascript_symbols(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for item in lines:
        offsets.append(offsets[-1] + len(item))
    return [(item.qualified_name, _line_index(offsets, item.start),
             max(_line_index(offsets, max(item.start, item.end - 1)), _line_index(offsets, item.start)))
            for item in symbols]


def _requested_symbol(context: str) -> str | None:
    patterns = (r"\b(?:async\s+)?def\s+([A-Za-z_]\w*)",
                r"\bclass\s+([A-Za-z_$][\w$]*)",
                r"\bfunction\s+([A-Za-z_$][\w$]*)",
                r"\b(?:public|private|protected|static|async)\s+([A-Za-z_$][\w$]*)\s*\(")
    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            return match.group(1)
    return None


def _candidate_alignment(source_lines: list[str], old_lines: list[str], ranges: list[tuple[str, int, int]],
                         old_start: int, context: str) -> tuple[str, dict[int, int]]:
    old_indexes = [index for index, line in enumerate(old_lines) if _norm(line)]
    old_tokens = [_norm(old_lines[index]) for index in old_indexes]
    if not old_tokens:
        raise PatchApplicationError("semantic fallback needs at least one nonblank context line")
    preferred = _requested_symbol(context)
    candidates: list[tuple[int, int, int, str, dict[int, int]]] = []
    scope_ranges = ranges or [("<module>", 0, len(source_lines) - 1)]
    for name, first, last in scope_ranges:
        if preferred and name.split(".")[-1] != preferred and name != preferred:
            continue
        visible = [(index, _norm(source_lines[index])) for index in range(max(0, first), min(last + 1, len(source_lines)))
                   if _norm(source_lines[index])]
        for begin in range(0, len(visible) - len(old_tokens) + 1):
            window = visible[begin:begin + len(old_tokens)]
            if [token for _, token in window] != old_tokens:
                continue
            mapping = dict(zip(old_indexes, (index for index, _ in window)))
            distance = abs(window[0][0] + 1 - old_start)
            candidates.append((distance, last - first, window[0][0], name, mapping))
    if not candidates:
        hint = f" in symbol {preferred}" if preferred else ""
        raise PatchApplicationError(f"AST-scoped context not found{hint}")
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    best = candidates[0]
    tied = [item for item in candidates if item[:3] == best[:3]]
    starts = {(item[2], item[3]) for item in tied}
    if len(starts) > 1:
        raise PatchApplicationError("semantic fallback context is ambiguous across symbols")
    return best[3], best[4]


def _indent_relative(line: str, source_anchor: str, patch_anchor: str) -> str:
    if not line.strip():
        return line
    source_indent = re.match(r"[ \t]*", source_anchor).group(0)
    patch_indent = re.match(r"[ \t]*", patch_anchor).group(0)
    line_indent = re.match(r"[ \t]*", line).group(0)
    if line_indent.startswith(patch_indent):
        return source_indent + line_indent[len(patch_indent):] + line[len(line_indent):]
    return line


def _apply_hunk(source: str, path: Path, hunk: dict) -> str:
    lines = source.splitlines(keepends=True)
    ranges = _symbol_ranges(source, path)
    symbol, mapping = _candidate_alignment(lines, hunk["old"], ranges,
                                           hunk["old_start"], hunk["context"])
    old = hunk["old"]
    new = hunk["new"]
    matcher = difflib.SequenceMatcher(a=[_norm(item) for item in old],
                                      b=[_norm(item) for item in new], autojunk=False)
    changes = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not changes:
        return source
    for _tag, a1, a2, b1, b2 in reversed(changes):
        consumed = [index for index in mapping if a1 <= index < a2]
        if consumed:
            start = mapping[min(consumed)]
            end = mapping[max(consumed)] + 1
        else:
            before = [index for index in mapping if index < a1]
            after = [index for index in mapping if index >= a2]
            if before:
                start = end = mapping[max(before)] + 1
            elif after:
                start = end = mapping[min(after)]
            else:
                raise PatchApplicationError(f"cannot anchor insertion in {symbol}")
        anchor_old_index = min(consumed) if consumed else (max([i for i in mapping if i < a1], default=min(mapping)))
        anchor_source_index = mapping[anchor_old_index]
        patch_anchor = old[anchor_old_index]
        source_anchor = lines[anchor_source_index]
        replacement = [_indent_relative(item, source_anchor, patch_anchor) for item in new[b1:b2]]
        lines[start:end] = replacement
    updated = "".join(lines)
    _validate(path, updated)
    return updated


def _semantic_fallback(root: Path, patch: str) -> dict[str, str]:
    parsed = _parse_file_hunks(patch)
    if not parsed:
        raise PatchApplicationError("patch contains no files")
    updated: dict[str, str] = {}
    for relative, hunks in parsed.items():
        if relative == "/dev/null":
            raise PatchApplicationError("semantic fallback does not create new files")
        target = _safe_target(root, relative)
        if not target.is_file():
            raise PatchApplicationError(f"semantic fallback target does not exist: {relative}")
        try:
            original = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise PatchApplicationError(f"cannot read semantic fallback target {relative}: {error}") from error
        current = original
        for hunk in sorted(hunks, key=lambda item: item["old_start"], reverse=True):
            current = _apply_hunk(current, target, hunk)
        _validate(target, current)
        updated[relative] = current
    return updated


def _atomic_write(target: Path, source: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, target.stat().st_mode)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def apply_patch(repo_root: str | Path, patch: str) -> dict:
    """Apply a unified diff, falling back to unique AST/block context matching.

    All changed files are syntax-checked after either path. Fallback edits are
    prepared in memory first and written atomically only after every hunk has a
    unique symbol-scoped match.
    """
    root = Path(repo_root).resolve()
    paths = _patch_paths(patch)
    if not paths:
        raise PatchApplicationError("patch contains no destination files")
    originals = {}
    for relative in paths:
        target = _safe_target(root, relative)
        originals[relative] = target.read_bytes() if target.exists() else None
    check = subprocess.run(["git", "apply", "--check", "--whitespace=error", "-"], cwd=root,
                           input=patch.encode("utf-8"), capture_output=True)
    if check.returncode == 0:
        applied = subprocess.run(["git", "apply", "--whitespace=error", "-"], cwd=root,
                                 input=patch.encode("utf-8"), capture_output=True)
        if applied.returncode == 0:
            try:
                for relative in paths:
                    target = _safe_target(root, relative)
                    _validate(target, target.read_text(encoding="utf-8"))
                return {"mode": "unified_diff", "files": paths, "fallback_reason": None}
            except (OSError, UnicodeError, PatchApplicationError):
                for relative, payload in originals.items():
                    target = _safe_target(root, relative)
                    if payload is not None:
                        target.write_bytes(payload)
                    else:
                        target.unlink(missing_ok=True)
                raise
    try:
        prepared = _semantic_fallback(root, patch)
        for relative, source in prepared.items():
            _atomic_write(_safe_target(root, relative), source)
        return {"mode": "semantic_fallback", "files": list(prepared),
                "fallback_reason": check.stderr.decode("utf-8", errors="replace").strip()}
    except BaseException:
        for relative, payload in originals.items():
            target = _safe_target(root, relative)
            if payload is None:
                target.unlink(missing_ok=True)
            elif target.exists():
                target.write_bytes(payload)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        raise
