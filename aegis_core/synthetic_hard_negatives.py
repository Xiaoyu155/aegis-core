"""Deterministic hard-negative generation from executed repair evidence.

The release writer fails closed: a candidate is emitted only after its chosen
patch passes the recorded test, its mutation is rejected by a real test or the
test-assertion guard, and both rendered samples fit the official tokenizer.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import tokenize
from io import StringIO

from .invariant_guard import inspect_test_assertion_delta


MUTATION_TYPES = (
    "off_by_one_boundary",
    "swallow_exception",
    "remove_timeout_validation",
    "break_idempotency_key",
    "unbounded_memory_growth",
    "concurrency_lock_loss",
    "authorization_bypass",
    "failure_as_success",
    "sql_command_injection",
    "weaken_test_assertion",
)


class HardNegativeError(RuntimeError):
    """Generation stopped because required proof or tokenization was missing."""


def _safe_relative(path: str) -> str:
    value = PurePosixPath(str(path).replace("\\", "/"))
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"unsafe source path: {path}")
    return value.as_posix()


def extract_record_payload(row: dict) -> dict | None:
    """Read embedded source and executed-test instructions from a repair row."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        content = next((item.get("content") for item in reversed(prompt)
                        if isinstance(item, dict) and item.get("role") == "user"), None)
    elif isinstance(prompt, str):
        content = prompt
    else:
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), dict):
        return None
    sources = payload["source"]
    if any(not isinstance(value, str) for value in sources.values()):
        return None
    if not isinstance(payload.get("frozen_test"), str) or not payload["frozen_test"].strip():
        return None
    return payload


def _python_operator_offset(source: str, operators: set[str]) -> tuple[int, int, str] | None:
    try:
        for token in tokenize.generate_tokens(StringIO(source).readline):
            if token.type == tokenize.OP and token.string in operators:
                lines = source.splitlines(keepends=True)
                start = sum(len(line) for line in lines[:token.start[0] - 1]) + token.start[1]
                end = start + len(token.string)
                return start, end, token.string
    except (tokenize.TokenError, IndentationError):
        return None
    return None


def mutate_source(source: str, mutation_type: str, *, language: str = "python") -> dict | None:
    """Apply one deterministic, source-level defect operator; return None if absent."""
    if mutation_type not in MUTATION_TYPES:
        raise ValueError(f"unknown mutation type: {mutation_type}")
    if language in {"javascript", "js", "typescript", "ts", "tsx", "jsx"}:
        return _mutate_javascript_source(source, mutation_type)
    if language not in {"python", "py"}:
        raise ValueError("language must be Python or JavaScript/TypeScript")
    if mutation_type == "off_by_one_boundary":
        match = re.search(r"\b(?:index|i|offset|position)\s*<\s*len\s*\(", source)
        if match:
            offset = source.index("<", match.start())
            return {"source": source[:offset] + "<=" + source[offset + 1:], "changed": True,
                    "reason": "included the collection length as a valid index"}
        token = _python_operator_offset(source, {"<=", ">="})
        if token:
            start, end, operator = token
            replacement = "<" if operator == "<=" else ">"
            return {"source": source[:start] + replacement + source[end:],
                    "changed": True, "reason": f"changed {operator} to {replacement}"}
        for pattern, replacement in ((r"\bmin\((\s*)(\d+)", None),
                                     (r"\brange\(([^()]*)\+\s*1\)", None)):
            match = re.search(pattern, source)
            if not match:
                continue
            if pattern.startswith(r"\bmin"):
                value = max(0, int(match.group(2)) - 1)
                updated = source[:match.start(2)] + str(value) + source[match.end(2):]
                return {"source": updated, "changed": True, "reason": "decreased inclusive bound by one"}
            inner = re.sub(r"\s*\+\s*1\s*$", "", match.group(1))
            updated = source[:match.start()] + f"range({inner})" + source[match.end():]
            return {"source": updated, "changed": True, "reason": "removed one from range bound"}
        return None

    if mutation_type == "swallow_exception":
        pattern = re.compile(r"(?m)^(?P<indent>[ \t]*)except(?:\s+[^:\n]+)?\s*:\s*\n(?P<body>(?:[ \t]+[^\n]*\n?)+)")
        for match in pattern.finditer(source):
            body = match.group("body")
            raise_match = re.search(r"(?m)^(?P<indent>[ \t]+)raise(?:\s+[^\n]*)?", body)
            if raise_match:
                changed = body[:raise_match.start()] + raise_match.group("indent") + "pass" + body[raise_match.end():]
                updated = source[:match.start("body")] + changed + source[match.end("body"):]
                return {"source": updated, "changed": True, "reason": "replaced exception propagation with pass"}
        return None

    if mutation_type == "remove_timeout_validation":
        request_timeout = re.search(r"(?P<prefix>\b(?:requests|httpx|urllib\.request)\.(?:get|post|put|patch|delete|request|urlopen)\s*\([^()\n]*?,\s*)timeout\s*=\s*[^,()\n]+,?", source)
        if request_timeout:
            return {"source": source[:request_timeout.start()] + request_timeout.group("prefix").rstrip(" ,") +
                    source[request_timeout.end():], "changed": True,
                    "reason": "removed explicit network request timeout"}
        patterns = (
            (r"\b(?:math\.)?isfinite\s*\([^()]*\)", "True"),
            (r"(?m)^(?P<indent>[ \t]*)if\s+[^\n]*\btimeout\b[^\n]*(?:<=|<|is None)[^\n]*:\s*$", None),
        )
        for pattern, replacement in patterns:
            match = re.search(pattern, source)
            if not match:
                continue
            if replacement is not None:
                return {"source": source[:match.start()] + replacement + source[match.end():],
                        "changed": True, "reason": "finite-value guard no longer rejects non-finite values"}
            line_start = source.rfind("\n", 0, match.start()) + 1
            line_end = source.find("\n", match.end())
            line_end = len(source) if line_end < 0 else line_end + 1
            guard = match.group(0)
            if "timeout" in guard:
                condition_start = source.rfind("\n", 0, match.start()) + 1
                condition_prefix = source[condition_start:match.start()]
                newline = "\r\n" if "\r\n" in source else "\n"
                return {"source": source[:condition_start] + condition_prefix + "if False:" + newline +
                        source[line_end:], "changed": True, "reason": "timeout rejection condition disabled"}
            return {"source": source[:line_start] + source[line_end:],
                    "changed": True, "reason": "removed timeout validation statement"}
        return None

    if mutation_type == "break_idempotency_key":
        patterns = (
            r"(?m)^(?P<prefix>[ \t]*(?:[\w.]+\.)?idempotency_key\s*=\s*).+$",
            r"(?m)^(?P<prefix>[ \t]*[\w.]+\[[\"']idempotency_key[\"']\]\s*=\s*).+$",
        )
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                value = "uuid.uuid4().hex"
                updated = source[:match.start()] + match.group("prefix") + value + source[match.end():]
                return {"source": updated, "changed": True, "reason": "replaced request identity with a random UUID"}
        return None

    if mutation_type == "unbounded_memory_growth":
        patterns = (
            r"(?m)^[ \t]*del\s+[\w.]+\s*\[\s*:\s*-[^\]]+\]\s*\n?",
            r"(?m)^[ \t]*[\w.]+\s*=\s*[\w.]+\s*\[\s*-[^\]]+\s*:\s*\]\s*\n?",
            r"\s*,?\s*maxlen\s*=\s*[^,)]+",
        )
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                updated = source[:match.start()] + source[match.end():]
                if pattern.startswith("(?m)^" ) and not updated.strip():
                    return None
                return {"source": updated, "changed": True, "reason": "removed bounded-history retention"}
        return None

    if mutation_type == "weaken_test_assertion":
        variants = mutate_test_assertions(source)
        return variants[0] if variants else None
    if mutation_type == "concurrency_lock_loss":
        pattern = re.compile(r"(?m)^(?P<indent>[ \t]*)async\s+with\s+[\w.]*?(?:lock|mutex)\s*:\s*\r?\n"
                             r"(?P<body>(?:(?P=indent)[ \t]+[^\n]*\r?\n?)+)", re.I)
        match = pattern.search(source)
        if match:
            body = match.group("body")
            prefix = match.group("indent")
            lines = body.splitlines(keepends=True)
            dedented = "".join(line[:len(prefix)] + line[len(prefix) + 4:] if line.startswith(prefix + "    ") else line
                               for line in lines)
            return {"source": source[:match.start()] + dedented + source[match.end():],
                    "changed": True, "reason": "removed async lock context around shared state"}
        return None
    if mutation_type == "authorization_bypass":
        pattern = re.compile(r"(?m)^(?P<indent>[ \t]*)if\s+(?P<condition>[^:\n]*(?:admin|authoriz|permission|role|access)[^:\n]*):\s*\r?\n"
                             r"(?P<body>(?:[ \t]+raise\s+(?:PermissionError|AuthorizationError|AccessDenied)[^\n]*\r?\n?)+)", re.I)
        match = pattern.search(source)
        if match:
            indent = match.group("indent")
            replacement = indent + "if True:\n" + indent + "    return True\n"
            return {"source": source[:match.start()] + replacement + source[match.end():],
                    "changed": True, "reason": "short-circuited authorization with a constant-true branch"}
        return None
    if mutation_type == "failure_as_success":
        pattern = re.compile(r"(?s)(except\b[^:]*:\s*(?:\n\s+[^\n]*)*?)(?P<key>[\"']?(?:status|status_code|statusCode)[\"']?\s*[:=]\s*)(?P<code>5\d\d)")
        match = pattern.search(source)
        if match:
            return {"source": source[:match.start("code")] + "200" + source[match.end("code"):],
                    "changed": True, "reason": "converted an exception response into HTTP 200"}
        return None
    if mutation_type == "sql_command_injection":
        pattern = re.compile(r"(?P<call>(?:\w+\.)?(?:execute|query)\s*\(\s*)(?P<quote>['\"])(?P<query>[^'\"\n]*\?[^'\"\n]*)(?P=quote)\s*,\s*\(\s*(?P<arg>[\w.]+)\s*,?\s*\)\s*\)")
        match = pattern.search(source)
        if match:
            query = match.group("query").replace("?", "'{ " + match.group("arg") + " }'")
            replacement = match.group("call") + "f" + match.group("quote") + query + match.group("quote") + ")"
            return {"source": source[:match.start()] + replacement + source[match.end():],
                    "changed": True, "reason": "interpolated untrusted SQL value into the command string"}
        return None
    return None


def _mutate_javascript_source(source: str, mutation_type: str) -> dict | None:
    """Apply a deterministic JavaScript/TypeScript defect operator."""
    if mutation_type == "off_by_one_boundary":
        match = re.search(r"(\b(?:index|i|offset|position)\s*<=\s*[\w$.]+\.(?:length|size))", source)
        if match:
            return {"source": source[:match.start()] + match.group(1).replace("<=", "<") + source[match.end():],
                    "changed": True, "reason": "changed inclusive collection bound to exclusive"}
        match = re.search(r"(\b(?:index|i|offset|position)\s*<\s*[\w$.]+\.(?:length|size))", source)
        if match:
            return {"source": source[:match.start()] + match.group(1).replace("<", "<=") + source[match.end():],
                    "changed": True, "reason": "included the collection length as a valid index"}
        return None
    if mutation_type == "swallow_exception":
        pattern = re.compile(r"(catch\s*\(\s*([\w$]+)\s*\)\s*\{)([\s\S]*?)(\})")
        for match in pattern.finditer(source):
            body = match.group(3)
            if re.search(r"\bthrow\s+" + re.escape(match.group(2)) + r"\s*;", body):
                replacement_body = re.sub(r"\bthrow\s+" + re.escape(match.group(2)) + r"\s*;", "return undefined;", body, count=1)
                return {"source": source[:match.start(3)] + replacement_body + source[match.end(3):],
                        "changed": True, "reason": "swallowed the caught JavaScript exception"}
            if not body.strip():
                return None
        return None
    if mutation_type == "remove_timeout_validation":
        signal = re.search(r"\bsignal\s*:\s*controller\.signal\s*,?", source)
        if signal:
            updated = source[:signal.start()] + source[signal.end():]
            return {"source": updated, "changed": True, "reason": "removed AbortController timeout signal"}
        controller = re.search(r"\bconst\s+controller\s*=\s*new\s+AbortController\s*\(\s*\)\s*;?\s*\n?", source)
        if controller:
            updated = source[:controller.start()] + source[controller.end():]
            return {"source": updated, "changed": True, "reason": "removed AbortController timeout signal"}
        timeout_guard = re.search(r"(?m)^(?P<indent>[ \t]*)if\s*\(\s*timeout\s*<=\s*0\s*\)\s*\{[^{}]*\}\s*;?", source)
        if timeout_guard:
            return {"source": source[:timeout_guard.start()] + timeout_guard.group("indent") + "if (false) {}" + source[timeout_guard.end():],
                    "changed": True, "reason": "disabled timeout validation"}
        return None
    if mutation_type == "break_idempotency_key":
        match = re.search(r"(?P<prefix>[\w$]*idempotency[\w$]*key\s*[:=]\s*)(?P<value>[^,;\n}]+)", source, re.I)
        if match:
            return {"source": source[:match.start("value")] + "crypto.randomUUID()" + source[match.end("value"):],
                    "changed": True, "reason": "replaced stable request identity with a random UUID"}
        return None
    if mutation_type == "unbounded_memory_growth":
        patterns = (r"\s*\bmax(?:Size|Entries)?\s*:\s*\d+\s*,?",)
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return {"source": source[:match.start()] + source[match.end():],
                        "changed": True, "reason": "removed bounded cache retention setting"}
        return None
    if mutation_type == "concurrency_lock_loss":
        match = re.search(r"(?s)(?P<prefix>return\s+)(?:mutex|lock)\.run\s*\(\s*async\s*\(\s*\)\s*=>\s*\{(?P<body>.*?)\}\s*\)", source)
        if match:
            return {"source": source[:match.start()] + "return (async () => {" + match.group("body") + "})()" + source[match.end():],
                    "changed": True, "reason": "removed async mutex wrapper around shared state"}
        return None
    if mutation_type == "authorization_bypass":
        match = re.search(r"(?s)(?:async\s+)?function\s+(?:authorize|isAuthorized|hasPermission|canAccess)\w*\s*\([^)]*\)\s*\{", source, re.I)
        if match:
            return {"source": source[:match.end()] + " if (true) return true; " + source[match.end():],
                    "changed": True, "reason": "inserted an unconditional authorization success"}
        return None
    if mutation_type == "failure_as_success":
        match = re.search(r"catch\s*\([^)]*\)\s*\{[\s\S]*?\bstatus(?:Code)?\s*[:=]\s*(?P<code>5\d\d)", source)
        if match:
            return {"source": source[:match.start("code")] + "200" + source[match.end("code"):],
                    "changed": True, "reason": "converted an exception response into HTTP 200"}
        return None
    if mutation_type == "sql_command_injection":
        prepared = re.search(r"(?P<call>(?:\w+\.)?prepare\s*\()(?P<quote>['\"])(?P<sql>[^'\"\n]*\?[^'\"\n]*)(?P=quote)\)\.all\(\s*(?P<arg>[\w$.]+)\s*\)", source)
        if prepared:
            sql = prepared.group("sql").replace("?", "'${" + prepared.group("arg") + "}'")
            replacement = prepared.group("call") + "`" + sql + "`).all()"
            return {"source": source[:prepared.start()] + replacement + source[prepared.end():],
                    "changed": True, "reason": "interpolated untrusted SQL value into a template string"}
        match = re.search(r"(?P<call>(?:\w+\.)?(?:query|execute)\s*\()(?P<quote>['\"])(?P<sql>[^'\"\n]*\?[^'\"\n]*)(?P=quote)\s*,\s*\[\s*(?P<arg>[\w$.]+)\s*\]\s*\)", source)
        if match:
            sql = match.group("sql").replace("?", "${" + match.group("arg") + "}")
            replacement = match.group("call") + "`" + sql + "`)"
            return {"source": source[:match.start()] + replacement + source[match.end():],
                    "changed": True, "reason": "interpolated untrusted SQL value into a template string"}
        return None
    if mutation_type == "weaken_test_assertion":
        pattern = re.compile(r"(?m)^(?P<indent>[ \t]*)assert\.(?:equal|strictEqual|ok)\([^\n]+\);?\s*$")
        match = pattern.search(source)
        if match:
            return {"source": source[:match.start()] + match.group("indent") + "assert.ok(true);" + source[match.end():],
                    "changed": True, "reason": "replaced a real JavaScript assertion with an always-true check"}
        return None
    return None


def mutate_test_assertions(source: str) -> list[dict]:
    """Return one independent always-true mutation for every Python assertion.

    AST locations include complete multiline ``self.assert*`` calls and avoid
    matching assertion-like text inside comments or strings.  UTF-8 byte
    columns are converted back to character offsets before editing.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    encoded = source.encode("utf-8")
    candidates = []
    for node in ast.walk(tree):
        is_assert = isinstance(node, ast.Assert)
        is_unittest_assert = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                              and isinstance(node.value.func, ast.Attribute)
                              and node.value.func.attr.startswith("assert"))
        if not (is_assert or is_unittest_assert) or not getattr(node, "end_lineno", None):
            continue
        start = (node.lineno, node.col_offset)
        end = (node.end_lineno, node.end_col_offset)
        candidates.append((start, end))
    variants = []
    seen = set()
    for start, end in sorted(set(candidates)):
        start_byte = sum(len(line) for line in encoded.splitlines(keepends=True)[:start[0] - 1]) + start[1]
        end_byte = sum(len(line) for line in encoded.splitlines(keepends=True)[:end[0] - 1]) + end[1]
        prefix = encoded[:start_byte]
        line_start = prefix.rfind(b"\n") + 1
        indentation = prefix[line_start:]
        if indentation.strip():
            continue
        replacement = b"assert True"
        mutated = (encoded[:start_byte] + replacement + encoded[end_byte:]).decode("utf-8")
        if mutated in seen or mutated == source:
            continue
        seen.add(mutated)
        variants.append({"source": mutated, "changed": True,
                         "reason": "replaced one behavioral assertion with assert True"})
    return variants


def load_official_tokenizer(model_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct"):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise HardNegativeError("official tokenizer validation requires transformers, which is not installed") from error
    model_path = Path(model_id)
    candidates = [model_path] if model_path.is_dir() else []
    cache_root = (Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE") else
                  Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub")
    snapshots = cache_root / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if snapshots.is_dir():
        candidates.extend(sorted((item for item in snapshots.iterdir() if item.is_dir()),
                                 key=lambda item: item.stat().st_mtime, reverse=True))
    if not candidates:
        raise HardNegativeError(f"official tokenizer {model_id} is not present in the local cache")
    last_error = None
    for candidate in candidates:
        if not (candidate / "tokenizer.json").is_file():
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(candidate), trust_remote_code=False,
                                                      local_files_only=True)
            tokenizer.name_or_path = model_id
            return tokenizer
        except Exception as error:
            last_error = error
    raise HardNegativeError(f"official tokenizer {model_id} is not present in the local cache: {last_error}")


def tokenizer_length(tokenizer, text: str) -> int:
    """Count untruncated tokens; callers must reject over-budget samples."""
    encoded = tokenizer(text, truncation=False, add_special_tokens=True)
    token_ids = encoded["input_ids"]
    return len(token_ids)


def verify_zero_truncation(tokenizer, text: str, max_tokens: int) -> dict:
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    count = tokenizer_length(tokenizer, text)
    return {"token_count": count, "max_tokens": max_tokens,
            "zero_truncation": count <= max_tokens}


def _message_patch(row: dict) -> str | None:
    value = row.get("completion")
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return None
    return decoded.get("patch") if isinstance(decoded, dict) and isinstance(decoded.get("patch"), str) else None


def _git(root: Path, *args: str, input: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, input=input, capture_output=True, timeout=15, check=False)


def _initialize_fixture(root: Path, source: dict, frozen_test: str) -> str:
    for relative, content in source.items():
        safe = _safe_relative(relative)
        target = root / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
    tests = root / "tests"
    tests.mkdir(exist_ok=True)
    probe = tests / "test_aegis_frozen_probe.py"
    suffix = 0
    while probe.exists():
        suffix += 1
        probe = tests / f"test_aegis_frozen_probe_{suffix}.py"
    probe.write_text(frozen_test, encoding="utf-8", newline="")
    for args in (("init", "-q", "-b", "main"), ("config", "user.name", "Aegis Hard Negative"),
                 ("config", "user.email", "aegis@example.invalid"), ("add", "-A"), ("commit", "-qm", "verified baseline")):
        result = _git(root, *args)
        if result.returncode:
            raise HardNegativeError(f"cannot initialize isolated fixture: {result.stderr.decode(errors='replace')}")
    return probe.relative_to(root).as_posix()


def _run_test(root: Path, test_path: str, command: list[str] | None, timeout: float) -> dict:
    actual = command or [sys.executable, "-m", "pytest", "-q", test_path]
    try:
        result = subprocess.run(actual, cwd=root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        return {"passed": False, "exit_code": 124, "stdout": str(error.stdout or ""),
                "stderr": "red-light command timed out", "command": actual}
    output = result.stdout + "\n" + result.stderr
    if "No module named pytest" in output or "No module named 'pytest'" in output:
        raise HardNegativeError("pytest is required to prove a real red-light; it is not installed")
    return {"passed": result.returncode == 0, "exit_code": result.returncode,
            "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:], "command": actual}


def _candidate_files(patch: str) -> list[str]:
    return [line[6:].strip() for line in patch.splitlines() if line.startswith("+++ b/")]


def _diff_file(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                       fromfile=f"a/{path}", tofile=f"b/{path}"))


def _canonical_line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_release(repo_root: str | Path = ".", *, source_dataset: str | Path | None = None,
                     output_dir: str | Path | None = None, target_pairs: int = 400,
                     tokenizer_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct", max_tokens: int = 4096,
                     test_command: list[str] | None = None, timeout: float = 45) -> dict:
    """Create only red-light-proven pairs and package them atomically."""
    if type(target_pairs) is not int or target_pairs < 1:
        raise ValueError("target_pairs must be a positive integer")
    root = Path(repo_root).resolve()
    source_path = Path(source_dataset) if source_dataset else root / "datasets" / "grpo_repair_pairs.jsonl"
    source_path = source_path.resolve()
    destination = Path(output_dir) if output_dir else root / "datasets" / "release_v13_augmented"
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    tokenizer = load_official_tokenizer(tokenizer_id)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    existing_file = destination / "hard_negative_pairs.jsonl"
    existing = []
    if existing_file.is_file():
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            raise HardNegativeError("existing augmented release has no manifest; refusing to append")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != _sha256_file(existing_file):
            raise HardNegativeError("existing augmented release checksum mismatch")
        existing = [json.loads(line) for line in existing_file.read_text(encoding="utf-8").splitlines() if line]
    accepted = list(existing)
    known = {row.get("pair_sha256") for row in accepted}
    rejected_by_reason = {}
    redlight_count = 0
    with source_path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(accepted) >= target_pairs:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejected_by_reason["invalid_json"] = rejected_by_reason.get("invalid_json", 0) + 1
                continue
            if row.get("positive") is not True:
                continue
            payload = extract_record_payload(row)
            chosen_patch = _message_patch(row)
            if payload is None or not chosen_patch:
                rejected_by_reason["missing_executed_evidence"] = rejected_by_reason.get("missing_executed_evidence", 0) + 1
                continue
            with tempfile.TemporaryDirectory(prefix="aegis-hard-negative-") as directory:
                fixture = Path(directory)
                probe_path = _initialize_fixture(fixture, payload["source"], payload["frozen_test"])
                applied = _git(fixture, "apply", "--check", "-", input=chosen_patch.encode("utf-8"))
                if applied.returncode:
                    rejected_by_reason["chosen_patch_did_not_apply"] = rejected_by_reason.get("chosen_patch_did_not_apply", 0) + 1
                    continue
                applied = _git(fixture, "apply", "-", input=chosen_patch.encode("utf-8"))
                if applied.returncode:
                    rejected_by_reason["chosen_patch_apply_error"] = rejected_by_reason.get("chosen_patch_apply_error", 0) + 1
                    continue
                chosen_result = _run_test(fixture, probe_path, test_command, timeout)
                if not chosen_result["passed"]:
                    rejected_by_reason["chosen_patch_not_red_green_verified"] = rejected_by_reason.get("chosen_patch_not_red_green_verified", 0) + 1
                    continue
                affected = _candidate_files(chosen_patch)
                for mutation_type in MUTATION_TYPES:
                    if len(accepted) >= target_pairs:
                        break
                    if mutation_type == "weaken_test_assertion":
                        test_text = (fixture / probe_path).read_text(encoding="utf-8")
                        mutations = [(probe_path, item) for item in mutate_test_assertions(test_text)]
                    else:
                        mutation_target = None
                        mutation = None
                        for relative in affected:
                            path = fixture / _safe_relative(relative)
                            if not path.is_file() or path.suffix.lower() not in {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                                continue
                            original = path.read_text(encoding="utf-8")
                            candidate = mutate_source(original, mutation_type)
                            if candidate and candidate["source"] != original:
                                mutation_target, mutation = relative, candidate
                                break
                        mutations = [(mutation_target, mutation)] if mutation else []
                    if not mutations:
                        rejected_by_reason[f"no_{mutation_type}_site"] = rejected_by_reason.get(f"no_{mutation_type}_site", 0) + 1
                        continue
                    for variant_index, (mutation_target, mutation) in enumerate(mutations):
                        if len(accepted) >= target_pairs:
                            break
                        target = fixture / _safe_relative(mutation_target)
                        original_text = target.read_text(encoding="utf-8")
                        if mutation_type == "weaken_test_assertion":
                            proof = inspect_test_assertion_delta(original_text, mutation["source"], path=mutation_target)
                            redlight = {"passed": not proof["passed"], "checker": "invariant_guard.inspect_test_assertion_delta",
                                        "findings": proof["findings"]}
                            negative_patch = chosen_patch + _diff_file(mutation_target, original_text, mutation["source"])
                        else:
                            target.write_text(mutation["source"], encoding="utf-8", newline="")
                            probe = _run_test(fixture, probe_path, test_command, timeout)
                            redlight = {"passed": not probe["passed"], "checker": "frozen_test_suite",
                                        "exit_code": probe["exit_code"],
                                        "output_sha256": hashlib.sha256((probe["stdout"] + probe["stderr"]).encode()).hexdigest()}
                            negative_patch = "".join(_git(fixture, "diff", "--", *affected).stdout.decode("utf-8", errors="replace").splitlines(keepends=True))
                            target.write_text(original_text, encoding="utf-8", newline="")
                        if not redlight["passed"]:
                            key = f"{mutation_type}_not_caught"
                            rejected_by_reason[key] = rejected_by_reason.get(key, 0) + 1
                            continue
                        prompt = str(row.get("prompt", ""))
                        candidate_id = hashlib.sha256(
                            f"{row.get('group_id', row.get('patch_sha256', line_number))}\0{mutation_type}\0{variant_index}".encode()).hexdigest()
                        pair = {"schema_version": "aegis.hard_negative.v1", "pair_sha256": candidate_id,
                                "group_id": candidate_id, "prompt": prompt,
                                "chosen_patch": chosen_patch, "rejected_patch": negative_patch,
                                "chosen_evidence": {"command": chosen_result["command"], "passed": True},
                                "rejection_evidence": {"mutation_type": mutation_type, "reason": mutation["reason"],
                                                       "redlight": redlight},
                                "source_record": row.get("group_id") or row.get("patch_sha256") or line_number}
                        rendered = _canonical_line(pair)
                        length = verify_zero_truncation(tokenizer, rendered, max_tokens)
                        if not length["zero_truncation"]:
                            rejected_by_reason["token_limit"] = rejected_by_reason.get("token_limit", 0) + 1
                        elif candidate_id not in known:
                            pair["tokenizer"] = {"model_id": tokenizer_id, **length}
                            accepted.append(pair)
                            known.add(candidate_id)
                            redlight_count += 1
    if len(accepted) < target_pairs:
        raise HardNegativeError(f"only {len(accepted)} of {target_pairs} verified pairs available; "
                                f"refusing incomplete release: {json.dumps(rejected_by_reason, sort_keys=True)}")
    destination.mkdir(parents=True, exist_ok=True)
    payload = "".join(_canonical_line(row) + "\n" for row in accepted)
    temporary = destination / ".hard_negative_pairs.jsonl.tmp"
    temporary.write_text(payload, encoding="utf-8", newline="")
    os.replace(temporary, existing_file)
    manifest = {"schema_version": "aegis.hard_negative_manifest.v1", "pair_count": len(accepted),
                "target_pairs": target_pairs, "official_tokenizer": tokenizer_id,
                "zero_truncation_verified": all(item.get("tokenizer", {}).get("zero_truncation") for item in accepted),
                "redlight_verified_pairs": len(accepted), "sha256": _sha256_file(existing_file),
                "rejected_candidates_by_reason": rejected_by_reason}
    manifest_tmp = destination / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, destination / "manifest.json")
    return {"status": "complete", "pair_count": len(accepted), "manifest": manifest,
            "output": str(destination), "newly_verified": redlight_count}
