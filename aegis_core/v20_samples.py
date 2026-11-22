"""Executable, deterministic bilingual samples for the v2.0 gate release."""
from __future__ import annotations

import asyncio
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .invariant_guard import inspect_test_assertion_delta
from .security_rules import inspect_javascript_test_assertion_delta
from .synthetic_hard_negatives import (MUTATION_TYPES, _canonical_line, load_official_tokenizer,
                                       mutate_source, verify_zero_truncation)
from .security_gate import scan_text


_RULE_CODE = {
    "off_by_one_boundary": "OFF_BY_ONE_BOUNDARY",
    "swallow_exception": "EXCEPTION_SWALLOWED",
    "remove_timeout_validation": "MISSING_TIMEOUT",
    "break_idempotency_key": "NONDETERMINISTIC_IDEMPOTENCY_KEY",
    "unbounded_memory_growth": "UNBOUNDED_GROWTH",
    "concurrency_lock_loss": "SHARED_STATE_WITHOUT_LOCK",
    "authorization_bypass": "AUTHORIZATION_BYPASS",
    "failure_as_success": "FAILURE_AS_SUCCESS",
    "sql_command_injection": "SQL_COMMAND_INJECTION",
    "weaken_test_assertion": "TEST_ASSERTION_ALWAYS_TRUE",
}


def _case_sources(defect: str, language: str, index: int) -> tuple[str, str]:
    suffix = f"{index:03d}"
    name = f"case_{suffix}"
    if language == "python":
        if defect == "off_by_one_boundary":
            source = f"def {name}(index, values):\n    return index < len(values)\n"
            test = (f"values = list(range({1 + index % 11}))\n"
                    f"assert {name}(0, values) is True\n"
                    f"assert {name}(len(values), values) is False\n")
        elif defect == "swallow_exception":
            source = (f"def {name}(value):\n    try:\n        return int(value)\n"
                      "    except ValueError as error:\n        raise ValueError('invalid integer') from error\n")
            test = (f"try:\n    {name}('not-an-integer')\nexcept ValueError:\n    pass\n"
                    "else:\n    raise AssertionError('conversion failure must propagate')\n")
        elif defect == "remove_timeout_validation":
            source = ("import urllib.request\n"
                      f"def {name}(url, timeout):\n"
                      "    return urllib.request.urlopen(url, timeout=timeout).read()\n")
            test = ("from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                    "from threading import Thread\nimport time\n"
                    "class _Slow(BaseHTTPRequestHandler):\n"
                    "    def do_GET(self):\n"
                    "        time.sleep(0.025)\n        self.send_response(200)\n"
                    "        self.end_headers()\n"
                    "        try:\n            self.wfile.write(b'ok')\n        except OSError:\n            pass\n"
                    "    def log_message(self, *args):\n        pass\n"
                    "server = HTTPServer(('127.0.0.1', 0), _Slow)\n"
                    "thread = Thread(target=server.serve_forever, daemon=True)\nthread.start()\n"
                    "try:\n"
                    f"    try:\n        {name}(f'http://127.0.0.1:{{server.server_port}}/', 0.002)\n"
                    "    except TimeoutError:\n        pass\n"
                    "    else:\n        raise AssertionError('request exceeded its timeout but completed')\n"
                    "finally:\n    server.shutdown()\n    server.server_close()\n    thread.join(1)\n")
        elif defect == "break_idempotency_key":
            source = ("import uuid\n"
                      f"def {name}(request_id):\n    idempotency_key = str(request_id)\n"
                      "    return {'Idempotency-Key': idempotency_key}\n")
            test = (f"assert {name}('request-{suffix}') == {name}('request-{suffix}')\n"
                    f"assert {name}('request-{suffix}')['Idempotency-Key'] == 'request-{suffix}'\n")
        elif defect == "unbounded_memory_growth":
            bound = 2 + index % 13
            source = ("from collections import deque\nclass History:\n"
                      f"    def __init__(self):\n        self.items = deque(maxlen={bound})\n"
                      "    def append(self, item):\n        self.items.append(item)\n")
            test = (f"history = History()\nfor item in range({bound * 4}):\n    history.append(item)\n"
                    f"assert len(history.items) == {bound}\n")
        elif defect == "concurrency_lock_loss":
            source = ("import asyncio\nclass Counter:\n    def __init__(self):\n"
                      "        self.counter = 0\n        self.lock = asyncio.Lock()\n"
                      "    async def increment(self):\n        async with self.lock:\n"
                      "            previous = self.counter\n            await asyncio.sleep(0)\n"
                      "            self.counter = previous + 1\n")
            test = ("async def _run():\n    counter = Counter()\n"
                    f"    await asyncio.gather(*[counter.increment() for _ in range({20 + index % 21})])\n"
                    f"    assert counter.counter == {20 + index % 21}\n"
                    "asyncio.run(_run())\n")
        elif defect == "authorization_bypass":
            source = (f"def authorize_{suffix}(user):\n"
                      "    if not user.is_admin:\n        raise PermissionError('denied')\n"
                      "    return True\n")
            test = ("class _User:\n    def __init__(self, is_admin):\n        self.is_admin = is_admin\n"
                    f"assert authorize_{suffix}(_User(True)) is True\n"
                    f"try:\n    authorize_{suffix}(_User(False))\n"
                    "except PermissionError:\n    pass\nelse:\n    raise AssertionError('non-admin was authorized')\n")
        elif defect == "failure_as_success":
            source = (f"def {name}():\n    try:\n        raise OSError('offline')\n"
                      "    except OSError:\n        return {'status': 503, 'body': 'unavailable'}\n")
            test = f"assert {name}()['status'] == 503\n"
        elif defect == "sql_command_injection":
            source = (f"def {name}(connection, user_id):\n"
                      "    return connection.execute(\n"
                      "        \"SELECT name FROM users WHERE name = ?\", (user_id,)\n"
                      "    ).fetchall()\n")
            test = ("import sqlite3\nconnection = sqlite3.connect(':memory:')\n"
                    "connection.execute('CREATE TABLE users(name TEXT)')\n"
                    "connection.executemany('INSERT INTO users VALUES (?)', [('Alice',), ('Bob',)])\n"
                    f"assert {name}(connection, \"' OR '1'='1\") == []\n"
                    "connection.close()\n")
        elif defect == "weaken_test_assertion":
            source = f"def value_{suffix}(number):\n    return number + 1\n"
            test = f"assert value_{suffix}(2) == 3\n"
        else:
            raise ValueError(f"unsupported defect class: {defect}")
        return source, test

    if language != "javascript":
        raise ValueError("language must be python or javascript")
    if defect == "off_by_one_boundary":
        source = f"function {name}(index, values) {{ return index < values.length; }}\n"
        test = f"assert.equal({name}(0, [1]), true); assert.equal({name}(1, [1]), false);\n"
    elif defect == "swallow_exception":
        source = (f"function {name}(value) {{ try {{ const result = Number(value); "
                  "if (!Number.isFinite(result)) throw new TypeError('invalid'); return result; "
                  "} catch (error) { throw error; } }\n")
        test = f"assert.throws(() => {name}('bad'), TypeError);\n"
    elif defect == "remove_timeout_validation":
        source = (f"async function {name}(url, timeoutMs) {{\n"
                  "  const controller = new AbortController();\n"
                  "  const timer = setTimeout(() => controller.abort(), timeoutMs);\n"
                  "  try { return await fetch(url, { signal: controller.signal }); }\n"
                  "  finally { clearTimeout(timer); }\n}\n")
        test = ("const http = require('node:http');\n"
                "const server = http.createServer((_request, response) => {\n"
                "  setTimeout(() => { response.end('late'); }, 25);\n});\n"
                "await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));\n"
                "const url = `http://127.0.0.1:${server.address().port}/`;\n"
                "try {\n"
                f"  await assert.rejects({name}(url, 2), error => error.name === 'AbortError');\n"
                "} finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }\n")
    elif defect == "break_idempotency_key":
        source = ("const crypto = require('node:crypto');\n"
                  f"function {name}(requestId) {{ const idempotencyKey = String(requestId); return {{'Idempotency-Key': idempotencyKey}}; }}\n")
        test = (f"assert.equal({name}('request-{suffix}')['Idempotency-Key'], {name}('request-{suffix}')['Idempotency-Key']);\n"
                f"assert.equal({name}('request-{suffix}')['Idempotency-Key'], 'request-{suffix}');\n")
    elif defect == "unbounded_memory_growth":
        bound = 2 + index % 13
        source = ("class LRUCache {\n"
                  "  constructor(options) { this.limit = options.maxEntries; this.map = new Map(); }\n"
                  "  set(key, value) { this.map.set(key, value); while (this.map.size > this.limit) this.map.delete(this.map.keys().next().value); }\n"
                  "}\n"
                  f"const cache = new LRUCache({{ maxEntries: {bound} }});\n")
        test = (f"for (let i = 0; i < {bound * 4}; i += 1) cache.set(i, i);\n"
                f"assert.equal(cache.map.size, {bound});\n")
    elif defect == "concurrency_lock_loss":
        count = 20 + index % 21
        source = ("let count = 0;\n"
                  "class Mutex { constructor() { this.tail = Promise.resolve(); }\n"
                  "  run(work) { const result = this.tail.then(work, work); this.tail = result.then(() => undefined, () => undefined); return result; }\n}\n"
                  "const mutex = new Mutex();\n"
                  "async function increment() { return mutex.run(async () => {\n"
                  "  const previous = count; await Promise.resolve(); count = previous + 1;\n"
                  "}); }\n")
        test = f"await Promise.all(Array.from({{length: {count}}}, () => increment())); assert.equal(count, {count});\n"
    elif defect == "authorization_bypass":
        source = (f"function authorizeUser{suffix}(user) {{\n"
                  "  if (!user.isAdmin) throw new Error('denied');\n  return true;\n}\n")
        test = (f"assert.equal(authorizeUser{suffix}({{isAdmin:true}}), true);\n"
                f"assert.throws(() => authorizeUser{suffix}({{isAdmin:false}}), /denied/);\n")
    elif defect == "failure_as_success":
        source = (f"async function {name}() {{ try {{ throw new Error('offline'); }} "
                  "catch (_error) { return { status: 503, body: 'unavailable' }; } }\n")
        test = f"await assert.doesNotReject(async () => assert.equal((await {name}()).status, 503));\n"
    elif defect == "sql_command_injection":
        source = (f"function {name}(db, userId) {{\n"
                  "  return db.prepare('SELECT name FROM users WHERE name = ?').all(userId);\n}\n")
        test = ("const { DatabaseSync } = require('node:sqlite');\n"
                "const db = new DatabaseSync(':memory:');\n"
                "db.exec(\"CREATE TABLE users(name TEXT); INSERT INTO users VALUES ('Alice'), ('Bob');\");\n"
                f"assert.equal({name}(db, \"' OR '1'='1\").length, 0); db.close();\n")
    elif defect == "weaken_test_assertion":
        source = f"function value{suffix}(number) {{ return number + 1; }}\n"
        test = f"assert.equal(value{suffix}(2), 3);\n"
    else:
        raise ValueError(f"unsupported defect class: {defect}")
    return source, test


def build_case(defect: str, language: str, index: int, *, positive: bool = False) -> dict:
    if defect not in MUTATION_TYPES:
        raise ValueError(f"unknown defect class: {defect}")
    source, test = _case_sources(defect, language, index)
    row = {"defect_type": defect, "language": language, "case_index": index,
           "source": source, "frozen_test": test, "sample_kind": "hard_positive" if positive else "hard_negative"}
    if positive:
        if defect == "unbounded_memory_growth":
            if language == "python":
                source = ("from itertools import count\n"
                          f"def ticks_{index:03d}(start):\n    yield from count(start)\n")
                test = (f"stream = ticks_{index:03d}({index})\n"
                        f"assert next(stream) == {index}\nassert next(stream) == {index + 1}\n")
            else:
                source = (f"function* ticks{index:03d}(start) {{ let value = start; while (true) yield value++; }}\n")
                test = f"const stream = ticks{index:03d}({index}); assert.equal(stream.next().value, {index}); assert.equal(stream.next().value, {index+1});\n"
        if defect == "failure_as_success":
            if language == "python":
                source = (f"def handle_{index:03d}(send):\n    try:\n        send()\n"
                          "    except OSError as error:\n        return {'status': 503, 'reason': type(error).__name__}\n"
                          "    return {'status': 200}\n")
                test = ("from unittest.mock import Mock\n"
                        "send = Mock(side_effect=OSError('offline'))\n"
                        f"assert handle_{index:03d}(send)['status'] == 503\nassert send.call_count == 1\n")
            else:
                source = (f"async function handle{index:03d}(send) {{ try {{ await send(); }} "
                          "catch (_error) { return { status: 503 }; } return { status: 200 }; }\n")
                test = ("const send = async () => { throw new Error('offline'); };\n"
                        f"assert.equal((await handle{index:03d}(send)).status, 503);\n")
        row["source"], row["frozen_test"] = source, test
        row["expected_findings"] = []
        return row

    if defect == "weaken_test_assertion":
        changed = mutate_source(test, defect, language=language)
        if changed is None:
            raise ValueError("assertion mutation template did not contain an assertion")
        row["rejected_source"] = source
        row["rejected_test"] = changed["source"]
        row["mutation_reason"] = changed["reason"]
    else:
        changed = mutate_source(source, defect, language=language)
        if changed is None:
            raise ValueError(f"no {defect} mutation site in {language} template")
        row["rejected_source"] = changed["source"]
        row["rejected_test"] = test
        row["mutation_reason"] = changed["reason"]
    return row


def _python_runner_source() -> str:
    return '''import json, sys, traceback
rows = json.load(open(sys.argv[1], encoding="utf-8"))
out = []
for row in rows:
    scope = {"__name__": "__aegis_generated_case__"}
    try:
        exec(compile(row["source"], row["case_id"] + ".py", "exec"), scope, scope)
        exec(compile(row["test"], row["case_id"] + "_test.py", "exec"), scope, scope)
        out.append({"passed": True, "error": None})
    except BaseException as error:
        out.append({"passed": False, "error": type(error).__name__})
json.dump(out, sys.stdout, sort_keys=True)
'''


def _javascript_runner_source() -> str:
    return '''const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const rows = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const timeout = ms => new Promise((_, reject) => setTimeout(() => reject(new Error('sample timeout')), ms));
(async () => {
  const output = [];
  for (const row of rows) {
    try {
      const context = { require, assert, console, setTimeout, clearTimeout, URL, AbortController, fetch,
                        Buffer, TextEncoder, TextDecoder, Promise, Map, Set, Array, Object, Number, String,
                        Error, TypeError, RangeError, Math, JSON, Date };
      const program = `(async () => {\\n${row.source}\\n${row.test}\\n})()`;
      const result = vm.runInNewContext(program, context,
                                        {timeout: 1500, filename: row.case_id + '.cjs'});
      if (result && typeof result.then === 'function') await Promise.race([result, timeout(2500)]);
      output.push({passed: true, error: null});
    } catch (error) {
      output.push({passed: false, error: error && error.constructor ? error.constructor.name : 'Error',
                   message: String(error && error.message || error).slice(0, 300)});
    }
  }
  process.stdout.write(JSON.stringify(output));
})().catch(error => { process.stderr.write(String(error)); process.exitCode = 2; });
'''


def _run_case_tests(root: Path, rows: list[dict], *, language: str, timeout: float = 180) -> list[dict]:
    if not rows:
        return []
    with tempfile.TemporaryDirectory(prefix="aegis-v20-proof-") as temporary:
        directory = Path(temporary)
        payload = directory / "cases.json"
        payload.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        if language == "python":
            runner = directory / "verify_cases.py"
            runner.write_text(_python_runner_source(), encoding="utf-8")
            command = [sys.executable, str(runner), str(payload)]
        else:
            node = shutil.which("node")
            if not node:
                raise RuntimeError("Node.js is required to execute JavaScript hard samples")
            runner = directory / "verify_cases.cjs"
            runner.write_text(_javascript_runner_source(), encoding="utf-8")
            command = [node, str(runner), str(payload)]
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout, check=False)
        if result.returncode:
            raise RuntimeError(f"{language} proof runner failed: {result.stderr[-4000:]}")
        try:
            outcomes = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{language} proof runner returned invalid evidence") from error
        if len(outcomes) != len(rows):
            raise RuntimeError("proof runner did not return one result per isolated case")
        return outcomes


def _diff(before: str, after: str, filename: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                       fromfile="a/" + filename, tofile="b/" + filename))


def generate_v20_release(repo_root: str | Path = ".", *, output_dir: str | Path | None = None,
                         negatives_per_type: int = 80, positives_per_type: int = 20,
                         tokenizer_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
                         max_tokens: int = 4096, timeout: float = 180) -> dict:
    """Build a complete bilingual release only after executing every frozen test."""
    if negatives_per_type < 1 or positives_per_type < 1:
        raise ValueError("sample counts must be positive")
    root = Path(repo_root).resolve()
    output = Path(output_dir) if output_dir else root / "datasets" / "release_v20"
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty release directory: {output}")
    tokenizer = load_official_tokenizer(tokenizer_id)
    samples: list[dict] = []
    jobs: dict[str, list[dict]] = {"python": [], "javascript": []}
    refs: list[tuple[dict, str, int]] = []
    for defect in MUTATION_TYPES:
        for index in range(negatives_per_type):
            for language in ("python", "javascript"):
                sample = build_case(defect, language, index)
                sample_id = hashlib.sha256(f"negative\0{defect}\0{language}\0{index}".encode()).hexdigest()
                sample["sample_id"] = sample_id
                if defect == "weaken_test_assertion":
                    guard = (inspect_test_assertion_delta(sample["frozen_test"], sample["rejected_test"],
                                                          path=f"tests/{sample_id}.py") if language == "python"
                             else inspect_javascript_test_assertion_delta(sample["frozen_test"], sample["rejected_test"],
                                                                          path=f"tests/{sample_id}.js"))
                    if guard["passed"]:
                        raise RuntimeError("assertion mutation escaped the invariant guard")
                    sample["redlight_proof"] = {"checker": "invariant_guard.test_assertion_delta",
                                                "passed": True, "findings": guard["findings"]}
                    sample["chosen_diff"] = _diff("", sample["frozen_test"], sample_id + ".test")
                    sample["rejected_diff"] = _diff(sample["frozen_test"], sample["rejected_test"], sample_id + ".test")
                else:
                    expected_code = _RULE_CODE[defect]
                    findings = scan_text(sample["rejected_source"], sample_id + (".py" if language == "python" else ".js"))
                    matching = [item for item in findings if item["code"] == expected_code]
                    if not matching:
                        raise RuntimeError(f"{language} {defect} mutant was not caught by the security scanner")
                    sample["expected_findings"] = matching
                    sample["chosen_diff"] = _diff("", sample["source"], sample_id + (".py" if language == "python" else ".js"))
                    sample["rejected_diff"] = _diff(sample["source"], sample["rejected_source"],
                                                     sample_id + (".py" if language == "python" else ".js"))
                    job = {"case_id": sample_id + "_chosen", "source": sample["source"],
                           "test": sample["frozen_test"]}
                    rejected = {"case_id": sample_id + "_rejected", "source": sample["rejected_source"],
                                "test": sample["rejected_test"]}
                    jobs[language].extend((job, rejected))
                    refs.append((sample, language, len(jobs[language]) - 2))
                samples.append(sample)
        for index in range(positives_per_type):
            for language in ("python", "javascript"):
                sample = build_case(defect, language, index, positive=True)
                sample_id = hashlib.sha256(f"positive\0{defect}\0{language}\0{index}".encode()).hexdigest()
                sample["sample_id"] = sample_id
                findings = scan_text(sample["source"], sample_id + (".py" if language == "python" else ".js"))
                if findings:
                    raise RuntimeError(f"hard positive falsely triggered security rules: {defect}/{language}: {findings}")
                sample["expected_findings"] = []
                job = {"case_id": sample_id, "source": sample["source"], "test": sample["frozen_test"]}
                jobs[language].append(job)
                refs.append((sample, language, len(jobs[language]) - 1))
                samples.append(sample)

    outcomes: dict[str, list[dict]] = {}
    for language in ("python", "javascript"):
        outcomes[language] = _run_case_tests(root, jobs[language], language=language, timeout=timeout)
    for sample, language, offset in refs:
        if sample["sample_kind"] == "hard_positive":
            proof = outcomes[language][offset]
            if not proof["passed"]:
                raise RuntimeError(f"hard-positive frozen test failed: {sample['sample_id']}: {proof}")
            sample["test_proof"] = {"passed": True, "error": None}
        elif sample["defect_type"] != "weaken_test_assertion":
            chosen, rejected = outcomes[language][offset:offset + 2]
            if not chosen["passed"] or rejected["passed"]:
                raise RuntimeError(f"red/green proof failed for {sample['sample_id']}: {chosen}, {rejected}")
            sample["redlight_proof"] = {"checker": "executed_frozen_test", "passed": True,
                                        "chosen_exit": 0, "rejected_exit": 1,
                                        "rejected_error": rejected["error"]}
            sample["test_proof"] = {"chosen_passed": True, "rejected_failed": True}
        else:
            chosen = outcomes[language][offset]
            if not chosen["passed"]:
                raise RuntimeError(f"chosen code did not pass after test mutation: {sample['sample_id']}")
            sample["test_proof"] = {"chosen_passed": True, "weakened_test_runtime_passes": True,
                                    "guard_rejects_weakened_test": True}

    category_counts = {name: {"hard_negative": 0, "hard_positive": 0, "python": 0, "javascript": 0}
                       for name in MUTATION_TYPES}
    seen = set()
    for sample in samples:
        body = {key: value for key, value in sample.items() if key != "sample_sha256"}
        digest = hashlib.sha256(_canonical_line(body).encode("utf-8")).hexdigest()
        if digest in seen:
            raise RuntimeError("sample generator produced a duplicate serialized case")
        seen.add(digest)
        sample["sample_sha256"] = digest
        sample["tokenizer"] = {"model_id": tokenizer_id,
                               **verify_zero_truncation(tokenizer, _canonical_line(sample), max_tokens)}
        if not sample["tokenizer"]["zero_truncation"]:
            raise RuntimeError(f"tokenizer length exceeds budget for {sample['sample_id']}")
        kind = sample["sample_kind"]
        category_counts[sample["defect_type"]][kind] += 1
        category_counts[sample["defect_type"]][sample["language"]] += 1

    if len(samples) != len(MUTATION_TYPES) * (negatives_per_type + positives_per_type) * 2:
        raise RuntimeError("v2 release count does not match the requested matrix")
    if any(item["hard_negative"] != negatives_per_type * 2 or
           item["hard_positive"] != positives_per_type * 2 or
           item["python"] != item["javascript"] for item in category_counts.values()):
        raise RuntimeError("v2 release language or category balance is incomplete")
    payload = "".join(_canonical_line(sample) + "\n" for sample in samples)
    dataset_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {"schema_version": "aegis.hard_samples_manifest.v2",
                "sample_count": len(samples), "hard_negative_count": len(MUTATION_TYPES) * negatives_per_type * 2,
                "hard_positive_count": len(MUTATION_TYPES) * positives_per_type * 2,
                "negative_samples_per_class_per_language": negatives_per_type,
                "positive_samples_per_class_per_language": positives_per_type,
                "defect_types": list(MUTATION_TYPES), "category_counts": category_counts,
                "languages": ["python", "javascript"], "official_tokenizer": tokenizer_id,
                "max_tokens": max_tokens, "zero_truncation_verified": True,
                "redlight_verified_negative_count": len(MUTATION_TYPES) * negatives_per_type * 2,
                "hard_positive_tests_passed": len(MUTATION_TYPES) * positives_per_type * 2,
                "dataset_sha256": dataset_sha,
                "generator": "aegis_core.v20_samples.generate_v20_release"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aegis-v20-release-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        (staging / "hard_samples_v2.jsonl").write_text(payload, encoding="utf-8", newline="")
        (staging / "manifest_v2.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                              sort_keys=True, indent=2) + "\n",
                                                   encoding="utf-8", newline="")
        if output.exists() and not any(output.iterdir()):
            output.rmdir()
        os.replace(staging, output)
    return {"status": "complete", "output": output.as_posix(), "manifest": manifest}


_V21_NUMERATORS = (
    "amount", "(amount + fee)", "(amount - fee)", "(amount * rate)",
    "(amount + fee * rate)", "max(amount, fee)", "min(amount, fee)",
    "abs(amount)", "(amount * rate + fee)", "sum((amount, fee))",
)
_V21_OPERATORS = ("/", "//", "%")
_V21_DENOMINATORS = ("principal", "reserve", "rate", "fee", "amount")
_V21_WRAPPERS = (
    "return {expression}",
    "value = {expression}\nreturn value",
    "if enabled:\n    return {expression}\nreturn 0",
    "try:\n    return {expression}\nexcept TypeError:\n    return 0",
    "for _index in range(2):\n    value = {expression}\n    return value\nreturn 0",
    "return {{'risk': {expression}}}",
    "return ({expression}, amount)",
    "return max(0, {expression})",
)


class _SkeletonNormalizer(ast.NodeTransformer):
    """Erase identifiers and literals so diversity counts syntax, not naming."""

    def visit_Name(self, node):
        node.id = "IDENT"
        return node

    def visit_arg(self, node):
        node.arg = "ARG"
        return node

    def visit_FunctionDef(self, node):
        node.name = "FUNCTION"
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if type(node.value) in {int, float}:
            node.value = 0
        elif type(node.value) is str:
            node.value = "STRING"
        return node


def ast_structure_skeleton(source: str) -> str:
    tree = _SkeletonNormalizer().visit(ast.parse(source))
    return ast.dump(ast.fix_missing_locations(tree), annotate_fields=True, include_attributes=False)


def _v21_source(index: int) -> str:
    shape_index = index % (len(_V21_NUMERATORS) * len(_V21_OPERATORS))
    numerator = _V21_NUMERATORS[shape_index // len(_V21_OPERATORS)]
    operator = _V21_OPERATORS[shape_index % len(_V21_OPERATORS)]
    denominator = _V21_DENOMINATORS[index % len(_V21_DENOMINATORS)]
    wrapper = _V21_WRAPPERS[(index // (len(_V21_NUMERATORS) * len(_V21_OPERATORS))) % len(_V21_WRAPPERS)]
    expression = f"{numerator} {operator} {denominator}"
    body = wrapper.format(expression=expression)
    return ("def calculate_risk(amount, principal, fee=1, rate=2, reserve=3, enabled=True):\n"
            + "\n".join("    " + line for line in body.splitlines()) + "\n")


def generate_v21_diverse_release(repo_root: str | Path = ".", *, output_dir: str | Path | None = None,
                                 sample_count: int = 240) -> dict:
    """Create executable-to-scan finance negatives with audited AST diversity."""
    if type(sample_count) is not int or not 200 <= sample_count <= 2000:
        raise ValueError("sample_count must be in 200..2000")
    root = Path(repo_root).resolve()
    output = Path(output_dir) if output_dir else root / "datasets" / "release_v21_diverse"
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite a nonempty release directory: {output}")
    templates = []
    unique_templates = []
    repeated_templates = []
    skeleton_hashes = set()
    for candidate_index in range(240):
        source = _v21_source(candidate_index)
        findings = [item for item in scan_text(source, f"samples/risk_{candidate_index:04d}.py")
                    if item["code"] == "UNGUARDED_DIVISOR"]
        if not findings:
            raise RuntimeError(f"sample {candidate_index} did not trigger its expected financial redlight")
        skeleton = ast_structure_skeleton(source)
        skeleton_hash = hashlib.sha256(skeleton.encode("utf-8")).hexdigest()
        template = (source, skeleton_hash)
        templates.append(template)
        if skeleton_hash in skeleton_hashes:
            repeated_templates.append(template)
        else:
            skeleton_hashes.add(skeleton_hash)
            unique_templates.append(template)
    ordered_templates = [*unique_templates, *repeated_templates]
    while len(ordered_templates) < sample_count:
        ordered_templates.extend(templates[:min(len(templates), sample_count - len(ordered_templates))])
    samples = []
    selected_skeleton_hashes = set()
    for index, (source, skeleton_hash) in enumerate(ordered_templates[:sample_count]):
        selected_skeleton_hashes.add(skeleton_hash)
        row = {"sample_id": f"v21-risk-{index:04d}", "defect_type": "unguarded_financial_division",
               "language": "python", "source": source,
               "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
               "ast_skeleton_sha256": skeleton_hash,
               "expected_finding": "UNGUARDED_DIVISOR"}
        row["sample_sha256"] = hashlib.sha256(_canonical_line(row).encode("utf-8")).hexdigest()
        samples.append(row)
    if len(selected_skeleton_hashes) < 200:
        raise RuntimeError(f"only {len(selected_skeleton_hashes)} distinct normalized AST skeletons were generated")
    payload = "".join(_canonical_line(row) + "\n" for row in samples)
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {"schema_version": "aegis.release_v21_diverse.v1", "sample_count": sample_count,
                "distinct_ast_skeleton_count": len(selected_skeleton_hashes),
                "distinct_ast_skeleton_sha256": sorted(selected_skeleton_hashes),
                "normalized_ast_skeleton": "ast.dump after replacing identifiers and numeric/string literals",
                "redlight_finding": "UNGUARDED_DIVISOR",
                "samples_sha256": payload_hash, "generator": "aegis_core.v20_samples.generate_v21_diverse_release"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aegis-v21-diverse-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        (staging / "samples.jsonl").write_text(payload, encoding="utf-8", newline="")
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                            sort_keys=True, indent=2) + "\n",
                                                 encoding="utf-8", newline="")
        if output.exists() and not any(output.iterdir()):
            output.rmdir()
        os.replace(staging, output)
    return {"status": "complete", "output": output.as_posix(),
            "manifest": manifest, "samples_sha256": payload_hash}

