"""Compile explicit acceptance rules into frozen, executable assertions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

from .sandbox import ProbeNotFailingError, Sandbox, run_exam_gate

NUMBER = r"-?(?:0|[1-9][0-9]{0,8})"
CLAMP = re.compile(
    rf"(?P<path>[\w./-]+\.(?:py|mjs|cjs|js))::(?P<symbol>[A-Za-z_]\w*) "
    rf"clamps? integers? to \[(?P<low>{NUMBER}),\s*(?P<high>{NUMBER})\] "
    rf"and returns (?P<fallback>{NUMBER}|null|None) for (?:invalid inputs?|non-integers?)\.?",
    re.IGNORECASE,
)
COLLECTION = re.compile(
    rf"(?P<path>[\w./-]+\.(?:py|mjs|cjs|js))::(?P<symbol>[A-Za-z_]\w*) "
    rf"counts? distinct string events and returns (?P<fallback>{NUMBER}|null|None) "
    rf"for (?:invalid inputs?|non-string events)\.?",
    re.IGNORECASE,
)
PROBE_NAMES = {".py": "tests/test_aegis_probe.py", ".js": "tests/test_aegis_probe.mjs",
               ".mjs": "tests/test_aegis_probe.mjs", ".cjs": "tests/test_aegis_probe.mjs"}


def acceptance_rule(intent):
    match = CLAMP.fullmatch(intent.strip()) or COLLECTION.fullmatch(intent.strip())
    if match is None:
        return None
    rule = match.groupdict()
    rule["kind"] = "clamp" if "low" in rule else "distinct_events"
    if rule["kind"] == "clamp":
        rule["low"], rule["high"] = int(rule["low"]), int(rule["high"])
    rule["fallback"] = None if rule["fallback"].lower() in {"none", "null"} else int(rule["fallback"])
    path = Path(rule["path"])
    if path.is_absolute() or ".." in path.parts or rule.get("low", 0) > rule.get("high", 0):
        raise ValueError("invalid acceptance target or integer interval")
    return rule


def expected_value(rule, value):
    if rule["kind"] == "distinct_events":
        if type(value) is list and all(type(event) is str for event in value):
            return len(set(value))
    elif type(value) is int:
        return min(rule["high"], max(rule["low"], value))
    return rule["fallback"]


def static_cases(rule):
    if rule["kind"] == "distinct_events":
        values = [[], ["open"], ["open", "close"], ["open", "open"], None, ["open", 1]]
        return [(value, expected_value(rule, value)) for value in values]
    low, high = rule["low"], rule["high"]
    return [(low + (high - low) // 2, low + (high - low) // 2),
            (low - 1, low), (low, low), (high, high), (high + 1, high),
            (None, rule["fallback"])]


def render_probe(rule, cases):
    """Use one assertion per subcase so failure evidence names the input."""
    suffix = Path(rule["path"]).suffix
    filename = PROBE_NAMES[suffix]
    if suffix == ".py":
        source = f"""import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path.cwd()))
target = importlib.import_module({rule['path'][:-3].replace('/', '.')!r})
function = getattr(target, {rule['symbol']!r})


@pytest.mark.parametrize("value, expected", {cases!r})
def test_aegis_acceptance(value, expected):
    result = function(value)
    assert type(result) is type(expected) and result == expected, "AEGIS_ACCEPTANCE"
"""
        command = ["{python}", "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", filename]
    else:
        source = f"""import test from 'node:test';
import assert from 'node:assert/strict';
import {{resolve}} from 'node:path';
import {{pathToFileURL}} from 'node:url';

const target = await import(pathToFileURL(resolve({json.dumps(rule['path'])})).href);
const fn = target[{json.dumps(rule['symbol'])}] ?? target.default?.[{json.dumps(rule['symbol'])}];
if (typeof fn !== 'function') throw new TypeError('Invalid acceptance target');

for (const [value, expected] of {json.dumps(cases)}) {{
  test(JSON.stringify(value), () => assert.strictEqual(fn(value), expected, 'AEGIS_ACCEPTANCE'));
}}
"""
        command = ["node", "--test", "--test-isolation=none", "--test-reporter=tap", filename]
    return filename, source, command


def compile_probe(spec, repo, base, *, timeout=60, budget=None):
    rule = spec.context.get("acceptance_rule")
    if spec.questions or rule is None:
        raise ValueError("acceptance rules must specify an executable target, input domain and expected behavior")
    root = Path(repo).resolve()
    filename, source, command = render_probe(rule, static_cases(rule))
    lines = source.splitlines()
    patch = (f"diff --git a/{filename} b/{filename}\nnew file mode 100644\n"
             f"--- /dev/null\n+++ b/{filename}\n@@ -0,0 +1,{len(lines)} @@\n"
             + "".join(f"+{line}\n" for line in lines))
    hashes = {filename: hashlib.sha256(source.encode("utf-8")).hexdigest()}
    sandbox = Sandbox(root)
    ticket = "synth-" + uuid.uuid4().hex[:12]
    path = sandbox.create(ticket, base=base)
    try:
        sandbox.apply_patch(ticket, patch)
        from functools import partial
        exam = run_exam_gate if budget is None else partial(budget.run_gate, run_exam_gate)
        red = exam(path, [sys.executable if arg == "{python}" else arg for arg in command],
                   timeout=timeout, probe_script_sha256=hashes)
        if not is_true_red(red):
            raise ProbeNotFailingError("synthesized probe must fail a real acceptance assertion: " +
                                       json.dumps(red, ensure_ascii=False))
    finally:
        sandbox.cleanup(ticket)
    return {"command": command, "patch": patch, "hashes": hashes, "red": red, "rule": rule}


def is_true_red(receipt):
    output = receipt["stdout"] + "\n" + receipt["stderr"]
    return (receipt["exit_code"] == 1 and receipt["failure_code"] == "ASSERTION_FAILED"
            and not receipt["changed"] and
            bool(re.search(r"AssertionError: AEGIS_ACCEPTANCE|(?m:^    AEGIS_ACCEPTANCE\s*$)", output)))
