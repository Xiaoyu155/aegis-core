"""Compile explicit acceptance rules into frozen, executable assertions."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
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
def _new_file_patch(filename, source):
    lines = source.splitlines()
    return (f"diff --git a/{filename} b/{filename}\nnew file mode 100644\n"
            f"--- /dev/null\n+++ b/{filename}\n@@ -0,0 +1,{len(lines)} @@\n"
            + "".join(f"+{line}\n" for line in lines))


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
        command = ["{python}", "-B", "-m", "pytest", "--noconftest", "-o", "addopts=",
                   "-p", "no:cacheprovider", "-q", filename]
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
    patch = _new_file_patch(filename, source)
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


def is_true_green(receipt, cases):
    """Require evidence that all generated cases ran, even on an exit code of zero."""
    if not receipt["passed"]:
        return False
    output = receipt["stdout"]
    if receipt["command"][0] == "node":
        return all(re.search(rf"(?m)^# {key} {expected}\s*$", output)
                   for key, expected in (("tests", cases), ("pass", cases), ("fail", 0)))
    return bool(re.search(rf"(?m)^\s*{cases} passed(?:,| in )", output))


STRUCTURED_KINDS = {
    "numeric_bounds", "state_machine", "idempotency", "bounded_collection",
    "async_timeout", "contract_schema",
}


def validate_structured_rule(rule: dict) -> dict:
    """Validate the JSON-only input language for the six high-level probe families."""
    if type(rule) is not dict:
        raise ValueError("structured probe must be a JSON object")
    if set(rule) - {"kind", "path", "symbol", "seed", "min", "max", "fallback", "transitions",
                    "invalid_result", "invalid_exception", "args", "expected", "replays",
                    "conflict_args", "conflict_expected", "capacity", "values", "eviction",
                    "timeout_ms", "timeout_args", "timeout_fallback", "input_schema",
                    "return_schema", "valid_args", "invalid_args", "invalid_exception",
                    "valid_mutations"}:
        raise ValueError("structured probe contains unknown fields")
    kind, path, symbol = rule.get("kind"), rule.get("path"), rule.get("symbol")
    if kind not in STRUCTURED_KINDS:
        raise ValueError("unsupported structured probe kind")
    if type(path) is not str or not path.endswith(".py") or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("structured probe path must be a repository-relative Python file")
    if type(symbol) is not str or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", symbol):
        raise ValueError("structured probe symbol must be a Python symbol path")
    canonical = dict(rule)
    canonical.setdefault("seed", 13_110_018)
    if type(canonical["seed"]) is not int or not 0 <= canonical["seed"] <= 2**32 - 1:
        raise ValueError("seed must be a 32-bit nonnegative integer")
    try:
        json.dumps(canonical, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("structured probe values must be finite JSON data") from error
    if kind == "numeric_bounds":
        low, high = canonical.get("min"), canonical.get("max")
        if type(low) is not int or type(high) is not int or low > high:
            raise ValueError("numeric_bounds requires integer min <= max")
        canonical.setdefault("fallback", None)
    elif kind == "state_machine":
        transitions = canonical.get("transitions")
        if type(transitions) is not dict or not transitions or any(
                type(state) is not str or type(actions) is not dict or not actions
                or any(type(action) is not str or type(dest) is not str for action, dest in actions.items())
                for state, actions in transitions.items()):
            raise ValueError("state_machine requires a nonempty state-to-action transition map")
        if "invalid_result" not in canonical and "invalid_exception" not in canonical:
            canonical["invalid_result"] = None
        _validate_exception(canonical.get("invalid_exception"))
    elif kind == "idempotency":
        if "args" not in canonical or "expected" not in canonical:
            raise ValueError("idempotency requires args and expected")
        replays = canonical.setdefault("replays", 3)
        if type(replays) is not int or not 2 <= replays <= 20:
            raise ValueError("idempotency replays must be in 2..20")
    elif kind == "bounded_collection":
        capacity, values = canonical.get("capacity"), canonical.get("values")
        if type(capacity) is not int or not 1 <= capacity <= 10000 or type(values) is not list:
            raise ValueError("bounded_collection requires positive capacity and a values list")
        if canonical.get("eviction", "oldest") not in {"oldest", "newest"}:
            raise ValueError("eviction must be oldest or newest")
        canonical.setdefault("eviction", "oldest")
    elif kind == "async_timeout":
        if "args" not in canonical or "expected" not in canonical or "timeout_args" not in canonical:
            raise ValueError("async_timeout requires args, expected and timeout_args")
        timeout_ms = canonical.get("timeout_ms")
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 10000:
            raise ValueError("timeout_ms must be in 1..10000")
        if "timeout_fallback" not in canonical:
            raise ValueError("async_timeout requires timeout_fallback")
    elif kind == "contract_schema":
        for key in ("input_schema", "return_schema", "valid_args", "invalid_args"):
            if key not in canonical:
                raise ValueError("contract_schema requires input_schema, return_schema, valid_args and invalid_args")
        if (type(canonical["valid_args"]) is not list or not canonical["valid_args"]
                or any(type(item) is not list for item in canonical["valid_args"])):
            raise ValueError("contract_schema needs at least one valid argument vector")
        if (type(canonical["invalid_args"]) is not list or not canonical["invalid_args"]
                or any(type(item) is not list for item in canonical["invalid_args"])):
            raise ValueError("contract_schema needs at least one invalid argument vector")
        _validate_exception(canonical.get("invalid_exception", "ValueError"))
        canonical.setdefault("invalid_exception", "ValueError")
        _validate_schema(canonical["input_schema"])
        _validate_schema(canonical["return_schema"])
    if "valid_mutations" in canonical and type(canonical["valid_mutations"]) is not list:
        raise ValueError("valid_mutations must be an array")
    return canonical


def structured_cases(rule: dict) -> list[dict]:
    """Compile category-specific valid, adversarial and fixed-seed property cases."""
    rule = validate_structured_rule(rule)
    rng = random.Random(rule["seed"])
    kind = rule["kind"]
    cases = []
    if kind == "numeric_bounds":
        low, high, fallback = rule["min"], rule["max"], rule["fallback"]
        midpoint = (low + high) // 2
        values = [("positive_low", low, low), ("positive_high", high, high),
                  ("negative_underflow", low - 1, low), ("negative_overflow", high + 1, high),
                  ("negative_type", None, fallback), ("negative_fraction", 1.5, fallback)]
        cases.extend(_case(label, [value], expected) for label, value, expected in values)
        cases.append(_case("positive_midpoint", [midpoint], midpoint))
        for index in range(5):
            value = rng.randint(low - 8, high + 8)
            cases.append(_case(f"randomized_{index}", [value], min(high, max(low, value))))
    elif kind == "state_machine":
        transitions = rule["transitions"]
        states = sorted(set(transitions) | {dest for actions in transitions.values() for dest in actions.values()})
        actions = sorted({action for choices in transitions.values() for action in choices})
        for state, mapping in transitions.items():
            for action, dest in mapping.items():
                cases.append(_case(f"positive_{state}_{action}", [state, action], dest))
        invalid = [(state, action) for state in states for action in actions
                   if action not in transitions.get(state, {})]
        invalid.extend([("__unknown_state__", actions[0]), (states[0], "__unknown_action__")])
        exception = rule.get("invalid_exception")
        for index, (state, action) in enumerate(invalid):
            cases.append(_case(f"negative_transition_{index}", [state, action],
                               rule.get("invalid_result"), exception=exception))
        pool = [(state, action, dest) for state, mapping in transitions.items() for action, dest in mapping.items()]
        for index in range(5):
            if index % 2 == 0:
                state, action, expected = rng.choice(pool)
            else:
                state, action, expected = "__mutated_state__", "__mutated_action__", rule.get("invalid_result")
            cases.append(_case(f"randomized_transition_{index}", [state, action], expected,
                               exception=exception if state == "__mutated_state__" else None))
    elif kind == "idempotency":
        args, expected, repeats = rule["args"], rule["expected"], rule["replays"]
        cases.append(_case("positive_first_submission", args, expected))
        cases.append(_case("negative_duplicate_replay", args, expected, repeats=repeats))
        conflict = rule.get("conflict_args")
        if conflict is not None:
            cases.append(_case("negative_key_payload_conflict", conflict, rule.get("conflict_expected"),
                               exception=rule.get("invalid_exception")))
        for index in range(5):
            count = rng.randint(2, 10)
            cases.append(_case(f"randomized_replay_count_{index}", args, expected, repeats=count))
    elif kind == "bounded_collection":
        capacity, values = rule["capacity"], rule["values"]
        eviction = rule["eviction"]
        expected = values[-capacity:] if eviction == "oldest" else values[:capacity]
        cases.append(_case("positive_within_capacity", [values[:capacity], capacity], values[:capacity]))
        attack_values = values + list(range(capacity + 1, capacity + 6))
        attack_expected = attack_values[-capacity:] if eviction == "oldest" else attack_values[:capacity]
        cases.append(_case("negative_capacity_overflow", [attack_values, capacity], attack_expected))
        for index in range(5):
            mutated = [rng.randint(-1000, 1000) for _ in range(capacity + rng.randint(1, 7))]
            actual = mutated[-capacity:] if eviction == "oldest" else mutated[:capacity]
            cases.append(_case(f"randomized_collection_{index}", [mutated, capacity], actual))
    elif kind == "async_timeout":
        timeout = rule["timeout_ms"]
        cases.extend([_case("positive_async_completion", rule["args"], rule["expected"], async_timeout_ms=timeout * 4 + 200),
                      _case("negative_timeout_fallback", rule["timeout_args"], rule["timeout_fallback"], async_timeout_ms=timeout * 4 + 200)])
        mutation_args = [item["args"] for item in rule.get("valid_mutations", []) if type(item) is dict and type(item.get("args")) is list]
        mutation_args = mutation_args or [rule["args"]] * 5
        for index, args in enumerate(mutation_args[:5] if not rule.get("valid_mutations") else []):
            cases.append(_case(f"randomized_async_{index}", args, rule["expected"], async_timeout_ms=timeout * 4 + 200))
    elif kind == "contract_schema":
        for index, args in enumerate(rule["valid_args"]):
            cases.append(_case(f"positive_contract_{index}", args, None, return_schema=rule["return_schema"]))
        for index, args in enumerate(rule["invalid_args"]):
            cases.append(_case(f"negative_contract_{index}", args, None,
                               exception=rule["invalid_exception"], input_schema=rule["input_schema"]))
        for index in range(5):
            value = _random_schema_value(rule["input_schema"], rng)
            cases.append(_case(f"randomized_contract_{index}", [value], None,
                               input_schema=rule["input_schema"], return_schema=rule["return_schema"]))
    for index, mutation in enumerate(rule.get("valid_mutations", [])[:5]):
        if type(mutation) is dict and type(mutation.get("args")) is list:
            cases.append(_case(f"randomized_declared_{index}", mutation["args"], mutation.get("expected"),
                               async_timeout_ms=rule.get("timeout_ms", 100) * 4 + 200
                               if kind == "async_timeout" else None,
                               return_schema=rule.get("return_schema") if kind == "contract_schema" else None))
    return cases


def render_structured_probe(rule: dict) -> tuple[str, str, list[str], list[dict]]:
    rule = validate_structured_rule(rule)
    cases = structured_cases(rule)
    module = rule["path"][:-3].replace("/", ".")
    source = f'''import asyncio
import builtins
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path.cwd()))
target_module = importlib.import_module({module!r})
target = target_module
for part in {rule["symbol"].split(".")!r}:
    target = getattr(target, part)
CASES = json.loads({json.dumps(cases, ensure_ascii=False)!r})

def _schema_matches(value, schema):
    kind = schema.get("type")
    checks = {{
        "object": lambda item: type(item) is dict,
        "array": lambda item: type(item) is list,
        "string": lambda item: type(item) is str,
        "integer": lambda item: type(item) is int,
        "number": lambda item: type(item) in (int, float) and math.isfinite(item),
        "boolean": lambda item: type(item) is bool,
        "null": lambda item: item is None,
    }}
    if kind in checks and not checks[kind](value):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if kind == "object":
        if any(key not in value for key in schema.get("required", [])):
            return False
        return all(key not in value or _schema_matches(value[key], child)
                   for key, child in schema.get("properties", {{}}).items())
    if kind == "array" and "items" in schema:
        return all(_schema_matches(item, schema["items"]) for item in value)
    if kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]: return False
        if "maximum" in schema and value > schema["maximum"]: return False
    return True

@pytest.mark.parametrize("case", CASES, ids=[case["label"] for case in CASES])
def test_aegis_structured_acceptance(case):
    args = case["args"]
    def invoke():
        value = target(*args)
        if case.get("async_timeout_ms") is not None:
            if not hasattr(value, "__await__"):
                raise TypeError("async contract requires an awaitable result")
            return asyncio.run(asyncio.wait_for(value, timeout=case["async_timeout_ms"] / 1000))
        return value
    expected_exception = case.get("exception")
    if expected_exception:
        with pytest.raises(getattr(builtins, expected_exception)):
            invoke()
        return
    try:
        actual = invoke()
        for _ in range(max(1, case.get("repeats", 1) - 1)):
            replay = invoke()
            assert type(replay) is type(actual) and replay == actual, "AEGIS_ACCEPTANCE: replay changed result"
    except asyncio.TimeoutError as error:
        raise AssertionError("AEGIS_ACCEPTANCE: async fallback was not returned") from error
    if case.get("input_schema") is not None:
        assert args and _schema_matches(args[0], case["input_schema"]), "AEGIS_ACCEPTANCE: invalid input accepted"
    if case.get("return_schema") is not None:
        assert _schema_matches(actual, case["return_schema"]), "AEGIS_ACCEPTANCE: return schema mismatch"
    else:
        expected = case.get("expected")
        assert type(actual) is type(expected) and actual == expected, "AEGIS_ACCEPTANCE: contract mismatch"
'''
    # math is imported only when executing the generated contract checks.
    source = source.replace("import json\nimport sys", "import json\nimport math\nimport sys")
    filename = PROBE_NAMES[".py"]
    command = ["{python}", "-B", "-m", "pytest", "--noconftest", "-o", "addopts=",
               "-p", "no:cacheprovider", "-q", filename]
    return filename, source, command, cases


def compile_structured_probe(rule: dict, repo: str | Path, base: str, *, timeout=60, budget=None) -> dict:
    """Prove the generated v2 probe is red in a genuine isolated baseline worktree."""
    canonical = validate_structured_rule(rule)
    root = Path(repo).resolve()
    filename, source, command, cases = render_structured_probe(canonical)
    patch = _new_file_patch(filename, source)
    hashes = {filename: hashlib.sha256(source.encode("utf-8")).hexdigest()}
    sandbox = Sandbox(root)
    ticket = "synth-v2-" + uuid.uuid4().hex[:12]
    path = sandbox.create(ticket, base=base)
    try:
        sandbox.apply_patch(ticket, patch)
        from functools import partial
        exam = run_exam_gate if budget is None else partial(budget.run_gate, run_exam_gate)
        red = exam(path, [sys.executable if arg == "{python}" else arg for arg in command],
                   timeout=timeout, probe_script_sha256=hashes)
        if not is_true_red(red):
            raise ProbeNotFailingError("structured probe must fail a real AEGIS_ACCEPTANCE assertion: "
                                       + json.dumps(red, ensure_ascii=False))
    finally:
        sandbox.cleanup(ticket)
    categories = {}
    for case in cases:
        group = "randomized" if case["label"].startswith("randomized") else (
            "negative" if case["label"].startswith("negative") else "positive")
        categories[group] = categories.get(group, 0) + 1
    return {"command": command, "patch": patch, "hashes": hashes, "red": red, "rule": canonical,
            "case_count": len(cases), "case_categories": categories, "random_seed": canonical["seed"]}


def _case(label, args, expected, *, exception=None, repeats=None, async_timeout_ms=None,
          input_schema=None, return_schema=None):
    value = {"label": label, "args": args, "expected": expected}
    if exception:
        value["exception"] = exception
    if repeats is not None:
        value["repeats"] = repeats
    if async_timeout_ms is not None:
        value["async_timeout_ms"] = async_timeout_ms
    if input_schema is not None:
        value["input_schema"] = input_schema
    if return_schema is not None:
        value["return_schema"] = return_schema
    return value


def _validate_exception(name):
    if name is None:
        return
    if type(name) is not str or name not in {"ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError"}:
        raise ValueError("invalid structured-probe exception class")


def _validate_schema(schema, depth=0):
    if depth > 12 or type(schema) is not dict:
        raise ValueError("contract schema must be a bounded object")
    allowed = {"type", "required", "properties", "items", "enum", "minimum", "maximum"}
    if set(schema) - allowed or schema.get("type") not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ValueError("unsupported contract schema fields or type")
    if "required" in schema and (type(schema["required"]) is not list or any(type(item) is not str for item in schema["required"])):
        raise ValueError("required must be an array of field names")
    if "properties" in schema:
        if type(schema["properties"]) is not dict:
            raise ValueError("properties must be an object")
        for child in schema["properties"].values():
            _validate_schema(child, depth + 1)
    if "items" in schema:
        _validate_schema(schema["items"], depth + 1)
    if "enum" in schema and type(schema["enum"]) is not list:
        raise ValueError("enum must be an array")
    if any(key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key]))
           for key in ("minimum", "maximum")):
        raise ValueError("schema numeric bounds must be finite")
    if schema.get("minimum", -math.inf) > schema.get("maximum", math.inf):
        raise ValueError("schema minimum exceeds maximum")


def _random_schema_value(schema, rng):
    kind = schema["type"]
    if kind == "object":
        return {key: _random_schema_value(child, rng) for key, child in schema.get("properties", {}).items()
                if key in schema.get("required", []) or rng.random() < 0.5}
    if kind == "array":
        return [_random_schema_value(schema["items"], rng) for _ in range(rng.randint(0, 3))] if "items" in schema else []
    if kind == "string":
        return rng.choice(schema["enum"]) if schema.get("enum") else f"case-{rng.randrange(100000)}"
    if kind == "integer":
        low, high = int(schema.get("minimum", 0)), int(schema.get("maximum", 100))
        if high - low > 10000:
            high = low + 10000
        return rng.randint(low, high)
    if kind == "number":
        low, high = schema.get("minimum", 0), schema.get("maximum", 100)
        if high - low > 10000:
            high = low + 10000
        return rng.uniform(low, high)
    if kind == "boolean":
        return bool(rng.randrange(2))
    return None
