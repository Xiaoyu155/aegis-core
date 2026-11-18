"""Generate deterministic unseen inputs from a frozen acceptance rule and baseline AST."""
import ast
import hashlib
from pathlib import Path
import random
import sys
import tempfile

from .probe_synth import expected_value, render_probe, static_cases

MUTATION_PROBE_FAILED = "MUTATION_PROBE_FAILED"
DEFAULT_SEED = 20261118


def mutation_cases(rule, *, seed=DEFAULT_SEED):
    randomizer = random.Random(seed)
    static = {repr(value) for value, _ in static_cases(rule)}
    inputs = [0, -1, 2**53 - 1, -(2**53 - 1), [], {}, "", "1",
              {"value": "invalid"}, True, False, 1.5]
    if rule["kind"] == "clamp":
        low, high = rule["low"], rule["high"]
        inputs.extend((low - 2, high + 2))
        for _ in range(5):
            inputs.extend((low - randomizer.randrange(2, 100000), high + randomizer.randrange(2, 100000),
                           randomizer.randint(low, high)))
    else:
        inputs.extend(([{}], [[]], [None], ["", ""], ["open", "close", "open", "reset"],
                       ["open", {"event": "close"}], ["open"] * 1024))
        for _ in range(5):
            names = [f"event-{randomizer.randrange(1000)}" for _ in range(4)]
            inputs.extend((names + names[:2], [*names, 5]))
    cases = []
    for value in inputs:
        if repr(value) in static:
            continue
        static.add(repr(value))
        cases.append((value, expected_value(rule, value)))
    return cases


def validate_target(path, rule):
    if Path(rule["path"]).suffix != ".py":
        return
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == rule["symbol"]]
    if len(functions) != 1:
        raise ValueError("mutation target must be a unique synchronous top-level function")
    arguments = functions[0].args
    positional = arguments.posonlyargs + arguments.args
    if len(positional) != 1 or arguments.kwonlyargs or arguments.vararg or arguments.kwarg:
        raise ValueError("mutation target must accept one positional input")
    annotation = positional[0].annotation
    allowed = {"int"} if rule["kind"] == "clamp" else {"list", "list[str]", "Sequence[str]"}
    if annotation is not None and ast.unparse(annotation).strip("'\"") not in allowed:
        raise ValueError("mutation target annotation disagrees with acceptance domain")


def run_mutations(path, rule, *, timeout, seed=DEFAULT_SEED, exam=None):
    from .sandbox import run_exam_gate
    cases = mutation_cases(rule, seed=seed)
    filename, source, command = render_probe(rule, cases)
    with tempfile.TemporaryDirectory(prefix="aegis-mutations-") as folder:
        script = Path(folder) / Path(filename).name
        payload = source.encode("utf-8")
        script.write_bytes(payload)
        command = [sys.executable if arg == "{python}" else str(script) if arg == filename else arg
                   for arg in command]
        result = (exam or run_exam_gate)(path, command, timeout=timeout,
                               probe_script_sha256={str(script): hashlib.sha256(payload).hexdigest()})
    return {**result, "seed": seed, "cases": len(cases)}
