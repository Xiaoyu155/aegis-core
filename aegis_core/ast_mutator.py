"""Generate and execute deterministic Python AST mutation tests.

Mutants and their frozen acceptance test run in separate temporary source trees.
The engine reports a zero score when the source has no mutation sites or when
the unmodified candidate does not first pass its acceptance test.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


_COMPARISON_REVERSE = {
    ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}

_DIRECT_PROBE_RUNNER = '''\
import importlib.util
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
probe_path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("aegis_frozen_probe", probe_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
tests = [value for name, value in vars(module).items()
         if name.startswith("test_") and callable(value)]
if not tests:
    raise SystemExit(78)
for test in tests:
    marks = getattr(test, "pytestmark", [])
    if not isinstance(marks, list):
        marks = [marks]
    parameters = [mark for mark in marks if getattr(mark, "name", None) == "parametrize"]
    if len(parameters) != 1:
        raise SystemExit(78)
    mark = parameters[0]
    if len(mark.args) < 2:
        raise SystemExit(78)
    names = [name.strip() for name in mark.args[0].split(",")]
    values = mark.args[1]
    try:
        iterator = iter(values)
    except TypeError:
        raise SystemExit(78)
    for value in iterator:
        arguments = (value,) if len(names) == 1 else tuple(value)
        if len(arguments) != len(names):
            raise SystemExit(78)
        test(*arguments)
'''


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _replace_child(parent: ast.AST, original: ast.AST, replacements: list[ast.stmt]) -> None:
    for field, value in ast.iter_fields(parent):
        if value is original:
            setattr(parent, field, replacements[0] if replacements else None)
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                if item is original:
                    value[index:index + 1] = replacements
                    return
    raise ValueError("mutation target is detached from its AST parent")


def _has_exit(node: ast.If) -> bool:
    return any(isinstance(item, (ast.Raise, ast.Return)) for statement in node.body
               for item in ast.walk(statement))


def generate_mutants(source: str, *, symbol: str | None = None, max_mutants: int = 128) -> list[dict]:
    """Return unique parseable mutants made from AST-level defect operators."""
    if type(source) is not str or type(max_mutants) is not int or not 1 <= max_mutants <= 512:
        raise ValueError("source must be text and max_mutants must be in 1..512")
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    node_indices = {node: index for index, node in enumerate(nodes)}
    if symbol is not None:
        if type(symbol) is not str or not symbol or any(not part.isidentifier() for part in symbol.split(".")):
            raise ValueError("symbol must be a dotted Python identifier")
        target_functions = [node for node in nodes
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name == symbol.split(".")[-1]]
        if not target_functions:
            raise ValueError(f"mutation symbol was not found: {symbol}")
        scoped_nodes = {item for function in target_functions for item in ast.walk(function)}
    else:
        scoped_nodes = set(nodes)
    candidates: list[tuple[str, ast.AST, int | None]] = []
    for node in nodes:
        if node not in scoped_nodes:
            continue
        if isinstance(node, ast.Compare):
            for index, operator in enumerate(node.ops):
                if type(operator) in _COMPARISON_REVERSE:
                    candidates.append(("reverse_comparison", node, index))
        elif isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            candidates.append(("change_numeric_boundary", node, None))
        elif isinstance(node, ast.Raise):
            candidates.append(("raise_to_pass", node, None))
        elif isinstance(node, ast.If) and _has_exit(node):
            candidates.append(("remove_guard", node, None))

    mutants = []
    seen = {source}
    for kind, original, operator_index in candidates:
        cloned = copy.deepcopy(tree)
        target_index = node_indices[original]
        target = list(ast.walk(cloned))[target_index]
        if kind == "reverse_comparison":
            assert isinstance(target, ast.Compare) and operator_index is not None
            target.ops[operator_index] = _COMPARISON_REVERSE[type(target.ops[operator_index])]()
        elif kind == "change_numeric_boundary":
            assert isinstance(target, ast.Constant)
            target.value = (target.value + 1 if target.value >= 0 else target.value - 1)
        elif kind == "raise_to_pass":
            parent = _parent_map(cloned)[target]
            _replace_child(parent, target, [ast.copy_location(ast.Pass(), target)])
        else:
            assert isinstance(target, ast.If)
            parent = _parent_map(cloned)[target]
            replacement = target.orelse or [ast.copy_location(ast.Pass(), target)]
            _replace_child(parent, target, replacement)
        mutated = ast.fix_missing_locations(cloned)
        try:
            mutated_source = ast.unparse(mutated) + "\n"
            compile(mutated_source, "<aegis-mutant>", "exec")
        except (SyntaxError, ValueError):
            continue
        if mutated_source in seen:
            continue
        seen.add(mutated_source)
        line = getattr(original, "lineno", 1)
        digest = hashlib.sha256(mutated_source.encode("utf-8")).hexdigest()
        mutants.append({"id": f"{kind}:{line}:{digest[:12]}", "kind": kind,
                        "line": line, "sha256": digest, "source": mutated_source})
        if len(mutants) >= max_mutants:
            break
    return mutants


def _mirror_package(source: Path, destination: Path, target: Path) -> None:
    """Mirror one import package with symlinks, substituting exactly one module."""
    destination.mkdir(parents=True, exist_ok=True)
    target_relative = target.relative_to(source)
    for child in source.iterdir():
        relative = child.relative_to(source)
        output = destination / child.name
        if relative == target_relative:
            continue
        if child.is_dir() and target_relative.is_relative_to(relative):
            _mirror_package(child, output, target)
        else:
            try:
                output.symlink_to(child, target_is_directory=child.is_dir())
            except (NotImplementedError, OSError):
                # Hosted Windows runners may lack SeCreateSymbolicLinkPrivilege.
                # Copying keeps the mutant tree isolated and the campaign real.
                if child.is_dir():
                    shutil.copytree(child, output, symlinks=True)
                else:
                    shutil.copy2(child, output)


def _isolated_tree(repo_root: Path, target_path: Path, test_path: Path,
                   mutant_source: str, destination: Path) -> None:
    relative_target = target_path.relative_to(repo_root)
    parts = relative_target.parts
    package_root = None
    for end in range(1, len(parts)):
        candidate = repo_root.joinpath(*parts[:end])
        if candidate.is_dir() and (candidate / "__init__.py").is_file():
            package_root = candidate
            break
    if package_root is None:
        # Standalone modules such as service.py are importable from the isolated
        # working directory used by the frozen probe.
        mutant_target = destination / relative_target
    else:
        target_package_path = relative_target.relative_to(package_root.relative_to(repo_root))
        package_destination = destination / package_root.relative_to(repo_root)
        _mirror_package(package_root, package_destination, target_path)
        mutant_target = package_destination / target_package_path
    mutant_target.parent.mkdir(parents=True, exist_ok=True)
    mutant_target.write_text(mutant_source, encoding="utf-8", newline="\n")
    relative_test = test_path.relative_to(repo_root)
    isolated_test = destination / relative_test
    isolated_test.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(test_path, isolated_test)


def _run_probe(repo_root: Path, target_path: Path, test_path: Path, mutant_source: str,
               *, timeout: float) -> dict:
    from .sandbox import run_exam_gate

    with tempfile.TemporaryDirectory(prefix="aegis-ast-mutant-") as temporary:
        isolated = Path(temporary)
        _isolated_tree(repo_root, target_path, test_path, mutant_source, isolated)
        relative_test = test_path.relative_to(repo_root).as_posix()
        runner = isolated / ".aegis_ast_probe_runner.py"
        runner.write_text(_DIRECT_PROBE_RUNNER, encoding="utf-8")
        command = [sys.executable, "-B", str(runner), str(isolated / relative_test)]
        digest = hashlib.sha256((isolated / relative_test).read_bytes()).hexdigest()
        receipt = run_exam_gate(isolated, command, timeout=timeout,
                                probe_script_sha256={relative_test: digest},
                                environment_overrides={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
        # Generated probes are parametrized assertion functions and can be run
        # directly without starting pytest for every mutant. Retain the generic
        # pytest path for user supplied probes with fixtures or multiple marks.
        if receipt["exit_code"] == 78 and not receipt["timed_out"]:
            command = [sys.executable, "-B", "-m", "pytest", "--noconftest", "-o", "addopts=",
                       "-p", "no:cacheprovider", "-q", relative_test]
            receipt = run_exam_gate(isolated, command, timeout=timeout,
                                    probe_script_sha256={relative_test: digest},
                                    environment_overrides={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
        return {"passed": receipt["passed"], "exit_code": receipt["exit_code"],
                "stdout": receipt["stdout"][-2000:], "stderr": receipt["stderr"][-2000:],
                "timed_out": receipt["timed_out"]}


def run_mutation_campaign(repo_root: str | Path, target: str | Path, test: str | Path,
                          *, symbol: str | None = None, max_mutants: int = 64,
                          timeout: float = 3.0) -> dict:
    """Execute a frozen pytest acceptance file against every generated mutant."""
    root = Path(repo_root).resolve()
    target_path = (root / target).resolve()
    test_path = (root / test).resolve()
    if (not target_path.is_relative_to(root) or not test_path.is_relative_to(root)
            or target_path.is_symlink() or test_path.is_symlink()
            or not target_path.is_file() or not test_path.is_file()
            or target_path.suffix != ".py" or test_path.suffix != ".py"):
        raise ValueError("target and test must be regular Python files inside repo_root")
    source = target_path.read_text(encoding="utf-8")
    mutants = generate_mutants(source, symbol=symbol, max_mutants=max_mutants)
    baseline = _run_probe(root, target_path, test_path, source, timeout=timeout)
    outcomes = []
    if baseline["passed"]:
        for mutant in mutants:
            receipt = _run_probe(root, target_path, test_path, mutant["source"], timeout=timeout)
            outcomes.append({"id": mutant["id"], "kind": mutant["kind"], "line": mutant["line"],
                             "killed": not receipt["passed"], "timed_out": receipt["timed_out"],
                             "exit_code": receipt["exit_code"]})
    generated = len(mutants)
    killed = sum(item["killed"] for item in outcomes)
    score = killed / generated if baseline["passed"] and generated else 0.0
    return {"status": "complete" if baseline["passed"] and generated else "baseline_failed" if not baseline["passed"] else "no_mutants",
            "target": target_path.relative_to(root).as_posix(),
            "test": test_path.relative_to(root).as_posix(), "baseline": baseline,
            "generated": generated, "killed": killed, "survived": generated - killed,
            "mutation_kill_rate": score, "mutants": outcomes,
            "passed": bool(baseline["passed"] and generated and killed == generated)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--path", required=True, help="tracked Python target relative to repo")
    parser.add_argument("--test", required=True, help="frozen pytest acceptance file relative to repo")
    parser.add_argument("--symbol", help="mutate only this function or method name")
    parser.add_argument("--max-mutants", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)
    try:
        if not 0 < args.timeout <= 30:
            raise ValueError("timeout must be in (0, 30]")
        result = run_mutation_campaign(args.repo, args.path, args.test, symbol=args.symbol,
                                       max_mutants=args.max_mutants, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, SyntaxError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
