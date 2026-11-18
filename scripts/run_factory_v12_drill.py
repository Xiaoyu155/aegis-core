"""Run four v12 acceptance scenarios against physically isolated Git worktrees."""
from __future__ import annotations

import ast
import difflib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def cli(*arguments, expected=0):
    run = subprocess.run([sys.executable, "-m", "aegis_core", *map(str, arguments)],
                         text=True, encoding="utf-8", capture_output=True)
    if run.returncode != expected:
        raise RuntimeError(f"CLI exit {run.returncode}, expected {expected}\n{run.stdout}\n{run.stderr}")
    return json.loads(run.stdout)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def repository(path, *, node=False):
    path.mkdir()
    git(path, "init", "-q")
    (path / "src").mkdir()
    (path / "src/limits.py").write_text("def clamp(value: int):\n    return value\n", encoding="utf-8")
    (path / "src/limits.mjs").write_text("export function clamp(value) { return value; }\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "-c", "user.name=Drill", "-c", "user.email=drill@example.invalid", "commit", "-qm", "Baseline")
    return "src/limits.mjs" if node else "src/limits.py"


def authored_patch(repo, target, output, *, cheat=False):
    before = (repo / target).read_text(encoding="utf-8")
    if target.endswith(".py"):
        function = ast.parse(before).body[0]
        value = function.args.args[0].arg
        if cheat:
            statements = f"""
if {value} is None:
    return None
if {value} == -1:
    return 0
if {value} == 11:
    return 10
return {value}
"""
        else:
            statements = f"return min(10, max(0, {value})) if type({value}) is int else None"
        function.body = ast.parse(statements).body
        after = ast.unparse(function) + "\n"
    else:
        after = ("export function clamp(value) {\n"
                 "  return Number.isInteger(value) ? Math.min(10, Math.max(0, value)) : null;\n"
                 "}\n")
    patch = f"diff --git a/{target} b/{target}\n" + "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{target}", f"b/{target}"
    ))
    output.write_bytes(patch.encode("utf-8"))
    return output


def wish(directory, repo, target, name, patches):
    manifest = directory / f"{name}.json"
    contract = cli("wish", f"{target}::clamp clamps integers to [0, 10] and returns null for invalid inputs.",
                   "--repo", repo, "--ticket-id", name, "--out", manifest,
                   *(argument for patch in patches for argument in ("--patch", patch)))
    assert contract["synthesized_probe"] and contract["probe_script_sha256"]
    return manifest, contract


def budget_dag(directory, repo):
    marker = directory / "unreachable"
    return write_json(directory / "low-budget.json", {
        "schema_version": "aegis.task_dag.v1",
        "dag_id": "drill-budget",
        "repo_root": str(repo),
        "base_commit": git(repo, "rev-parse", "HEAD"),
        "max_token_budget": 0,
        "max_cost_usd": 0,
        "max_wall_clock_sec": 0.8,
        "tasks": [
            {"task_id": "slow", "kind": "validation", "depends_on": [],
             "command": ["{python}", "-c", "import time; time.sleep(10)"]},
            {"task_id": "unreachable", "kind": "validation", "depends_on": ["slow"],
             "command": ["{python}", "-c", "from pathlib import Path; Path(__import__('sys').argv[1]).touch()", str(marker)]},
        ],
    }), marker


def main():
    with tempfile.TemporaryDirectory(prefix="aegis-v12-drill-") as temporary:
        directory = Path(temporary)
        python_repo, node_repo = directory / "python-repo", directory / "node-repo"
        python_target = repository(python_repo)
        node_target = repository(node_repo, node=True)
        good = authored_patch(python_repo, python_target, directory / "good.diff")
        cheat = authored_patch(python_repo, python_target, directory / "cheat.diff", cheat=True)
        node_patch = authored_patch(node_repo, node_target, directory / "node.diff")

        contract, frozen = wish(directory, python_repo, python_target, "mutation", [cheat, good])
        result = cli("run", "--contract", contract)
        first, second = result["results"]
        assert first["after_passed"] and first["gate_failure"] == "MUTATION_PROBE_FAILED"
        assert first["reward"] < 0 and not first["passed"] and not first["mutation"]["passed"]
        assert second["after_passed"] and second["mutation"]["passed"] and second["passed"]
        print(f"1 | static=PASS cheat=MUTATION_PROBE_FAILED reward={first['reward']:.4f} "
              f"generic=PASS mutations={second['mutation']['cases']} seed={second['mutation']['seed']}", flush=True)

        assert frozen["expanded_spec"]["kind"] == "ExpandedSpec"
        assert [item["kind"] for item in frozen["expanded_spec"]["acceptance_criteria"]] == [
            "normal", "boundary", "fallback"
        ]
        baseline = second["baseline"]
        sha = next(iter(frozen["probe_script_sha256"].values()))
        assert baseline["failure_code"] == "ASSERTION_FAILED" and baseline["exit_code"] == 1
        assert baseline["probe_script_sha256"] == frozen["probe_script_sha256"]
        assert baseline["probe_runtime_sha256"] == second["final"]["probe_runtime_sha256"]
        print(f"2 | ExpandedSpec -> pytest red={baseline['failure_code']} sha256={sha} run={result['status']}",
              flush=True)

        dag, marker = budget_dag(directory, python_repo)
        halt = cli("run", "--dag", dag, expected=1)
        assert halt["reason"] == "BUDGET_EXCEEDED_HALT" and not marker.exists()
        assert halt["budget"]["halted_by"] == "max_wall_clock_sec"
        from aegis_core.ledger import Ledger
        ledger = Ledger(python_repo / ".git/aegis/ledger.sqlite")
        events = ledger.budget_events(halt["budget"]["scope"])
        assert [event["operation"] for event in events] == ["sandbox_gate", "halt"]
        print(f"3 | {halt['reason']} limit={halt['budget']['halted_by']} "
              f"elapsed={halt['budget']['wall_clock_sec']:.3f}s events={len(events)} dependent_executed={marker.exists()}",
              flush=True)

        py_profile = cli("attach", "--repo", python_repo)["profile_path"]
        js_profile = cli("attach", "--repo", node_repo)["profile_path"]
        node_contract, _ = wish(directory, node_repo, node_target, "batch-node", [node_patch])
        config = write_json(directory / "factory.json", {"schema_version": "aegis.factory.v1", "jobs": [
            {"id": "node", "priority": 1, "profile": js_profile, "contract": str(node_contract)},
            {"id": "python", "priority": 2, "profile": py_profile, "dag": str(dag)},
        ]})
        destination = directory / "factory-report.json"
        report = cli("batch", "--config", config, "--out", destination, expected=1)
        assert json.loads(destination.read_text(encoding="utf-8")) == report
        python_result, node_result = report["jobs"]
        assert python_result["id"] == "python" and python_result["result"]["reason"] == "BUDGET_EXCEEDED_HALT"
        assert node_result["id"] == "node" and node_result["status"] == "succeeded"
        node_gate = node_result["result"]["results"][0]
        assert node_gate["baseline"]["failure_code"] == "ASSERTION_FAILED" and node_gate["mutation"]["passed"]
        assert report["passed"] == report["failed"] == 1
        assert all(all(job["languages"][lang]["present"] for lang in ("python", "javascript"))
                   for job in report["jobs"])
        for repo in [python_repo, node_repo]:
            assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
        cost = report["budget"]
        print(f"4 | python={python_result['result']['reason']} node={node_result['status']} "
              f"languages=Python+JavaScript repositories=2 passed={report['passed']} failed={report['failed']} "
              f"tokens={cost['token_consumption']} usd={cost['estimated_cost_usd']:.6f} "
              f"reward={cost['reward']:.4f} ledger=verified", flush=True)


if __name__ == "__main__":
    main()
