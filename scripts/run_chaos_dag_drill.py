"""Run a real four-node Aegis DAG drill in a disposable Git repository."""
from __future__ import annotations

import difflib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(command, *, cwd, env, check=True):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise RuntimeError(json.dumps({"command": command, "exit_code": result.returncode,
                                       "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False))
    return result


def _git(root, *args):
    return _run(["git", *args], cwd=root, env=os.environ.copy()).stdout.strip()


def _patch(before: str, after: str, path: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile="a/" + path, tofile="b/" + path))


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _pid_state(pid: int):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return None
    try:
        return stat.read_text(encoding="ascii").split()[2]
    except (OSError, IndexError):
        return "unknown"


def _assert_child_reaped(pid: int):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = _pid_state(pid)
        if state is None:
            return {"pid": pid, "state": None, "reaped": True}
        if state == "Z":
            time.sleep(0.02)
            continue
        time.sleep(0.02)
    state = _pid_state(pid)
    return {"pid": pid, "state": state, "reaped": state is None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-db", help="durable Aegis ledger to receive this drill's causal traces")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="aegis-pr8-chaos-") as temporary:
        temp = Path(temporary)
        repo = temp / "repo"
        (repo / "tests").mkdir(parents=True)
        guard = "def double(value):\n    return value + 1\n"
        poison = "def run():\n    return 0\n"
        test = "from guard import double\n\ndef test_double():\n    assert double(3) == 6\n"
        _write(repo / "guard.py", guard)
        _write(repo / "poison.py", poison)
        _write(repo / "tests" / "test_guard.py", test)
        for command in (["git", "init", "-b", "main"], ["git", "add", "."],
                        ["git", "-c", "user.name=Aegis Drill", "-c", "user.email=drill@localhost",
                         "commit", "-m", "chaos drill baseline"]):
            _run(command, cwd=repo, env=os.environ.copy())
        baseline_commit = _git(repo, "rev-parse", "HEAD")

        env = os.environ.copy()
        inherited = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + inherited if inherited else "")
        env["AEGIS_WORLD_MODEL_DIR"] = str(ROOT / "artifacts" / "causal_world_model")
        test_cmd = [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_guard.py"]

        failed_source = guard.replace("value + 1", "value + 2")
        failed_patch = _write(temp / "node1-first.diff", _patch(guard, failed_source, "guard.py"))
        preflight_contract = temp / "preflight.contract.json"
        wish_command = [sys.executable, "-m", "aegis_core", "wish",
                        "Fix double so double(3) returns 6", "--repo", str(repo), "--target", "guard.py",
                        "--source-dir", ".", "--fail-before", json.dumps(test_cmd),
                        "--pass-after", json.dumps(test_cmd), "--patch", str(failed_patch),
                        "--max-attempts", "1", "--no-auto-probe", "--out", str(preflight_contract)]
        if options.evidence_db:
            wish_command += ["--db", options.evidence_db]
        _run(wish_command, cwd=ROOT, env=env)
        preflight_command = [sys.executable, "-m", "aegis_core", "run", "--contract", str(preflight_contract)]
        if options.evidence_db:
            preflight_command += ["--db", options.evidence_db]
        preflight_process = _run(preflight_command, cwd=ROOT, env=env, check=False)
        if preflight_process.returncode != 1:
            raise RuntimeError("the initial faulty repair was expected to fail: " + preflight_process.stdout)
        preflight = json.loads(preflight_process.stdout)
        first_attempt = preflight["results"][0]
        diagnosis = first_attempt.get("diagnosis")
        if not isinstance(diagnosis, dict) or diagnosis.get("kind") != "FailureDiagnosis":
            raise RuntimeError("the failed repair did not produce FailureDiagnosis")
        if diagnosis.get("category") != "assertion" or "assert" not in diagnosis.get("evidence_excerpt", ""):
            raise RuntimeError("FailureDiagnosis did not identify the assertion failure")
        corrected_source = guard.replace("value + 1", "value * 2")
        corrected_patch = _write(temp / "node1-self-heal.diff", _patch(guard, corrected_source, "guard.py"))

        cheat_test = test.replace("== 6", "== 5")
        cheat_patch = _write(temp / "node2-cheat.diff", _patch(test, cheat_test, "tests/test_guard.py"))
        poisoned_source = "def run():\n    while True:\n        pass\n"
        poison_patch = _write(temp / "node4-timeout.diff", _patch(poison, poisoned_source, "poison.py"))

        marker = temp / "node3-executed.marker"
        child_pid_file = temp / "timeout-child.pid"
        hang_command = (
            "import pathlib,subprocess,sys; "
            f"child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
            "import poison; poison.run()"
        )
        timeout_before = [sys.executable, "-c", "import poison; assert poison.run() == 1"]
        timeout_after = [sys.executable, "-c", hang_command]
        command_module = [sys.executable, "-c",
                          f"from pathlib import Path; Path({str(marker)!r}).write_text('executed')"]

        base = {"schema_version": "aegis.task_dag.v1", "dag_id": "pr8-chaos-drill",
                "repo_root": str(repo), "base_commit": baseline_commit,
                "requirement": "Four-node failure recovery, pre-block, cascade fuse, and timeout kill drill",
                "topological_layers": [], "dependency_edges": [], "retrieval": {}, "tasks": [
                    {"task_id": "node1-self-heal", "kind": "repair", "title": "Read diagnosis and self-heal",
                     "depends_on": [], "contract": {"intent": "Fix double so double(3) returns 6",
                         "target_files": ["guard.py"], "success_criteria": ["assertion probe passes"]},
                     "risk_estimate": 0.2, "patches": [str(failed_patch), str(corrected_patch)],
                     "max_attempts": 2, "source_dirs": ["."], "fail_before_cmd": test_cmd,
                     "pass_after_cmd": test_cmd, "timeout_sec": 20},
                    {"task_id": "node2-cheat", "kind": "repair", "title": "Pre-block protected-test tampering",
                     "depends_on": [], "contract": {"intent": "Reject a test assertion edit",
                         "target_files": ["guard.py"], "success_criteria": ["HIGH_RISK_PRE_BLOCK"]},
                     "risk_estimate": 0.95, "patches": [str(cheat_patch)], "max_attempts": 1,
                     "source_dirs": ["."], "fail_before_cmd": test_cmd, "pass_after_cmd": test_cmd,
                     "timeout_sec": 10},
                    {"task_id": "node3-downstream", "kind": "command", "title": "Must be cascade-skipped",
                     "depends_on": ["node2-cheat"], "contract": {"intent": "Must not execute after failed prerequisite"},
                     "risk_estimate": 0.8, "command": command_module, "timeout_sec": 2},
                    {"task_id": "node4-timeout", "kind": "repair", "title": "Kill infinite-loop process group",
                     "depends_on": [], "contract": {"intent": "Terminate an unbounded repair probe",
                         "target_files": ["poison.py"], "success_criteria": ["TIMEOUT_KILLED"]},
                     "risk_estimate": 0.99, "patches": [str(poison_patch)], "max_attempts": 1,
                     "source_dirs": ["."], "fail_before_cmd": timeout_before,
                     "pass_after_cmd": timeout_after, "timeout_sec": 3},
                ]}
        plan_path = temp / "chaos-dag.json"
        plan_path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        dag_command = [sys.executable, "-m", "aegis_core", "run", "--dag", str(plan_path),
                       "--repo", str(repo), "--workers", "3"]
        if options.evidence_db:
            dag_command += ["--db", options.evidence_db]
        dag_process = _run(dag_command, cwd=ROOT, env=env, check=False)
        if dag_process.returncode != 1:
            raise RuntimeError("the chaos DAG itself should report blocked/failed poison nodes: " + dag_process.stdout)
        dag = json.loads(dag_process.stdout)
        by_id = {row["task_id"]: row for row in dag["tasks"]}
        first = by_id["node1-self-heal"]["aegis_result"]["results"]
        cheat = by_id["node2-cheat"]["aegis_result"]["results"][0]
        skipped = by_id["node3-downstream"]
        timeout_result = by_id["node4-timeout"]["aegis_result"]["results"][0]
        if len(first) != 2 or first[0].get("diagnosis", {}).get("kind") != "FailureDiagnosis" or not first[1]["passed"]:
            raise RuntimeError("node 1 did not fail, diagnose, and self-heal")
        if not cheat.get("pre_blocked") or cheat.get("pre_block_reason") != "HIGH_RISK_PRE_BLOCK":
            raise RuntimeError("node 2 was not deterministically pre-blocked")
        if skipped.get("cascade_status") != "SKIPPED_UPSTREAM_FAILED" or marker.exists():
            raise RuntimeError("node 3 ran or did not carry the cascade skip status")
        if not timeout_result.get("timeout_killed") or timeout_result.get("reward", 0) >= 0:
            raise RuntimeError("node 4 did not receive a negative timeout-kill reward")
        child_result = _assert_child_reaped(int(child_pid_file.read_text(encoding="ascii")))
        worktree_paths = sorted(path.name for path in (repo / ".worktrees").glob("*") if path.exists())
        remaining_git_worktrees = [line[8:] for line in _git(repo, "worktree", "list", "--porcelain").splitlines()
                                   if line.startswith("worktree ")]
        if not child_result["reaped"] or worktree_paths or len(remaining_git_worktrees) != 1:
            raise RuntimeError("process or isolated worktree cleanup failed")
        if _git(repo, "status", "--porcelain"):
            raise RuntimeError("the temporary target repository was modified")

        evidence = {"status": "passed", "baseline_commit": baseline_commit,
                    "preflight_raw_result": preflight,
                    "failure_diagnosis_used": {"kind": diagnosis["kind"], "category": diagnosis["category"],
                                               "assertions": diagnosis["assertions"],
                                               "evidence_excerpt": diagnosis["evidence_excerpt"]},
                    "dag_raw_result": dag,
                    "timeout_child_cleanup": child_result,
                    "remaining_git_worktrees": remaining_git_worktrees,
                    "residual_worktree_paths": worktree_paths,
                    "downstream_marker_exists": marker.exists(),
                    "main_worktree_clean": True}
        print(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
