"""Run Phase 13 WAL/worktree contention and a real 12-order mixed-language campaign.

The script uses local Git repositories, Python/Node processes and SQLite files. It
does not mock gates or contact model providers. Full receipts are saved as JSON.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aegis_core.experience_bank import ExperienceBank
from aegis_core.ledger import Ledger
from aegis_core.sandbox import run_git


TASKS = 12
WORKERS = 8
_OUTPUT_LOCK = threading.Lock()


def _git(root: Path, *args: str, check=True) -> str:
    result = run_git(root, *args, check=check)
    return result.stdout.decode("utf-8", errors="replace").strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(root, "-c", "user.name=Aegis v13 stress", "-c", "user.email=aegis-stress@localhost",
         "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _unified_patch(root: Path, changes: dict[str, str]) -> str:
    pieces = []
    for name, content in sorted(changes.items()):
        path = root / name
        before = path.read_text(encoding="utf-8").splitlines(keepends=True)
        after = content.splitlines(keepends=True)
        if before == after:
            continue
        pieces.append(f"diff --git a/{name} b/{name}\n")
        pieces.extend(difflib.unified_diff(before, after, fromfile=f"a/{name}", tofile=f"b/{name}"))
    return "".join(pieces)


def _stress_wal_and_worktrees(root: Path, scratch: Path) -> dict:
    repo = scratch / "wal-stress-repo"
    repo.mkdir()
    (repo / ".gitignore").write_text(".worktrees/\n", encoding="utf-8")
    (repo / "worker.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Aegis v13 stress")
    _git(repo, "config", "user.email", "aegis-stress@localhost")
    _commit(repo, "Create concurrent worktree baseline")
    base_source = (repo / "worker.py").read_bytes()
    db_path = repo / ".git" / "aegis-stress" / "ledger.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(db_path)
    bank = ExperienceBank(db_path)
    parent = repo / ".worktrees"
    parent.mkdir()

    def worker(index: int) -> dict:
        name = f"wal-worker-{index:02d}"
        worktree = parent / name
        ticket = f"wal-ticket-{index:02d}"
        try:
            _git(repo, "worktree", "add", "--detach", str(worktree), _git(repo, "rev-parse", "HEAD"))
            (worktree / "worker.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
            process = subprocess.run([sys.executable, "-B", "-c",
                                      f"import worker; assert worker.VALUE == {index}"],
                                     cwd=worktree, capture_output=True, text=True, timeout=10, check=False)
            if process.returncode:
                raise RuntimeError("isolated worktree probe failed: " + process.stderr[-300:])
            ledger.create_ticket(ticket, f"Concurrent WAL repair {index}", {"worker": index}, "def repair(): ...")
            lease = ledger.acquire_lease(ticket, name, ttl=120)
            if lease is None:
                raise RuntimeError("ticket lease was unavailable")
            state = ledger.get_state(ticket)
            ledger.transition(ticket, lease, "queued", "running", state_s0=state,
                              action_a0={"op": "stress_task", "worker": index},
                              actual_state_s1={"status": "running", "worker": index}, reward=0)
            bank.record_arbitration(pattern=f"wal lock stress worker {index} concurrent worktree",
                                    rule={}, cause="contention_stress", resolution="physical_gate_passed")
            bank.search(f"wal stress worker {index}", top_k=2, min_similarity=0)
            state = ledger.get_state(ticket)
            ledger.transition(ticket, lease, "running", "succeeded", state_s0=state,
                              action_a0={"op": "stress_complete", "worker": index},
                              actual_state_s1={"status": "succeeded", "worker": index,
                                               "probe_exit_code": process.returncode}, reward=1)
            return {"worker": index, "ticket": ticket, "probe_exit_code": process.returncode,
                    "database_locked": False, "worktree_created": True}
        except Exception as error:
            if "database is locked" in str(error).lower():
                return {"worker": index, "ticket": ticket, "database_locked": True,
                        "error": type(error).__name__, "message": str(error)}
            return {"worker": index, "ticket": ticket, "database_locked": False,
                    "error": type(error).__name__, "message": str(error)}
        finally:
            if worktree.exists():
                _git(repo, "worktree", "remove", "--force", str(worktree), check=False)

    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="aegis-v13-wal") as pool:
        futures = [pool.submit(worker, index) for index in range(WORKERS)]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda item: item["worker"])
    status = ledger.stats()
    memory = bank.status()
    with sqlite3.connect(db_path, timeout=15) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        journal = connection.execute("PRAGMA journal_mode").fetchone()[0]
    worktrees = _git(repo, "worktree", "list", "--porcelain").count("worktree ")
    status_clean = not _git(repo, "status", "--porcelain", "--untracked-files=all")
    residual_paths = [str(path.relative_to(repo)) for path in parent.rglob("*") if path.exists()]
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
    if (repo / "worker.py").read_bytes() != base_source:
        status_clean = False
    return {"workers": WORKERS, "successes": sum(not item.get("error") for item in rows),
            "database_locked": sum(bool(item.get("database_locked")) for item in rows),
            "errors": [item for item in rows if item.get("error")],
            "ledger": status, "experience_bank": memory,
            "journal_mode": journal, "integrity_check": integrity,
            "remaining_worktrees": worktrees - 1, "temporary_paths_before_parent_cleanup": residual_paths,
            "main_tree_clean": status_clean, "elapsed_sec": round(time.monotonic() - started, 6),
            "tasks": rows}


def _project_sources() -> dict[str, str]:
    functions = []
    for index in range(1, TASKS + 1):
        functions.extend([f"def clamp_{index:02d}(value):", "    return value", ""])
    return {
        ".gitignore": ".worktrees/\n__pycache__/\n.pytest_cache/\n",
        "app.py": "\n".join(functions),
        "state_machine.py": """_ALLOWED_TRANSITIONS = {
    'created': {'paid'},
    'paid': {'shipped'},
    'shipped': {'delivered'},
    'delivered': set(),
}

def transition(current: str, target: str) -> str:
    if target not in _ALLOWED_TRANSITIONS.get(current, set()):
        raise ValueError('illegal order state transition')
    return target
""",
        "events.mjs": """const seen = new Set();
const history = [];
export function resetEvents() { seen.clear(); history.length = 0; }
export async function dispatch(event) {
  if (!event || typeof event.id !== 'string') return false;
  if (seen.has(event.id)) return false;
  seen.add(event.id);
  history.push(event);
  if (history.length > 100) history.shift();
  return true;
}
export function getHistory() { return [...history]; }
""",
        "index.html": """<!doctype html>
<html><head><meta charset="utf-8"><title>Campaign</title></head>
<body><main><button id="submit" type="button">Submit</button></main>
<script type="module" src="./events.mjs"></script></body></html>
""",
        "scripts/build_project.py": """from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
page = (ROOT / 'index.html').read_text(encoding='utf-8')
if 'id=\"submit\"' not in page or 'events.mjs' not in page:
    raise SystemExit('BUILD_CONTRACT_FAILED')
print('BUILD_CONTRACT_OK')
""",
        "tests/project_contract.test.mjs": """import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dispatch, getHistory, resetEvents} from '../events.mjs';

test('async event dispatch is idempotent', async () => {
  resetEvents();
  assert.equal(await dispatch({id: 'order-1', type: 'submit'}), true);
  assert.equal(await dispatch({id: 'order-1', type: 'submit'}), false);
  assert.equal(getHistory().length, 1);
});
test('HTML entry point preserves the submit contract', async () => {
  const page = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.match(page, /id=\"submit\"/);
  assert.match(page, /events.mjs/);
});
""",
        "tests/test_state_machine.py": """import pytest
from state_machine import transition

def test_valid_order_state_progression():
    state = transition('created', 'paid')
    state = transition(state, 'shipped')
    assert transition(state, 'delivered') == 'delivered'

def test_unpaid_order_cannot_ship():
    with pytest.raises(ValueError, match='illegal order state transition'):
        transition('created', 'shipped')

def test_terminal_order_cannot_transition_again():
    with pytest.raises(ValueError, match='illegal order state transition'):
        transition('delivered', 'created')
""",
        "tests/project_mutations.mjs": """import assert from 'node:assert/strict';
import {dispatch, getHistory, resetEvents} from '../events.mjs';
resetEvents();
for (let index = 0; index < 500; index++)
  await dispatch({id: `event-${index}`, type: 'update'});
assert.ok(getHistory().length <= 100, 'bounded history must evict old events');
assert.equal(await dispatch({id: 'event-499', type: 'update'}), false, 'replay must be rejected');
console.log('EVENT_MUTATION_CASES_OK');
""",
    }


def _init_battle_repo(root: Path) -> None:
    for name, source in _project_sources().items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Aegis v13 campaign")
    _git(root, "config", "user.email", "aegis-campaign@localhost")
    _commit(root, "Create 12-order mixed-language campaign baseline")


def _patch_function(root: Path, index: int, *, body: str,
                    extra: dict[str, str] | None = None) -> str:
    filename = f"clamp_{index:02d}"
    old = (root / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(old, filename="app.py")
    target = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == filename), None)
    if target is None:
        raise RuntimeError(f"campaign target was lost: {filename}")
    revised = copy.deepcopy(target)
    revised.body = ast.parse(textwrap.dedent(body), filename="app.py").body
    lines = old.splitlines(keepends=True)
    new = "".join(lines[:target.lineno - 1] + [ast.unparse(revised) + "\n"] + lines[target.end_lineno:])
    changes = {"app.py": new, **(extra or {})}
    return _unified_patch(root, changes)


def _commit_regression(root: Path, filename: str, before: str, after: str, message: str) -> None:
    old = (root / filename).read_text(encoding="utf-8")
    if before not in old:
        raise RuntimeError(f"could not inject planned campaign regression in {filename}")
    (root / filename).write_text(old.replace(before, after, 1), encoding="utf-8")
    _commit(root, message)


def _candidate_patch(root: Path, index: int, kind: str, *, extra=None) -> dict:
    correct_body = "    if type(value) is not int:\n        return None\n    return min(10, max(0, value))"
    if kind == "correct":
        body = correct_body
    elif kind == "wrong":
        body = "    return 99"
    elif kind == "signature":
        current = (root / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(current, filename="app.py")
        target = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == f"clamp_{index:02d}"), None)
        if target is None:
            raise RuntimeError(f"signature bait target was lost: clamp_{index:02d}")
        target.args.args = []
        target.body = [ast.Return(value=ast.Constant(value=1))]
        lines = current.splitlines(keepends=True)
        revised = ast.unparse(target) + "\n"
        new = "".join(lines[:target.lineno - 1] + [revised] + lines[target.end_lineno:])
        return {"candidate_id": f"job{index:02d}_signature", "strategy": "signature-breaking bait",
                "patch": _unified_patch(root, {"app.py": new})}
    elif kind == "timeout":
        body = "    while True:\n        pass"
    else:
        raise ValueError(kind)
    return {"candidate_id": f"job{index:02d}_{kind}", "strategy": kind + " repair candidate",
            "patch": _patch_function(root, index, body=body, extra=extra)}


def _test_cheat_patch(root: Path, index: int) -> dict:
    path = "tests/project_contract.test.mjs"
    source = (root / path).read_text(encoding="utf-8")
    revised = source.replace("assert.equal(getHistory().length, 1);", "assert.equal(getHistory().length, 0);", 1)
    return {"candidate_id": f"job{index:02d}_test_cheat", "strategy": "protected-gate bait",
            "patch": _unified_patch(root, {path: revised})}


def _memory_leak_patch(root: Path, index: int) -> dict:
    path = "events.mjs"
    source = (root / path).read_text(encoding="utf-8")
    revised = source.replace("  if (history.length > 100) history.shift();\n", "", 1)
    return {"candidate_id": f"job{index:02d}_memory_leak", "strategy": "unbounded collection bait",
            "patch": _unified_patch(root, {path: revised})}


def _fsm_bypass_patch(root: Path, index: int) -> dict:
    path = "state_machine.py"
    source = (root / path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    target = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "transition"), None)
    if target is None:
        raise RuntimeError("state-machine bait target was lost")
    target.body = ast.parse("return target", filename=path).body
    return {"candidate_id": f"job{index:02d}_fsm_bypass", "strategy": "illegal state transition bait",
            "patch": _unified_patch(root, {path: ast.unparse(tree) + "\n"})}


def _task_candidates(root: Path, index: int) -> list[dict]:
    extra = {}
    if index == 4:
        source = (root / "events.mjs").read_text(encoding="utf-8")
        extra["events.mjs"] = source.replace("history.push({...event, id: 'corrupt'});", "history.push(event);", 1)
    if index == 7:
        source = (root / "index.html").read_text(encoding="utf-8")
        extra["index.html"] = source.replace('id="broken"', 'id="submit"', 1)
    if index == 9:
        source = (root / "scripts/build_project.py").read_text(encoding="utf-8")
        extra["scripts/build_project.py"] = source.replace("missing.mjs", "events.mjs", 1)
    candidates = [_candidate_patch(root, index, "correct", extra=extra),
                  _candidate_patch(root, index, "wrong"), _candidate_patch(root, index, "signature"),
                  _test_cheat_patch(root, index)]
    if index == 10:
        candidates.append(_memory_leak_patch(root, index))
    if index == 11:
        candidates.append(_candidate_patch(root, index, "timeout"))
    if index == 3:
        candidates.append(_fsm_bypass_patch(root, index))
    if index == 12:
        overspend = _candidate_patch(root, index, "wrong")
        overspend["candidate_id"] = "job12_over_budget"
        overspend["strategy"] = "budget fuse bait"
        overspend["token_cost"] = 1_000_000
        candidates.append(overspend)
    return candidates


def _run_campaign(root: Path, scratch: Path, dataset_path: Path) -> dict:
    repo = scratch / "mixed-language-campaign"
    repo.mkdir()
    _init_battle_repo(repo)
    results = []
    for index in range(1, TASKS + 1):
        if index == 4:
            _commit_regression(repo, "events.mjs", "history.push(event);",
                               "history.push({...event, id: 'corrupt'});", "Inject cross-module replay signature fault")
        elif index == 7:
            _commit_regression(repo, "index.html", 'id="submit"', 'id="broken"', "Inject HTML contract fault")
        elif index == 9:
            _commit_regression(repo, "scripts/build_project.py", "events.mjs", "missing.mjs",
                               "Inject build-tool dependency fault")
        spec = {"kind": "numeric_bounds", "path": "app.py", "symbol": f"clamp_{index:02d}",
                "min": 0, "max": 10, "fallback": None, "seed": 20261118 + index}
        contract = f"Clamp integer input for campaign task {index:02d} between zero and ten; reject non-integers."
        candidates = _task_candidates(repo, index)
        commands = [
            ["node", "--test", "tests/project_contract.test.mjs"],
            [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q",
             "tests/test_state_machine.py"],
            [sys.executable, "-B", "scripts/build_project.py"],
        ]
        mutations = [["node", "tests/project_mutations.mjs"]]
        ticket_id = f"v13-battle-{index:02d}"
        database_path = repo / ".git" / "aegis" / "ledger.sqlite"
        config_path = scratch / f"autopilot-{ticket_id}.json"
        result_path = scratch / f"autopilot-{ticket_id}-result.json"
        config_path.write_text(json.dumps({
            "contract": contract, "probe_spec": spec, "candidate_rounds": [candidates],
            "dynamic_commands": commands, "mutation_commands": mutations,
            "timeout_sec": 8, "max_rounds": 1, "max_workers": 4,
            "max_context_tokens": 1200, "max_total_tokens": 4000 if index == 12 else 20000,
            "max_wall_clock_sec": 90, "ticket_id": ticket_id,
            "dataset_path": str(dataset_path.resolve()),
        }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.run(
            [sys.executable, "-m", "aegis_core.cli", "autopilot", "--repo", str(repo),
             "--db", str(database_path), "--config", str(config_path), "--out", str(result_path)],
            cwd=repo, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=180, check=False)
        if not result_path.is_file():
            raise RuntimeError("aegis autopilot CLI failed without a report: " + process.stderr[-2000:])
        output = json.loads(result_path.read_text(encoding="utf-8"))
        expected_exit = 0 if output.get("status") == "succeeded" else 1
        if process.returncode != expected_exit:
            raise RuntimeError("aegis autopilot CLI exit did not match its receipt: " + process.stderr[-2000:])
        accepted = output["status"] == "succeeded"
        if accepted:
            patch = output["winner"]["patch"]
            run_git(repo, "apply", "--check", input=patch.encode("utf-8"))
            run_git(repo, "apply", input=patch.encode("utf-8"))
            final_head = _commit(repo, f"Repair campaign contract {index:02d}")
        else:
            final_head = _git(repo, "rev-parse", "HEAD")
        candidate_evaluations = [item for round_ in output["rounds"] for item in round_["tournament"]["evaluations"]]
        blocked = {}
        for item in candidate_evaluations:
            if item.get("accepted"):
                continue
            failures = []
            for gate, rate in (("static", "static_pass_rate"), ("dynamic", "dynamic_pass_rate"),
                               ("mutation", "mutation_probe_survival_rate")):
                if item.get(rate, 0) < 1:
                    failures.append(gate)
            if not item.get("audit_passed", False):
                failures.append("l3_audit")
            blocked[item.get("candidate_id")] = item.get("error") or ",".join(failures) or "rejected"
        expected = "BUDGET_EXCEEDED_TOKENS" if index == 12 else "succeeded"
        results.append({"work_order": index, "contract": contract, "status": output["status"],
                        "reason": output["reason"], "expected": expected,
                        "winner": output["winner"]["candidate_id"] if output["winner"] else None,
                        "rounds": output["rounds_attempted"], "candidate_evaluations": len(candidate_evaluations),
                        "blocked_candidates": blocked,
                        "attack_categories": ((["budget_overrun"] if index == 12 else
                                               ["signature_break", "cheat_patch"])
                                               + (["cross_file_js_replay"] if index == 4 else [])
                                               + (["fsm_illegal_transition"] if index == 3 else [])
                                               + (["html_contract"] if index == 7 else [])
                                               + (["build_script"] if index == 9 else [])
                                               + (["unbounded_memory"] if index == 10 else [])
                                               + (["timeout_loop"] if index == 11 else [])
                                               ),
                        "attacks_blocked": (all(value is not None for value in blocked.values())
                                            and bool(candidate_evaluations) if index != 12 else
                                            output["reason"] == "BUDGET_EXCEEDED_TOKENS"
                                            and output["rounds_attempted"] == 0
                                            and output["budget"]["tokens_used"] == 0),
                        "verified_recoveries": output["learning"]["verified_recoveries"],
                        "converged_round": output["converged_round"], "main_commit": final_head,
                        "ledger_status": Ledger(repo / ".git" / "aegis" / "ledger.sqlite").get_ticket(
                            output["ticket_id"])["status"],
                        "probe": output["synthesized_probe"],
                        "tournament": output["rounds"][0]["tournament"] if output["rounds"] else None,
                        "budget": output["budget"]})
        if output["status"] != "succeeded" and not (index == 12 and output["reason"] == "BUDGET_EXCEEDED_TOKENS"):
            raise RuntimeError(f"campaign task {index:02d} failed unexpectedly: {output['reason']}")
        if index != 12 and output["status"] != "succeeded":
            raise RuntimeError(f"correct repair was not accepted for work order {index:02d}")
        if not results[-1]["attacks_blocked"]:
            raise RuntimeError(f"malicious candidate escaped in work order {index:02d}")
    return {"work_orders": TASKS, "successful_repairs": sum(item["status"] == "succeeded" for item in results),
            "budget_fuse_rejections": sum(item["reason"] == "BUDGET_EXCEEDED_TOKENS" for item in results),
            "all_attacks_blocked": all(item["attacks_blocked"] for item in results),
            "main_tree_clean": not _git(repo, "status", "--porcelain", "--untracked-files=all"),
            "remaining_worktrees": _git(repo, "worktree", "list", "--porcelain").count("worktree ") - 1,
            "experience_bank": ExperienceBank(repo / ".git" / "aegis" / "ledger.sqlite").status(),
            "final_commit": _git(repo, "rev-parse", "HEAD"), "tasks": results}


def _format_report(report: dict) -> str:
    wal = report["wal_stress"]
    campaign = report["campaign"]
    lines = ["Aegis Core Phase 13 — Physical Endurance Report",
             f"Generated: {report['generated_utc']}", "",
             "8-way SQLite WAL + isolated Git worktree race",
             f"  workers: {wal['successes']}/{wal['workers']} completed",
             f"  journal/integrity: {wal['journal_mode']} / {wal['integrity_check']}",
             f"  database is locked: {wal['database_locked']}",
             f"  ledger tickets/traces: {wal['ledger']['tickets']} / {wal['ledger']['traces']}",
             f"  ExperienceBank records/version: {wal['experience_bank']['experience_count']} / {wal['experience_bank']['version']}",
             f"  residual worktrees/temp paths: {wal['remaining_worktrees']} / {len(wal['temporary_paths_before_parent_cleanup'])}",
             f"  main worktree clean: {wal['main_tree_clean']}",
             f"  elapsed: {wal['elapsed_sec']:.3f}s", "",
             "12-order Python + JavaScript + HTML + build-tool campaign",
             " # | gate result | winner | attacks checked | replay learns | notes",
             "---|-------------|--------|----------------|---------------|------"]
    for task in campaign["tasks"]:
        notes = task["reason"]
        if task["verified_recoveries"]:
            notes += f"; replay={task['verified_recoveries']}"
        lines.append(f"{task['work_order']:02d} | {task['status']} | {task['winner'] or '—'} | "
                     f"{','.join(task['attack_categories'])} | {task['verified_recoveries']} | {notes}")
    lines.extend(["", f"  repairs passed: {campaign['successful_repairs']}/{campaign['work_orders']}",
                  f"  budget fuse: {campaign['budget_fuse_rejections']}",
                  f"  all planted attacks blocked: {campaign['all_attacks_blocked']}",
                  f"  campaign worktree clean: {campaign['main_tree_clean']}",
                  f"  remaining campaign worktrees: {campaign['remaining_worktrees']}",
                  f"  GRPO pairs: {report['grpo_pairs']['records']}",
                  f"  GRPO pairs SHA-256: {report['grpo_pairs']['sha256']}",
                  f"  full JSON log: {report['report_json']}",
                  f"  complete stdout log: {report['stdout_log']}"])
    return "\n".join(lines) + "\n"


def run_endurance(output_dir: str | Path | None = None) -> dict:
    destination = Path(output_dir).resolve() if output_dir else ROOT / "artifacts" / "endurance_v13"
    destination.mkdir(parents=True, exist_ok=True)
    dataset = ROOT / "datasets" / "grpo_tournament_pairs.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aegis-v13-endurance-") as raw:
        scratch = Path(raw)
        wal = _stress_wal_and_worktrees(ROOT, scratch)
        campaign = _run_campaign(ROOT, scratch, dataset)
    report = {"schema_version": "aegis.endurance.v13", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "repository_head": _git(ROOT, "rev-parse", "HEAD"), "wal_stress": wal, "campaign": campaign,
              "grpo_pairs": {"path": str(dataset),
                             "records": sum(1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()),
                             "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest() if dataset.exists() else None}}
    report_json = destination / "report.json"
    stdout_log = destination / "stdout.txt"
    report["report_json"] = str(report_json)
    report["stdout_log"] = str(stdout_log)
    output = _format_report(report)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    stdout_log.write_text(output, encoding="utf-8")
    print(output, end="")
    if (wal["successes"] != WORKERS or wal["database_locked"] or wal["remaining_worktrees"]
            or wal["temporary_paths_before_parent_cleanup"] or not wal["main_tree_clean"]
            or wal["integrity_check"] != "ok" or campaign["work_orders"] != TASKS
            or not campaign["all_attacks_blocked"] or not campaign["main_tree_clean"]
            or campaign["remaining_worktrees"] or campaign["successful_repairs"] != TASKS - 1
            or campaign["budget_fuse_rejections"] != 1):
        raise RuntimeError("Phase 13 endurance acceptance criteria failed; inspect report.json")
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run_endurance(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
