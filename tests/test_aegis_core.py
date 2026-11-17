import ast
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from aegis_core.ledger import Ledger
from aegis_core.slicer import extract_causal_slice
from aegis_core.sandbox import Sandbox, run_exam_gate, tree_hash
from aegis_core.interface_schema import validate_message, JSON_SCHEMA


def _new(ledger, ticket="t"):
    ledger.create_ticket(ticket, "wish", {"goal": "test"}, "def run() -> None: ...")


def _transition(ledger, token, ticket="t", reward=0):
    return ledger.transition(ticket, token, "queued", "running", state_s0={"status": "queued"},
                             action_a0={"op": "start"}, actual_state_s1={"status": "running"},
                             reward=reward, predicted_next_state={"status": "running"}, token_cost=7)


def _writer(args):
    path, index = args
    ledger = Ledger(path)
    ticket = f"t{index:03}"
    _new(ledger, ticket)
    token = ledger.acquire_lease(ticket, "worker")
    _transition(ledger, token, ticket)
    return ticket


def test_wal_multiprocess_writes_and_jsonl(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = Ledger(path)
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert len(list(pool.map(_writer, [(path, i) for i in range(24)]))) == 24
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("SELECT count(*) FROM tickets").fetchone()[0] == 24
    with ledger._connection() as db:
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 1
    destination = tmp_path / "traces.jsonl"
    assert ledger.export_jsonl(destination) == 48
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [(r["ticket_id"], r["step_idx"]) for r in rows] == sorted((r["ticket_id"], r["step_idx"]) for r in rows)
    assert all(r["state_s1"] == r["actual_state_s1"] for r in rows)
    assert rows[1]["token_cost"] == 7
    assert rows[1]["predicted_next_state"] == {"status": "running"}
    assert rows[1]["state_s0"] == rows[0]["state_s1"]
    assert ledger.export_jsonl(destination, "t000") == 2


def test_competing_lease_single_winner_and_fencing(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "db")
    _new(ledger)
    with ThreadPoolExecutor(max_workers=12) as pool:
        tokens = list(pool.map(lambda i: ledger.acquire_lease("t", f"w{i}", 10), range(24)))
    winners = [token for token in tokens if token]
    assert len(winners) == 1
    now = time.time()
    monkeypatch.setattr("aegis_core.ledger.time.time", lambda: now + 20)
    fresh = ledger.acquire_lease("t", "new")
    assert fresh and fresh != winners[0]
    with pytest.raises(PermissionError):
        _transition(ledger, winners[0])
    assert _transition(ledger, fresh) == 1


def test_ledger_atomic_rollback_and_append_only(tmp_path):
    ledger = Ledger(tmp_path / "db")
    _new(ledger)
    token = ledger.acquire_lease("t", "worker")
    with pytest.raises(ValueError):
        _transition(ledger, token, reward=float("nan"))
    assert ledger.get_ticket("t")["status"] == "queued"
    assert _transition(ledger, token) == 1
    with pytest.raises(ValueError):
        ledger.transition("t", token, "running", "succeeded", state_s0={"status": "running", "invented": 1},
                          action_a0={}, actual_state_s1={"status": "succeeded"}, reward=1)
    assert ledger.get_ticket("t")["status"] == "running"
    with ledger._connection() as db:
        for statement in ("UPDATE causal_traces SET reward=100", "DELETE FROM causal_traces"):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(statement)
    with pytest.raises(ValueError):
        ledger.export_jsonl(tmp_path / "db")


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan")])
def test_bad_lease_ttl(tmp_path, ttl):
    ledger = Ledger(tmp_path / "db")
    _new(ledger)
    with pytest.raises(ValueError):
        ledger.acquire_lease("t", "worker", ttl)


def test_slicer_aliases_methods_docstrings_and_compression(tmp_path):
    (tmp_path / "dep.py").write_text('''def helper(x: int, /, *, scale: int = 2) -> int:
    """Multiply x."""
    hidden = 'secret implementation'
    return x * scale

class Tool:
    """A tool interface."""
    def run(self, x: int) -> int:
        """Run tool."""
        return x + 999
''' + '\n# irrelevant padding\n' * 200)
    (tmp_path / "main.py").write_text('''from dep import helper as h
import dep as d
from dep import Tool

def target(x: int) -> int:
    """Preserve complete body."""
    return h(x) + d.helper(x) + Tool().run(x)

def local():
    return target(2)
''')
    (tmp_path / "caller.py").write_text('''from main import target as invoke
import main as m
async def upstream():
    return invoke(3) + m.target(4)
''')
    result = extract_causal_slice(tmp_path, "main.py", "target")
    assert result == extract_causal_slice(tmp_path, "main.py", "target")
    assert "return h(x)" in result["target"]["source"]
    dependencies = {item["symbol"]: item["stub"] for item in result["downstream"]}
    assert set(dependencies) == {"helper", "Tool", "Tool.run"}
    signature = ast.parse(dependencies['helper']).body[0]
    assert signature.args.kwonlyargs[0].arg == 'scale'
    assert ast.unparse(signature.args.kwonlyargs[0].annotation) == 'int'
    assert signature.args.kw_defaults[0].value == 2
    assert 'Multiply x.' in dependencies["helper"]
    assert 'secret implementation' not in dependencies["helper"]
    assert '999' not in dependencies["Tool"]
    for stub in dependencies.values():
        ast.parse(stub)
    assert {(item["file"], item["caller"], item["call"]) for item in result["upstream"]} == {
        ("main.py", "local", "target(2)"), ("caller.py", "upstream", "invoke(3)"),
        ("caller.py", "upstream", "m.target(4)")}
    assert result["compression_ratio"] > 0.8
    assert result["unresolved_calls"] == []


def test_slicer_decorators_async_dynamic_and_path_escape(tmp_path):
    (tmp_path / "sample.py").write_text('''def decorate(f):
    return f

@decorate
async def target(callback):
    return await callback()
''')
    result = extract_causal_slice(tmp_path, "sample.py", "target")
    assert result["target"]["source"].startswith("@decorate\nasync def")
    assert result["unresolved_calls"] == ["callback"]
    with pytest.raises(ValueError):
        extract_causal_slice(tmp_path, "../outside.py", "target")
    with pytest.raises(KeyError):
        extract_causal_slice(tmp_path, "sample.py", "absent")


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Aegis Tests")
    _git(path, "config", "user.email", "tests@localhost")
    (path / ".gitignore").write_text(".worktrees/\n__pycache__/\n.pytest_cache/\n")
    (path / "value.py").write_text("VALUE = 0\n")
    (path / "test_value.py").write_text("from value import VALUE\ndef test_value():\n    assert VALUE == 1\n")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "baseline")
    return path


_PATCH = '''diff --git a/value.py b/value.py
--- a/value.py
+++ b/value.py
@@ -1 +1 @@
-VALUE = 0
+VALUE = 1
'''


def test_sandbox_failure_success_merge_and_causal_export(repo, tmp_path):
    sandbox = Sandbox(repo)
    path = sandbox.create("ticket-1")
    assert path == repo / ".worktrees" / "ticket-1"
    ledger = Ledger(tmp_path / "ledger.db")
    _new(ledger)
    token = ledger.acquire_lease("t", "worker")
    _transition(ledger, token)
    failed = sandbox.exam("ticket-1", [sys.executable, "-m", "pytest", "-v"])
    assert failed["exit_code"] == 1 and failed["reward"] == -1
    assert "AssertionError" in failed["stdout"]
    with pytest.raises(ValueError):
        sandbox.merge("ticket-1", "must fail")
    sandbox.apply_patch("ticket-1", _PATCH)
    passed = sandbox.exam("ticket-1", [sys.executable, "-B", "-m", "pytest", "-v"])
    assert passed["passed"] and passed["reward"] == 1
    assert not passed["changed"]
    ledger.transition("t", token, "running", "succeeded", state_s0={"status": "running"},
                      action_a0={"op": "exam_gate", "prior_failure": failed},
                      predicted_next_state={"status": "succeeded"},
                      actual_state_s1={"status": "succeeded", "gate": passed}, reward=passed["reward"])
    assert ledger.acquire_lease("t", "other") is None
    destination = tmp_path / "training.jsonl"
    assert ledger.export_jsonl(destination) == 3
    assert json.loads(destination.read_text().splitlines()[-1])["reward"] == 1
    sha = sandbox.merge("ticket-1", "implement value")
    assert _git(repo, "rev-parse", "HEAD") == sha
    assert (repo / "value.py").read_text() == "VALUE = 1\n"
    sandbox.cleanup("ticket-1")
    assert not path.exists()
    assert "aegis/ticket-1" not in _git(repo, "branch", "--list")


def test_atomic_patch_failure_cleanup_and_gate_invalidation(repo):
    sandbox = Sandbox(repo)
    path = sandbox.create("atomic")
    before = tree_hash(path)
    bad_patch = _PATCH + '''diff --git a/missing.py b/missing.py
--- a/missing.py
+++ b/missing.py
@@ -1 +1 @@
-no
+yes
'''
    with pytest.raises(subprocess.CalledProcessError):
        sandbox.apply_patch("atomic", bad_patch)
    assert tree_hash(path) == before
    sandbox.apply_patch("atomic", _PATCH)
    assert sandbox.exam("atomic", [sys.executable, "-B", "-m", "pytest"])["passed"]
    (path / "value.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError):
        sandbox.merge("atomic", "stale gate")
    sandbox.cleanup("atomic")
    assert (repo / "value.py").read_text() == "VALUE = 0\n"


@pytest.mark.parametrize("ticket", ["../escape", "x/y", "-bad", "x;echo pwn", ""])
def test_sandbox_rejects_invalid_ids(repo, ticket):
    with pytest.raises(ValueError):
        Sandbox(repo).create(ticket)


def test_gate_timeout_missing_executable_and_mutation(tmp_path):
    assert run_exam_gate(tmp_path, ["aegis-nonexistent-command"])["exit_code"] == 127
    result = run_exam_gate(tmp_path, [sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.05)
    assert result["exit_code"] == 124 and result["reward"] == -1
    result = run_exam_gate(tmp_path, [sys.executable, "-c", "from pathlib import Path; Path('new').write_text('x')"])
    assert result["changed"] and result["exit_code"] == 0 and result["reward"] == -1


def test_tree_hash_tracks_names_content_and_symlinks(tmp_path):
    (tmp_path / "f").write_bytes(b"data")
    before = tree_hash(tmp_path)
    (tmp_path / "f").rename(tmp_path / "g")
    assert tree_hash(tmp_path) != before
    (tmp_path / "g").rename(tmp_path / "f")
    assert tree_hash(tmp_path) == before
    try:
        (tmp_path / "link").symlink_to("f")
    except OSError:
        if os.name != "nt":
            raise
    else:
        assert tree_hash(tmp_path) != before


def _plan():
    return {"kind": "CausalRolloutPlan", "ticket_id": "parent", "initial_state": {"x": 0},
            "steps": [{"step_idx": 0, "state_s0": {"x": 0}, "action_a0": {"op": "inc"},
                       "predicted_next_state": {"x": 1}}],
            "sub_tickets": [{"ticket_id": "a", "intent": "first", "contract_stub": "def a(): ...", "depends_on": []},
                            {"ticket_id": "b", "intent": "second", "contract_stub": "def b(): ...", "depends_on": ["a"]}]}


def test_schema_accepts_all_primitives_and_serializes_schema():
    plan = _plan()
    assert validate_message(plan) == plan
    assert validate_message(plan) is not plan
    question = {"kind": "MissingConstraintQuestion", "ticket_id": "t", "question": "Which target?",
                "missing_constraints": ["target"], "options": ["Linux", "Windows"]}
    assert validate_message(question) == question
    fault = {"kind": "FaultAttribution", "ticket_id": "t", "step_idx": 0,
             "predicted_next_state": {}, "actual_state_s1": {}, "cause": "assertion failed",
             "evidence": ["exit code 1"], "confidence": 0.8, "repair_action": {"op": "patch"}}
    assert validate_message(fault) == fault
    assert json.loads(json.dumps(JSON_SCHEMA))["oneOf"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(extra=1),
    lambda p: p.pop("initial_state"),
    lambda p: p["steps"][0].update(step_idx=True),
    lambda p: p["steps"][0].update(state_s0={"x": 99}),
    lambda p: p["initial_state"].update(x=float("nan")),
    lambda p: p["sub_tickets"][0].update(depends_on=["b"]),
    lambda p: p["sub_tickets"][0].update(depends_on=["missing"]),
    lambda p: p["sub_tickets"][1].update(ticket_id="a"),
    lambda p: p["initial_state"].update(bad=(1, 2)),
])
def test_schema_rejects_invalid_protocol_and_dag(mutation):
    plan = _plan()
    mutation(plan)
    with pytest.raises(ValueError):
        validate_message(plan)


def test_slicer_does_not_invent_edges_for_shadowed_symbols(tmp_path):
    (tmp_path / "scope.py").write_text('''def helper():
    return 1

def target(helper):
    return helper()

def caller(target):
    return target(2)
''')
    result = extract_causal_slice(tmp_path, "scope.py", "target")
    assert result["downstream"] == []
    assert result["upstream"] == []
    assert result["unresolved_calls"] == ["helper"]


def test_failed_gate_records_negative_training_reward(repo, tmp_path):
    ledger = Ledger(tmp_path / "db")
    _new(ledger)
    token = ledger.acquire_lease("t", "worker")
    _transition(ledger, token)
    sandbox = Sandbox(repo)
    path = sandbox.create("negative")
    gate = sandbox.exam("negative", [sys.executable, "-m", "pytest"])
    assert gate["reward"] == -1
    ledger.transition("t", token, "running", "failed", state_s0={"status": "running"},
                      action_a0={"op": "exam_gate", "before_hash": tree_hash(path)},
                      actual_state_s1={"status": "failed", "feedback": gate},
                      predicted_next_state={"status": "succeeded"}, reward=gate["reward"])
    exported = tmp_path / "negative.jsonl"
    ledger.export_jsonl(exported)
    trace = json.loads(exported.read_text().splitlines()[-1])
    assert trace["reward"] == -1
    assert trace["predicted_next_state"] != trace["actual_state_s1"]
    sandbox.cleanup("negative")


def test_parent_advance_blocks_merge(repo):
    sandbox = Sandbox(repo)
    sandbox.create("advance")
    assert sandbox.exam("advance", [sys.executable, "-c", "pass"])["passed"]
    (repo / "new_file").write_text("parent advanced")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "advance parent")
    with pytest.raises(ValueError, match="parent advanced"):
        sandbox.merge("advance", "must be re-examined")
    sandbox.cleanup("advance")


def test_default_gate_command_and_unignored_worktree_parent(repo, monkeypatch):
    # Lifecycle management also works in repositories without a worktree ignore rule.
    (repo / '.gitignore').write_text('__pycache__/\n.pytest_cache/\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'remove worktree ignore')
    sandbox = Sandbox(repo)
    sandbox.create('default')
    sandbox.apply_patch('default', _PATCH)
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    assert sandbox.exam('default')['passed']
    sandbox.merge('default', 'verified default pytest')
    sandbox.cleanup('default')


def test_slicer_relative_module_import(tmp_path):
    (tmp_path / 'dep.py').write_text('def run():\n    return 42\n')
    (tmp_path / 'main.py').write_text('from . import dep as d\ndef target():\n    return d.run()\n')
    result = extract_causal_slice(tmp_path, 'main.py', 'target')
    assert [item['symbol'] for item in result['downstream']] == ['run']
    assert result['unresolved_calls'] == []
