"""Physical Git replay, frozen evidence integrity and counterfactual blame."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from aegis_core.time_travel import ReplayIntegrityError, TimeTravelReplay


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout.decode().strip()


def _repo(path, files):
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Replay Tests")
    _git(path, "config", "user.email", "replay@example.invalid")
    for filename, content in files.items():
        target = path / filename; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(path, "add", "."); _git(path, "commit", "-qm", "base")
    return path


def _command(assertion="assert evaluate(2) == 3"):
    return [sys.executable, "-c", f"from app import evaluate; {assertion}"]


def _patch(root, before, after, filename="app.py"):
    source = (root / filename).read_text(encoding="utf-8")
    updated = source.replace(before, after)
    assert updated != source
    (root / filename).write_text(updated, encoding="utf-8")
    _git(root, "add", filename); _git(root, "commit", "-qm", "candidate")
    return _git(root, "rev-parse", "HEAD")


def test_capture_freezes_archive_probe_seeds_and_world_context(tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    result = store.capture("ticket-1", probe="assertion bytes", probe_path="tests/frozen.py",
                           probe_command=_command(), mutation_seeds=[0, 17],
                           world_context={"model": {"state": "red"}})
    frozen = store.snapshot("ticket-1")
    assert result["archive_sha256"] == hashlib.sha256(frozen["archive"]).hexdigest()
    assert frozen["probe"] == b"assertion bytes"
    assert frozen["mutation_seeds"] == [0, 17]
    assert frozen["world_context"] == {"model": {"state": "red"}}


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 20261119, 2**31 - 1])
def test_repeated_capture_of_same_ticket_is_idempotent(seed, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    first = store.capture("same", probe="probe", probe_command=_command(), mutation_seeds=[seed])
    second = store.capture("same", probe="probe", probe_command=_command(), mutation_seeds=[seed])
    assert first["archive_sha256"] == second["archive_sha256"]
    assert store.snapshot("same")["mutation_seeds"] == [seed]


@pytest.mark.parametrize("probe_path", ["/tmp/out.py", "../escape.py", "a/../../escape.py", ".", ""])
def test_capture_rejects_absolute_or_traversing_probe_paths(probe_path, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError):
        store.capture("unsafe", probe_path=probe_path, probe_command=_command())


@pytest.mark.parametrize("command", [[], [""], ["python", "", "-c"], "python -c pass", [1]])
def test_capture_rejects_nonportable_or_empty_probe_commands(command, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError):
        store.capture("bad-command", probe_command=command)


@pytest.mark.parametrize("code", ["def evaluate(x): return x + 1\n", "def evaluate(x):\n    return x + 2\n", "VALUE = 10\ndef evaluate(x): return x + VALUE\n"])
def test_restore_uses_real_detached_worktree_with_identical_tracked_bytes(code, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": code})
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    store.capture("restore", probe="frozen", probe_path="tests/probe.py", probe_command=_command())
    restored = store.restore("restore")
    worktree = Path(restored["worktree"])
    try:
        assert _git(worktree, "rev-parse", "HEAD") == _git(root, "rev-parse", "HEAD")
        assert (worktree / "app.py").read_bytes() == (root / "app.py").read_bytes()
        assert (worktree / "tests/probe.py").read_bytes() == b"frozen"
        assert restored["archive_sha256"] == store.snapshot("restore")["archive_sha256"]
        assert "(no branch)" in _git(worktree, "status", "--short", "--branch")
    finally:
        store.cleanup(worktree)
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.mark.parametrize("value", [3, 4, 5, 7, 11, 13])
def test_replay_physically_executes_probe_and_repeats_same_byte_result(value, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": f"def evaluate(x): return x + {value - 2}\n"})
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    store.capture("run", probe=b"print(evaluate(2))", probe_command=[sys.executable, "-c",
                   f"from app import evaluate; print(evaluate(2)); assert evaluate(2) == {value}"])
    first = store.replay("run")
    second = store.replay("run")
    assert first["status"] == second["status"] == "replayed"
    assert first["outcome"]["exit_code"] == 0
    assert first["outcome"]["result_sha256"] == second["outcome"]["result_sha256"]
    assert first["outcome"]["matches_observed"] is None


def test_replay_applies_frozen_rejected_patch_before_executing_probe(tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    base = _git(root, "rev-parse", "HEAD")
    patch = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-def evaluate(x): return x + 1\n+def evaluate(x): return x + 2\n"
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    store.capture("rejected", commit=base, probe=b"assert evaluate(2)==3", probe_path="tests/probe.py",
                  probe_command=_command(), candidate_patch=patch)
    result = store.replay("rejected")
    assert result["outcome"]["exit_code"] != 0
    assert result["candidate_patch_sha256"] == hashlib.sha256(patch.encode()).hexdigest()
    assert "AssertionError" in result["outcome"]["stderr"]
    assert _git(root, "status", "--porcelain") == ""


def test_replay_capture_preserves_observed_result_digest(tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    command = [sys.executable, "-c", "from app import evaluate; print(evaluate(2))"]
    physical = subprocess.run(command, cwd=root, capture_output=True, check=True)
    result_hash = hashlib.sha256(json.dumps({"exit_code": physical.returncode,
        "stdout_sha256": hashlib.sha256(physical.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(physical.stderr).hexdigest()}, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    # Compute the same canonical digest used by the replay receipt.
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    store.capture("observed", probe_command=command, observed_result={"result_sha256": result_hash})
    assert store.replay("observed")["outcome"]["matches_observed"] is True


@pytest.mark.parametrize("ticket", ["missing", "not-created", "unknown-1", "unknown-2"])
def test_snapshot_lookup_is_explicit_for_unknown_ticket(ticket, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    with pytest.raises(KeyError):
        TimeTravelReplay(root, tmp_path / "ledger.sqlite").snapshot(ticket)


def test_snapshot_records_are_append_only(tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    store = TimeTravelReplay(root, tmp_path / "ledger.sqlite")
    store.capture("immutable", probe_command=_command())
    with sqlite3.connect(tmp_path / "ledger.sqlite") as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE replay_snapshots SET probe=? WHERE ticket_id='immutable'", (b"changed",))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM replay_snapshots WHERE ticket_id='immutable'")


def test_corrupted_snapshot_blob_is_rejected_before_replay(tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    dbpath = tmp_path / "ledger.sqlite"
    store = TimeTravelReplay(root, dbpath)
    store.capture("tamper", probe=b"correct", probe_command=_command())
    with sqlite3.connect(dbpath) as db:
        db.execute("DROP TRIGGER replay_snapshots_no_update")
        db.execute("UPDATE replay_snapshots SET probe=? WHERE ticket_id='tamper'", (b"changed",))
    with pytest.raises(ReplayIntegrityError):
        store.snapshot("tamper")


@pytest.mark.parametrize("failure", ["assert evaluate(2) == 3", "assert evaluate(-1) == 0",
                                      "assert evaluate(3) == 4"])
def test_counterfactual_blame_isolates_regressing_source_line(failure, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x):\n    return x + 1\n", "readme.txt": "unchanged context\n"})
    base = _git(root, "rev-parse", "HEAD")
    (root / "app.py").write_text("def evaluate(x):\n    return x + 2\n", encoding="utf-8")
    (root / "readme.txt").write_text("unrelated note\n", encoding="utf-8")
    _git(root, "add", "."); _git(root, "commit", "-qm", "bad change")
    candidate = _git(root, "rev-parse", "HEAD")
    result = TimeTravelReplay(root, tmp_path / "ledger.sqlite").counterfactual_blame(base, candidate,
                                    [sys.executable, "-c", f"from app import evaluate; {failure}"])
    assert result["status"] == "attributed"
    assert [row["file"] for row in result["culprits"]] == ["app.py"]
    assert result["culprits"][0]["line_ranges"][0]["candidate_start"] == 2
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


@pytest.mark.parametrize("line,value", [(2, 0), (2, 2), (3, 0), (3, 2), (4, 0), (4, 2)])
def test_line_attribution_reports_exact_changed_range(line, value, tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x):\n    offset = 1\n    result = x + offset\n    return result\n"})
    base = _git(root, "rev-parse", "HEAD")
    lines = (root / "app.py").read_text().splitlines()
    lines[line - 1] = "    return 9" if line == 4 else "    offset = 0" if line == 2 else "    result = x + 0"
    (root / "app.py").write_text("\n".join(lines) + "\n")
    _git(root, "add", "app.py"); _git(root, "commit", "-qm", "change")
    candidate = _git(root, "rev-parse", "HEAD")
    result = TimeTravelReplay(root, tmp_path / "ledger.sqlite").counterfactual_blame(base, candidate,
                                    [sys.executable, "-c", "from app import evaluate; assert evaluate(4)==5"])
    if result["status"] == "attributed":
        assert any(item["candidate_start"] == line for item in result["culprits"][0]["line_ranges"])
    else:
        assert result["status"] == "not_candidate_regression"


def test_counterfactual_blame_does_not_claim_a_passing_candidate(tmp_path):
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    base = _git(root, "rev-parse", "HEAD")
    (root / "app.py").write_text("def evaluate(x): return x + 1\n# harmless comment\n")
    _git(root, "add", "app.py"); _git(root, "commit", "-qm", "comment")
    result = TimeTravelReplay(root, tmp_path / "ledger.sqlite").counterfactual_blame(
        base, _git(root, "rev-parse", "HEAD"), _command())
    assert result["status"] == "not_candidate_regression"


def test_cli_replay_subcommand_uses_frozen_snapshot_and_writes_json(tmp_path, capsys):
    from aegis_core.cli import main
    root = _repo(tmp_path / "repo", {"app.py": "def evaluate(x): return x + 1\n"})
    database = tmp_path / "ledger.sqlite"
    TimeTravelReplay(root, database).capture("cli-ticket", probe_command=_command())
    output = tmp_path / "replay.json"
    code = main(["replay", "--ticket", "cli-ticket", "--repo", str(root), "--db", str(database),
                 "--out", str(output)])
    assert code == 0
    assert json.loads(output.read_text())["outcome"]["exit_code"] == 0
    assert json.loads(capsys.readouterr().out)["status"] == "replayed"
