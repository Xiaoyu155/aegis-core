"""Explicit JSON config scopes stay isolated and get a real syntax gate."""
import difflib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aegis_core.reviewer import audit_candidate
from aegis_core.tournament import Candidate, CandidateTournament
from aegis_core.self_heal_loop import run_autopilot


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _repo(path):
    path.mkdir()
    (path / ".gitignore").write_text(".worktrees/\n__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    (path / "app.py").write_text("def compute(value):\n    return value\n", encoding="utf-8")
    (path / "package.json").write_text('{"scripts":{"check":"node --version"}}\n', encoding="utf-8")
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Config Scope Tests")
    _git(path, "config", "user.email", "scope@example.invalid")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "baseline")
    return path


def _multi_patch(root, replacements):
    chunks = []
    for filename, replacement in replacements.items():
        before = (root / filename).read_text(encoding="utf-8").splitlines(keepends=True)
        after = replacement.splitlines(keepends=True)
        chunks.append(f"diff --git a/{filename} b/{filename}\n")
        chunks.extend(difflib.unified_diff(before, after,
                                           fromfile=f"a/{filename}", tofile=f"b/{filename}"))
    return "".join(chunks)


def _tournament(root, tmp_path, allowed_paths=None):
    return CandidateTournament(
        root,
        dynamic_commands=[[sys.executable, "-c", "from app import compute; assert compute(-1) == 0"]],
        mutation_commands=[[sys.executable, "-c", "from app import compute; assert compute(11) == 10"]],
        static_commands=[], timeout_sec=5, max_workers=1,
        dataset_path=tmp_path / "pairs.jsonl", allowed_paths=allowed_paths,
    )


def test_explicit_config_allowlist_runs_json_syntax_gate_and_candidate_probes(tmp_path):
    root = _repo(tmp_path / "repo")
    package = json.loads((root / "package.json").read_text())
    package["phase15"] = {"validated": True}
    candidate = Candidate("scoped-config", "clamp and update allowed config", _multi_patch(root, {
        "app.py": "def compute(value):\n    if type(value) is not int:\n        return None\n    return min(10, max(0, value))\n",
        "package.json": json.dumps(package, separators=(",", ":")) + "\n",
    }))

    result = _tournament(root, tmp_path, allowed_paths=["app.py", "package.json"]).run([candidate])

    evaluation = result["evaluations"][0]
    assert evaluation["accepted"] is True
    assert any("json.tool" in command["command"] for command in evaluation["static"])
    assert evaluation["dynamic_pass_rate"] == 1
    assert evaluation["mutation_probe_survival_rate"] == 1
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_json_config_patch_is_rejected_without_an_explicit_scope(tmp_path):
    root = _repo(tmp_path / "repo")
    package = json.loads((root / "package.json").read_text())
    package["unexpected"] = True
    candidate = Candidate("unscoped-config", "change config", _multi_patch(root, {
        "app.py": "def compute(value):\n    if type(value) is not int:\n        return None\n    return min(10, max(0, value))\n",
        "package.json": json.dumps(package) + "\n",
    }))

    result = _tournament(root, tmp_path).run([candidate])

    evaluation = result["evaluations"][0]
    assert evaluation["accepted"] is False
    assert "package.json" in evaluation["message"]
    assert "dynamic" not in evaluation


def test_l3_json_reviewer_rejects_malformed_allowlisted_configuration(tmp_path):
    root = _repo(tmp_path / "repo")
    baseline = _git(root, "rev-parse", "HEAD")
    (root / "package.json").write_text('{"scripts": ', encoding="utf-8")

    result = audit_candidate(root, baseline, ["package.json"])

    assert result["passed"] is False
    assert result["audited_files"] == ["package.json"]
    assert result["findings"][0]["code"] == "AUDIT_SOURCE_UNREADABLE"
    assert result["findings"][0]["path"] == "package.json"


@pytest.mark.parametrize("path", ["../package.json", "/tmp/package.json", ""])
def test_allowlist_rejects_unsafe_paths_before_candidate_execution(tmp_path, path):
    root = _repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="safe repository-relative"):
        _tournament(root, tmp_path, allowed_paths=[path])


def test_autopilot_forwards_explicit_config_scope_to_worktree_tournament(tmp_path):
    root = _repo(tmp_path / "repo")
    package = json.loads((root / "package.json").read_text())
    package["phase15"] = True
    patch = _multi_patch(root, {
        "app.py": "def compute(value):\n    if type(value) is not int:\n        return None\n    return min(10, max(0, value))\n",
        "package.json": json.dumps(package) + "\n",
    })
    result = run_autopilot(
        root,
        contract="Clamp integer values from zero through ten and mark the package config.",
        probe_spec={"kind": "numeric_bounds", "path": "app.py", "symbol": "compute",
                    "min": 0, "max": 10, "fallback": None, "seed": 15},
        candidate_rounds=[[{"candidate_id": "config-scope", "strategy": "bounded clamp", "patch": patch}]],
        allowed_paths=["app.py", "package.json"],
        database_path=tmp_path / "autopilot.sqlite", dataset_path=tmp_path / "pairs.jsonl",
        timeout_sec=5, max_rounds=1, max_context_tokens=512, max_total_tokens=2000,
        max_wall_clock_sec=60, ticket_id="phase15-config-scope",
    )

    assert result["status"] == "succeeded"
    evaluation = result["rounds"][0]["tournament"]["evaluations"][0]
    assert evaluation["accepted"] is True
    assert any("json.tool" in gate["command"] for gate in evaluation["static"])
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "worktree", "list", "--porcelain").count("worktree ") == 1
