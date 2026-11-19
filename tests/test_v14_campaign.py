"""Physical campaign workorders, Aegis batch gates and autopilot replay cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_21repos_mega_campaign import (
    _aegis_env, _batch_command, _code_files, _patch_for_comment, _select_target,
    _autopilot_ticket, _candidate_metrics, re_slug,
)
from aegis_core.probe_synth import compile_structured_probe, is_true_red


ROOT = Path(__file__).resolve().parents[1]


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout.decode().strip()


def _repo(root, files):
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Campaign Tests")
    _git(root, "config", "user.email", "campaign-test@example.invalid")
    for name, source in files.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source if isinstance(source, bytes) else source.encode("utf-8"))
    _git(root, "add", "."); _git(root, "commit", "-qm", "business source baseline")
    return root


@pytest.mark.parametrize("repo_name", ["my project", "pet/mini app", "仓库-v14", "A__B", "repo@name"])
def test_ticket_slug_is_safe_and_bounded(repo_name):
    slug = re_slug(repo_name)
    assert slug and len(slug) <= 48 and all(ch.isalnum() or ch in "_.-" for ch in slug)


def test_candidate_campaign_metrics_use_only_physically_completed_scenarios():
    rows = [{"status": "succeeded", "candidate_rejections": 3},
            {"status": "command_failed", "candidate_rejections": 0}]
    assert _candidate_metrics(rows) == {"succeeded_scenarios": 1, "evaluations": 4,
        "rejected": 3, "intentionally_invalid": 3, "interception_rate": 1.0}


def test_code_inventory_uses_tracked_source_files_and_skips_vendor_and_tests(tmp_path):
    repo = _repo(tmp_path / "repo", {"src/core.py": "def run(): return 1\n",
        "tests/test_core.py": "def test_run(): pass\n", "node_modules/pkg/index.js": "function x(){}\n",
        "dist/out.js": "function y(){}\n", "docs/readme.md": "docs\n", "src/app.ts": "export const ok = true;\n"})
    rows = _code_files(repo)
    assert [item["path"] for item in rows] == ["src/app.ts", "src/core.py"]
    assert all(item["raw"] and item["encoding"] for item in rows)


def test_target_selection_prefers_existing_application_source_over_generated_vendor(tmp_path):
    rows = [{"path": "vendor/source.py", "suffix": ".py", "bytes": 1000},
            {"path": "src/server.ts", "suffix": ".ts", "bytes": 200},
            {"path": "src/service.py", "suffix": ".py", "bytes": 80}]
    assert _select_target(rows)["path"] == "src/service.py"


def test_target_selection_ignores_tests_and_non_tool_scripts():
    rows = [{"path": "scripts/ui-deep-e2e.py", "suffix": ".py", "bytes": 800},
            {"path": "tests/test_app.py", "suffix": ".py", "bytes": 400},
            {"path": "app/index.ts", "suffix": ".ts", "bytes": 120}]
    assert _select_target(rows)["path"] == "app/index.ts"


def test_target_selection_refuses_repositories_without_gate_eligible_business_source():
    with pytest.raises(ValueError, match="eligible for Aegis"):
        _select_target([{"path": "scripts/audit.py", "suffix": ".py", "bytes": 800},
                        {"path": "test_compare.py", "suffix": ".py", "bytes": 100}])


@pytest.mark.parametrize("filename,source", [
    ("src/app.py", "def f(): return 1\n"), ("src/app.js", "export function f(){ return 1; }\n"),
    ("src/app.ts", "export function f(): number { return 1; }\n"),
    ("src/index.html", "<!doctype html><body>ok</body>\n"),
    ("src/query.sql", "SELECT 1;\n"), ("scripts/check.sh", "#!/bin/sh\nexit 0\n"),
    ("scripts/check.ps1", "Write-Output 'ok'\n"),
])
def test_language_specific_campaign_patch_applies_in_detached_worktree(filename, source, tmp_path):
    repo = _repo(tmp_path / "repo", {filename: source})
    target = next(item for item in _code_files(repo) if item["path"] == filename)
    patch = _patch_for_comment(target, "AEGIS_V14_TEST_MARKER")
    worktree = tmp_path / "candidate"
    _git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
    try:
        subprocess.run(["git", "apply", "--check", "-"], cwd=worktree, input=patch,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["git", "apply", "-"], cwd=worktree, input=patch,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        assert "AEGIS_V14_TEST_MARKER" in (worktree / filename).read_text(encoding="utf-8")
        assert not _git(repo, "diff", "--name-only", "HEAD")
    finally:
        _git(repo, "worktree", "remove", "--force", str(worktree))
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_campaign_patch_preserves_crlf_and_adds_exactly_one_line(tmp_path):
    repo = _repo(tmp_path / "repo", {"src/app.py": b"def f():\r\n    return 1\r\n"})
    target = _code_files(repo)[0]
    patch = _patch_for_comment(target, "AEGIS_V14_CRLF")
    assert b"\r\n" in patch
    worktree = tmp_path / "candidate"
    _git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
    try:
        subprocess.run(["git", "apply", "-"], cwd=worktree, input=patch,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        assert (worktree / "src/app.py").read_bytes().endswith(b"# AEGIS_V14_CRLF\r\n")
    finally:
        _git(repo, "worktree", "remove", "--force", str(worktree))


@pytest.mark.parametrize("suffix", ["py", "ts"])
def test_real_aegis_batch_compiles_red_contract_and_applies_source_patch(suffix, tmp_path):
    filename, source = (("src/app.py", "def value(): return 1\n") if suffix == "py" else
                        ("src/app.ts", "export const value = (): number => 1;\n"))
    repo = _repo(tmp_path / "repo", {filename: source})
    attach = subprocess.run([sys.executable, "-m", "aegis_core.cli", "attach", "--repo", str(repo)],
                            cwd=repo, env=_aegis_env(), capture_output=True, text=True, timeout=120)
    assert attach.returncode == 0, attach.stderr
    target = next(item for item in _code_files(repo) if item["path"] == filename)
    marker = f"AEGIS_V14_BATCH_{suffix.upper()}"
    patch_path = tmp_path / "candidate.patch"
    patch_path.write_bytes(_patch_for_comment(target, marker))
    ticket = f"batch-{suffix}"
    record = _batch_command(repo, ticket, target, patch_path, marker,
                            repo / ".aegis/repo_profile.json", tmp_path / "ledger.sqlite", tmp_path)
    assert record["exit_code"] == 0
    assert record["status"] == "succeeded"
    assert _git(repo, "rev-parse", "HEAD") == _git(repo, "rev-parse", "HEAD")
    assert not _git(repo, "diff", "--name-only", "HEAD")


def test_real_autopilot_rejects_mutants_then_replays_winning_patch(tmp_path):
    repo = _repo(tmp_path / "repo", {"README.md": "temporary campaign root\n"})
    record = _autopilot_ticket("Xiaoyu155/campaign-physical", repo, tmp_path / "ledger.sqlite",
                               tmp_path / "pairs.jsonl", tmp_path / "reports", timeout=180)
    assert record["status"] == "succeeded"
    assert record["winner"] == "conservative_repair"
    assert record["candidate_rejections"] == 3
    assert record["rounds_attempted"] == 2
    assert record["replay_status"] == "replayed"
    assert record["replay_probe_exit_code"] == 0
    assert record["default_head_unchanged"] and record["tracked_main_worktree_clean"]
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_compiled_acceptance_probe_isolated_from_unavailable_repository_conftest(tmp_path):
    repo = _repo(tmp_path / "repo", {"service.py": "def bounded(value):\n    return value\n",
                                      "conftest.py": "raise ImportError('optional project dependencies missing')\n",
                                      "pyproject.toml": "[tool.pytest.ini_options]\naddopts = ['--dist=load']\n"})
    result = compile_structured_probe({"kind": "numeric_bounds", "path": "service.py",
        "symbol": "bounded", "min": 0, "max": 10, "seed": 401}, repo, _git(repo, "rev-parse", "HEAD"))
    assert "--noconftest" in result["command"]
    assert result["command"][result["command"].index("-o") + 1] == "addopts="
    assert is_true_red(result["red"])
    assert "AEGIS_ACCEPTANCE" in result["red"]["stdout"] + result["red"]["stderr"]
