"""EnvDoctor runs against real temporary directories and child test processes."""
from collections import namedtuple
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aegis_core.env_doctor import EnvDoctor


Usage = namedtuple("Usage", "total used free")


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("free,threshold,expected", [
    (0, 1, True), (1, 1, False), (2, 1, False), (99, 100, True),
    (100, 100, False), (101, 100, False), (5, 10, True), (10, 10, False),
    (11, 10, False), (2**30, 2**31, True),
])
def test_disk_threshold_uses_remaining_bytes_and_exact_boundary(free, threshold, expected, tmp_path):
    doctor = EnvDoctor(tmp_path, min_free_bytes=threshold,
                       usage_provider=lambda _path: Usage(1000, 1000 - free, free), cleanup=False)
    report = doctor.check()
    assert report["disk"]["low_space"] is expected
    assert report["disk"]["free_bytes"] == free


def test_low_disk_cleanup_removes_only_named_caches_and_proven_orphan_worktrees(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / ".pytest_cache" / "v" / "cache").mkdir(parents=True)
    (tmp_path / ".pytest_cache" / "v" / "cache" / "nodeids").write_text("[]", encoding="utf-8")
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / "src" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    orphan = tmp_path / ".worktrees" / "stale"
    orphan.mkdir(parents=True)
    (orphan / ".git").write_text(f"gitdir: {tmp_path / '.git' / 'worktrees' / 'stale'}\n", encoding="utf-8")
    (tmp_path / ".worktrees" / "unverified").mkdir()
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "model.json").write_text("keep", encoding="utf-8")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "release.jsonl").write_text("keep", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bootstrap_local.ps1").write_text("keep", encoding="utf-8")
    usage = lambda _path: Usage(100, 1, 0)
    report = EnvDoctor(tmp_path, min_free_bytes=1, usage_provider=usage).check(changed_files=[])
    assert report["disk"]["cleaned"] == [
        {"path": ".pytest_cache", "kind": "cache"},
        {"path": "src/__pycache__", "kind": "cache"},
        {"path": ".worktrees/stale", "kind": "orphan_worktree"},
    ]
    assert not (tmp_path / ".pytest_cache").exists()
    assert not (tmp_path / "src" / "__pycache__").exists()
    assert not orphan.exists()
    assert (tmp_path / ".worktrees" / "unverified").is_dir()
    assert (tmp_path / "training" / "model.json").read_text() == "keep"
    assert (tmp_path / "datasets" / "release.jsonl").read_text() == "keep"
    assert (tmp_path / "scripts" / "bootstrap_local.ps1").read_text() == "keep"


def test_healthy_disk_does_not_remove_existing_caches(tmp_path):
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    file = cache / "keep.pyc"
    file.write_bytes(b"keep")
    doctor = EnvDoctor(tmp_path, min_free_bytes=10, usage_provider=lambda _p: Usage(100, 0, 100))
    report = doctor.check()
    assert report["disk"]["low_space"] is False
    assert file.read_bytes() == b"keep"
    assert report["disk"]["cleaned"] == []


def test_no_cleanup_option_reports_low_space_without_mutating_cache(tmp_path):
    cache = tmp_path / ".ruff_cache"
    cache.mkdir()
    sentinel = cache / "state"
    sentinel.write_text("keep", encoding="utf-8")
    report = EnvDoctor(tmp_path, min_free_bytes=2, usage_provider=lambda _p: Usage(10, 10, 0),
                       cleanup=False).check()
    assert report["disk"]["low_space"] is True
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("template,destination", [
    (".env.example", ".env"), ("config.json.template", "config.json"),
    ("settings.example.toml", "settings.toml"), ("local.sample", "local"),
    ("secrets.example.json", "secrets.json"), ("app.template.yaml", "app.yaml"),
])
def test_missing_configuration_templates_are_materialized_outside_project(template, destination, tmp_path):
    template_path = tmp_path / template
    template_path.write_text("mode = 'example'\n", encoding="utf-8")
    doctor = EnvDoctor(tmp_path, min_free_bytes=0, usage_provider=lambda _p: Usage(10, 1, 9))
    report = doctor.check()
    assert len(report["temporary_configs"]) == 1
    entry = report["temporary_configs"][0]
    assert entry["template"] == template
    assert entry["expected_path"] == destination
    assert entry["isolated"] is True
    temporary = Path(entry["temporary_path"])
    assert temporary.is_file()
    assert temporary.parent != tmp_path
    assert temporary.read_text(encoding="utf-8") == template_path.read_text(encoding="utf-8")
    assert not (tmp_path / destination).exists()
    doctor.cleanup_temporary_configs()
    assert not temporary.exists()


def test_existing_local_configuration_is_never_replaced_by_example(tmp_path):
    (tmp_path / ".env.example").write_text("SAFE=example\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SAFE=local\n", encoding="utf-8")
    report = EnvDoctor(tmp_path, min_free_bytes=0).check()
    assert report["temporary_configs"] == []
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "SAFE=local\n"


@pytest.mark.parametrize("package", [
    "missing-aegis-package-one", "missing-aegis-package-two", "missing-aegis-package-three",
    "missing-aegis-package-four", "missing-aegis-package-five", "missing-aegis-package-six",
])
def test_missing_python_dependencies_are_reported_from_project_metadata(package, tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\nname='probe'\nversion='0.1'\nrequires-python='>=3.11'\ndependencies=['{package}>=1']\n",
        encoding="utf-8")
    report = EnvDoctor(tmp_path, min_free_bytes=0).check()
    assert {item["name"] for item in report["missing_dependencies"]} == {package}
    assert report["sparse_mode"]["enabled"] is True
    assert "missing_dependencies_or_runtime" in report["sparse_mode"]["reasons"]


def test_python_and_node_runtime_versions_are_reported_without_assuming_node_exists(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"engines": {"node": ">=18"}, "scripts": {"test": "node --test"}}),
                                             encoding="utf-8")
    report = EnvDoctor(tmp_path, min_free_bytes=0).check()
    assert report["python"]["version"].startswith("3.")
    assert report["python"]["executable"] == sys.executable
    assert report["node"]["expected"] == ">=18"
    assert "version" in report["node"] and "missing" in report["node"]


def test_sparse_plan_runs_only_import_related_python_tests_and_marks_boundary(tmp_path):
    (tmp_path / "aegis_core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "aegis_core" / "env_doctor.py").write_text("VALUE = 1\n", encoding="utf-8")
    test = tmp_path / "tests" / "test_env_doctor.py"
    test.write_text("from aegis_core.env_doctor import VALUE\n\ndef test_value():\n    assert VALUE == 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_other.py").write_text("def test_other(): assert False\n", encoding="utf-8")
    doctor = EnvDoctor(tmp_path, min_free_bytes=0)
    plan = doctor.sparse_plan(["aegis_core/env_doctor.py"])
    assert len(plan["commands"]) == 1
    assert plan["commands"][0][-1] == "tests/test_env_doctor.py"
    assert plan["coverage_boundary"]["selected_test_files"] == ["tests/test_env_doctor.py"]
    assert plan["coverage_boundary"]["full_repository_suite_run"] is False


def test_sparse_gate_executes_a_real_targeted_test_process(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "module.py").write_text("VALUE = 6\n", encoding="utf-8")
    (tmp_path / "tests" / "test_probe.py").write_text(
        "from module import VALUE\n\ndef test_real_process():\n    assert VALUE == 6\n", encoding="utf-8")
    doctor = EnvDoctor(tmp_path, min_free_bytes=0)
    result = doctor.run_sparse_gate(["module.py"])
    assert result["status"] == "passed"
    assert result["results"][0]["exit_code"] == 0
    assert result["coverage_boundary"]["full_repository_suite_run"] is False


def test_sparse_gate_does_not_claim_success_when_no_affected_tests_exist(tmp_path):
    (tmp_path / "tests").mkdir()
    doctor = EnvDoctor(tmp_path, min_free_bytes=0)
    result = doctor.run_sparse_gate(["pkg/uncovered.py"])
    assert result["status"] == "failed"
    assert result["results"] == []
    assert result["coverage_boundary"]["untested_changed_files"] == ["pkg/uncovered.py"]


@pytest.mark.parametrize("mode", ["invalid", "full", "yes", "automatic", "disabled", "AUTO"])
def test_invalid_sparse_mode_values_are_rejected(mode, tmp_path):
    with pytest.raises(ValueError, match="sparse_mode"):
        EnvDoctor(tmp_path).check(sparse_mode=mode)


@pytest.mark.parametrize("threshold", [-1, 1.5, True, "100"])
def test_invalid_disk_thresholds_are_rejected(threshold, tmp_path):
    with pytest.raises(ValueError, match="min_free_bytes"):
        EnvDoctor(tmp_path, min_free_bytes=threshold)


@pytest.mark.parametrize("path", ["src/changed.py", "aegis_core/env_doctor.py", "tests/test_env_doctor.py"])
def test_sparse_coverage_report_always_lists_the_requested_changed_path(path, tmp_path):
    doctor = EnvDoctor(tmp_path, min_free_bytes=0)
    plan = doctor.sparse_plan([path])
    assert path in plan["coverage_boundary"]["changed_files"]
    assert plan["coverage_boundary"]["full_repository_suite_run"] is False


def test_orphan_worktree_without_a_valid_git_admin_target_is_preserved(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    unknown = tmp_path / ".worktrees" / "unknown"
    unknown.mkdir(parents=True)
    (unknown / ".git").write_text("gitdir: /tmp/unrelated/nonexistent\n", encoding="utf-8")
    report = EnvDoctor(tmp_path, min_free_bytes=2, usage_provider=lambda _p: Usage(10, 10, 0)).check()
    assert unknown.is_dir()
    assert not any(item["kind"] == "orphan_worktree" for item in report["disk"]["cleaned"])


def test_report_is_json_serializable_and_records_sparse_coverage_boundary(tmp_path):
    doctor = EnvDoctor(tmp_path, min_free_bytes=100,
                       usage_provider=lambda _p: Usage(100, 99, 1), cleanup=False)
    report = doctor.check(changed_files=["a.py"])
    encoded = json.dumps(report, sort_keys=True)
    assert '"schema_version": "aegis.env_doctor.v1"' in encoded
    assert report["status"] == "SPARSE_MODE"
    assert report["sparse_mode"]["coverage_boundary"]["full_repository_suite_run"] is False
