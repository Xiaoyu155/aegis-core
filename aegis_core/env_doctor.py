"""Preflight and bounded recovery for local repair environments."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib


_CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache"}
_IGNORED_SCAN_DIRS = {".git", ".venv", "venv", "node_modules", "training", "datasets"}


class EnvDoctor:
    """Inspect runtimes, free space, project requirements, and config templates.

    Cleanup is intentionally narrow: only named caches and provably stale
    Aegis worktrees below ``.worktrees`` are eligible, and only when free space
    is below the configured threshold.
    """

    def __init__(self, repo_root: str | Path, *, min_free_bytes: int = 2 * 1024**3,
                 usage_provider=None, cleanup: bool = True) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ValueError("repo_root must be an existing directory")
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError("min_free_bytes must be a nonnegative integer")
        self.min_free_bytes = min_free_bytes
        self._usage_provider = usage_provider or shutil.disk_usage
        self.cleanup_enabled = cleanup
        self._temporary_dirs: list[str] = []

    def cleanup_temporary_configs(self) -> None:
        for directory in self._temporary_dirs:
            shutil.rmtree(directory, ignore_errors=True)
        self._temporary_dirs.clear()

    def _python_requirements(self) -> list[dict]:
        path = self.repo_root / "pyproject.toml"
        declarations = []
        if path.is_file():
            try:
                project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
                declarations.extend(project.get("dependencies", []))
                optional = project.get("optional-dependencies", {})
                declarations.extend(optional.get("test", []))
            except (OSError, tomllib.TOMLDecodeError):
                return [{"name": "pyproject.toml", "status": "invalid_metadata"}]
        for filename in ("requirements.txt", "requirements-test.txt"):
            requirements = self.repo_root / filename
            if requirements.is_file():
                declarations.extend(line.strip() for line in requirements.read_text(encoding="utf-8").splitlines()
                                    if line.strip() and not line.lstrip().startswith(("#", "-")))
        seen = set()
        missing = []
        for declaration in declarations:
            name = re.split(r"[<>=!~;\[]", declaration, maxsplit=1)[0].strip()
            normalized = re.sub(r"[-.]", "_", name).lower()
            if not name or normalized in seen or normalized in {"pytest"} and not self._has_module("pytest"):
                if name:
                    seen.add(normalized)
                if name and not self._has_module(self._import_name(name)):
                    missing.append({"name": name, "import_name": self._import_name(name), "kind": "python"})
                continue
            seen.add(normalized)
            module = self._import_name(name)
            if not self._has_module(module):
                missing.append({"name": name, "import_name": module, "kind": "python"})
        return missing

    @staticmethod
    def _import_name(package: str) -> str:
        aliases = {"scikit-learn": "sklearn", "pillow": "PIL", "pyyaml": "yaml",
                   "beautifulsoup4": "bs4", "opencv-python": "cv2"}
        return aliases.get(package.lower(), re.split(r"[<>=!~;\[]", package, maxsplit=1)[0].replace("-", "_"))

    @staticmethod
    def _has_module(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    def _node_requirements(self) -> dict:
        package = self.repo_root / "package.json"
        if not package.is_file():
            return {"required": False, "version": self._node_version(), "missing": []}
        try:
            metadata = json.loads(package.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"required": True, "version": self._node_version(), "missing": ["invalid package.json"]}
        expected = metadata.get("engines", {}).get("node")
        installed = self._node_version()
        missing = sorted(set(metadata.get("dependencies", {})) | set(metadata.get("devDependencies", {})))
        modules_dir = self.repo_root / "node_modules"
        if modules_dir.is_dir():
            missing = [name for name in missing if not (modules_dir / name).exists()]
        return {"required": bool(missing or expected), "expected": expected, "version": installed,
                "missing": missing if modules_dir.is_dir() else missing}

    @staticmethod
    def _node_version() -> str | None:
        node = shutil.which("node")
        if not node:
            return None
        try:
            result = subprocess.run([node, "--version"], capture_output=True, text=True,
                                    timeout=3, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def _templates(self) -> list[Path]:
        matches = []
        for current, dirs, filenames in os.walk(self.repo_root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in _IGNORED_SCAN_DIRS and not (Path(current) / name).is_symlink()]
            for filename in filenames:
                lower = filename.lower()
                if (lower.endswith((".example", ".template", ".sample"))
                        or ".example." in lower or ".template." in lower):
                    matches.append(Path(current) / filename)
        return sorted(matches)

    @staticmethod
    def _destination_for_template(template: Path) -> Path:
        name = template.name
        name = re.sub(r"\.(?:example|template|sample)(?=\.|$)", "", name, flags=re.I)
        return template.with_name(name)

    def _materialize_temporary_configs(self) -> list[dict]:
        pairs = []
        for template in self._templates():
            destination = self._destination_for_template(template)
            if destination.exists():
                continue
            temporary_root = Path(tempfile.mkdtemp(prefix="aegis-env-doctor-"))
            self._temporary_dirs.append(str(temporary_root))
            temp_target = temporary_root / destination.name
            shutil.copyfile(template, temp_target)
            pairs.append({"template": template.relative_to(self.repo_root).as_posix(),
                          "expected_path": destination.relative_to(self.repo_root).as_posix(),
                          "temporary_path": str(temp_target), "isolated": True})
        return pairs

    def _registered_worktrees(self) -> set[Path]:
        if not (self.repo_root / ".git").exists():
            return set()
        result = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=self.repo_root,
                                capture_output=True, text=True, timeout=5, check=False)
        paths = set()
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("worktree "):
                    try:
                        paths.add(Path(line[9:]).resolve())
                    except OSError:
                        pass
        return paths

    def _cleanup_caches_and_orphans(self) -> list[dict]:
        removed = []
        for current, dirs, _files in os.walk(self.repo_root, topdown=True, followlinks=False):
            base = Path(current)
            retained = []
            for name in dirs:
                target = base / name
                if name in _IGNORED_SCAN_DIRS:
                    continue
                if name in _CACHE_NAMES and not target.is_symlink():
                    shutil.rmtree(target, ignore_errors=True)
                    removed.append({"path": target.relative_to(self.repo_root).as_posix(), "kind": "cache"})
                    continue
                if target.is_symlink():
                    continue
                retained.append(name)
            dirs[:] = retained
        worktrees = self.repo_root / ".worktrees"
        if worktrees.is_dir() and not worktrees.is_symlink():
            registered = self._registered_worktrees()
            admin_dir = (self.repo_root / ".git" / "worktrees").resolve()
            for candidate in worktrees.iterdir():
                if not candidate.is_dir() or candidate.is_symlink() or candidate.resolve() in registered:
                    continue
                marker = candidate / ".git"
                if not marker.is_file():
                    continue
                try:
                    value = marker.read_text(encoding="utf-8").strip()
                    match = re.fullmatch(r"gitdir:\s*(.+)", value, re.I)
                    admin = Path(match.group(1)).resolve() if match else None
                except OSError:
                    admin = None
                if (admin is not None and admin_dir in admin.parents and not admin.exists()):
                    shutil.rmtree(candidate)
                    removed.append({"path": candidate.relative_to(self.repo_root).as_posix(), "kind": "orphan_worktree"})
        if (self.repo_root / ".git").exists():
            subprocess.run(["git", "worktree", "prune", "--expire=now"], cwd=self.repo_root,
                           capture_output=True, timeout=5, check=False)
        return removed

    def sparse_plan(self, changed_files: list[str] | None) -> dict:
        changed = sorted({str(item).replace("\\", "/").lstrip("./") for item in (changed_files or [])})
        tests_dir = self.repo_root / "tests"
        selected = set()
        for relative in changed:
            path = Path(relative)
            if path.suffix == ".py":
                stem = path.stem.removeprefix("__init__")
                candidate = tests_dir / f"test_{stem}.py"
                if candidate.is_file():
                    selected.add(candidate)
                else:
                    import_name = path.with_suffix("").as_posix().replace("/", ".")
                    for test_file in tests_dir.glob("test_*.py"):
                        try:
                            content = test_file.read_text(encoding="utf-8")
                        except (OSError, UnicodeError):
                            continue
                        if import_name in content:
                            selected.add(test_file)
            elif path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                for test_file in (self.repo_root / "tests").rglob("*.test.*"):
                    if path.stem.lower() in test_file.name.lower():
                        selected.add(test_file)
        commands = []
        python_tests = sorted(item for item in selected if item.suffix == ".py")
        js_tests = sorted(item for item in selected if item.suffix != ".py")
        if python_tests:
            commands.append([sys.executable, "-m", "pytest", "-q", *[item.relative_to(self.repo_root).as_posix() for item in python_tests]])
        if js_tests:
            node = shutil.which("node")
            commands.append([node or "node", "--test",
                             *[item.relative_to(self.repo_root).as_posix() for item in js_tests]])
        boundary = {"changed_files": changed,
                    "selected_test_files": [item.relative_to(self.repo_root).as_posix() for item in sorted(selected)],
                    "untested_changed_files": [item for item in changed if not any(
                        Path(item).stem.lower() in Path(test).stem.lower() for test in
                        (path.relative_to(self.repo_root).as_posix() for path in selected))],
                    "full_repository_suite_run": False}
        return {"commands": commands, "coverage_boundary": boundary}

    def run_sparse_gate(self, changed_files: list[str] | None, *, timeout: float = 120) -> dict:
        plan = self.sparse_plan(changed_files)
        results = []
        for command in plan["commands"]:
            try:
                completed = subprocess.run(command, cwd=self.repo_root, capture_output=True,
                                           text=True, encoding="utf-8", errors="replace",
                                           timeout=timeout, check=False)
                results.append({"command": command, "exit_code": completed.returncode,
                                "passed": completed.returncode == 0,
                                "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:]})
            except subprocess.TimeoutExpired as error:
                results.append({"command": command, "exit_code": 124, "passed": False,
                                "stdout": str(error.stdout or "")[-8000:],
                                "stderr": "targeted gate timed out"})
        return {"status": "passed" if results and all(item["passed"] for item in results) else "failed",
                "results": results, "coverage_boundary": plan["coverage_boundary"]}

    def check(self, *, changed_files: list[str] | None = None, sparse_mode: str = "auto") -> dict:
        if sparse_mode not in {"auto", "on", "off"}:
            raise ValueError("sparse_mode must be auto, on, or off")
        usage = self._usage_provider(self.repo_root)
        low_disk = int(usage.free) < self.min_free_bytes
        cleaned = self._cleanup_caches_and_orphans() if low_disk and self.cleanup_enabled else []
        try:
            usage_after = self._usage_provider(self.repo_root)
            disk = {"total_bytes": int(usage_after.total), "used_bytes": int(usage_after.used),
                    "free_bytes": int(usage_after.free), "minimum_free_bytes": self.min_free_bytes,
                    "low_space": int(usage_after.free) < self.min_free_bytes,
                    "cleaned": cleaned}
        except OSError as error:
            disk = {"error": str(error), "minimum_free_bytes": self.min_free_bytes,
                    "low_space": True, "cleaned": cleaned}
        missing_python = self._python_requirements()
        node = self._node_requirements()
        expected_python = None
        metadata = self.repo_root / "pyproject.toml"
        if metadata.is_file():
            try:
                expected_python = tomllib.loads(metadata.read_text(encoding="utf-8")).get("project", {}).get("requires-python")
            except (OSError, tomllib.TOMLDecodeError):
                pass
        python_report = {"version": platform.python_version(), "executable": sys.executable,
                         "expected": expected_python}
        missing = ([*missing_python] if missing_python else [])
        if node["required"] and node["version"] is None:
            missing.append({"name": "node", "kind": "runtime"})
        if node["required"] and node["missing"]:
            missing.extend({"name": item, "kind": "node"} for item in node["missing"])
        configs = self._materialize_temporary_configs()
        sparse_reasons = []
        if disk.get("low_space"):
            sparse_reasons.append("disk_below_threshold")
        if missing:
            sparse_reasons.append("missing_dependencies_or_runtime")
        if sparse_mode == "on":
            sparse_reasons.append("explicit_sparse_mode")
        enabled = sparse_mode == "on" or (sparse_mode == "auto" and bool(sparse_reasons))
        sparse = self.sparse_plan(changed_files) if enabled else None
        status = "SPARSE_MODE" if enabled else ("READY_WITH_MISSING_CONFIG_TEMPLATES" if configs else "READY")
        if sparse_mode == "off" and sparse_reasons:
            status = "BLOCKED"
        return {"schema_version": "aegis.env_doctor.v1", "status": status,
                "ready": status != "BLOCKED", "blocking": status == "BLOCKED",
                "disk": disk, "python": python_report, "node": node,
                "missing_dependencies": missing, "temporary_configs": configs,
                "sparse_mode": {"enabled": enabled, "reasons": sparse_reasons,
                                "commands": sparse["commands"] if sparse else [],
                                "coverage_boundary": sparse["coverage_boundary"] if sparse else None},
                "generated_config_directory": self._temporary_dirs[0] if self._temporary_dirs else None}
