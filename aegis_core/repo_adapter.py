"""Read-only project reconnaissance and explicit Aegis profile attachment."""
from __future__ import annotations

import json
from pathlib import Path
import re
import uuid


SKIP_DIRS = {".git", ".aegis", ".venv", "venv", "node_modules", "dist", "build", ".worktrees"}
PROTECTED_TEST_PATTERNS = [
    "**/tests/**", "**/test/**", "**/__tests__/**", "**/test_*.py", "**/test_*.js",
    "**/test_*.jsx", "**/test_*.mjs", "**/test_*.cjs", "**/test_*.ts",
    "**/test_*.tsx", "**/test_*.mts", "**/test_*.cts",
    "**/*_test.py", "**/*_test.js", "**/*_test.jsx", "**/*_test.mjs", "**/*_test.cjs",
    "**/*_test.ts", "**/*_test.tsx", "**/*_test.mts", "**/*_test.cts",
    "**/*.test.js", "**/*.test.jsx", "**/*.test.ts", "**/*.test.tsx",
    "**/*.spec.js", "**/*.spec.jsx", "**/*.spec.ts", "**/*.spec.tsx",
    "**/*.test.mjs", "**/*.test.cjs", "**/*.test.mts", "**/*.test.cts",
    "**/*.spec.mjs", "**/*.spec.cjs", "**/*.spec.mts", "**/*.spec.cts",
]


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        yield path, relative.as_posix()


def _test_name(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    parts = normalized.split("/")
    return (
        any(part in {"test", "tests", "__tests__"} for part in parts)
        or name.startswith("test_")
        or re.search(r"_test\.(?:py|js|jsx|mjs|cjs|ts|tsx|mts|cts)$", name)
        or bool(re.search(r"\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs|mts|cts)$", name))
    )


def inspect_repository(repo: str | Path) -> dict:
    root = Path(repo).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--repo must be a directory")
    files = list(_iter_files(root))
    extension_counts = {}
    paths = []
    test_paths = []
    for _, relative in files:
        suffix = Path(relative).suffix.lower()
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        if _test_name(relative):
            test_paths.append(relative)
        paths.append(relative)
    languages = {
        "python": sum(extension_counts.get(suffix, 0) for suffix in (".py",)),
        "javascript": sum(extension_counts.get(suffix, 0) for suffix in (".js", ".jsx", ".mjs", ".cjs")),
        "typescript": sum(extension_counts.get(suffix, 0) for suffix in (".ts", ".tsx", ".mts", ".cts")),
    }
    package_json = root / "package.json"
    package = {}
    if package_json.is_file() and not package_json.is_symlink():
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            package = {}
    tests = set()
    if any(Path(path).suffix.lower() == ".py" and _test_name(path) for path in paths):
        tests.add("pytest")
    if any(Path(path).suffix.lower() in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
           and _test_name(path) for path in paths):
        tests.add("node --test")
    scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
    if isinstance(scripts, dict) and isinstance(scripts.get("test"), str) and scripts["test"].strip():
        tests.add("npm test")
        if re.search(r"\bnode\s+--test\b", scripts["test"]):
            tests.add("node --test")
    if any((root / name).exists() for name in ("pytest.ini", "tox.ini")):
        tests.add("pytest")

    entry_names = {"main.py", "app.py", "__main__.py", "index.js", "index.ts", "main.js", "main.ts"}
    entry_points = [path for path in paths if Path(path).name in entry_names]
    if isinstance(package, dict) and isinstance(package.get("main"), str):
        entry = package["main"].replace("\\", "/")
        if entry in paths and entry not in entry_points:
            entry_points.insert(0, entry)
    entry_points = sorted(set(entry_points), key=lambda name: (name.count("/"), name))
    return {
        "schema_version": "aegis.repo_profile.v1",
        "repo_root": str(root),
        "file_count": len(files),
        "languages": {key: {"files": value, "present": value > 0} for key, value in languages.items()},
        "extension_counts": dict(sorted(extension_counts.items())),
        "entry_modules": entry_points,
        "test_frameworks": sorted(tests),
        "test_files": sorted(test_paths),
        "protected_test_patterns": list(PROTECTED_TEST_PATTERNS),
        "package_scripts": scripts if isinstance(scripts, dict) else {},
    }


def attach_repository(repo: str | Path) -> dict:
    """Write a profile atomically under the selected repository's .aegis dir."""
    profile = inspect_repository(repo)
    root = Path(profile["repo_root"])
    directory = root / ".aegis"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "repo_profile.json"
    temp = directory / ("repo_profile." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return {"profile": profile, "profile_path": str(destination)}
