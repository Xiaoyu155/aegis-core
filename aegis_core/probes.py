"""Frozen probe metadata and failure codes shared by repair and validation gates."""
import hashlib
from pathlib import Path
import re

from .ast_slicer import JS_SUFFIXES


PROBE_KINDS = ("unit", "browser_visual")
PROBE_SUFFIXES = JS_SUFFIXES | {".py", ".sh", ".ps1", ".bat", ".cmd"}
PROBE_INTEGRITY_VIOLATION = "PROBE_INTEGRITY_VIOLATION"


def _probe_path(root, name):
    path = Path(name)
    absolute = path if path.is_absolute() else root / path
    absolute = absolute.resolve(strict=True)
    if not path.is_absolute() and not absolute.is_relative_to(root):
        raise ValueError("probe script escapes the repository; use an explicit absolute path")
    if not absolute.is_file():
        raise ValueError("probe script must be a file")
    return absolute


def freeze_probes(root, commands, *, kind="unit", scripts=()):
    if kind not in PROBE_KINDS:
        raise ValueError("probe_kind must be unit or browser_visual")
    if type(scripts) not in (tuple, list) or any(type(name) is not str or not name for name in scripts):
        raise ValueError("probe_scripts must be a list of file paths")
    root = Path(root).resolve()
    frozen = {}

    def freeze(name):
        path = _probe_path(root, name)
        key = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        frozen[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return key

    rewritten = []
    for command in commands:
        argv = []
        for arg in command:
            name, separator, selector = arg.partition("::")
            path = Path(name)
            if not arg.startswith("-") and path.suffix.lower() in PROBE_SUFFIXES and (root / path).is_file():
                # This also makes an absolute in-repository test execute in the candidate worktree.
                key = freeze(name)
                arg = f"./{key}" if not Path(key).is_absolute() else key
                arg += separator + selector
            argv.append(arg)
        rewritten.append(argv)
    for script in scripts:
        freeze(script)
    if kind == "browser_visual" and not frozen:
        raise ValueError("browser_visual requires at least one frozen probe script")
    return rewritten, frozen


def verify_probe_files(root, locked):
    if not isinstance(locked, dict):
        raise ValueError("probe_script_sha256 must be a path-to-hash object")
    root = Path(root).resolve()
    for name, expected in locked.items():
        if not isinstance(name, str) or not name or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("invalid probe path or SHA-256")
        try:
            path = _probe_path(root, name)
            # Reject substitution through a changed symbolic link.
            expected_path = Path(name) if Path(name).is_absolute() else root / name
            if path != expected_path or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("probe content changed")
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"{PROBE_INTEGRITY_VIOLATION}: {name}") from error


def classify_probe_failure(exit_code, output, *, kind, changed=False):
    if changed:
        return "PROBE_MUTATED_WORKTREE"
    if exit_code == 0:
        return None
    if exit_code == 124:
        return "TIMEOUT_KILLED"
    if exit_code == 127 or re.search(
            r"Executable doesn.t exist|no usable sandbox|browserType\.launch.*(?:failed|closed)"
            r"|could not find (?:chromium|firefox|browser)|host system is missing dependencies"
            r"|(?:Cannot find (?:module|package)|No module named|ModuleNotFoundError:)\s*['\"]?(?:@playwright/|playwright\b|puppeteer\b|pytest\b)"
            r"|(?:ECONNREFUSED|ERR_CONNECTION_REFUSED|EADDRINUSE|EACCES|EPERM)", output, re.I):
        return "ENVIRONMENT_UNAVAILABLE"
    if kind == "browser_visual" and re.search(
            r"TimeoutError|(?:page|locator)\.[^\n]*Timeout\s+\d+\s*ms exceeded", output, re.I):
        return "BROWSER_WAIT_TIMEOUT"
    if re.search(r"ModuleNotFoundError|Cannot find module|ERR_MODULE_NOT_FOUND|No module named"
                 r"|(?:\w+\.)+\w+ is not a function", output, re.I):
        return "MODULE_NOT_IMPLEMENTED"
    if re.search(r"AssertionError|ERR_ASSERTION|\bassert\b|(?m:^\s*not ok\b)|test failed", output, re.I):
        return "ASSERTION_FAILED"
    return "EXECUTION_ERROR"
