"""Deterministic multi-file requirement planning and gated DAG execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

from .ast_slicer import ImportGraph, is_protected_test_path
from .runner import retrieve, terms


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout.strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:36] or "task"


def _is_business_path(path: str) -> bool:
    from .mediator import is_business_source
    return is_business_source(path) and not is_protected_test_path(path)


def _topological_layers(tasks: list[dict]) -> list[list[str]]:
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("TaskDAG needs at least one task")
    by_id = {task["task_id"]: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("TaskDAG task IDs must be unique")
    for task in tasks:
        if any(parent not in by_id for parent in task["depends_on"]):
            raise ValueError("TaskDAG has an unknown dependency")
        if task["task_id"] in task["depends_on"]:
            raise ValueError("TaskDAG contains a self dependency")
    completed, layers = set(), []
    while len(completed) < len(tasks):
        layer = sorted(task_id for task_id, task in by_id.items()
                       if task_id not in completed and set(task["depends_on"]) <= completed)
        if not layer:
            raise ValueError("TaskDAG contains a dependency cycle")
        layers.append(layer)
        completed.update(layer)
    return layers


def build_task_dag(requirement: str, repo: str | Path, *, max_tasks: int = 8) -> dict:
    """Plan a requirement from repository symbols, imports and matching tests."""
    root = Path(repo).resolve(strict=True)
    if not (root / ".git").exists() and not (root / ".git").is_file():
        raise ValueError("--repo must be a Git repository")
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("requirement text is required")
    if type(max_tasks) is not int or not 1 <= max_tasks <= 32:
        raise ValueError("max_tasks must be 1..32")
    graph = ImportGraph(root)
    retrieval = retrieve(root, requirement, token_budget=6000, graph=graph)
    query = terms(requirement)
    candidates = {}
    for row in retrieval["candidates"]:
        if _is_business_path(row["file"]):
            candidates[row["file"]] = max(float(row["score"]), candidates.get(row["file"], 0.0))
    # Score complete source files as well as top symbol hits so multi-module
    # requirements don't collapse to a single retrieval result.
    for relative in [*graph.paths, *graph.js_modules]:
        if not _is_business_path(relative):
            continue
        if relative in graph.paths:
            module = graph.paths[relative]
            content = " ".join(key + " " + __import__("ast").unparse(node)
                               for key, node in graph.symbols.items()
                               if graph.symbol_modules[key] == module)
        else:
            content = " ".join(key + " " + item["source"]
                               for key, item in graph.js_symbols.items()
                               if key.startswith(relative + ":"))
        overlap = query & terms(relative + " " + content)
        score = len(overlap) / max(1, len(query))
        if score:
            candidates[relative] = max(score, candidates.get(relative, 0.0))
    request_lower = requirement.lower().replace("\\", "/")
    mentioned_source_paths = set(re.findall(
        r"(?<![a-z0-9_./-])((?:[a-z0-9_.-]+/)*[a-z0-9_.-]+\.(?:py|js|jsx|mjs|cjs|ts|tsx|mts|cts))(?=$|[^a-z0-9_])",
        request_lower))
    planned_new_paths = sorted(relative for relative in mentioned_source_paths
                               if relative not in graph.paths and relative not in graph.js_modules
                               and "/" in relative
                               and _is_business_path(relative)
                               and (root / Path(relative).parent).is_dir())
    mentioned_stems = {Path(relative).stem for relative in planned_new_paths}
    mentioned_paths = {relative for relative in candidates if relative.lower() in request_lower}
    mentioned_stems.update(Path(relative).stem for relative in candidates
                           if re.search(rf"(?<![a-z0-9_]){re.escape(Path(relative).stem.lower())}(?![a-z0-9_])",
                                        request_lower))
    if mentioned_paths:
        candidates = {relative: score for relative, score in candidates.items()
                      if relative in mentioned_paths}
    elif len(mentioned_stems) >= 2:
        candidates = {relative: score for relative, score in candidates.items()
                      if Path(relative).stem in mentioned_stems}
    integration_modules = []
    validation_requested = bool(query & {"test", "tests", "validate", "validation", "integration", "e2e"})
    if validation_requested and len(candidates) >= 2:
        for relative in list(candidates):
            dependencies = set(graph.file_dependencies.get(relative, []))
            is_consumer = any(dependency in candidates and dependency != relative for dependency in dependencies)
            explicitly_named = Path(relative).stem in mentioned_stems
            if is_consumer and explicitly_named:
                integration_modules.append(relative)
                del candidates[relative]
    if not candidates:
        raise ValueError("the requirement did not identify an indexed business module")
    ranked = sorted(candidates, key=lambda path: (-candidates[path], path))[:max_tasks]
    repair_tasks = []
    task_by_file = {}
    for index, relative in enumerate(ranked):
        task_id = f"repair-{_slug(Path(relative).stem)}-{index + 1}"
        module = graph.paths.get(relative)
        if module is not None:
            symbols = sorted(key.split(":", 1)[1] for key in graph.symbols
                             if graph.symbol_modules[key] == module)
            context = graph.slice(relative, symbols[0], token_budget=2400) if symbols else {}
        else:
            js_names = sorted(key.split(":", 1)[1] for key in graph.js_symbols if key.startswith(relative + ":"))
            context = graph.slice(relative, js_names[0], token_budget=2400) if js_names else {}
        repair_tasks.append({
            "task_id": task_id,
            "kind": "repair",
            "title": f"Inspect and update {relative}",
            "depends_on": [],
            "contract": {
                "intent": requirement,
                "target_files": [relative],
                "success_criteria": ["preserve the module interface", "satisfy the frozen acceptance probe"],
                "ast_context": context,
            },
            "risk_estimate": round(max(0.05, min(0.95, 0.15 + 0.75 * candidates[relative])), 4),
            "patches": [],
            "fail_before_cmd": [],
            "pass_after_cmd": [],
            "source_dirs": [str(Path(relative).parent).replace(".", "", 1) or "."],
        })
        task_by_file[relative] = task_id
    for index, relative in enumerate(planned_new_paths):
        task_id = f"create-{_slug(Path(relative).stem)}-{index + 1}"
        repair_tasks.append({
            "task_id": task_id,
            "kind": "repair",
            "title": f"Create business module {relative}",
            "depends_on": [],
            "contract": {
                "intent": requirement,
                "target_files": [relative],
                "success_criteria": ["create the explicitly requested business module",
                                     "satisfy the frozen acceptance probe"],
                "ast_context": {"new_module": relative, "consumer_context": []},
            },
            "risk_estimate": 0.3,
            "patches": [],
            "fail_before_cmd": [],
            "pass_after_cmd": [],
            "source_dirs": [str(Path(relative).parent)],
        })
        task_by_file[relative] = task_id
    selected = set(ranked)
    selected.update(planned_new_paths)
    # A caller depends on the module it imports, so the lower-level contract
    # is completed first. Lexical fallback makes otherwise disconnected work
    # deterministic and keeps each task independently addressable.
    for relative, task_id in task_by_file.items():
        imports = set(graph.file_dependencies.get(relative, []))
        if relative in ranked and Path(relative).stem in mentioned_stems:
            imports.update(new_path for new_path in planned_new_paths
                           if new_path != relative and Path(new_path).parent == Path(relative).parent)
        task_by_file[relative]
        repair = next(task for task in repair_tasks if task["task_id"] == task_id)
        repair["depends_on"] = sorted(task_by_file[dependency] for dependency in imports
                                      if dependency in selected and task_by_file[dependency] != task_id)
    test_files = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if is_protected_test_path(relative) and path.suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            if any(Path(source).parent == Path(relative).parent for source in ranked):
                test_files.append(relative)
    test_files = sorted(test_files)
    validation_cmd = ([sys.executable, "-m", "pytest", "-q", *test_files]
                      if test_files and all(Path(path).suffix == ".py" for path in test_files)
                      else ["npm", "test"] if (root / "package.json").is_file()
                      else [sys.executable, "-m", "pytest", "-q"])
    validation = {
        "task_id": "validate-integration",
        "kind": "validation",
        "title": "Run integration tests after module repairs",
        "depends_on": sorted(task["task_id"] for task in repair_tasks),
        "contract": {"intent": requirement, "test_files": test_files,
                     "integration_modules": sorted(integration_modules),
                     "success_criteria": ["all selected integration tests pass", "test tree remains unchanged"]},
        "risk_estimate": 0.08,
        "command": validation_cmd,
        "timeout_sec": 180,
    }
    tasks = repair_tasks + [validation]
    layers = _topological_layers(tasks)
    return {
        "schema_version": "aegis.task_dag.v1",
        "dag_id": "dag-" + hashlib.sha256((requirement + str(root)).encode()).hexdigest()[:12],
        "repo_root": str(root),
        "base_commit": _git(root, "rev-parse", "HEAD"),
        "requirement": requirement,
        "tasks": tasks,
        "topological_layers": layers,
        "dependency_edges": [{"from": parent, "to": task["task_id"]}
                             for task in tasks for parent in task["depends_on"]],
        "retrieval": {"candidates": retrieval["candidates"], "diagnostics": graph.diagnostics},
    }


def execute_task_dag(plan_path: str | Path, *, db: str | Path | None = None,
                     workers: int = 4, repo: str | Path | None = None) -> tuple[dict, int]:
    """Run each DAG layer through ordinary Aegis contracts and a staged worktree."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "aegis.task_dag.v1":
        raise ValueError("unsupported TaskDAG schema")
    root = Path(repo or plan.get("repo_root", "")).resolve(strict=True)
    if Path(plan["repo_root"]).resolve() != root:
        raise ValueError("TaskDAG repository does not match --repo")
    if _git(root, "rev-parse", "HEAD") != plan["base_commit"]:
        raise ValueError("TaskDAG baseline changed; regenerate the plan")
    if _git(root, "status", "--porcelain"):
        raise ValueError("DAG execution requires a clean repository")
    layers = _topological_layers(plan["tasks"])
    task_map = {task["task_id"]: task for task in plan["tasks"]}
    ledger_path = Path(db).resolve() if db else Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "aegis/ledger.sqlite"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    stage_id = "aegis/dag-" + uuid.uuid4().hex[:10]
    stage_dir = root / ".worktrees" / ("dag-" + uuid.uuid4().hex[:10])
    if stage_dir.parent.is_symlink():
        raise ValueError("DAG worktree parent must not be a symlink")
    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "worktree", "add", "-b", stage_id, str(stage_dir), plan["base_commit"])
    statuses, results = {}, []
    try:
        for layer_index, layer in enumerate(layers):
            ready = [task_id for task_id in layer
                     if all(statuses.get(parent) == "passed" for parent in task_map[task_id]["depends_on"])]
            blocked = [task_id for task_id in layer if task_id not in ready]
            for task_id in blocked:
                statuses[task_id] = "blocked"
                results.append({"task_id": task_id, "status": "blocked", "reason": "upstream_gate_failed",
                                "cascade_status": "SKIPPED_UPSTREAM_FAILED",
                                "failed_dependencies": [parent for parent in task_map[task_id]["depends_on"]
                                                        if statuses.get(parent) != "passed"]})
            repair_tasks = [task_map[task_id] for task_id in ready if task_map[task_id]["kind"] == "repair"]
            manifests = []
            manifest_tasks = {}
            for task in repair_tasks:
                patches = task.get("patches", [])
                if not patches:
                    statuses[task["task_id"]] = "failed"
                    results.append({"task_id": task["task_id"], "status": "failed",
                                    "reason": "no_patch_artifact"})
                    continue
                targets = task["contract"]["target_files"]
                probe = task.get("fail_before_cmd") or task.get("pass_after_cmd")
                if not probe:
                    statuses[task["task_id"]] = "failed"
                    results.append({"task_id": task["task_id"], "status": "failed",
                                    "reason": "frozen_probe_required"})
                    continue
                from .cli import _wish
                from .mediator import is_business_source
                ticket_id = "dag-" + _slug(task["task_id"]) + "-" + uuid.uuid4().hex[:8]
                manifest = plan_path.parent / (ticket_id + ".contract.json")
                args = argparse.Namespace(
                    repo=str(stage_dir), text=task["contract"]["intent"], auto_probe=False,
                    allow_model=False, models=None, target=targets, proposal=None, constraints=None,
                    selected_module=None, db=str(ledger_path), ticket_id=ticket_id,
                    fail_before=json.dumps(probe), pass_after=json.dumps(task.get("pass_after_cmd") or probe),
                    source_dir=task.get("source_dirs") or ["."], patch=patches,
                    max_attempts=task.get("max_attempts", len(patches)),
                    timeout=float(task.get("timeout_sec", 180)), out=str(manifest))
                document, code = _wish(args)
                if code or document["compilation"]["status"] != "COMPILED":
                    statuses[task["task_id"]] = "failed"
                    results.append({"task_id": task["task_id"], "status": "failed", "reason": "contract_compile_failed"})
                    manifest.unlink(missing_ok=True)
                    continue
                manifests.append(manifest)
                manifest_tasks[str(manifest)] = task
            if manifests:
                from .queue_worker import run_contracts
                executions = run_contracts(manifests, db=ledger_path, workers=min(workers, max(1, len(manifests))))
                for manifest, (result, code) in zip(manifests, executions):
                    task = manifest_tasks[str(manifest)]
                    task_id = task["task_id"]
                    if code == 0 and result.get("status") == "succeeded":
                        passed_attempts = [attempt for attempt in result.get("results", [])
                                           if attempt.get("passed") and attempt.get("patch_sha256")]
                        winning_hash = passed_attempts[-1]["patch_sha256"] if passed_attempts else None
                        matching = []
                        for patch_path in task["patches"]:
                            patch_text = Path(patch_path).read_text(encoding="utf-8")
                            if hashlib.sha256(patch_text.encode("utf-8")).hexdigest() == winning_hash:
                                matching.append(Path(patch_path).read_bytes())
                        if len(matching) != 1:
                            statuses[task_id] = "failed"
                            results.append({"task_id": task_id, "status": "failed",
                                            "reason": "passing_patch_artifact_not_identified",
                                            "aegis_result": result, "exit_code": code})
                            manifest.unlink(missing_ok=True)
                            continue
                        statuses[task_id] = "passed"
                        # Materialize exactly the candidate whose two probes passed.
                        patch_bytes = matching[0]
                        subprocess.run(["git", "apply", "--check", "--whitespace=error", "-"], cwd=stage_dir,
                                       input=patch_bytes, capture_output=True, check=True)
                        subprocess.run(["git", "apply", "--whitespace=error", "-"], cwd=stage_dir,
                                       input=patch_bytes, capture_output=True, check=True)
                        subprocess.run(["git", "add", "--all"], cwd=stage_dir, check=True, capture_output=True)
                        subprocess.run(["git", "-c", "user.name=Aegis", "-c", "user.email=aegis@localhost",
                                        "commit", "-m", f"Aegis DAG: {task['title']}"], cwd=stage_dir,
                                       check=True, capture_output=True)
                    else:
                        statuses[task_id] = "failed"
                    results.append({"task_id": task_id, "status": statuses[task_id],
                                    "aegis_result": result, "exit_code": code})
                    manifest.unlink(missing_ok=True)
            for task_id in ready:
                task = task_map[task_id]
                if task["kind"] not in {"validation", "command"} or statuses.get(task_id):
                    continue
                command = task.get("command")
                if not isinstance(command, list) or not command:
                    statuses[task_id] = "failed"
                    results.append({"task_id": task_id, "status": "failed", "reason": "validation_command_required"})
                    continue
                from .sandbox import run_exam_gate
                gate = run_exam_gate(stage_dir, command, timeout=float(task.get("timeout_sec", 180)))
                statuses[task_id] = "passed" if gate["passed"] else "failed"
                results.append({"task_id": task_id, "status": statuses[task_id], "gate": gate})
            # A task that was selected but did not yield an explicit result is
            # an error, preventing silent skips from unlocking descendants.
            for task_id in ready:
                if task_id not in statuses:
                    statuses[task_id] = "failed"
                    results.append({"task_id": task_id, "status": "failed", "reason": "unsupported_task_kind"})
        passed = all(statuses.get(task["task_id"]) == "passed" for task in plan["tasks"])
        return {"dag_id": plan["dag_id"], "status": "succeeded" if passed else "failed",
                "tasks": results, "task_statuses": statuses,
                "staged_commit": _git(stage_dir, "rev-parse", "HEAD")}, 0 if passed else 1
    finally:
        try:
            _git(root, "worktree", "remove", "--force", str(stage_dir))
        finally:
            try: _git(root, "branch", "-D", stage_id)
            except subprocess.CalledProcessError: pass
