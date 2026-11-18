"""Execute attached-repository work orders in priority order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _relative(base, path):
    if type(path) is not str or not path:
        raise ValueError("expected a nonempty path")
    return (base / path).resolve()


def _execute(job, base, workers):
    from .cli import _load, _repo, _run
    from .planner import execute_task_dag

    profile_path = _relative(base, job["profile"])
    profile = _load(profile_path)
    if profile.get("schema_version") != "aegis.repo_profile.v1":
        raise ValueError("factory requires an aegis attach profile")
    root = _repo(profile["repo_root"])
    if profile_path != root / ".aegis" / "repo_profile.json":
        raise ValueError("profile does not belong to the attached repository")
    command = "contract" if "contract" in job else "dag"
    path = _relative(base, job[command])
    if Path(_load(path)["repo_root"]).resolve() != root:
        raise ValueError("work order repository does not match attached profile")
    db = _relative(base, job["db"]) if "db" in job else None
    if command == "contract":
        result, code = _run(argparse.Namespace(contract=path, db=db))
    else:
        models = _relative(base, job["models"]) if "models" in job else None
        result, code = execute_task_dag(path, db=db, repo=root, workers=workers, models=models)
    return {"repo_root": str(root), "languages": profile["languages"], "result": result,
            "status": result["status"], "exit_code": code}


def run_factory_queue(config, *, workers=4):
    from .cli import _load

    config = Path(config).resolve(strict=True)
    queue = _load(config)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be 1..32")
    if queue.get("schema_version") != "aegis.factory.v1" or not isinstance(queue.get("jobs"), list) or not queue["jobs"]:
        raise ValueError("factory requires a nonempty aegis.factory.v1 job list")
    jobs = queue["jobs"]
    ids = set()
    for job in jobs:
        if (not isinstance(job, dict) or set(job) - {"id", "priority", "profile", "contract", "dag", "db", "models"}
                or type(job.get("id")) is not str or not job["id"] or job["id"] in ids
                or type(job.get("priority")) is not int or "profile" not in job
                or len({"contract", "dag"} & job.keys()) != 1 or ("models" in job and "dag" not in job)):
            raise ValueError("invalid factory job")
        ids.add(job["id"])
    reports = []
    for job in sorted(jobs, key=lambda item: -item["priority"]):
        try:
            report = _execute(job, config.parent, workers)
        except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
            report = {"status": "error", "exit_code": 1, "error_type": type(error).__name__}
        reports.append({"id": job["id"], "priority": job["priority"], **report})
    passed = sum(item["exit_code"] == 0 for item in reports)
    budgets = [item["result"]["budget"] for item in reports if "budget" in item.get("result", {})]
    totals = {key: sum(report[key] for report in budgets)
              for key in ("token_consumption", "estimated_cost_usd", "reward", "wall_clock_sec")}
    totals["reward_per_usd"] = totals["reward"] / totals["estimated_cost_usd"] if totals["estimated_cost_usd"] else None
    failed = len(reports) - passed
    return {"status": "failed" if failed else "succeeded", "jobs": reports, "passed": passed,
            "failed": failed, "budget": totals}, int(bool(failed))
