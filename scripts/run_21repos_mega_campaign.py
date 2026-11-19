#!/usr/bin/env python3
"""Run read-only, worktree-isolated Aegis campaign tickets on Xiaoyu155 repositories.

Every repository is cloned or supplied as a disposable source checkout.  The two
batch tickets are temporary source annotations verified by real fail-before / pass-
after gates.  The autopilot ticket seeds a tracked contract fixture in an outer
detached worktree, races bad and correct AST-authored patches, and replays the
selected result.  No commit is pushed or merged to any business repository.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "Xiaoyu155"
LANGUAGE_SUFFIXES = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts",
                     ".html", ".htm", ".sql", ".sh", ".ps1", ".css", ".vue"}
SKIP_PARTS = {".git", ".aegis", ".worktrees", ".venv", "venv", "node_modules", "dist", "build",
              "vendor", "out", "target", "coverage", "__pycache__", ".next"}


def _run(command, *, cwd=None, env=None, timeout=300):
    return subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout, check=False)


def _git(repo, *args, check=True):
    result = _run(["git", *args], cwd=repo, timeout=600)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-2000:] or "git command failed")
    return result.stdout.decode("utf-8", "replace").strip()


def _aegis_env():
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + old if old else "")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _metadata(account):
    result = _run(["gh", "repo", "list", account, "--limit", "1000", "--json",
                   "nameWithOwner,isPrivate,isArchived,url,defaultBranchRef,primaryLanguage"] ,
                  env=_aegis_env(), timeout=120)
    if result.returncode:
        raise RuntimeError("gh repo list failed: " + result.stderr.decode("utf-8", "replace")[-1000:])
    rows = json.loads(result.stdout)
    selected = [item for item in rows if item["nameWithOwner"].lower() != f"{account}/aegis-core".lower()
                and not item.get("isArchived")]
    return sorted(selected, key=lambda item: item["nameWithOwner"].lower())


def _sparse_clone(repo, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    clone = _run(["gh", "repo", "clone", repo["nameWithOwner"], destination, "--",
                  "--depth=1", "--filter=blob:none", "--no-checkout"], env=_aegis_env(), timeout=1200)
    if clone.returncode:
        raise RuntimeError(clone.stderr.decode("utf-8", "replace")[-1500:] or "clone failed")
    patterns = ["*.py", "*.js", "*.jsx", "*.mjs", "*.cjs", "*.ts", "*.tsx", "*.mts", "*.cts",
                "*.html", "*.htm", "*.sql", "*.sh", "*.ps1", "*.css", "*.vue", "package.json",
                "pyproject.toml", "setup.cfg", "pytest.ini", "requirements*.txt", "Makefile"]
    sparse = _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=destination, timeout=60)
    if sparse.returncode:
        raise RuntimeError(sparse.stderr.decode("utf-8", "replace")[-1000:])
    pattern_file = destination / ".git" / "info" / "sparse-checkout"
    pattern_file.write_text("\n".join(patterns) + "\n", encoding="utf-8")
    checkout = _run(["git", "checkout", "--detach", "HEAD"], cwd=destination, env=_aegis_env(), timeout=600)
    if checkout.returncode:
        raise RuntimeError(checkout.stderr.decode("utf-8", "replace")[-1500:])


def _code_files(repo):
    names = _git(repo, "ls-files", "-z").split("\0")
    files = []
    for name in names:
        if not name:
            continue
        path = Path(name)
        if path.suffix.lower() not in LANGUAGE_SUFFIXES or any(part in SKIP_PARTS for part in path.parts):
            continue
        if any(part.lower() in {"tests", "test", "__tests__", "fixtures", "migrations"} for part in path.parts):
            continue
        full = repo / path
        try:
            if not full.is_file() or full.is_symlink() or full.stat().st_size > 512 * 1024:
                continue
            raw = full.read_bytes()
            decoded = None
            encoding = None
            for candidate in ("utf-8-sig", "gbk"):
                try:
                    decoded, encoding = raw.decode(candidate), candidate
                    break
                except UnicodeDecodeError:
                    pass
            if decoded is None:
                continue
        except OSError:
            continue
        files.append({"path": path.as_posix(), "source": decoded, "raw": raw, "encoding": encoding,
                      "suffix": path.suffix.lower(), "bytes": len(raw)})
    return files


def _select_target(files, repo=None):
    if not files:
        raise ValueError("no decodable tracked business source is available")
    from aegis_core.mediator import is_business_source
    from aegis_core.repo_adapter import load_tool_source_globs
    globs = load_tool_source_globs(repo) if repo is not None else None
    business = [row for row in files if is_business_source(row["path"],
                **({"tool_source_globs": globs} if globs is not None else {}))]
    if not business:
        raise ValueError("no tracked source eligible for Aegis business-change gates")
    files = business
    def score(row):
        path = PurePosixPath(row["path"])
        value = 0
        if path.suffix == ".py": value += 40
        elif path.suffix in {".js", ".ts", ".tsx", ".mjs", ".jsx"}: value += 30
        elif path.suffix in {".html", ".htm", ".vue"}: value += 20
        if path.parts[0] in {"src", "app", "agent", "server", "client", "frontend", "backend"}: value += 12
        if "generated" in path.parts or path.name.endswith((".min.js", ".bundle.js")): value -= 100
        return (-value, len(path.parts), -row["bytes"], row["path"])
    return sorted(files, key=score)[0]


def _patch_for_comment(target, marker):
    old = target["raw"]
    suffix = target["suffix"]
    if suffix in {".html", ".htm", ".vue"}:
        comment = f"<!-- {marker} -->"
    elif suffix == ".sql":
        comment = f"-- {marker}"
    elif suffix == ".css":
        comment = f"/* {marker} */"
    else:
        comment = f"// {marker}" if suffix in {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"} else f"# {marker}"
    newline = b"\r\n" if b"\r\n" in old else b"\n"
    updated = old + (b"" if old.endswith((b"\n", b"\r")) else newline) + comment.encode("utf-8") + newline
    before_lines, after_lines = old.splitlines(keepends=True), updated.splitlines(keepends=True)
    diff = difflib.diff_bytes(difflib.unified_diff, before_lines, after_lines,
                              fromfile=("a/" + target["path"]).encode(),
                              tofile=("b/" + target["path"]).encode())
    return b"diff --git a/" + target["path"].encode() + b" b/" + target["path"].encode() + b"\n" + b"".join(diff)


def _batch_command(repo, ticket_id, target, patch_path, marker, profile, database, output_dir):
    text = (f"Document the audited {target['suffix']} source contract for {target['path']} "
            f"under campaign ticket {ticket_id}; add this trace marker: {marker}.")
    proposal_path = output_dir / f"{ticket_id}.proposal.json"
    contract_path = output_dir / f"{ticket_id}.contract.json"
    proposal_path.write_text(json.dumps({"target_files": [target["path"]],
        "contract_stub": "def document_audited_contract() -> None:\n    ..."}), encoding="utf-8")
    command_source = ("from pathlib import Path; "
        f"p=Path({target['path']!r}); s=p.read_text(encoding={target['encoding']!r}); "
        f"assert {marker!r} in s; "
        + (f"compile(s, {target['path']!r}, 'exec')" if target["suffix"] == ".py" else "assert len(s) > 0"))
    probe = json.dumps([sys.executable, "-B", "-c", command_source], ensure_ascii=False)
    wish = _run([sys.executable, "-m", "aegis_core.cli", "wish", text, "--repo", repo,
                 "--ticket-id", ticket_id, "--target", target["path"], "--proposal", proposal_path,
                 "--no-auto-probe", "--fail-before", probe, "--pass-after", probe,
                 "--patch", patch_path, "--out", contract_path, "--db", database],
                cwd=repo, env=_aegis_env(), timeout=120)
    if wish.returncode:
        return {"ticket_id": ticket_id, "mode": "aegis batch", "target_file": target["path"],
                "status": "compile_failed", "exit_code": wish.returncode,
                "stderr": wish.stderr.decode("utf-8", "replace")[-1200:]}
    config_path = output_dir / f"{ticket_id}.factory.json"
    config_path.write_text(json.dumps({"schema_version": "aegis.factory.v1", "jobs": [{
        "id": ticket_id, "priority": 1, "profile": str(profile), "contract": str(contract_path),
        "db": str(database)}]}, ensure_ascii=False), encoding="utf-8")
    batch_out = output_dir / f"{ticket_id}.batch.json"
    result = _run([sys.executable, "-m", "aegis_core.cli", "batch", "--config", config_path,
                   "--workers", "1", "--out", batch_out], cwd=repo, env=_aegis_env(), timeout=300)
    record = {"ticket_id": ticket_id, "mode": "aegis batch", "target_file": target["path"],
              "marker": marker, "exit_code": result.returncode,
              "stdout_tail": result.stdout.decode("utf-8", "replace")[-500:],
              "stderr_tail": result.stderr.decode("utf-8", "replace")[-800:]}
    if batch_out.is_file():
        try:
            data = json.loads(batch_out.read_text(encoding="utf-8"))
            record.update(status=data.get("status"), passed=data.get("passed"), failed=data.get("failed"))
            job = (data.get("jobs") or [{}])[0]
            record["reason"] = (job.get("result") or {}).get("reason")
        except (ValueError, OSError):
            record["status"] = "invalid_batch_receipt"
    return record


def _autopilot_ticket(repo_name, repo, database, dataset, output_dir, timeout=120, ticket_suffix=""):
    slug = repo_name.split("/", 1)[-1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ticket_id = "v14-swarm-" + re_slug(slug) + ("-" + re_slug(ticket_suffix) if ticket_suffix else "")
    task_root = repo.parent / (".aegis-v14-autopilot-" + re_slug(slug))
    if task_root.exists():
        shutil.rmtree(task_root)
    head_before = _git(repo, "rev-parse", "HEAD")
    git_worktrees_before = _git(repo, "worktree", "list", "--porcelain").count("worktree ")
    _git(repo, "worktree", "add", "--detach", str(task_root), head_before)
    try:
        fixture = task_root / "aegis_campaign_probe.py"
        fixture.write_text("def bounded(value):\n    return value\n", encoding="utf-8")
        _git(task_root, "config", "user.name", "Aegis Campaign")
        _git(task_root, "config", "user.email", "aegis-campaign@example.invalid")
        _git(task_root, "add", "aegis_campaign_probe.py")
        _git(task_root, "commit", "-m", "Seed isolated campaign contract fixture")
        base = _git(task_root, "rev-parse", "HEAD")
        before = fixture.read_text(encoding="utf-8")
        patches = {
            "bait_constant": before.replace("return value", "return 99"),
            "broken_signature": before.replace("def bounded(value):", "def bounded():").replace("return value", "return 0"),
            "conservative_repair": "def bounded(value):\n    if type(value) is not int:\n        return None\n    return min(10, max(0, value))\n",
            "off_by_one": "def bounded(value):\n    if type(value) is not int:\n        return None\n    return min(10, max(0, value + 1))\n",
        }
        import difflib as _difflib
        def unified(candidate_source):
            patch = "diff --git a/aegis_campaign_probe.py b/aegis_campaign_probe.py\n"
            patch += "".join(_difflib.unified_diff(before.splitlines(True), candidate_source.splitlines(True),
                fromfile="a/aegis_campaign_probe.py", tofile="b/aegis_campaign_probe.py"))
            return patch
        candidate_rounds = [[
            {"candidate_id": "bait_constant", "strategy": "constant shortcut that ignores input",
             "patch": unified(patches["bait_constant"])},
            {"candidate_id": "broken_signature", "strategy": "remove a required parameter",
             "patch": unified(patches["broken_signature"])},
        ], [
            {"candidate_id": "conservative_repair", "strategy": "type-safe inclusive clamp with None fallback",
             "patch": unified(patches["conservative_repair"])},
            {"candidate_id": "off_by_one", "strategy": "plausible but incorrect upper/lower value shift",
             "patch": unified(patches["off_by_one"])},
        ]]
        config_path = output_dir / f"{ticket_id}.autopilot.json"
        config_path.write_text(json.dumps({"ticket_id": ticket_id,
            "contract": f"Clamp the campaign contract value to the inclusive range [0, 10] in {repo_name}.",
            "probe_spec": {"kind": "numeric_bounds", "path": "aegis_campaign_probe.py",
                           "symbol": "bounded", "min": 0, "max": 10, "fallback": None,
                           "seed": 1419},
            "candidate_rounds": candidate_rounds, "max_rounds": 2, "max_workers": 2,
            "timeout_sec": 15, "max_context_tokens": 1200, "max_total_tokens": 8000,
            "max_wall_clock_sec": timeout, "dataset_path": str(dataset)},
            ensure_ascii=False), encoding="utf-8")
        report_path = output_dir / f"{ticket_id}.autopilot-result.json"
        start = time.monotonic()
        command = _run([sys.executable, "-m", "aegis_core.cli", "autopilot", "--config", config_path,
                        "--repo", task_root, "--db", database, "--out", report_path],
                       cwd=task_root, env=_aegis_env(), timeout=timeout + 60)
        elapsed = round(time.monotonic() - start, 4)
        record = {"ticket_id": ticket_id, "mode": "aegis autopilot", "repo": repo_name,
                  "status": "command_failed" if command.returncode else "unknown",
                  "exit_code": command.returncode, "elapsed_sec": elapsed,
                  "seeded_fault": "identity body replaced with return value (true red expected)",
                  "candidate_strategies": ["constant bait", "signature break", "correct clamp", "off by one"],
                  "stdout_tail": command.stdout.decode("utf-8", "replace")[-1000:],
                  "stderr_tail": command.stderr.decode("utf-8", "replace")[-1500:],
                  "isolated_fixture": "aegis_campaign_probe.py", "base_commit": base}
        if report_path.is_file():
            try:
                result = json.loads(report_path.read_text(encoding="utf-8"))
                record.update(status=result.get("status"), reason=result.get("reason"),
                              rounds_attempted=result.get("rounds_attempted"),
                              candidate_rejections=sum(int(round_.get("tournament", {}).get("candidate_count", 0))-
                                  int(round_.get("tournament", {}).get("accepted_count", 0))
                                  for round_ in result.get("rounds", [])),
                              winner=(result.get("winner") or {}).get("candidate_id"),
                              time_travel_snapshot=result.get("time_travel_snapshot"),
                              learning=result.get("learning", {}))
                for round_ in result.get("rounds", []):
                    for evaluation in round_.get("tournament", {}).get("evaluations", []):
                        audit = evaluation.get("audit", {})
                        for finding in audit.get("findings", []):
                            record.setdefault("audit_findings", []).append(finding)
            except (ValueError, OSError) as error:
                record["status"] = "invalid_autopilot_receipt"
                record["receipt_error"] = str(error)
        replay_path = output_dir / f"{ticket_id}.replay.json"
        replay = _run([sys.executable, "-m", "aegis_core.cli", "replay", "--ticket", ticket_id,
                       "--repo", task_root, "--db", database, "--out", replay_path],
                      cwd=task_root, env=_aegis_env(), timeout=60)
        record["replay_exit_code"] = replay.returncode
        if replay_path.is_file():
            replay_document = json.loads(replay_path.read_text(encoding="utf-8"))
            record["replay_status"] = replay_document.get("status")
            record["replay_probe_exit_code"] = (replay_document.get("outcome") or {}).get("exit_code")
        record["default_head_unchanged"] = _git(repo, "rev-parse", "HEAD") == head_before
        record["tracked_main_worktree_clean"] = not bool(_git(repo, "diff", "--name-only", "HEAD"))
        return record
    finally:
        _git(repo, "worktree", "remove", "--force", str(task_root), check=False)
        _git(repo, "worktree", "prune", check=False)
        if task_root.exists(): shutil.rmtree(task_root, ignore_errors=True)


def re_slug(value):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")[:48] or "repo"


def _analyze(repo, files):
    from aegis_core.ast_slicer import ImportGraph
    from aegis_core.impact_analyzer import ImpactAnalyzer
    from aegis_core.semantic_contract import SemanticContractAnalyzer
    graph = ImportGraph(repo)
    impact = ImpactAnalyzer(repo)
    contracts = SemanticContractAnalyzer(repo).analyze()
    languages = {"python": len(graph.modules), "javascript_typescript_html": len(graph.js_modules),
                 "html_documents": len(graph.frontend_documents),
                 "tracked_source_files": len(files)}
    return {"languages": languages, "impact_graph": impact.graph_metrics,
            "semantic_contracts": contracts["metrics"],
            "parse_diagnostics": len(graph.diagnostics)}


def _db_summary(database):
    if not database.is_file(): return {"causal_traces": 0, "tickets": 0, "experience_records": 0,
                                      "journal_mode": None}
    with sqlite3.connect(str(database), timeout=30) as db:
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        traces = db.execute("SELECT COUNT(*) FROM causal_traces").fetchone()[0] if "causal_traces" in tables else 0
        tickets = db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] if "tickets" in tables else 0
        memories = db.execute("SELECT COUNT(*) FROM experience_records").fetchone()[0] if "experience_records" in tables else 0
        return {"causal_traces": traces, "tickets": tickets, "experience_records": memories,
                "journal_mode": mode}


def _candidate_metrics(rows):
    succeeded = sum(row.get("status") == "succeeded" for row in rows)
    rejected = sum(int(row.get("candidate_rejections", 0)) for row in rows)
    invalid = 3 * succeeded
    return {"succeeded_scenarios": succeeded, "evaluations": 4 * succeeded,
            "rejected": rejected, "intentionally_invalid": invalid,
            "interception_rate": rejected / invalid if invalid else None}


class _WalSampler:
    def __init__(self, database):
        self.path = Path(str(database) + "-wal")
        self.peak = 0; self.stop_event = threading.Event(); self.thread = None
    def start(self):
        def sample():
            while not self.stop_event.is_set():
                try: self.peak = max(self.peak, self.path.stat().st_size)
                except OSError: pass
                self.stop_event.wait(.01)
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)
        try: self.peak = max(self.peak, self.path.stat().st_size)
        except OSError: pass
        return self.peak


def _campaign(args):
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_lines = ["AEGIS V14 | 21-REPOSITORY READ-ONLY MEGA CAMPAIGN",
                 f"account={args.account} | requested_workorders={(0 if args.skip_batch else 2) * (len(args.repos) if args.repos else 21) + (0 if args.skip_autopilot else 1) * (len(args.repos) if args.repos else 21)} | parallel_autopilot_workers={args.workers}",
                 "execution=real Git clones + detached worktrees + actual aegis batch/autopilot/replay commands",
                 "remote_mutation=none (no push, no PR, no merge); all generated commits stay in temporary checkouts"]
    def log(line):
        log_lines.append(line)
        print(line, flush=True)
    repositories = _metadata(args.account)
    if args.repos:
        wanted = {item.casefold() for item in args.repos}
        repositories = [item for item in repositories
                        if item["nameWithOwner"].casefold() in wanted
                        or item["nameWithOwner"].split("/", 1)[-1].casefold() in wanted]
        found = {item["nameWithOwner"].casefold() for item in repositories}
        missing = sorted(item for item in wanted if item not in found
                         and not any(row["nameWithOwner"].split("/", 1)[-1].casefold() == item for row in repositories))
        if missing: raise ValueError("unknown campaign repositories: " + ", ".join(missing))
    if len(repositories) != 21:
        log(f"NOTICE repository_count={len(repositories)} (account inventory is not exactly 21)")
    owned = args.source_dir is None
    temporary = tempfile.TemporaryDirectory(prefix="aegis-v14-mega-") if owned else None
    source_root = Path(temporary.name) / "repos" if temporary else Path(args.source_dir).resolve()
    if owned:
        source_root.mkdir(parents=True, exist_ok=True)
        for metadata in repositories:
            _sparse_clone(metadata, source_root / metadata["nameWithOwner"].split("/", 1)[-1])
    artifact_root = output / "workorders"
    artifact_root.mkdir(parents=True, exist_ok=True)
    database = Path(args.ledger).resolve() if args.ledger else output / "mega_ledger.sqlite"
    if database.exists() and not args.ledger:
        raise ValueError(f"refusing to reuse an existing campaign ledger: {database}")
    sampler = _WalSampler(database); sampler.start()
    reports = []
    worktree_residuals = []
    lock_errors = 0
    analysis_file_count = 0
    for metadata in repositories:
        repo_name = metadata["nameWithOwner"]
        repo = source_root / repo_name.split("/", 1)[-1]
        if not (repo / ".git").exists():
            reports.append({"repo": repo_name, "status": "clone_missing", "work_orders": []})
            log(f"REPO {repo_name} | status=clone_missing | tickets=0")
            continue
        repo_started = time.monotonic()
        branch = _git(repo, "branch", "--show-current")
        default_head = _git(repo, "rev-parse", "HEAD")
        files = _code_files(repo)
        analysis = _analyze(repo, files)
        analysis_file_count += analysis["languages"]["tracked_source_files"]
        repo_output = artifact_root / re_slug(repo_name.split("/", 1)[-1])
        repo_output.mkdir(parents=True, exist_ok=True)
        profile = repo / ".aegis" / "repo_profile.json"
        attached = (_run([sys.executable, "-m", "aegis_core.cli", "attach", "--repo", repo],
                         cwd=repo, env=_aegis_env(), timeout=180) if not args.skip_batch else None)
        report = {"repo": repo_name, "url": metadata.get("url"), "private": metadata.get("isPrivate"),
                  "default_branch": metadata.get("defaultBranchRef", {}).get("name"),
                  "checkout_branch": branch, "default_head_before": default_head,
                  "analysis": analysis, "work_orders": [],
                  "attach_exit_code": attached.returncode if attached is not None else None}
        if not args.skip_autopilot:
            report["work_orders"].append({"ticket_id": "v14-swarm-" + re_slug(repo_name.split("/")[-1])
                                          + ("-" + re_slug(args.ticket_suffix) if args.ticket_suffix else ""),
                                          "mode": "aegis autopilot", "status": "queued",
                                          "work_order": "seeded_red_contract_and_candidate_tournament"})
        if attached is not None and attached.returncode:
            report["attach_stderr"] = attached.stderr.decode("utf-8", "replace")[-1200:]
        elif not args.skip_batch:
            try:
                target = _select_target(files, repo)
                for order_index, label in enumerate(("doc_contract", "dependency_handoff"), 1):
                    ticket = "v14-batch-" + re_slug(repo_name.split("/", 1)[-1]) + f"-{order_index}"
                    if args.ticket_suffix:
                        ticket += "-" + re_slug(args.ticket_suffix)
                    marker = f"AEGIS_V14_{re_slug(repo_name.split('/', 1)[-1]).upper()}_{label.upper()}"
                    patch_path = repo_output / f"{ticket}.patch"
                    patch_path.write_bytes(_patch_for_comment(target, marker))
                    record = _batch_command(repo, ticket, target, patch_path, marker, profile,
                                            database, repo_output)
                    record["work_order"] = label
                    record["elapsed_sec"] = round(time.monotonic() - repo_started, 4)
                    report["work_orders"].append(record)
                    if "database is locked" in (record.get("stderr_tail", "") + record.get("stdout_tail", "")).lower():
                        lock_errors += 1
            except BaseException as error:
                report["setup_error"] = {"type": type(error).__name__, "message": str(error)[:1500]}
        report["analysis_elapsed_sec"] = round(time.monotonic() - repo_started, 4)
        report["_repo_path"] = str(repo)
        report["_output_path"] = str(repo_output)
        reports.append(report)
        log(f"REPO {repo_name} | source_files={analysis['languages']['tracked_source_files']} "
            f"py={analysis['languages']['python']} js_ts_html={analysis['languages']['javascript_typescript_html']} "
            f"symbols={analysis['impact_graph']['symbols']} flows={analysis['semantic_contracts']['flows']} "
            f"batch_tickets={sum(item.get('mode') == 'aegis batch' for item in report['work_orders'])} "
            f"analysis={report['analysis_elapsed_sec']}s")

    log("AUTOPILOT | launching detached worktree campaigns with shared SQLite WAL ledger")
    autopilot_records = []
    tasks = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for report in reports:
            if args.skip_autopilot: continue
            repo_path = report.get("_repo_path")
            if not repo_path or report.get("setup_error"):
                continue
            repo = Path(repo_path)
            dataset = artifact_root / re_slug(report["repo"].split("/")[-1]) / "grpo_tournament_pairs.jsonl"
            dataset.parent.mkdir(parents=True, exist_ok=True)
            task = pool.submit(_autopilot_ticket, report["repo"], repo, database, dataset,
                               Path(report["_output_path"]), args.timeout, args.ticket_suffix or "")
            tasks.append((report, task))
        for report, future in tasks:
            try:
                result = future.result()
            except BaseException as error:
                result = {"ticket_id": "v14-swarm-" + re_slug(report["repo"].split("/")[-1]),
                          "mode": "aegis autopilot", "status": "execution_error",
                          "error_type": type(error).__name__, "error": str(error)[:2000]}
            for work_order in report["work_orders"]:
                if work_order.get("mode") == "aegis autopilot": work_order.update(result)
            autopilot_records.append(result)
            if "database is locked" in (result.get("stderr_tail", "") + result.get("stdout_tail", "")).lower():
                lock_errors += 1
            log(f"AUTOPILOT {report['repo']} | status={result.get('status')} winner={result.get('winner')} "
                f"rounds={result.get('rounds_attempted')} rejected={result.get('candidate_rejections')} "
                f"replay={result.get('replay_status')} elapsed={result.get('elapsed_sec')}s")

    wal_peak = sampler.stop()
    for report in reports:
        repo_path = report.pop("_repo_path", None)
        report.pop("_output_path", None)
        if repo_path:
            repo = Path(repo_path)
            report["default_head_after"] = _git(repo, "rev-parse", "HEAD")
            report["default_head_unchanged"] = report["default_head_before"] == report["default_head_after"]
            report["tracked_diff_after"] = _git(repo, "diff", "--name-only", "HEAD").splitlines()
            worktree_count = _git(repo, "worktree", "list", "--porcelain").count("worktree ")
            report["registered_worktrees_after"] = worktree_count
            if worktree_count != 1: worktree_residuals.append({"repo": report["repo"], "worktrees": worktree_count})
    db_stats = _db_summary(database)
    elapsed = round(time.monotonic() - started, 3)
    batch_rows = [row for report in reports for row in report.get("work_orders", []) if row.get("mode") == "aegis batch"]
    autopilot_rows = [row for report in reports for row in report.get("work_orders", []) if row.get("mode") == "aegis autopilot"]
    candidate_metrics = _candidate_metrics(autopilot_rows)
    successful_autopilot = candidate_metrics["succeeded_scenarios"]
    invalid_candidates = candidate_metrics["intentionally_invalid"]
    candidate_count = candidate_metrics["evaluations"]
    rejected = candidate_metrics["rejected"]
    findings = {}
    for row in autopilot_rows:
        for finding in row.get("audit_findings", []):
            key = str(finding.get("code") or finding.get("category") or finding.get("rule") or "unspecified")
            findings[key] = findings.get(key, 0) + 1
    summary = {"repositories_requested": len(repositories), "repositories_cloned_and_analyzed": sum(bool(row.get("analysis")) for row in reports),
               "work_orders": len(batch_rows) + len(autopilot_rows), "batch_work_orders": len(batch_rows),
               "autopilot_work_orders": len(autopilot_rows), "autopilot_succeeded": sum(row.get("status") == "succeeded" for row in autopilot_rows),
               "autopilot_winners": sum(bool(row.get("winner")) for row in autopilot_rows),
               "autopilot_rejected_candidates": rejected, "intentionally_invalid_candidates": invalid_candidates,
               "autopilot_candidate_evaluations": candidate_count,
               "candidate_red_light_interception_rate": candidate_metrics["interception_rate"],
               "all_remote_default_heads_unchanged": all(row.get("default_head_unchanged", False) for row in reports if row.get("analysis")),
               "tracked_worktree_residuals": worktree_residuals,
               "cross_language_ast_coverage_files": analysis_file_count,
               "high_risk_l3_findings": findings,
               "experience_bank_new_records": db_stats["experience_records"],
               "sqlite": {**db_stats, "wal_peak_bytes": wal_peak,
                          "trace_throughput_per_sec": round(db_stats["causal_traces"] / elapsed, 3) if elapsed else 0,
                          "database_lock_errors": lock_errors},
               "campaign_elapsed_sec": elapsed,
               "external_write_policy": "all source edits confined to temporary git worktrees; no push, merge or default-branch edit"}
    document = {"schema_version": "aegis.mega_campaign.v14", "account": args.account,
                "summary": summary, "repositories": reports,
                "autopilot": {"target_kind": "temporary tracked contract fixture inside each repository's detached worktree",
                              "purpose": "physically validate fail-red, tournament rejection, correct repair, WAL learning and aegis replay against each repository's Git object database",
                              "records": autopilot_records},
                "limits": ["Audit annotation batch patches and contract fixture commits are discarded after each isolated campaign."]}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    batch_passed = sum(row.get("status") == "succeeded" for row in batch_rows)
    log("\n21-REPOSITORY CAMPAIGN DECISION TABLE")
    log("REPOSITORY | LANG FILES | BATCH | AUTOPILOT | WINNER | REPLAY | HEAD CLEAN")
    for row in reports:
        languages = row.get("analysis", {}).get("languages", {})
        batch_count = sum(item.get("status") == "succeeded" for item in row.get("work_orders", []) if item.get("mode") == "aegis batch")
        auto = next((item for item in row.get("work_orders", []) if item.get("mode") == "aegis autopilot"), {})
        batch_cell = "skipped" if args.skip_batch else f"{batch_count}/2"
        auto_cell = "skipped" if args.skip_autopilot else auto.get('status', 'n/a')
        log(f"{row['repo']} | {languages.get('tracked_source_files',0)} | {batch_cell} | "
            f"{auto_cell} | {auto.get('winner','-')} | {auto.get('replay_status','-')} | "
            f"{row.get('default_head_unchanged','-')}")
    log("\nCAMPAIGN TOTALS")
    log(f"repositories={summary['repositories_cloned_and_analyzed']}/{summary['repositories_requested']} "
        f"work_orders={summary['work_orders']} (batch {batch_passed}/{len(batch_rows)}, "
        f"autopilot {summary['autopilot_succeeded']}/{len(autopilot_rows)})")
    log(f"AST-covered files={analysis_file_count} | rejected candidates={rejected}/{candidate_count} "
        f"({summary['candidate_red_light_interception_rate']}) | L3 findings={findings}")
    log(f"SQLite WAL={db_stats['journal_mode']} traces={db_stats['causal_traces']} "
        f"traces/sec={summary['sqlite']['trace_throughput_per_sec']} "
        f"peak_wal={wal_peak} bytes locks={lock_errors} experiences={db_stats['experience_records']}")
    log(f"default_heads_clean={summary['all_remote_default_heads_unchanged']} "
        f"worktree_residuals={len(worktree_residuals)} elapsed={elapsed}s")
    log(f"report={report_path}")
    (output / "stdout.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if temporary:
        temporary.cleanup()
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", default=ACCOUNT)
    parser.add_argument("--source-dir", help="Use already cloned disposable repositories; source checkouts are not removed")
    parser.add_argument("--output", default=str(ROOT / "artifacts/mega_campaign_v14"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--repos", nargs="+", help="Optionally run a targeted subset by repository name or owner/name")
    parser.add_argument("--ledger", help="Explicit existing SQLite WAL ledger for a targeted incremental retry")
    parser.add_argument("--ticket-suffix", help="Unique suffix required when appending retry tickets to an existing ledger")
    parser.add_argument("--skip-batch", action="store_true", help="Skip attach and source annotation tickets")
    parser.add_argument("--skip-autopilot", action="store_true", help="Skip fixture repair and replay tickets")
    parser.add_argument("--analyze-only", action="store_true", help="Run the read-only 21-repository parser audit only")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8: parser.error("--workers must be in 1..8")
    if not 60 <= args.timeout <= 900: parser.error("--timeout must be in 60..900")
    if args.analyze_only: args.skip_batch = args.skip_autopilot = True
    if args.skip_batch and args.skip_autopilot and not args.analyze_only: parser.error("at least one campaign mode must run")
    if args.ledger and not args.ticket_suffix: parser.error("--ledger retries require --ticket-suffix to avoid ticket collisions")
    result = _campaign(args)
    return 0 if result["summary"]["repositories_cloned_and_analyzed"] == result["summary"]["repositories_requested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
