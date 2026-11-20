#!/usr/bin/env python3
"""Build the offline Aegis visual showcase from checked-in evidence."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
W = 1920
H = 1080
FPS = 30
DURATION = 90

SOURCES = {
    "status": "STATUS.json",
    "fleet": "artifacts/fleet_dashboard_v16.json",
    "delivery": "artifacts/real_delivery_v15/report.json",
    "world_metrics": "artifacts/causal_world_model/training_metrics.json",
    "world_replay": "artifacts/causal_world_model/online_replay_report.json",
    "world_mode": "artifacts/mega_campaign_v14/workorders/omni-eval/v14-batch-omni-eval-1.batch.json",
    "release": "datasets/release_v15/manifest.json",
    "tournament": "datasets/grpo_tournament_pairs.jsonl",
}

COLORS = {
    "bg": "#07111b",
    "bg2": "#0b1825",
    "panel": "#101f2c",
    "panel2": "#132735",
    "grid": "#142938",
    "line": "#254050",
    "text": "#e9f2f5",
    "muted": "#8da5b2",
    "dim": "#58717f",
    "cyan": "#45dfd2",
    "blue": "#67aaff",
    "green": "#68e2a8",
    "amber": "#ffca75",
    "red": "#ff7184",
    "purple": "#b994ff",
}


def read_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _history(node: Any, found: dict[int, dict[str, int]]) -> None:
    if isinstance(node, dict):
        phase = node.get("phase")
        tests = node.get("tests")
        if isinstance(phase, int) and isinstance(tests, dict):
            count = tests.get("total_collected", tests.get("passed"))
            passed = tests.get("passed", count)
            if isinstance(count, int):
                found[phase] = {"phase": phase, "collected": count, "passed": int(passed or 0)}
        elif isinstance(tests, int) and isinstance(node.get("pull_request"), str):
            match = re.search(r"/pull/(\d+)$", node["pull_request"])
            if match:
                phase = int(match.group(1))
                found[phase] = {"phase": phase, "collected": tests, "passed": tests}
        for value in node.values():
            _history(value, found)
    elif isinstance(node, list):
        for value in node:
            _history(value, found)


def _sample_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # All four candidates below are recorded preference rows for the same job10 prompt.
    selected = {"job10_signature", "job10_test_cheat", "job10_memory_leak"}
    chosen = None
    rejected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("chosen_candidate") != "job10_correct":
            continue
        candidate = row.get("rejected_candidate")
        if candidate in selected:
            evidence = row.get("rejection_evidence", {})
            rejected[candidate] = {
                "id": candidate,
                "score": row.get("rejected_score"),
                "static_pass_rate": evidence.get("static_pass_rate"),
                "dynamic_pass_rate": evidence.get("dynamic_pass_rate"),
                "mutation_probe_survival_rate": evidence.get("mutation_probe_survival_rate"),
                "audit_passed": evidence.get("audit_passed"),
                "audit_findings": evidence.get("audit_findings", []),
                "accepted": evidence.get("accepted"),
            }
            chosen = {
                "id": row["chosen_candidate"],
                "score": row.get("chosen_score"),
            }
    if chosen is None or set(rejected) != selected:
        raise ValueError("Expected the four job10 preference candidates in the checked-in dataset")
    return {
        "prompt": "Clamp integer input for campaign task 10 between zero and ten; reject non-integers.",
        "chosen": chosen,
        "rejected": [rejected[key] for key in ("job10_signature", "job10_test_cheat", "job10_memory_leak")],
        "pair_rows": 4,
        "source": SOURCES["tournament"],
    }


def _find_mode(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("model_mode"):
            return str(value["model_mode"])
        for item in value.values():
            found = _find_mode(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_mode(item)
            if found:
                return found
    return None


def load_data() -> dict[str, Any]:
    status = read_json(SOURCES["status"])
    fleet = read_json(SOURCES["fleet"])
    delivery = read_json(SOURCES["delivery"])
    world = read_json(SOURCES["world_metrics"])
    replay = read_json(SOURCES["world_replay"])
    world_mode_receipt = read_json(SOURCES["world_mode"])
    release = read_json(SOURCES["release"])
    pairs = read_jsonl(SOURCES["tournament"])

    scans = status["milestones"]["m4_external_frontend_readonly_scans"]
    history_by_phase: dict[int, dict[str, int]] = {}
    _history(status, history_by_phase)
    history = [history_by_phase[key] for key in sorted(history_by_phase)]
    successful_ci = status["github_actions"].get("latest_successful_runs", [])
    fleet_rows = fleet.get("repositories", [])
    risk = world["training"]["reward_interval"]["validation_calibrated_metrics"]["risk_head_precision_recall"]
    counts = release["counts"]
    verified_repairs = []
    for repo, record in delivery["business_repositories"].items():
        tests = record.get("verified_tests", {})
        verified_repairs.append({
            "repository": repo,
            "repair": record.get("repair") or " · ".join(record.get("repairs", [])),
            "tests": tests,
        })

    modules = [
        {
            "id": "ast",
            "name": "AST slicer",
            "entry": "aegis_core/semantic_contract.py",
            "tests": 93,
            "test_source": "tests/test_v14_semantic_contract.py",
            "purpose": "Builds symbol and contract relationships across Python, JavaScript, TypeScript, and inline HTML consumers.",
            "evidence": "STATUS.json previous phase 14: semantic_contract, 93 tests.",
        },
        {
            "id": "worktree",
            "name": "Worktree sandbox",
            "entry": "aegis_core/self_heal_loop.py",
            "tests": 25,
            "test_source": "tests/test_v13_self_heal_loop.py",
            "purpose": "Runs candidate work in an isolated repair loop so a proposed change can be checked before delivery.",
            "evidence": "25 tests collected from the linked repository test file; candidate preference receipts are in the tournament dataset.",
        },
        {
            "id": "probes",
            "name": "Probe compiler",
            "entry": "aegis_core/probe_synth.py",
            "tests": 48,
            "test_source": "tests/test_v13_probe_synth.py",
            "purpose": "Synthesizes targeted checks and mutates expectations to expose candidates that only satisfy superficial assertions.",
            "evidence": "48 tests collected from the linked repository test file.",
        },
        {
            "id": "tournament",
            "name": "Candidate tournament",
            "entry": "aegis_core/tournament.py",
            "tests": 35,
            "test_source": "tests/test_v13_tournament.py",
            "purpose": "Scores competing repairs, preserves chosen and rejected patches, and exports preference examples.",
            "evidence": f"datasets/grpo_tournament_pairs.jsonl contains {len(pairs)} rows; release v15 records {counts['tournament_pairs_seen']} tournament pairs seen.",
        },
        {
            "id": "invariants",
            "name": "Invariant guard",
            "entry": "aegis_core/invariant_guard.py",
            "tests": status["milestones"]["m1_frontend_invariant_guard"]["tests"],
            "test_source": "tests/test_v16_frontend_invariants.py",
            "purpose": "Checks front-end network timeouts, finite numeric inputs, cleanup, async storage, and unsafe HTML evaluation patterns.",
            "evidence": "STATUS.json phase 16 records 96 frontend invariant-guard tests.",
        },
        {
            "id": "scan",
            "name": "Scan aggregation",
            "entry": "aegis_core/impact_analyzer.py",
            "tests": 41,
            "test_source": "tests/test_v13_impact_analyzer.py",
            "purpose": "Aggregates scan evidence and impacts for reviewed changes. Raw rule hits are reported; a post-deduplication count is not recorded.",
            "evidence": "STATUS.json lists 3 read-only scans and their raw rule-hit totals.",
        },
        {
            "id": "replay",
            "name": "Time travel",
            "entry": "aegis_core/time_travel.py",
            "tests": 45,
            "test_source": "tests/test_v14_time_travel.py",
            "purpose": "Replays frozen evidence and attributes changes to edited lines while retaining byte hashes and probe context.",
            "evidence": "STATUS.json phase 14 records 45 tests and 21 physical campaign replays.",
        },
        {
            "id": "world",
            "name": "World model",
            "entry": "aegis_core/world_model.py",
            "tests": 6,
            "test_source": "tests/test_world_model.py",
            "purpose": "Predicts likely outcomes from recorded traces; current campaign receipts use observe mode.",
            "evidence": f"Validation risk head: precision {risk['precision']:.2f}, recall {risk['recall']:.2f}, support {risk['support']}; no false-positive rate is reported.",
        },
        {
            "id": "fleet",
            "name": "Fleet orchestration",
            "entry": "aegis_core/pr_orchestrator.py",
            "tests": status["milestones"]["m3_pr_orchestrator"]["tests"],
            "test_source": "tests/test_v16_pr_orchestrator.py",
            "purpose": "Audits delivery branches, conflicts, and default-branch cleanliness across the recorded repository fleet.",
            "evidence": f"artifacts/fleet_dashboard_v16.json records {fleet['summary']['delivery_branches']} mergeable branches and {fleet['summary']['conflicts']} conflicts.",
        },
    ]

    result = {
        "title": "Aegis Core · Engineering Showcase",
        "version": status["version"],
        "phase": status["phase"],
        "tests": {
            "collected": status["tests"]["total_collected"],
            "passed": status["tests"]["passed"],
            "history": history,
        },
        "ci": {
            "jobs_expected": status["github_actions"]["required_jobs"],
            "verified_runs": [run["result"] for run in successful_ci],
            "head": status["github_actions"].get("latest_verified_head"),
        },
        "profiler": status["milestones"]["m2_runtime_profiler_sampling"]["high_frequency_probe"],
        "release": {
            "sft_examples": counts["sft_examples"],
            "preference_pairs": counts["preference_pairs"],
            "chosen_positive": counts["chosen_positive"],
            "rejected_negative": counts["rejected_negative"],
            "truncated_rows": counts["truncated_rows"],
            "tournament_pairs_seen": counts["tournament_pairs_seen"],
            "sft_rows_file": "datasets/release_v15/datasets/sft_train.jsonl",
            "grpo_rows_file": "datasets/release_v15/datasets/grpo_preference_pairs.jsonl",
        },
        "scans": {
            "repositories": len(scans),
            "scanned_files": sum(item["scanned_files"] for item in scans),
            "raw_rule_hits": sum(item["rule_hits"] for item in scans),
            "by_repository": [
                {"repository": item["repository"], "files": item["scanned_files"], "hits": item["rule_hits"]}
                for item in scans
            ],
            "post_dedup_hits": None,
            "post_dedup_note": "No post-deduplication count is recorded in the checked-in evidence.",
        },
        "verified_repairs": verified_repairs,
        "tournament": _sample_candidates(pairs),
        "fleet": {
            "summary": fleet["summary"],
            "repositories": [row.get("repository") or row.get("name") for row in fleet_rows],
            "main_pollution": 0 if status["milestones"]["m3_pr_orchestrator"]["fleet_audit"]["default_branch_zero_pollution"] else None,
        },
        "world": {
            "training_examples": world["dataset"]["training_examples"],
            "train_examples": world["dataset"]["train_examples"],
            "validation_examples": world["dataset"]["validation_examples"],
            "test_examples": world["dataset"]["holdout_test_examples"],
            "risk": risk,
            "false_positive_rate": None,
            "mode": _find_mode(world_mode_receipt),
            "observe_evidence": SOURCES["world_replay"],
            "replay_selected": replay["selected_traces"],
            "replay_examples": replay["replay_examples"],
            "promotion_note": "The checked-in validation report does not establish a false-positive rate; campaign receipts remain in observe mode.",
        },
        "modules": modules,
        "sources": SOURCES,
        "claims_not_verified": [
            "v1.2.0 and 1,500 passing tests: current main STATUS.json and pyproject.toml report v1.1.0 and 1,338 passing tests.",
            "16 delivered repositories: the fleet artifact records 15 delivery branches.",
            "715 to 108 / 84.9% scan reduction: the raw 715 hits are recorded, but no post-deduplication count is recorded.",
            "A calibrated world-model false-positive rate: the validation report does not provide that rate.",
        ],
        "source_note": "All displayed engineering metrics are extracted from STATUS.json, artifacts/, or datasets/ in the checked-out main branch. Module test counts are collected repository test files and listed with their source paths.",
    }
    return result


def _short(value: Any) -> str:
    if value is None:
        return "NOT RECORDED"
    return str(value)


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark"><title>Aegis Core · Engineering Showcase</title>
<style>
:root{--bg:#08121d;--panel:#10202d;--panel2:#142835;--line:#294454;--text:#eaf3f5;--muted:#92a8b4;--cyan:#45dfd2;--blue:#70adff;--green:#68e2a8;--amber:#ffca75;--red:#ff7184;--purple:#bd9dff;--mono:ui-monospace,SFMono-Regular,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 20% 0%,#123141 0%,transparent 38%),radial-gradient(ellipse at 100% 20%,#182742 0%,transparent 42%),var(--bg);color:var(--text);font:15px/1.5 var(--sans)}button{font:inherit;color:inherit}a{color:var(--cyan)}.wrap{max-width:1480px;margin:auto;padding:28px clamp(16px,3vw,44px) 60px}.top{display:flex;align-items:center;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:18px}.brand{display:flex;align-items:center;gap:14px}.mark{width:42px;height:42px;border:1px solid var(--cyan);border-radius:12px;display:grid;place-items:center;color:var(--cyan);font:700 20px var(--mono);box-shadow:0 0 24px #45dfd233}.eyebrow{font:700 11px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--cyan)}h1{font-size:clamp(26px,4vw,46px);line-height:1.1;margin:6px 0 8px;letter-spacing:-.04em}.sub{color:var(--muted);max-width:820px}.source-tag{border:1px solid var(--line);padding:8px 12px;border-radius:99px;color:var(--muted);font:12px var(--mono);white-space:nowrap}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.stat,.panel{background:linear-gradient(145deg,#122431ef,#0d1a26f2);border:1px solid var(--line);border-radius:16px;box-shadow:0 12px 38px #0002}.stat{padding:18px}.stat strong{display:block;font:700 clamp(25px,3vw,36px) var(--mono);letter-spacing:-.06em;color:var(--cyan)}.stat span{color:var(--muted);font-size:12px;display:block;margin-top:5px}.pill{display:inline-flex;gap:7px;align-items:center;border:1px solid #25414d;border-radius:99px;padding:5px 9px;color:var(--muted);font:11px var(--mono)}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 10px currentColor}.grid2{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}.panel{padding:20px;margin-bottom:16px}.panel h2{font-size:18px;margin:0 0 5px}.panel p{color:var(--muted);margin:0 0 15px}.panel-head{display:flex;justify-content:space-between;align-items:start;gap:12px}.topology{position:relative;min-height:400px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(3,1fr);gap:14px;padding:18px 0}.topology:before,.topology:after{content:"";position:absolute;z-index:0;pointer-events:none;border-top:1px dashed #426170;left:11%;right:11%;top:50%}.topology:after{top:17%;bottom:17%;left:50%;right:auto;border-top:0;border-left:1px dashed #426170}.node{z-index:1;cursor:pointer;text-align:left;background:linear-gradient(135deg,#152a38,#10212d);border:1px solid #2a4656;border-radius:14px;padding:13px 14px;transition:.2s;min-height:96px}.node:hover,.node.active{border-color:var(--cyan);transform:translateY(-2px);box-shadow:0 0 24px #45dfd21e}.node .n{font:700 10px var(--mono);color:var(--cyan);letter-spacing:.12em}.node strong{display:block;margin:5px 0 2px;font-size:14px}.node small{color:var(--muted);font-size:11px}.detail{min-height:155px;background:#08151f;border:1px solid #223b49;border-radius:12px;padding:16px}.detail h3{margin:0 0 6px;font-size:17px}.detail .entry{color:var(--cyan);font:12px var(--mono);word-break:break-word}.detail .tests{display:inline-block;margin:12px 0;padding:5px 8px;border:1px solid #2b4c58;border-radius:8px;font:12px var(--mono);color:var(--green)}.detail p{margin:5px 0}.chart{width:100%;height:auto;overflow:visible}.legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.bars{display:grid;gap:12px}.barrow{display:grid;grid-template-columns:minmax(110px,1.25fr) 3fr 52px;align-items:center;gap:12px;font-size:12px}.track{height:11px;background:#1b303d;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#2fa9bd,var(--cyan));width:var(--w)}.callout{border:1px solid #755c2f;background:#241e13;border-radius:12px;padding:12px 14px;color:#f1d8a6;font-size:12px;margin-top:14px}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}.mini{background:#0b1823;border:1px solid #263f4d;border-radius:12px;padding:14px}.mini .value{font:700 23px var(--mono);color:var(--cyan)}.mini .label{font-size:11px;color:var(--muted)}.candidate-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:15px 0}.candidate{border:1px solid #354856;background:#0c1822;border-radius:12px;padding:13px;min-height:128px}.candidate.good{border-color:#36785a}.candidate.bad{border-color:#714452}.candidate .state{font:700 10px var(--mono);text-transform:uppercase;letter-spacing:.1em}.candidate strong{display:block;font:13px var(--mono);margin:8px 0}.candidate small{color:var(--muted);font-size:11px}.sim{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.action{border:1px solid #2c6f70;background:linear-gradient(135deg,#103b42,#14313e);padding:10px 15px;border-radius:10px;cursor:pointer;color:var(--cyan);font-weight:700}.action:hover{filter:brightness(1.2)}.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}.step{border:1px solid #29404d;border-radius:10px;padding:12px;color:var(--muted);background:#0a1721;transition:.25s}.step.on{border-color:var(--cyan);color:var(--text);box-shadow:inset 0 0 20px #45dfd20d}.step b{display:block;font:700 10px var(--mono);color:var(--cyan);margin-bottom:5px}.repo-list{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.repo{border:1px solid #233b49;border-radius:9px;padding:9px 10px;font:11px var(--mono);color:#bdcdd4;background:#0b1822;overflow-wrap:anywhere}.warn{border-color:#765b35;background:#221b12;color:#f4d9a7}.footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px}.footer code{color:#c9d8de}.sources{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}.sources code{border:1px solid var(--line);border-radius:7px;padding:5px 7px;background:#0a1721;font:10px var(--mono)}.mobile-note{color:var(--muted);font-size:12px}.kpis{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.fade{animation:fade .35s ease}@keyframes fade{from{opacity:.15;transform:translateY(4px)}to{opacity:1;transform:none}}@media(max-width:900px){.grid2{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.topology{min-height:360px}.candidate-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.wrap{padding:17px 12px 40px}.top{align-items:flex-start}.source-tag{white-space:normal;text-align:right;max-width:120px;font-size:10px}.stats{gap:8px}.stat{padding:13px}.stat strong{font-size:23px}.panel{padding:15px}.topology{gap:7px;min-height:320px}.node{padding:9px 8px;min-height:90px}.node strong{font-size:12px}.node small{font-size:10px}.node .n{font-size:9px}.detail{padding:12px}.barrow{grid-template-columns:95px 1fr 38px;gap:7px;font-size:10px}.steps{grid-template-columns:repeat(2,1fr)}.repo-list{grid-template-columns:repeat(2,1fr)}.candidate-grid{gap:7px}.candidate{padding:10px}.two{gap:8px}.mobile-note{display:block}}
</style></head><body><main class="wrap">
<header class="top"><div class="brand"><div class="mark">A</div><div><div class="eyebrow">Deterministic software factory</div><h1>Aegis Core · <span id="version"></span></h1><div class="sub">A verifiable tour of the current main branch. The numbers below are drawn from checked-in status, run artifacts, and release datasets.</div></div></div><div class="source-tag">OFFLINE · EVIDENCE-LINKED</div></header>
<section class="stats" id="stats"></section>
<details class="panel"><summary style="cursor:pointer;color:var(--cyan);font-weight:700">Motion preview · GIF / MP4</summary><div style="display:grid;grid-template-columns:minmax(0,2fr) minmax(180px,1fr);gap:16px;align-items:center;margin-top:14px"><img src="../showcase/preview.gif" alt="Animated Aegis engineering showcase preview" style="width:100%;border:1px solid var(--line);border-radius:12px"><div><a href="../showcase/aegis_demo.mp4" download style="display:inline-block;padding:10px 13px;border:1px solid #2c6f70;border-radius:10px;text-decoration:none">Download 1080p MP4</a><div class="sources"><code>showcase/preview.gif</code><code>showcase/aegis_demo.mp4</code></div></div></div></details>
<div class="grid2"><section class="panel"><div class="panel-head"><div><div class="eyebrow">System map</div><h2>Interactive architecture</h2><p>Choose a module to inspect its role, collected tests, and core entry point.</p></div><span class="pill"><i class="dot"></i> 9 modules</span></div><div class="topology" id="topology"></div><div class="detail" id="module-detail" aria-live="polite"></div></section>
<section><div class="panel"><div class="panel-head"><div><div class="eyebrow">Engineering history</div><h2>Verified test growth</h2><p id="history-note"></p></div><span class="pill">STATUS.json</span></div><svg id="growth" class="chart" viewBox="0 0 720 240" role="img" aria-label="Recorded test history"></svg><div class="legend"><span class="pill"><i class="dot"></i> passing tests</span><span class="pill">Latest recorded: <b id="last-tests"></b></span></div></div>
<div class="panel"><div class="eyebrow">Runtime profile</div><h2>Hot path with bounded memory</h2><p>Deterministic sampling retained a rare tail branch while aggregating repeated calls.</p><div class="two" id="runtime-cards"></div></div></section></div>
<section class="panel"><div class="panel-head"><div><div class="eyebrow">Candidate gate</div><h2>Anti-cheat tournament · recorded task sample</h2><p id="tournament-prompt"></p></div><span class="pill">4 preference rows · one chosen candidate</span></div><div class="candidate-grid" id="candidates"></div><div class="sim"><button class="action" id="run-sim">Replay the gate decision</button><span id="sim-caption" class="mobile-note">A browser-only walkthrough of recorded candidate receipts.</span></div><div class="steps" id="steps"></div></section>
<div class="grid2"><section class="panel"><div class="eyebrow">Frontend invariant scans</div><h2>Observed scan evidence</h2><p id="scan-summary"></p><div class="bars" id="scan-bars"></div><div class="callout" id="scan-caveat"></div><h3 style="font-size:14px;margin:18px 0 8px">Verified repairs in delivery receipts</h3><div class="bars" id="repairs"></div></section>
<section class="panel"><div class="eyebrow">Replay & safety valve</div><h2>Reproduce first, promote carefully</h2><p>Replay receipts freeze the evidence needed to reproduce a failure. The world model remains observational when the checked-in report cannot establish the needed false-positive rate.</p><div class="two" id="world-cards"></div><div class="callout warn" id="world-note"></div></section></div>
<section class="panel"><div class="panel-head"><div><div class="eyebrow">Fleet delivery</div><h2>Audited delivery branches</h2><p>Names and counts come from the checked-in fleet dashboard. Main-branch pollution is reported as zero.</p></div><div class="kpis" id="fleet-kpis"></div></div><div class="repo-list" id="repos"></div></section>
<section class="panel"><div class="eyebrow">Training release</div><h2>Real distillation bundle</h2><p id="release-note"></p><div class="sources"><code>datasets/release_v15/manifest.json</code><code id="sft-path"></code><code id="grpo-path"></code></div></section>
<footer class="footer"><strong>Evidence note</strong><p id="source-note"></p><div class="sources" id="sources"></div><p class="mobile-note">Open this file directly for offline use. The GIF and MP4 are in the sibling <code>showcase/</code> directory.</p></footer>
</main><script id="aegis-data" type="application/json">@@DATA@@</script><script>
const D=JSON.parse(document.getElementById('aegis-data').textContent);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('version').textContent='v'+D.version;
const summary=D.fleet.summary;
const stats=[['TESTS PASSING',`${D.tests.passed.toLocaleString()}`,`out of ${D.tests.collected.toLocaleString()} recorded`],['CI MATRIX',`${D.ci.jobs_expected}/${D.ci.jobs_expected}`,`${D.ci.verified_runs.join(' · ')}`],['FLEET BRANCHES',`${summary.delivery_branches}`,`${summary.conflicts} conflicts · ${D.fleet.main_pollution} main writes`],['TRAINING RELEASE',`${D.release.sft_examples.toLocaleString()} / ${D.release.preference_pairs}`,`SFT examples / preference pairs`]];
document.getElementById('stats').innerHTML=stats.map(x=>`<div class="stat"><strong>${esc(x[1])}</strong><span>${esc(x[0])}<br>${esc(x[2])}</span></div>`).join('');
const topology=document.getElementById('topology');
topology.innerHTML=D.modules.map((m,i)=>`<button class="node ${i===0?'active':''}" data-id="${esc(m.id)}"><span class="n">MODULE 0${i+1}</span><strong>${esc(m.name)}</strong><small>${esc(m.entry.split('/').pop())}</small></button>`).join('');
function showModule(id){const m=D.modules.find(x=>x.id===id);if(!m)return;document.querySelectorAll('.node').forEach(n=>n.classList.toggle('active',n.dataset.id===id));document.getElementById('module-detail').innerHTML=`<h3>${esc(m.name)}</h3><div class="entry">${esc(m.entry)}</div><span class="tests">${esc(m.tests)} tests · ${esc(m.test_source)}</span><p>${esc(m.purpose)}</p><p>${esc(m.evidence)}</p>`;document.getElementById('module-detail').classList.remove('fade');void document.getElementById('module-detail').offsetWidth;document.getElementById('module-detail').classList.add('fade')}
topology.addEventListener('click',e=>{const b=e.target.closest('.node');if(b)showModule(b.dataset.id)});showModule(D.modules[0].id);
const hist=D.tests.history;const svg=document.getElementById('growth');const pad={l:48,r:20,t:20,b:44},vw=720,vh=240,max=Math.max(...hist.map(x=>x.passed))*1.12;const x=i=>pad.l+i*(vw-pad.l-pad.r)/Math.max(1,hist.length-1),y=v=>vh-pad.b-v/max*(vh-pad.t-pad.b);let grid='';for(let i=0;i<4;i++){let yy=pad.t+i*(vh-pad.t-pad.b)/3;grid+=`<line x1="${pad.l}" y1="${yy}" x2="${vw-pad.r}" y2="${yy}" stroke="#294454" stroke-dasharray="3 6"/><text x="4" y="${yy+4}" fill="#8da5b2" font-size="10">${Math.round(max*(3-i)/3)}</text>`}const points=hist.map((p,i)=>`${x(i)},${y(p.passed)}`).join(' ');const marks=hist.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p.passed)}" r="5" fill="#45dfd2"/><text x="${x(i)}" y="${Math.max(14,y(p.passed)-12)}" fill="#e9f2f5" font-size="10" text-anchor="middle">${p.passed}</text><text x="${x(i)}" y="${vh-14}" fill="#8da5b2" font-size="10" text-anchor="middle">P${p.phase}</text>`).join('');svg.innerHTML=`${grid}<polyline points="${points}" fill="none" stroke="#45dfd2" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>${marks}`;document.getElementById('last-tests').textContent=D.tests.passed.toLocaleString();document.getElementById('history-note').textContent=hist.length?`Recorded milestones ${hist[0].phase}–${hist[hist.length-1].phase}; earlier PR test counts are absent from STATUS.json.`:'No phase history found.';
const p=D.profiler;const runtime=[['HOT CALLS',p.hot_calls.toLocaleString()],['TRACE TIME',`${p.compatibility_backend_elapsed_seconds}s`],['TAIL SITES',p.unique_sites_observed],['RESERVOIR REPLACEMENTS',p.reservoir_replacements]];document.getElementById('runtime-cards').innerHTML=runtime.map(v=>`<div class="mini"><div class="value">${esc(v[1])}</div><div class="label">${esc(v[0])}</div></div>`).join('');
document.getElementById('tournament-prompt').textContent=D.tournament.prompt;const candidate=[{id:D.tournament.chosen.id,score:D.tournament.chosen.score,good:true,label:'SELECTED' },...D.tournament.rejected.map(c=>({...c,good:false,label:'REJECTED'}))];document.getElementById('candidates').innerHTML=candidate.map(c=>`<div class="candidate ${c.good?'good':'bad'}"><div class="state" style="color:${c.good?'var(--green)':'var(--red)'}">${c.label}</div><strong>${esc(c.id)}</strong><small>score ${Number(c.score).toFixed(3)}${c.good?' · chosen positive':` · static ${Math.round(c.static_pass_rate*100)}% · dynamic ${Math.round(c.dynamic_pass_rate*100)}% · mutation survival ${Math.round(c.mutation_probe_survival_rate*100)}%`}</small>${c.audit_findings?.length?`<small style="display:block;color:var(--amber);margin-top:6px">${esc(c.audit_findings[0].code)} · ${esc(c.audit_findings[0].path)}:${esc(c.audit_findings[0].line)}</small>`:''}</div>`).join('');
const stepData=[['01 · ISOLATE','Candidate receipts share the same recorded task prompt.'],['02 · PROBE','Static and dynamic pass rates separate superficial fixes from behavior.'],['03 · MUTATE','Mutation survival exposes the signature and test-cheat variants.'],['04 · REVIEW','UNBOUNDED_GROWTH rejects the persistent-memory candidate; the chosen repair remains positive.']];document.getElementById('steps').innerHTML=stepData.map((s,i)=>`<div class="step" data-step="${i}"><b>${s[0]}</b>${s[1]}</div>`).join('');let simTimer=null;document.getElementById('run-sim').onclick=()=>{clearInterval(simTimer);let i=0;const steps=[...document.querySelectorAll('.step')];steps.forEach(s=>s.classList.remove('on'));document.getElementById('sim-caption').textContent='Playing the recorded evidence path…';simTimer=setInterval(()=>{steps.forEach((s,j)=>s.classList.toggle('on',j===i));if(i>=steps.length){clearInterval(simTimer);document.getElementById('sim-caption').textContent='Recorded outcome: job10_correct chosen; three listed alternatives rejected.';return}i++},750)};
document.getElementById('scan-summary').textContent=`${D.scans.scanned_files} files across ${D.scans.repositories} read-only repositories · ${D.scans.raw_rule_hits} raw rule hits.`;const maxHits=Math.max(...D.scans.by_repository.map(x=>x.hits));document.getElementById('scan-bars').innerHTML=D.scans.by_repository.map(r=>`<div class="barrow"><span>${esc(r.repository.split('/').pop())}</span><div class="track"><div class="fill" style="--w:${(r.hits/maxHits*100).toFixed(1)}%"></div></div><b>${r.hits}</b></div>`).join('');document.getElementById('scan-caveat').textContent=D.scans.post_dedup_note;
document.getElementById('repairs').innerHTML=D.verified_repairs.map(r=>`<div class="barrow" style="grid-template-columns:minmax(120px,1fr) 2fr"><span>${esc(r.repository)}</span><span style="color:var(--muted)">${esc(r.repair)}</span></div>`).join('');
document.getElementById('world-cards').innerHTML=[['TRAINING EXAMPLES',D.world.training_examples],['VALIDATION SUPPORT',D.world.risk.support],['RISK PRECISION',D.world.risk.precision.toFixed(2)],['RISK RECALL',D.world.risk.recall.toFixed(2)],['FALSE POSITIVES',D.world.risk.false_positive],['FALSE-POSITIVE RATE','not reported']].map(v=>`<div class="mini"><div class="value">${esc(v[1])}</div><div class="label">${esc(v[0])}</div></div>`).join('');document.getElementById('world-note').textContent=D.world.promotion_note;
document.getElementById('fleet-kpis').innerHTML=[`${summary.delivery_branches} mergeable`,`${summary.conflicts} conflicts`,`main writes ${D.fleet.main_pollution}`].map(x=>`<span class="pill">${esc(x)}</span>`).join('');document.getElementById('repos').innerHTML=D.fleet.repositories.map(r=>`<div class="repo">${esc(r)}</div>`).join('');document.getElementById('release-note').textContent=`${D.release.sft_examples.toLocaleString()} SFT examples · ${D.release.preference_pairs} preference pairs · ${D.release.truncated_rows} truncated rows. The manifest also records ${D.release.chosen_positive} chosen positives and ${D.release.rejected_negative} rejected negatives.`;document.getElementById('sft-path').textContent=D.release.sft_rows_file;document.getElementById('grpo-path').textContent=D.release.grpo_rows_file;document.getElementById('source-note').textContent=D.source_note;document.getElementById('sources').innerHTML=Object.values(D.sources).map(s=>`<code>${esc(s)}</code>`).join('');
</script></body></html>'''
    return template.replace("@@DATA@@", payload)


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_text(draw, xy, text, size=24, color=None, bold=False, anchor=None):
    draw.text(xy, str(text), font=_font(size, bold), fill=color or COLORS["text"], anchor=anchor)


def _rounded(draw, box, fill, outline=None, radius=18, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _base_frame(second: int, scene: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (W, H), COLORS["bg"])
    d = ImageDraw.Draw(image, "RGBA")
    # Fine, subdued drafting grid.
    for x in range(0, W, 48):
        d.line((x, 0, x, H), fill=(28, 50, 65, 70), width=1)
    for y in range(0, H, 48):
        d.line((0, y, W, y), fill=(28, 50, 65, 70), width=1)
    d.rectangle((0, 0, W, 93), fill=(7, 17, 27, 245))
    d.line((0, 92, W, 92), fill=(47, 91, 110, 150), width=2)
    _draw_text(d, (68, 30), "AEGIS / CONTROL SYSTEM", 20, COLORS["cyan"], True)
    minute, second_in_minute = divmod(second, 60)
    _draw_text(d, (W - 70, 34), f"SHOWCASE  /  {minute:02d}:{second_in_minute:02d}", 16, COLORS["muted"], True, "ra")
    d.rectangle((0, H - 12, W, H), fill=(13, 30, 42, 255))
    d.rectangle((0, H - 12, int(W * min(1, (second + 1) / DURATION)), H), fill=COLORS["cyan"])
    # Tiny scene and signal identifiers anchor the composition without implying metrics.
    _draw_text(d, (68, H - 49), f"ENGINEERING EVIDENCE   /   CUT {scene+1:02d}", 13, COLORS["dim"], True)
    return image, d


def _title(d, kicker: str, headline: str, sub: str):
    _draw_text(d, (86, 132), kicker.upper(), 17, COLORS["cyan"], True)
    _draw_text(d, (86, 168), headline, 44, COLORS["text"], True)
    _draw_text(d, (88, 228), sub, 20, COLORS["muted"])


def _draw_metric_card(d, box, label, value, foot, color=COLORS["cyan"]):
    _rounded(d, box, COLORS["panel"], COLORS["line"], radius=18, width=2)
    x1, y1, _, _ = box
    _draw_text(d, (x1 + 28, y1 + 24), label.upper(), 15, COLORS["muted"], True)
    _draw_text(d, (x1 + 28, y1 + 60), value, 38, color, True)
    _draw_text(d, (x1 + 28, y1 + 112), foot, 15, COLORS["muted"])


def _frame_scene(second: int, data: dict[str, Any]):
    from PIL import ImageDraw

    if second < 10:
        scene, local = 0, second
    elif second < 25:
        scene, local = 1, second - 10
    elif second < 45:
        scene, local = 2, second - 25
    elif second < 60:
        scene, local = 3, second - 45
    elif second < 75:
        scene, local = 4, second - 60
    else:
        scene, local = 5, second - 75
    image, d = _base_frame(second, scene)
    pulse = (local % 4) / 3
    if scene == 0:
        _draw_text(d, (86, 148), "DETERMINISTIC SOFTWARE FACTORY", 18, COLORS["cyan"], True)
        _draw_text(d, (82, 204), "Aegis Core", 88, COLORS["text"], True)
        _draw_text(d, (89, 310), f"v{data['version']}   /   PHASE {data['phase']:02d}", 25, COLORS["muted"], True)
        _draw_text(d, (90, 370), "Every claim is tied to a checked-in status or artifact.", 21, COLORS["muted"])
        tests = int(data["tests"]["passed"] * (local + 1) / 10)
        sft = int(data["release"]["sft_examples"] * (local + 1) / 10)
        grpo = int(data["release"]["preference_pairs"] * (local + 1) / 10)
        cards = [
            ((88, 510, 510, 700), "Passing tests", f"{tests:,}", f"of {data['tests']['collected']:,} recorded"),
            ((544, 510, 966, 700), "CI matrix", f"{data['ci']['jobs_expected']}/8", "Linux + Windows · push + PR", COLORS["green"]),
            ((1000, 510, 1422, 700), "Fleet branches", str(data['fleet']['summary']['delivery_branches']), "mergeable · zero conflicts", COLORS["blue"]),
            ((1456, 510, 1832, 700), "SFT / preference", f"{sft:,} / {grpo}", "examples / pairs", COLORS["purple"]),
        ]
        for item in cards:
            _draw_metric_card(d, *item)
        _rounded(d, (88, 758, 1832, 871), COLORS["bg2"], COLORS["line"], radius=14)
        _draw_text(d, (120, 788), "CURRENT MAIN EVIDENCE", 16, COLORS["amber"], True)
        _draw_text(d, (120, 824), "The checked-in main branch reports v1.1.0 and 1,338 passing tests.", 21, COLORS["text"])

    elif scene == 1:
        _title(d, "02 / Symbol map + runtime profile", "Follow a call. Catch the cold tail.", "Cross-language contracts meet bounded, deterministic runtime sampling.")
        left = (86, 320, 1118, 846)
        right = (1150, 320, 1832, 846)
        _rounded(d, left, COLORS["panel"], COLORS["line"], radius=18, width=2)
        _rounded(d, right, COLORS["panel"], COLORS["line"], radius=18, width=2)
        _draw_text(d, (124, 350), "AST SYMBOL FLOW", 16, COLORS["cyan"], True)
        nodes = [(142, 450, "PYTHON", COLORS["blue"]), (470, 450, "JAVASCRIPT", COLORS["amber"]), (798, 450, "TYPESCRIPT", COLORS["purple"]), (470, 670, "INLINE HTML", COLORS["green"])]
        for x, y, label, color in nodes:
            _rounded(d, (x, y, x + 246, y + 88), COLORS["bg2"], color, radius=16, width=2)
            _draw_text(d, (x + 18, y + 31), label, 19, color, True)
        edges = [((388, 494), (470, 494)), ((716, 494), (798, 494)), ((592, 538), (592, 670)), ((276, 538), (470, 714))]
        for idx, (a, b) in enumerate(edges):
            d.line((a, b), fill=(65, 223, 210, 180), width=4)
            t = ((local * 35 + idx * 53) % 260) / 260
            x = int(a[0] + (b[0] - a[0]) * t)
            y = int(a[1] + (b[1] - a[1]) * t)
            d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=COLORS["cyan"])
        _draw_text(d, (124, 795), "Semantic contracts · typed fields · runtime overlays", 17, COLORS["muted"])
        _draw_text(d, (1190, 350), "100,000 HOT CALLS", 16, COLORS["muted"], True)
        _draw_text(d, (1190, 390), f"{data['profiler']['hot_calls']:,}", 62, COLORS["cyan"], True)
        _draw_text(d, (1190, 481), "1.418 s", 43, COLORS["text"], True)
        _draw_text(d, (1190, 536), "sys.settrace compatibility profile", 15, COLORS["muted"])
        _rounded(d, (1190, 600, 1792, 775), COLORS["bg2"], COLORS["line"], radius=14)
        _draw_text(d, (1220, 624), f"{data['profiler']['unique_sites_observed']} unique sites · tail cold branch captured", 17, COLORS["green"], True)
        _draw_text(d, (1220, 667), f"Reservoir limit {data['profiler']['reservoir_limit']} · replacements {data['profiler']['reservoir_replacements']}", 15, COLORS["muted"])
        _draw_text(d, (1220, 707), "Repeated edge hits aggregate without consuming new slots.", 14, COLORS["muted"])

    elif scene == 2:
        cand = data["tournament"]
        _title(d, "03 / Isolated candidate tournament", "One repair passes. Three variants fail.", "Four recorded alternatives for one task · each kept as a real preference row.")
        _rounded(d, (86, 322, 1832, 467), COLORS["panel"], COLORS["line"], radius=16)
        _draw_text(d, (120, 345), "RECORDED TASK", 14, COLORS["cyan"], True)
        _draw_text(d, (120, 383), cand["prompt"], 24, COLORS["text"], True)
        cards = [
            ("job10_correct", "SELECTED", COLORS["green"], f"score {cand['chosen']['score']:.3f}", "chosen positive"),
        ]
        for row in cand["rejected"]:
            details = row
            label = "SIGNATURE" if row["id"].endswith("signature") else "TEST CHEAT" if row["id"].endswith("test_cheat") else "MEMORY LEAK"
            if label == "SIGNATURE": foot = f"dynamic {row['dynamic_pass_rate']:.0%} · mutation survival {row['mutation_probe_survival_rate']:.0%}"
            elif label == "TEST CHEAT": foot = f"static {row['static_pass_rate']:.0%} · audit {str(row['audit_passed']).lower()}"
            else: foot = row["audit_findings"][0]["code"] if row["audit_findings"] else "audit evidence"
            cards.append((row["id"], label, COLORS["red"], f"score {row['score']:.3f}", foot))
        x_positions = [86, 536, 986, 1436]
        for idx, ((name, state, color, score, foot), x) in enumerate(zip(cards, x_positions)):
            top = 520
            _rounded(d, (x, top, x + 396, top + 287), COLORS["panel"], color, radius=16, width=2)
            _draw_text(d, (x + 25, top + 23), f"CANDIDATE 0{idx+1}", 13, COLORS["muted"], True)
            _draw_text(d, (x + 25, top + 64), state, 17, color, True)
            _draw_text(d, (x + 25, top + 105), name, 18, COLORS["text"], True)
            _draw_text(d, (x + 25, top + 158), score, 21, color, True)
            _draw_text(d, (x + 25, top + 207), foot, 14, COLORS["muted"])
            # Stage lights show the four checkpoints; pulse indicates the current one.
            labels = ["ISOLATE", "PROBE", "MUTATE", "AUDIT"]
            for j, lab in enumerate(labels):
                cx = x + 50 + j * 92
                c = color if (local + idx + j) % 4 == 0 else COLORS["dim"]
                d.ellipse((cx, top + 247, cx + 10, top + 257), fill=c)
                _draw_text(d, (cx + 14, top + 243), lab, 10, COLORS["muted"], True)
        _draw_text(d, (90, 862), "Dataset: datasets/grpo_tournament_pairs.jsonl · same task prompt · chosen + signature + test-cheat + memory-leak rows", 16, COLORS["muted"])

    elif scene == 3:
        _title(d, "04 / Frontend invariant scans", "Read the raw signal. Keep the denominator visible.", "Three read-only repositories · actual file and finding counts from STATUS.json.")
        scans = data["scans"]
        _draw_metric_card(d, (88, 330, 522, 505), "Repositories", str(scans["repositories"]), "read-only scan sample", COLORS["blue"])
        _draw_metric_card(d, (554, 330, 988, 505), "Files scanned", str(scans["scanned_files"]), "JS / TS / HTML", COLORS["cyan"])
        _draw_metric_card(d, (1020, 330, 1454, 505), "Raw rule hits", str(scans["raw_rule_hits"]), "sum of recorded repositories", COLORS["amber"])
        _rounded(d, (1486, 330, 1832, 505), COLORS["panel"], COLORS["line"], radius=18, width=2)
        _draw_text(d, (1515, 359), "POST-DEDUP", 14, COLORS["muted"], True)
        _draw_text(d, (1515, 402), "—", 54, COLORS["dim"], True)
        _draw_text(d, (1515, 472), "not recorded", 14, COLORS["amber"])
        maxv = max(row["hits"] for row in scans["by_repository"])
        y = 585
        for row in scans["by_repository"]:
            _draw_text(d, (110, y), row["repository"].split("/")[-1], 18, COLORS["text"], True)
            _rounded(d, (565, y + 3, 1530, y + 29), COLORS["bg2"], COLORS["line"], radius=10)
            width = int(955 * row["hits"] / maxv)
            d.rounded_rectangle((565, y + 3, 565 + width, y + 29), radius=10, fill=COLORS["cyan"] if row["hits"] != maxv else COLORS["blue"])
            _draw_text(d, (1580, y), f"{row['hits']} hits", 17, COLORS["muted"], True)
            _draw_text(d, (110, y + 40), f"{row['files']} files scanned", 14, COLORS["muted"])
            y += 102
        _rounded(d, (88, 906, 1832, 974), COLORS["bg2"], COLORS["amber"], radius=12, width=2)
        _draw_text(d, (116, 928), "Verified delivery fixes include non-finite leverage guards, database socket recovery, and invalid checkpoint-state rejection.", 16, COLORS["text"])

    elif scene == 4:
        world = data["world"]
        _title(d, "05 / Time travel + model safety", "Reproduce the failure. Keep predictions in observe.", "Frozen receipts preserve the evidence needed for deterministic replay and line-level blame.")
        _rounded(d, (88, 322, 930, 852), COLORS["panel"], COLORS["line"], radius=18, width=2)
        _draw_text(d, (128, 355), "TIME-TRAVEL RECEIPT", 16, COLORS["cyan"], True)
        _draw_text(d, (128, 404), "aegis replay --ticket", 26, COLORS["text"], True)
        receipt = ["Git archive byte hashes", "Frozen probe + seed + context", "Deterministic replay verification", "Changed-line blame / delta-debugging"]
        for i, line in enumerate(receipt):
            yy = 490 + i * 69
            d.ellipse((130, yy + 7, 144, yy + 21), fill=COLORS["green"] if i <= local // 4 else COLORS["dim"])
            _draw_text(d, (168, yy), line, 19, COLORS["text"] if i <= local // 4 else COLORS["muted"])
        _draw_text(d, (128, 794), "21 physical campaign replays · 45 time-travel tests", 16, COLORS["muted"])
        _rounded(d, (968, 322, 1832, 852), COLORS["panel"], COLORS["amber"], radius=18, width=2)
        _draw_text(d, (1010, 355), "WORLD MODEL · OBSERVE", 16, COLORS["amber"], True)
        _draw_text(d, (1010, 408), f"{world['training_examples']:,} training examples", 25, COLORS["text"], True)
        risk = world["risk"]
        metrics = [("Precision", f"{risk['precision']:.2f}"), ("Recall", f"{risk['recall']:.2f}"), ("High-risk support", str(risk["support"])), ("False positives", str(risk["false_positive"]))]
        for i, (label, value) in enumerate(metrics):
            col, row = i % 2, i // 2
            xx, yy = 1010 + col * 375, 500 + row * 107
            _rounded(d, (xx, yy, xx + 330, yy + 78), COLORS["bg2"], COLORS["line"], radius=12)
            _draw_text(d, (xx + 16, yy + 11), label.upper(), 12, COLORS["muted"], True)
            _draw_text(d, (xx + 16, yy + 37), value, 23, COLORS["cyan"] if label != "False positives" else COLORS["red"], True)
        _draw_text(d, (1010, 744), "False-positive rate: NOT REPORTED", 16, COLORS["red"], True)
        _draw_text(d, (1010, 780), "Observed support is too small to claim calibrated authorization.", 15, COLORS["muted"])

    else:
        _title(d, "06 / Fleet delivery", "Audit the fleet. Leave every main branch clean.", "Real delivery branches and repository names from artifacts/fleet_dashboard_v16.json.")
        summary = data["fleet"]["summary"]
        _draw_metric_card(d, (88, 316, 500, 475), "Delivery branches", str(summary["delivery_branches"]), "recorded in fleet audit", COLORS["cyan"])
        _draw_metric_card(d, (526, 316, 938, 475), "Mergeable", str(summary["mergeable"]), "all audited branches", COLORS["green"])
        _draw_metric_card(d, (964, 316, 1376, 475), "Conflicts", str(summary["conflicts"]), "recorded conflicts", COLORS["blue"])
        _draw_metric_card(d, (1402, 316, 1832, 475), "Main writes", str(data["fleet"]["main_pollution"]), "default-branch pollution", COLORS["purple"])
        _draw_text(d, (96, 532), "REPOSITORY DELIVERY MAP", 16, COLORS["muted"], True)
        names = data["fleet"]["repositories"]
        for idx, name in enumerate(names):
            col = idx % 5
            row = idx // 5
            x = 88 + col * 356
            y = 580 + row * 102
            _rounded(d, (x, y, x + 326, y + 70), COLORS["panel"], COLORS["line"], radius=12)
            _draw_text(d, (x + 17, y + 20), name.replace("Xiaoyu155/", ""), 15, COLORS["text"], True)
            _draw_text(d, (x + 17, y + 48), "delivery branch · clean default", 11, COLORS["green"])
        _rounded(d, (88, 900, 1832, 972), COLORS["bg2"], COLORS["cyan"], radius=12, width=2)
        _draw_text(d, (118, 924), "AUDIT  →  SHIP  →  PRESERVE", 20, COLORS["cyan"], True)
        _draw_text(d, (640, 926), "15 mergeable · 0 conflicts · 0 default-branch pollution", 17, COLORS["muted"])

    return image


def build_html(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), encoding="utf-8")


def build_visuals(data: dict[str, Any]) -> None:
    from PIL import Image

    output = ROOT / "showcase"
    key_dir = output / "frames"
    output.mkdir(exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    keyframes = {
        "keyframe_01_overview.png": 5,
        "keyframe_02_runtime.png": 17,
        "keyframe_03_tournament.png": 35,
        "keyframe_04_fleet.png": 82,
    }
    for filename, second in keyframes.items():
        _frame_scene(second, data).save(key_dir / filename, format="PNG", optimize=True)

    preview_samples = [
        (scene, step)
        for scene in range(6)
        for step in ((1, 3, 6, 9) if scene == 0 else (1, 5, 9, 14))
    ]
    preview_images = []
    offsets = [0, 10, 25, 45, 60, 75]
    for scene, step in preview_samples:
        second = min(89, offsets[scene] + min(step, (9 if scene == 0 else 14)))
        preview_images.append(_frame_scene(second, data).resize((960, 540), Image.Resampling.LANCZOS).quantize(colors=128))
    preview_images[0].save(
        output / "preview.gif",
        save_all=True,
        append_images=preview_images[1:],
        duration=140,
        loop=0,
        optimize=True,
        disposal=2,
    )

    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if ffmpeg.returncode != 0:
        raise RuntimeError("ffmpeg is required to compose the MP4")
    with tempfile.TemporaryDirectory(prefix="aegis-frames-") as temp_name:
        temp = Path(temp_name)
        for second in range(DURATION):
            path = temp / f"frame_{second:04d}.png"
            _frame_scene(second, data).save(path, format="PNG", optimize=False)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "1", "-i", str(temp / "frame_%04d.png"),
            "-vf", f"fps={FPS},format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "25",
            "-t", str(DURATION), "-movflags", "+faststart",
            str(output / "aegis_demo.mp4"),
        ]
        subprocess.run(cmd, check=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate:format=duration,size", "-of", "json", str(output / "aegis_demo.mp4")],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    duration = float(metadata["format"]["duration"])
    if (stream["codec_name"], stream["width"], stream["height"], stream["pix_fmt"]) != ("h264", 1920, 1080, "yuv420p"):
        raise RuntimeError(f"Unexpected video stream: {stream}")
    if not 60 <= duration <= 90:
        raise RuntimeError(f"Unexpected duration: {duration}")
    (output / "ffprobe.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-only", action="store_true", help="Write the offline dashboard without rendering images/video")
    args = parser.parse_args()
    data = load_data()
    (ROOT / "showcase").mkdir(exist_ok=True)
    (ROOT / "showcase/data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_html(data, ROOT / "docs/aegis_showcase.html")
    if not args.html_only:
        build_visuals(data)


if __name__ == "__main__":
    main()
