"""Luna requirement orchestration, backed by Aegis preflight and repair gates."""
from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile

from aegis_core.env_doctor import EnvDoctor
from aegis_core.sandbox import run_git

from .flight_recorder import FlightRecorder, sha256_file


class LunaBlocked(RuntimeError):
    """A fail-closed product run stopped because a required gate was unavailable."""


def _git(root: Path, *args: str) -> str:
    return run_git(root, *args).stdout.decode("utf-8").strip()


def _dirty_paths(root: Path) -> list[str]:
    output = _git(root, "status", "--porcelain", "--untracked-files=all")
    return [line[3:].strip().strip('"') for line in output.splitlines() if line.strip()]


def _patch_for(root: Path, relative: str, new_source: str) -> str:
    old_source = (root / relative).read_text(encoding="utf-8")
    before, after = old_source.splitlines(keepends=True), new_source.splitlines(keepends=True)
    diff = difflib.unified_diff(before, after, fromfile="a/" + relative, tofile="b/" + relative)
    return "diff --git a/" + relative + " b/" + relative + "\n" + "".join(diff)


def _demo_candidates(root: Path, target: str) -> list[list[dict]]:
    source = (root / target).read_text(encoding="utf-8")
    bad = source.replace("return min(10, value)", "return max(0, value)")
    good = source.replace(
        "    if type(value) is not int:\n        return 0\n    return min(10, value)",
        "    if type(value) is not int:\n        return None\n    return min(10, max(0, value))")
    if bad == source or good == source:
        raise LunaBlocked("embedded combat fixture no longer matches its frozen baseline")
    return [[{"candidate_id": "floor_only", "strategy": "first repair misses upper bound",
              "patch": _patch_for(root, target, bad), "token_cost": 12}],
            [{"candidate_id": "bounded_stock", "strategy": "clamp both bounds and reject invalid types",
              "patch": _patch_for(root, target, good), "token_cost": 20}]]


def _router(models_path: str | Path, root: Path):
    from .model_router import ModelRouter, QuotaLedger, load_config
    from aegis_core.router import Router, load_models
    path = Path(models_path).expanduser().resolve()
    if not path.is_file():
        raise LunaBlocked(f"model gateway config is missing: {path}")
    config = load_config(path, root)
    return ModelRouter(config, QuotaLedger(root, config)) if config else Router(load_models(path))


def _ask(router, system: str, payload: dict, *, role: str, timeout: float = 180) -> tuple[dict, dict]:
    from .model_router import ModelRouter
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    deadline = time.monotonic() + timeout
    if isinstance(router, ModelRouter):
        response = router.call(role, messages, deadline=deadline)
    else:
        response = router.call(router.chain[0], messages, deadline=deadline, require_json=True)
    if response.get("status") != "admitted":
        raise LunaBlocked("model gateway unavailable: " + str(response.get("reason", "blocked")))
    try:
        value = json.loads(response["output"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise LunaBlocked("model gateway returned an invalid JSON contract") from error
    if type(value) is not dict:
        raise LunaBlocked("model gateway response must be a JSON object")
    return value, {"tier": response.get("tier"), "slot_id": response.get("slot_id"),
                   "provider": response.get("provider")}


def _model_probe(router, requirement: str, plan: dict) -> tuple[dict, dict]:
    from aegis_core.probe_synth import validate_structured_rule
    result, route = _ask(router,
        "Translate the requirement into one executable Aegis structured probe. Return only JSON "
        "with the exact shape {\"probe_spec\":{...}}. Use a tracked source path and an existing Python "
        "symbol. Supported kinds: numeric_bounds, state_machine, idempotency, bounded_collection, "
        "async_timeout, contract_schema. Include every required field for the chosen kind. Do not invent "
        "external credentials, tests, paths or side effects.",
        {"requirement": requirement, "task_plan": plan}, role="planner")
    if set(result) != {"probe_spec"}:
        raise LunaBlocked("model gateway response has an unexpected probe envelope")
    try:
        spec = validate_structured_rule(result["probe_spec"])
    except (TypeError, ValueError) as error:
        raise LunaBlocked("model generated an invalid acceptance probe: " + str(error)) from error
    return spec, route


def _model_candidate_provider(router, allowed_paths: list[str]):
    def provider(context: dict) -> list[dict]:
        value, _route = _ask(router,
            "Generate up to three independent repair candidates for a local Aegis tournament. "
            "Return JSON exactly as {\"candidates\":[{\"candidate_id\":string,\"strategy\":string,"
            "\"patch\":unified_git_diff,\"token_cost\":integer}]}. Each diff may change only the "
            "listed target files. Do not edit or weaken tests, probes, gates, manifests, credentials or "
            "configuration. The first response should attempt a repair; later responses must address the "
            "recorded diagnosis. Do not claim a patch passed.",
            {"allowed_paths": allowed_paths, "context": context}, role="coder")
        if set(value) != {"candidates"} or type(value["candidates"]) is not list or not value["candidates"]:
            raise LunaBlocked("model generated no valid repair candidates")
        if len(value["candidates"]) > 3:
            raise LunaBlocked("model generated too many repair candidates")
        for candidate in value["candidates"]:
            if (type(candidate) is not dict or set(candidate) !=
                    {"candidate_id", "strategy", "patch", "token_cost"}):
                raise LunaBlocked("model candidate envelope is invalid")
            if type(candidate["patch"]) is not str or not candidate["patch"].startswith("diff --git "):
                raise LunaBlocked("model candidate must contain a unified Git diff")
        return value["candidates"]
    return provider


def _create_delivery(root: Path, run_id: str, base: str, patch: str,
                     requirement: str) -> dict:
    branch = f"luna/delivery/{run_id}"
    parent = root / ".worktrees" / f"luna-delivery-{run_id}"
    checkout = parent / "checkout"
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or checkout.exists():
        raise LunaBlocked("delivery worktree path already exists or is unsafe")
    run_git(root, "worktree", "add", "-b", branch, str(checkout), base)
    try:
        checked = run_git(checkout, "apply", "--check", input=patch.encode("utf-8"), check=False)
        if checked.returncode:
            raise LunaBlocked("winning patch no longer applies to the frozen base")
        run_git(checkout, "apply", input=patch.encode("utf-8"))
        changed = [name for name in _git(checkout, "diff", "--name-only", base).splitlines() if name]
        if not changed:
            raise LunaBlocked("delivery patch changed no files")
        run_git(checkout, "add", "--", *changed)
        subject = " ".join(requirement.strip().split())[:80] or "Aegis gated repair"
        run_git(checkout, "-c", "user.name=Luna Delivery", "-c", "user.email=luna@localhost",
                "commit", "-m", "luna: " + subject)
        commit = _git(checkout, "rev-parse", "HEAD")
        return {"branch": branch, "commit": commit, "base_commit": base,
                "changed_files": changed, "worktree": str(checkout)}
    finally:
        run_git(root, "worktree", "remove", "--force", str(checkout), check=False)


def _signed_package(root: Path, recorder: FlightRecorder, report_path: Path,
                    delivery: dict, engine_root: Path, keys_dir: str | Path | None,
                    *, demo: bool, autopilot: dict, requirement: str) -> dict:
    from aegis_core.enforce import (create_keypair, issue_token, key_directory,
                                    policy_sha256, verify_token)

    logs = recorder.directory
    patch_path = logs / f"{recorder.run_id}_winner.patch"
    patch_path.write_text(autopilot["winner"]["patch"], encoding="utf-8", newline="\n")
    manifest = {"schema_version": "luna.delivery.v1", "run_id": recorder.run_id,
                "requirement": requirement, "repository": str(root),
                "base_commit": delivery["base_commit"], "delivery_branch": delivery["branch"],
                "delivery_commit": delivery["commit"], "changed_files": delivery["changed_files"],
                "patch_sha256": autopilot["winner"]["patch_sha256"],
                "aegis_ticket_id": autopilot["ticket_id"],
                "aegis_gates": {"status": autopilot["status"],
                                "baseline_true_red": autopilot["synthesized_probe"]["baseline_true_red"],
                                "rounds_attempted": autopilot["rounds_attempted"],
                                "review_passed": any(item.get("candidate_id") == autopilot["winner"]["candidate_id"]
                                                     and item.get("accepted") and item.get("audit_passed")
                                                     for item in autopilot["rounds"][-1]["tournament"]["evaluations"])},
                "policy_sha256": policy_sha256(engine_root),
                "report_path": str(report_path.relative_to(root)),
                "machine_log_path": str(recorder.machine_path.relative_to(root)),
                "machine_log_sha256": sha256_file(recorder.machine_path)}
    manifest_path = logs / f"{recorder.run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary = None
    selected_keys = keys_dir
    if demo:
        temporary = tempfile.TemporaryDirectory(prefix="luna-demo-ed25519-")
        selected_keys = Path(temporary.name) / "keys"
        create_keypair(selected_keys)
    elif selected_keys is None:
        selected_keys = key_directory()
    try:
        signed = issue_token(manifest_path, policy_version="luna-v1",
                             expires_in_seconds=3600, directory=selected_keys,
                             policy_digest=manifest["policy_sha256"])
        checked = verify_token(signed["token"],
                               expected_manifest_sha256=sha256_file(manifest_path),
                               expected_policy_version="luna-v1",
                               expected_policy_sha256=manifest["policy_sha256"],
                               directory=selected_keys)
        if not checked.get("authorized"):
            raise LunaBlocked("Ed25519 verification rejected the delivery manifest")
        signature_path = logs / f"{recorder.run_id}_signature.json"
        signature_path.write_text(json.dumps({"status": "verified", "algorithm": "Ed25519",
                                              "fingerprint": signed["fingerprint"],
                                              "payload": signed["payload"], "token": signed["token"]},
                                             ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                  encoding="utf-8")
        if temporary:
            temporary.cleanup()
        return {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                "signature": str(signature_path), "signature_verified": True,
                "key_fingerprint": signed["fingerprint"], "ephemeral_key": demo,
                "patch": str(patch_path), "evidence_log_sha256": manifest["machine_log_sha256"],
                "package": str(logs / f"{recorder.run_id}_delivery.zip")}
    finally:
        if temporary:
            temporary.cleanup()


def run_requirement(requirement: str, repo: str | Path = ".", *, models: str | Path | None = None,
                    keys_dir: str | Path | None = None, db: str | Path | None = None,
                    run_id: str | None = None, timeout: float = 20,
                    max_rounds: int = 3, demo: bool = False) -> dict:
    """Run one product request and persist both operator-friendly and audit records."""
    root = Path(repo).expanduser().resolve(strict=True)
    if not root.is_dir() or not requirement.strip():
        raise ValueError("repo must be a directory and requirement must be nonempty")
    git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if git_root != root:
        raise ValueError("--repo must name the Git repository root")
    recorder = FlightRecorder(root, run_id=run_id, requirement=requirement)
    plan_steps, intercepts, evidence, deliverables = [], [], [], []
    autopilot = None
    final_status = "BLOCKED"
    understanding = requirement.strip()
    engine_root = Path(__file__).resolve().parent.parent
    doctor = EnvDoctor(root, cleanup=False)
    try:
        start = time.perf_counter()
        environment = doctor.check(changed_files=[], sparse_mode="auto", strict=False)
        recorder.record("環境自檢", command="EnvDoctor.check(cleanup=False)",
                        probe_elapsed_ms=(time.perf_counter() - start) * 1000,
                        verdict="READY" if not environment["blocking"] else "BLOCKED",
                        details={"python": environment["python"], "disk": environment["disk"],
                                 "missing_dependencies": environment["missing_dependencies"],
                                 "signing_key_valid": environment["security"]["signing_key"]["valid"]})
        if environment["blocking"]:
            raise LunaBlocked("environment preflight blocked the run")
        dirty = _dirty_paths(root)
        if dirty:
            intercepts.append("工作区不干净，已在写入交付分支前阻断：" + ", ".join(dirty[:12]))
            recorder.record("基线完整性检查", command="git status --porcelain",
                            verdict="BLOCKED_DIRTY_BASELINE", details={"paths": dirty[:50]})
            raise LunaBlocked("repository baseline must be clean before an Aegis run")
        base = _git(root, "rev-parse", "HEAD")
        recorder.record("冻结 Git 基线", command="git rev-parse HEAD", verdict="FROZEN",
                        details={"base_commit": base})

        from aegis_core.planner import build_task_dag
        plan_start = time.perf_counter()
        planning_text = requirement
        if demo:
            planning_text = ("examples/luna_combat_sandbox/inventory.py::clamp_stock clamps integers "
                            "to [0, 10] and returns None for invalid inputs.")
        dag = build_task_dag(planning_text, root, max_tasks=8,
                             experience_db=Path(db).resolve() if db else None)
        impact = dag.get("impact_analysis") or dag.get("impact") or {}
        recorder.record("影响面分析与任务拆解", command="Aegis build_task_dag",
                        probe_elapsed_ms=(time.perf_counter() - plan_start) * 1000,
                        verdict="PLANNED", details={"dag_id": dag.get("dag_id"),
                                                   "tasks": dag.get("tasks", []),
                                                   "topological_layers": dag.get("topological_layers", []),
                                                   "questions": dag.get("spec", {}).get("questions", []),
                                                   "impact": impact})
        plan_steps.append(f"生成 {len(dag.get('tasks', []))} 个任务，依赖层数 {len(dag.get('topological_layers', []))}。")
        if dag.get("spec", {}).get("questions") and not demo:
            recorder.record("验收约束检查", verdict="BLOCKED_MISSING_CONSTRAINT",
                            details={"questions": dag["spec"]["questions"]})
            raise LunaBlocked("requirement needs acceptance constraints before implementation")

        if demo:
            target = "examples/luna_combat_sandbox/inventory.py"
            if not (root / target).is_file():
                raise LunaBlocked("the embedded inventory combat fixture is not present")
            contract = ("examples/luna_combat_sandbox/inventory.py::clamp_stock clamps integers "
                        "to [0, 10] and returns None for invalid inputs.")
            probe_spec = {"kind": "numeric_bounds", "path": target, "symbol": "clamp_stock",
                          "min": 0, "max": 10, "fallback": None, "seed": 20261121}
            candidates = _demo_candidates(root, target)
            router = None
            route_info = {"provider": "deterministic combat fixture", "mode": "offline-demo"}
        else:
            from aegis_core.probe_synth import acceptance_rule
            rule = acceptance_rule(requirement)
            from .model_router import config_path
            router = _router(config_path(root, models), root)
            if rule and rule["kind"] == "clamp":
                probe_spec = {"kind": "numeric_bounds", "path": rule["path"], "symbol": rule["symbol"],
                              "min": rule["low"], "max": rule["high"], "fallback": rule["fallback"]}
                route_info = {"provider": "deterministic acceptance compiler", "mode": "offline-probe"}
            else:
                probe_spec, route_info = _model_probe(router, requirement, dag)
            target = probe_spec["path"]
            tracked = set(filter(None, _git(root, "ls-files", "-z").split("\0")))
            if target not in tracked or not (root / target).is_file():
                raise LunaBlocked("acceptance probe target must be a tracked source file")
            contract = requirement
            candidates = None
        recorder.record("红灯探针冻结", command="Aegis compile_structured_probe",
                        verdict="READY", details={"probe_spec": probe_spec, "route": route_info})
        if candidates:
            auto = {"candidate_rounds": candidates}
        else:
            auto = {"candidate_provider": _model_candidate_provider(router, [target])}
        from aegis_core.self_heal_loop import run_autopilot
        db_path = Path(db).resolve() if db else recorder.directory / f"{recorder.run_id}.sqlite"
        data_file = tempfile.NamedTemporaryFile(prefix="luna-tournament-", suffix=".jsonl", delete=False)
        data_file.close()
        try:
            gate_start = time.perf_counter()
            autopilot = run_autopilot(root, contract=contract, probe_spec=probe_spec,
                                      timeout_sec=timeout, max_rounds=max_rounds,
                                      max_workers=2, max_context_tokens=4000,
                                      max_total_tokens=20000, max_wall_clock_sec=600,
                                      allowed_paths=[target], database_path=db_path,
                                      dataset_path=Path(data_file.name), **auto)
        finally:
            Path(data_file.name).unlink(missing_ok=True)
        replay = autopilot.get("time_travel_snapshot", {}).get("ticket_id")
        recorder.record("隔离红绿探针与 L3 审查", command="Aegis SelfHealLoop.run",
                        diff_sha256=(autopilot.get("winner") or {}).get("patch_sha256"),
                        probe_elapsed_ms=(time.perf_counter() - gate_start) * 1000,
                        verdict=autopilot.get("status", "BLOCKED"), replay_handle=replay,
                        details={"ticket_id": autopilot.get("ticket_id"),
                                 "baseline_true_red": autopilot.get("synthesized_probe", {}).get("baseline_true_red"),
                                 "rounds": autopilot.get("rounds", []),
                                 "time_travel_snapshot": autopilot.get("time_travel_snapshot"),
                                 "diagnosis": autopilot.get("diagnosis")})
        if autopilot.get("status") != "succeeded" or not autopilot.get("winner"):
            intercepts.extend([f"Aegis 阻断候选修复：{row.get('candidate_id')}，未通过门禁。"
                               for round_row in autopilot.get("rounds", [])
                               for row in round_row.get("tournament", {}).get("evaluations", [])
                               if not row.get("accepted")])
            raise LunaBlocked("no candidate passed all Aegis acceptance, mutation and L3 gates")
        for round_row in autopilot.get("rounds", []):
            for row in round_row.get("tournament", {}).get("evaluations", []):
                if not row.get("accepted"):
                    intercepts.append(f"拦截候选 `{row.get('candidate_id')}`：静态 {row.get('static_pass_rate')}, "
                                      f"动态 {row.get('dynamic_pass_rate')}, 变异 {row.get('mutation_probe_survival_rate')}, "
                                      f"L3 {row.get('audit_passed')}。")
        evidence.append(f"基线探针确认为真实红灯：{autopilot['synthesized_probe']['baseline_true_red']}。")
        evidence.append(f"Aegis 自愈 {autopilot['rounds_attempted']} 轮，全门禁通过候选：{autopilot['winner']['candidate_id']}。")
        evidence.append(f"回放句柄：`{autopilot['time_travel_snapshot']['ticket_id']}`；补丁 SHA-256：`{autopilot['winner']['patch_sha256']}`。")
        last_tournament = autopilot["rounds"][-1]["tournament"]
        evidence.append("L3 审查：" + ("通过" if all(item.get("audit_passed") for item in
                                               last_tournament["evaluations"] if item.get("accepted"))
                                         else "未通过"))
        delivery = _create_delivery(root, recorder.run_id, base,
                                    autopilot["winner"]["patch"], requirement)
        recorder.record("生成隔离交付分支", command=f"git worktree add -b {delivery['branch']}",
                        diff_sha256=autopilot["winner"]["patch_sha256"], verdict="BRANCH_CREATED",
                        replay_handle=replay, details=delivery)
        deliverables.append(f"交付分支 `{delivery['branch']}`，提交 `{delivery['commit'][:12]}`，变更 {', '.join(delivery['changed_files'])}。")

        report_path = recorder.report_path
        # Render once before signing so the signed manifest binds to the exact report bytes.
        report_path = recorder.render_report(understanding=understanding, plan=plan_steps,
                                             interceptions=intercepts or ["未发现作弊或门禁绕过；错误候选均由 Aegis 门禁判负。"],
                                             evidence=evidence, deliverables=deliverables,
                                             final_status="READY_TO_REVIEW")
        signing = _signed_package(root, recorder, report_path, delivery, engine_root, keys_dir,
                                  demo=demo, autopilot=autopilot, requirement=requirement)
        recorder.record("Ed25519 签发与交付包封装", command="Aegis issue_token + verify_token",
                        verdict="SIGNATURE_VERIFIED",
                        replay_handle=replay, details={"manifest_sha256": signing["manifest_sha256"],
                                                      "key_fingerprint": signing["key_fingerprint"],
                                                      "ephemeral_key": signing["ephemeral_key"]})
        deliverables.extend([f"签名清单 SHA-256：`{signing['manifest_sha256']}`（Ed25519 验签通过）。",
                             f"交付包 `{signing['package']}`（包含最终战报与机器日志）。",
                             "建议：由负责人审阅交付分支后再合并；Luna 不会自动改写主分支。"])
        final_status = "SUCCEEDED"
        recorder.record("任务交付完成", command="luna run", verdict=final_status,
                        diff_sha256=autopilot["winner"]["patch_sha256"], replay_handle=replay,
                        details={"branch": delivery["branch"], "commit": delivery["commit"],
                                 "manifest_sha256": signing["manifest_sha256"]})
        report_path = recorder.render_report(understanding=understanding, plan=plan_steps,
                                             interceptions=intercepts or ["未发现作弊或门禁绕过；所有候选都经过固定验收探针、变异探针和 L3 审查。"],
                                             evidence=evidence, deliverables=deliverables,
                                             final_status=final_status)
        with zipfile.ZipFile(signing["package"], "w", compression=zipfile.ZIP_DEFLATED) as package:
            for path in (report_path, recorder.machine_path, Path(signing["manifest"]),
                         Path(signing["signature"]), Path(signing["patch"])):
                package.write(path, path.name)
        signing["package_sha256"] = sha256_file(signing["package"])
        return {"status": final_status, "run_id": recorder.run_id,
                "machine_log": str(recorder.machine_path), "report": str(report_path),
                "machine_log_sha256": sha256_file(recorder.machine_path),
                "chain": recorder.head_hash, "delivery": delivery, "signing": signing,
                "aegis": autopilot}
    except (LunaBlocked, ValueError, OSError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        final_status = "BLOCKED"
        if not recorder.events or recorder.events[-1]["verdict"] not in {"BLOCKED", "BLOCKED_DIRTY_BASELINE",
                                                                          "BLOCKED_MISSING_CONSTRAINT"}:
            recorder.record("任务阻断", command="luna run", verdict="BLOCKED",
                            details={"reason": str(error), "error_type": type(error).__name__})
        report = recorder.render_report(understanding=understanding, plan=plan_steps,
                                        interceptions=intercepts or [str(error)], evidence=evidence,
                                        deliverables=deliverables or ["未签发交付包；按阻断原因补齐前提后重跑。"],
                                        final_status=final_status)
        return {"status": final_status, "run_id": recorder.run_id,
                "reason": str(error), "machine_log": str(recorder.machine_path),
                "machine_log_sha256": sha256_file(recorder.machine_path),
                "report": str(report), "chain": recorder.head_hash}
    finally:
        doctor.cleanup_temporary_configs()
