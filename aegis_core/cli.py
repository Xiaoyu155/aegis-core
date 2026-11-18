"""Unified stdlib control tower. Commands print one JSON result to stdout."""
import argparse
import ast
import hashlib
import math
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from .ledger import Ledger
from .mediator import compile_wish, compile_request, is_business_source, validate_compilation
from .reward import compute_step_reward
from .router import Router, load_models
from .sandbox import Sandbox, tree_hash


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout.decode("utf-8").strip()


def _load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _digest(document):
    return hashlib.sha256(json.dumps({k: v for k, v in document.items() if k != "contract_sha256"},
                                    ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _repo(path):
    root = Path(path).resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("--repo must name the repository root")
    return root


def _ledger(root, db):
    path = Path(db).resolve() if db else Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "aegis/ledger.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return Ledger(path)


def _skeleton(root):
    return sorted(name for name in _git(root, "ls-files", "-z").split("\0") if name and (root / name).is_file())


def _argv(value):
    if value is None:
        return []
    # JSON arrays are portable across Windows and Linux; strings are convenience only.
    if value.lstrip().startswith("["):
        result = json.loads(value)
    else:
        result = shlex.split(value)
    if type(result) is not list or any(type(item) is not str or not item for item in result):
        raise ValueError("probe commands must be argument arrays")
    return result


def _wish(args):
    root = _repo(args.repo)
    skeleton = _skeleton(root)
    sources = [name for name in skeleton if is_business_source(name)]
    if not sources:
        raise ValueError("repository has no existing business source targets")
    enhanced = None
    if args.auto_probe:
        enhanced = compile_wish(root, args.text, targets=args.target, allow_model=args.allow_model, models=args.models)
    candidates = None
    if enhanced:
        proposal = {key: enhanced[key] for key in ('target_files', 'contract_stub')}
    elif args.proposal:
        proposal = _load(args.proposal)
    else:
        if args.target:
            targets = args.target
        else:
            scores = [(name, 1.0 if name in args.text or Path(name).stem in args.text else 0.0) for name in sources]
            ranked = sorted(scores, key=lambda item: (-item[1], item[0]))
            targets = [ranked[0][0]]
            candidates = [{"path": name, "score": score} for name, score in ranked[:3]]
        # Generic stub is explicit when no user-approved business signature is supplied.
        proposal = {"target_files": targets, "contract_stub": "def apply_change() -> None:\n    ..."}
    constraints = _load(args.constraints) if args.constraints else {}
    if enhanced:
        constraints = {**enhanced['constraints'], **constraints}
    selection = args.selected_module or constraints.get("selected_module")
    if selection and candidates:
        proposal["target_files"] = [selection]
    if args.selected_module:
        constraints["selected_module"] = args.selected_module
    ledger = _ledger(root, args.db)
    ticket = args.ticket_id or ("wish-" + uuid.uuid4().hex[:12])
    try:
        prior = ledger.get_ticket(ticket)
        if prior["wish_raw"] != args.text or prior["status"] not in {"queued", "MISSING_CONSTRAINT"}:
            raise ValueError("resume requires the same wish and a pending ticket")
    except KeyError:
        ledger.create_ticket(ticket, args.text, {}, proposal["contract_stub"])
    token = ledger.acquire_lease(ticket, "cli-compiler")
    if not token:
        raise PermissionError("ticket is leased")
    try:
        if enhanced and enhanced['missing']:
            result = {'kind': 'MissingConstraintQuestion', 'ticket_id': ticket, 'status': 'MISSING_CONSTRAINT',
                      'citations': enhanced['context'].get('retrieval', []),
                      'questions': [{'code': 'D_CONSTRAINT', 'question': ', '.join(enhanced['missing']),
                                     'options': [{'id': 'provide', 'label': '补充契约', 'value': {}}, {'id': 'stop', 'label': '停止', 'value': {}}], 'recommended': 'stop'}]}
            state = ledger.get_state(ticket)
            ledger.transition(ticket, token, state['status'], 'MISSING_CONSTRAINT', state_s0=state,
                              action_a0={'op': 'compile'}, actual_state_s1={'status': 'MISSING_CONSTRAINT', 'compilation': result}, reward=0)
        else:
            result = compile_request(ticket, args.text, skeleton, proposal, constraints=constraints,
                                 candidates=candidates, ledger=ledger, lease_token=token)
    finally:
        ledger.release_lease(ticket, token)
    if not enhanced:
        from .runner import retrieve
        legacy_context = retrieve(root, ' '.join(proposal['target_files']))['context']
    document = {"schema_version": "aegis.cli.v1", "ticket_id": ticket, "repo_root": str(root),
                "base_commit": _git(root, "rev-parse", "HEAD"), "compilation": result,
                "fail_before_cmd": _argv(args.fail_before) or (["{python}", "-B", "-m", "pytest", "-q", "tests/test_aegis_probe.py"] if enhanced else []), "pass_after_cmd": _argv(args.pass_after) or (["{python}", "-B", "-m", "pytest", "-q", "tests/test_aegis_probe.py"] if enhanced else []),
                "source_dirs": args.source_dir or ["."],
                "patches": [str(Path(path).resolve()) for path in args.patch or []],
                "models_path": str(Path(args.models).resolve()) if args.models else None,
                "probe_patch": enhanced['probe_patch'] if enhanced else "", "context": enhanced['context'] if enhanced else legacy_context,
                "allow_model": args.allow_model, "max_attempts": args.max_attempts,
                "timeout_sec": args.timeout}
    document["contract_sha256"] = _digest(document)
    if result["status"] == "COMPILED":
        token = ledger.acquire_lease(ticket, "cli-freeze")
        if token is None:
            raise PermissionError("ticket is leased before manifest freeze")
        try:
            state = ledger.get_state(ticket)
            ledger.transition(ticket, token, "COMPILED", "COMPILED", state_s0=state,
                              action_a0={"op": "freeze_cli_manifest"},
                              actual_state_s1={**state, "contract_sha256": document["contract_sha256"]}, reward=0)
        finally:
            ledger.release_lease(ticket, token)
    if args.out:
        _save(args.out, document)
    return document, 2 if result["status"] == "MISSING_CONSTRAINT" else 0


def _validate_manifest(document, root, ledger):
    fields = {"schema_version", "ticket_id", "repo_root", "base_commit", "compilation", "fail_before_cmd", "pass_after_cmd",
              "source_dirs", "patches", "models_path", "allow_model", "max_attempts", "timeout_sec", "contract_sha256", "probe_patch", "context"}
    if set(document) != fields or document["schema_version"] != "aegis.cli.v1" or _digest(document) != document["contract_sha256"]:
        raise ValueError("invalid or modified frozen manifest")
    if Path(document["repo_root"]).resolve() != root or _git(root, "rev-parse", "HEAD") != document["base_commit"]:
        raise ValueError("repository baseline changed; compile a new contract")
    if _git(root, "diff", "--name-only", "HEAD", "--"):
        raise ValueError("tracked working tree changed; commit or restore before run")
    compilation = validate_compilation(document["compilation"], _skeleton(root))
    if compilation["status"] != "COMPILED" or compilation["ticket_id"] != document["ticket_id"]:
        raise ValueError("unresolved constraint questions")
    ticket = ledger.get_ticket(document["ticket_id"])
    if (ticket["status"] != "COMPILED" or ticket["compiled_intent"] != compilation
            or ledger.get_state(document["ticket_id"]).get("contract_sha256") != document["contract_sha256"]):
        raise ValueError("contract is not compiled in this ledger")
    for key in ("fail_before_cmd", "pass_after_cmd"):
        command = document[key]
        if type(command) is not list or not command or any(type(arg) is not str or not arg for arg in command):
            raise ValueError("frozen probe argv required")
    if not document["source_dirs"] or any(type(p) is not str for p in document["source_dirs"]):
        raise ValueError("source directories required")
    if type(document["max_attempts"]) is not int or not 1 <= document["max_attempts"] <= 20:
        raise ValueError("max_attempts must be 1..20")
    if type(document["timeout_sec"]) not in (int, float) or not 0 < document["timeout_sec"] <= 3600:
        raise ValueError("timeout must be 0..3600 seconds")
    if type(document["allow_model"]) is not bool or type(document["patches"]) is not list or any(type(p) is not str for p in document["patches"]):
        raise ValueError("invalid patch/model settings")
    if document["allow_model"] and not document["models_path"]:
        raise ValueError("model execution requires --models")
    if not document["allow_model"] and not document["patches"]:
        raise ValueError("provide patch candidates or explicitly enable a model")


def _offline_router():
    # An offline patch is a submitted artifact, not a real local model invocation.
    node = {"id": "offline", "provider": "openai", "model": "offline-artifact", "model_env": "", "base_url": "", "base_url_env": "",
            "api_key_env": "", "timeout_sec": 60, "max_concurrency": 1, "group": "offline", "weight": 1, "enabled": True}
    return Router({"version": 1, "min_verify_reserve_sec": 1, "groups": {"offline": 1},
                   "tiers": {"offline": [node]}, "unattended_work_order": ["offline", "blocked_needs_human"]})


def _run(args):
    document = _load(args.contract)
    root = _repo(document['repo_root'])
    ledger = _ledger(root, args.db)
    try:
        return _execute(args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        ticket = document.get('ticket_id')
        if ticket:
            state = ledger.get_state(ticket)
            if state['status'] == 'COMPILED':
                token = ledger.acquire_lease(ticket, 'precheck')
                if token:
                    ledger.transition(ticket, token, 'COMPILED', 'PRECHECK_FAILED', state_s0=state,
                                      action_a0={'op': 'precheck'}, actual_state_s1={'status': 'PRECHECK_FAILED', 'reason': str(error), 'error_type': type(error).__name__}, reward=-1)
        raise


def _execute(args):
    document = _load(args.contract)
    root = _repo(document["repo_root"])
    ledger = _ledger(root, args.db)
    _validate_manifest(document, root, ledger)
    ticket = document["ticket_id"]
    # Validate lifecycle ID before taking a lease/changing state.
    manager = Sandbox(root)
    manager._paths(ticket)
    router = Router(load_models(Path(document["models_path"]))) if document["allow_model"] else _offline_router()
    from .world_model import CausalWorldModel
    model_path = Path(os.environ.get("AEGIS_WORLD_MODEL_DIR", root / "artifacts" / "causal_world_model")).resolve()
    world_model = CausalWorldModel.load_if_available(model_path)

    def verify_probe_baseline():
        from .sandbox import validate_probe_patch, run_exam_gate, ProbeNotFailingError
        validate_probe_patch(document['probe_patch'])
        check = 'check-' + uuid.uuid4().hex[:12]
        path = manager.create(check, base=document['base_commit'])
        try:
            manager.apply_patch(check, document['probe_patch'])
            before = run_exam_gate(path, [sys.executable if a == '{python}' else a for a in document['fail_before_cmd']], timeout=min(60, document['timeout_sec']))
            if before['exit_code'] != 1 or before['changed']:
                raise ProbeNotFailingError(json.dumps(before))
        finally:
            manager.cleanup(check)

    probe_checked = False
    # Legacy repositories without trained weights keep the existing precheck path.
    # With a model, each proposed action is predicted and durably recorded first.
    if document.get('probe_patch') and world_model is None:
        verify_probe_baseline()
        probe_checked = True
    for candidate in document['patches']:
        if not Path(candidate).is_file():
            raise FileNotFoundError(candidate)
    token = ledger.acquire_lease(ticket, "cli-runner", ttl=document["timeout_sec"] + 60)
    if not token:
        raise PermissionError("ticket is leased")
    tier = router.chain[0]
    deadline = time.monotonic() + document["timeout_sec"]
    results = []
    terminal, reason = "failed", "attempts_exhausted"
    try:
        state = ledger.get_state(ticket)
        running = {"status": "running", "contract_sha256": document["contract_sha256"]}
        ledger.transition(ticket, token, "COMPILED", "running", state_s0=state, action_a0={"op": "start_run"}, actual_state_s1=running, reward=0)
        for index in range(document["max_attempts"]):
            patch = None
            admitted = None
            breaker = {"repeated_patch": False}
            if document["allow_model"]:
                context = {"contract": document["compilation"], "prior_attempts": results,
                           "source": {name: (root / name).read_text(encoding="utf-8") for name in document["compilation"]["target_files"]},
                           "frozen_probes": {key: document[key] for key in ("fail_before_cmd", "pass_after_cmd")},
                           "frozen_tests": {name: (root / name).read_text(encoding="utf-8") for name in _skeleton(root)
                                            if name.startswith("tests/") and name.endswith(".py") and (root / name).stat().st_size < 32768}}
                admitted = router.call(tier, [{"role": "system", "content": 'Return only JSON {"patch":"unified diff"}. Never edit tests or configuration.'},
                                             {"role": "user", "content": json.dumps(context, ensure_ascii=False)}], deadline=deadline)
                if admitted["status"] != "admitted":
                    reason = admitted.get("reason", "blocked_needs_human")
                    break
                patch = json.loads(admitted["output"])["patch"]
                breaker = admitted["breaker"]
                slot = admitted["slot_id"]
            else:
                if index >= len(document["patches"]):
                    break
                admitted = router.acquire(tier, deadline=deadline)
                if admitted["status"] != "admitted":
                    reason = admitted["reason"]
                    break
                slot = admitted["slot_id"]
                try:
                    patch = Path(document["patches"][index]).read_text(encoding="utf-8")
                    breaker = router.record_output(slot, patch)
                finally:
                    router.release(admitted["lease_id"])
            if type(patch) is not str:
                raise ValueError("model patch must be text")
            patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
            target_files = document["compilation"]["target_files"]
            attempt_action = {"op": "repair_attempt", "patch": patch, "patch_sha256": patch_sha256,
                              "target_files": target_files, "slot_id": slot,
                              "context": document["context"], "contract": document["compilation"],
                              "repeated_patch": breaker["repeated_patch"]}
            prediction = None
            if world_model is not None:
                prediction_state = ledger.get_state(ticket)
                prediction = world_model.predict(prediction_state, attempt_action,
                                                 context=document["context"], prior_attempts=results)
                ledger.transition(ticket, token, "running", "running", state_s0=prediction_state,
                                  action_a0={**attempt_action, "op": "world_model_prediction"},
                                  actual_state_s1=prediction_state, reward=0,
                                  predicted_next_state=prediction["predicted_next_state"],
                                  predicted_reward=prediction["predicted_reward"],
                                  prediction_confidence=prediction["prediction_confidence"],
                                  predicted_reward_interval=prediction["predicted_reward_interval"])
            work_id = "run-" + uuid.uuid4().hex[:16]
            result = {"baseline_failed": False, "after_passed": False, "passed": False, "cheat_detected": False,
                      "changed_files": [], "diff_lines": 0, "repeated_patch": breaker["repeated_patch"],
                      "patch_sha256": patch_sha256}
            if prediction is not None:
                result.update({"predicted_next_state": prediction["predicted_next_state"],
                               "predicted_reward": prediction["predicted_reward"],
                               "predicted_reward_interval": prediction["predicted_reward_interval"],
                               "prediction_confidence": prediction["prediction_confidence"],
                               "risk_probability": prediction["risk_probability"],
                               "world_model_version": prediction["model_version"]})
            started = time.monotonic()
            pre_block_reason = (prediction["predicted_next_state"].get("pre_block_reason")
                                if prediction is not None else None)
            if pre_block_reason:
                paths = prediction["predicted_next_state"].get("changed_paths", [])
                result.update(pre_blocked=True, pre_block_reason="HIGH_RISK_PRE_BLOCK",
                              pre_block_cause=pre_block_reason,
                              cheat_detected=pre_block_reason in {"test_modification", "out_of_scope_path"},
                              repeated_patch=pre_block_reason == "repeated_patch",
                              changed_files=paths,
                              diff_lines=sum(line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
                                             for line in patch.splitlines()))
                reason = "HIGH_RISK_PRE_BLOCK"
            elif breaker["repeated_patch"]:
                reason = "blocked_needs_human"
            else:
                if document.get('probe_patch') and not probe_checked:
                    verify_probe_baseline()
                    probe_checked = True
                path = manager.create(work_id, base=document["base_commit"])
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= router.reserve:
                        reason = "verification_reserve"
                    else:
                        try:
                            resolve = lambda command: [sys.executable if arg == "{python}" else arg for arg in command]
                            result.update(manager.repair(work_id, patch, fail_before_cmd=resolve(document["fail_before_cmd"]),
                                                         pass_after_cmd=resolve(document["pass_after_cmd"]), source_dirs=tuple(document["source_dirs"]),
                                                         timeout=remaining, probe_patch=document.get('probe_patch', '')))
                        except (ValueError, subprocess.SubprocessError) as error:
                            result["error_type"] = type(error).__name__
                        outside = set(result["changed_files"]) - set(document["compilation"]["target_files"])
                        if outside:
                            result.update(cheat_detected=True, passed=False, outside_contract=sorted(outside))
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    result["error"] = type(error).__name__  # no vendor credentials or command payloads
                finally:
                    try:
                        if not result['passed']:
                            from .runner import diagnose
                            result['diagnosis'] = diagnose(path, result, attempt=index+1)
                    finally:
                        manager.cleanup(work_id)
            elapsed = time.monotonic() - started
            reward = compute_step_reward(baseline_failed=result["baseline_failed"], after_passed=result["after_passed"],
                                         target_files=document["compilation"]["target_files"], changed_files=result["changed_files"],
                                         diff_lines=result["diff_lines"], elapsed_sec=elapsed, latency_budget_sec=document["timeout_sec"],
                                         cheat_detected=result["cheat_detected"], repeated_patch=result["repeated_patch"],
                                         timeout_killed=result.get("timeout_killed", False))
            if not result['passed'] and 'diagnosis' not in result:
                if not result.get("pre_blocked"):
                    from .runner import diagnose
                    result['diagnosis'] = diagnose(root, result, attempt=index+1)
            result["reward"] = reward
            error = None
            if prediction is not None:
                from .runner import classify_causal_outcome, causal_prediction_error
                actual_outcome = classify_causal_outcome(result)
                result["world_model_outcome"] = actual_outcome
                error = causal_prediction_error(prediction["predicted_next_state"],
                                                prediction["predicted_reward"], result,
                                                prediction["predicted_reward_interval"])
                result["causal_prediction_error"] = error
            prior = ledger.get_state(ticket)
            current = {"status": "running", "attempt": index, **result}
            ledger.transition(ticket, token, "running", "running", state_s0=prior,
                              action_a0=attempt_action, actual_state_s1=current, reward=reward,
                              predicted_next_state=prediction["predicted_next_state"] if prediction else None,
                              predicted_reward=prediction["predicted_reward"] if prediction else None,
                              prediction_confidence=prediction["prediction_confidence"] if prediction else None,
                              prediction_error=error,
                              predicted_reward_interval=prediction["predicted_reward_interval"] if prediction else None)
            results.append(result)
            if result["passed"]:
                terminal, reason = "succeeded", "gate_passed"
                break
            router.record_rejection(slot, patch)
            if result.get("pre_blocked"):
                break
            if breaker["repeated_patch"]:
                if breaker["next_tier"] == "blocked_needs_human":
                    break
                tier = breaker["next_tier"]
        previous = ledger.get_state(ticket)
        ledger.transition(ticket, token, "running", terminal, state_s0=previous, action_a0={"op": "finish_run"},
                          actual_state_s1={"status": terminal, "reason": reason, "attempts": len(results)}, reward=0)
    except Exception:
        previous = ledger.get_state(ticket)
        if previous["status"] == "running":
            ledger.transition(ticket, token, "running", "failed", state_s0=previous, action_a0={"op": "abort_run"},
                              actual_state_s1={"status": "failed", "reason": "execution_error"}, reward=-1)
        raise
    finally:
        if ledger.get_ticket(ticket)["status"] not in {"succeeded", "failed"}:
            ledger.release_lease(ticket, token)
    return {"ticket_id": ticket, "status": terminal, "reason": reason, "attempts": len(results), "results": results}, 0 if terminal == "succeeded" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aegis", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wish = commands.add_parser("wish")
    wish.add_argument("text")
    wish.add_argument("--repo", default=".")
    wish.add_argument("--db")
    wish.add_argument("--ticket-id")
    wish.add_argument("--proposal")
    wish.add_argument("--constraints")
    wish.add_argument("--target", action="append")
    wish.add_argument("--selected-module")
    wish.add_argument("--source-dir", action="append")
    wish.add_argument("--fail-before")
    wish.add_argument("--pass-after")
    wish.add_argument("--patch", action="append")
    wish.add_argument("--models")
    wish.add_argument("--allow-model", action="store_true")
    wish.add_argument("--max-attempts", type=int, default=3)
    wish.add_argument("--timeout", type=float, default=120)
    wish.add_argument("--out", "--output")
    wish.add_argument("--auto-probe", action=argparse.BooleanOptionalAction, default=True)
    run = commands.add_parser("run")
    run_target = run.add_mutually_exclusive_group(required=True)
    run_target.add_argument("--contract")
    run_target.add_argument("--dag")
    run.add_argument("--db")
    run.add_argument("--repo")
    run.add_argument("--workers", type=int, default=4)
    plan = commands.add_parser("plan")
    plan.add_argument("text")
    plan.add_argument("--repo", default=".")
    plan.add_argument("--out", "--output", required=True)
    plan.add_argument("--max-tasks", type=int, default=8)
    learn = commands.add_parser("learn")
    learn.add_argument("--repo", default=".")
    learn.add_argument("--db")
    learn.add_argument("--min-surprise", type=float, default=0.3)
    learn.add_argument("--artifact-dir")
    learn.add_argument("--limit", type=int, default=256)
    learn.add_argument("--epochs", type=int, default=4)
    learn.add_argument("--seed", type=int, default=20261118)
    attach = commands.add_parser("attach")
    attach.add_argument("--repo", required=True)
    for name in ("stats", "export-traces"):
        command = commands.add_parser(name)
        command.add_argument("--repo", default=".")
        command.add_argument("--db")
        if name == "export-traces":
            command.add_argument("--out", "--output", default="datasets/latest_traces.jsonl")
            command.add_argument("--ticket-id")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        if args.command == "wish":
            result, code = _wish(args)
        elif args.command == "run":
            if args.dag:
                from .planner import execute_task_dag
                result, code = execute_task_dag(args.dag, db=args.db, workers=args.workers, repo=args.repo)
            else:
                result, code = _run(args)
        elif args.command == "plan":
            from .planner import build_task_dag
            result = build_task_dag(args.text, args.repo, max_tasks=args.max_tasks)
            destination = Path(args.out).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            _save(destination, result)
            result = {"status": "planned", "dag_id": result["dag_id"],
                      "tasks": len(result["tasks"]), "layers": result["topological_layers"],
                      "output": str(destination)}
            code = 0
        elif args.command == "learn":
            root = _repo(args.repo)
            ledger = _ledger(root, args.db)
            from .learning import learn_from_ledger
            result = learn_from_ledger(repo=root, db=ledger.path, min_surprise=args.min_surprise,
                                       artifact_dir=args.artifact_dir, limit=args.limit,
                                       epochs=args.epochs, seed=args.seed)
            code = 0
        elif args.command == "attach":
            from .repo_adapter import attach_repository
            result = attach_repository(args.repo)
            code = 0
        else:
            ledger = _ledger(_repo(args.repo), args.db)
            if args.command == "stats":
                result = ledger.stats()
            else:
                destination = Path(args.out).resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                result = {"exported": ledger.export_jsonl(destination, args.ticket_id), "output": str(destination)}
            code = 0
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return code
    except (ValueError, KeyError, OSError, PermissionError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
