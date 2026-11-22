"""Offline-first Contract-to-Replay orchestration for structured repair tournaments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

from .experience_bank import ExperienceBank
from .impact_analyzer import ImpactAnalyzer
from .ledger import Ledger
from .probe_synth import compile_structured_probe
from .tournament import Candidate, CandidateTournament


class SelfHealLoop:
    """Compile a contract, prove its probe red, then learn across gated repair rounds."""

    def __init__(self, repo_root: str | Path, *, database_path: str | Path | None = None,
                 dataset_path: str | Path | None = None):
        self.root = Path(repo_root).resolve()
        self.database_path = (Path(database_path).resolve() if database_path is not None
                              else Path(self._git("rev-parse", "--path-format=absolute", "--git-common-dir")) / "aegis" / "ledger.sqlite")
        self.dataset_path = (Path(dataset_path).resolve() if dataset_path is not None
                             else self.root / "datasets" / "grpo_tournament_pairs.jsonl")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self, *, contract: str, probe_spec: dict, candidate_rounds=None,
            candidate_provider=None, dynamic_commands=None, mutation_commands=None,
            static_commands=None, runtime_profile_command=None, allowed_paths=None,
            max_rounds: int = 3, max_workers: int = 4,
            timeout_sec: float = 60, max_context_tokens: int = 4000,
            max_total_tokens: int = 20000, max_wall_clock_sec: float = 600,
            ticket_id: str | None = None, answers: dict | None = None) -> dict:
        started = time.monotonic()
        if type(contract) is not str or not contract.strip() or len(contract) > 20000:
            raise ValueError("contract must be nonempty text of at most 20000 characters")
        if type(max_rounds) is not int or not 1 <= max_rounds <= 8:
            raise ValueError("max_rounds must be in 1..8")
        if type(max_total_tokens) is not int or max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")
        if type(max_context_tokens) is not int or max_context_tokens < 128:
            raise ValueError("max_context_tokens must be at least 128")
        if type(max_wall_clock_sec) not in (int, float) or not math.isfinite(max_wall_clock_sec) or max_wall_clock_sec <= 0:
            raise ValueError("max_wall_clock_sec must be positive and finite")
        if candidate_provider is None and (not isinstance(candidate_rounds, list) or not candidate_rounds):
            raise ValueError("candidate_rounds or candidate_provider is required")
        root_status = self._git("status", "--porcelain", "--untracked-files=all").splitlines()
        dataset_relative = (self.dataset_path.relative_to(self.root).as_posix()
                            if self.dataset_path.is_relative_to(self.root) else None)
        dirty = [line for line in root_status if line[3:].strip().strip('"') != dataset_relative]
        if dirty:
            raise ValueError("autopilot requires a clean repository baseline")
        base = self._git("rev-parse", "HEAD")
        source_path = (self.root / probe_spec.get("path", "")).resolve()
        if not source_path.is_relative_to(self.root) or not source_path.is_file():
            raise ValueError("probe target must be an existing source file inside the repository")
        tracked = set(filter(None, self._git("ls-files", "-z").split("\0")))
        relative_target = source_path.relative_to(self.root).as_posix()
        if relative_target not in tracked:
            raise ValueError("probe target must be tracked at the frozen baseline")

        ledger = Ledger(self.database_path)
        bank = ExperienceBank(self.database_path)
        expanded = self._expand(contract, answers or {}, ledger.path)
        if expanded.questions:
            return {"status": "MISSING_CONSTRAINT", "contract": contract,
                    "questions": list(expanded.questions),
                    "phases": ["contract", "expand"], "base_commit": base}
        impact = ImpactAnalyzer(self.root)
        context = impact.compress_context({"file": relative_target, "symbol": probe_spec.get("symbol")},
                                          max_context_tokens=max_context_tokens)
        expanded_document = expanded.to_dict()
        expanded_document["context"]["structured_probe"] = probe_spec
        expanded_document["context"]["impact_summary"] = {
            "changed_files": context["changed_files"], "token_count": context["token_count"],
            "compression_ratio": context["compression_ratio"],
            "critical_symbol_recall": context["critical_symbol_recall"],
            "graph_metrics": context["graph_metrics"]}
        probe = compile_structured_probe(probe_spec, self.root, base, timeout=timeout_sec)
        generated_command = probe["command"]
        dynamic = [generated_command, *(dynamic_commands or [])]
        mutation_probe_command = [sys.executable, "-B", "-m", "aegis_core.ast_mutator",
                                  "--repo", ".", "--path", relative_target,
                                  "--test", generated_command[-1], "--symbol", probe_spec["symbol"],
                                  "--max-mutants", "2",
                                  # Keep the probe viable inside short tournament budgets while
                                  # allowing slower Windows interpreters to start on normal runs.
                                  "--timeout", str(min(2.0, max(1.0, timeout_sec / 5.0)))]
        mutations = [mutation_probe_command, *(mutation_commands or [])]
        tournament = CandidateTournament(self.root, dynamic_commands=dynamic,
                                         mutation_commands=mutations, static_commands=static_commands,
                                         timeout_sec=timeout_sec, max_workers=max_workers,
                                         dataset_path=self.dataset_path, allowed_paths=allowed_paths,
                                         probe_patch=probe["patch"],
                                         runtime_profile_command=runtime_profile_command)
        ticket = ticket_id or "autopilot-" + uuid.uuid4().hex[:16]
        ledger.create_ticket(ticket, contract, expanded_document, "def repair() -> None:\n    ...")
        lease = ledger.acquire_lease(ticket, "aegis-autopilot", ttl=max(60.0, max_wall_clock_sec * 2))
        if not lease:
            raise PermissionError("autopilot ticket is already leased")
        state = ledger.get_state(ticket)
        ledger.transition(ticket, lease, "queued", "running", state_s0=state,
                          action_a0={"op": "autopilot_start", "contract_sha256": _sha(contract),
                                     "probe_sha256": probe["hashes"]},
                          actual_state_s1={"status": "running", "phase": "contract_expand_synth"}, reward=0)
        history, token_spend, terminal, reason, final_winner = [], 0, "failed", "no_candidate_passed", None
        replay_patch = ""
        previous_diagnosis = None
        rounds = min(max_rounds, len(candidate_rounds)) if candidate_provider is None else max_rounds
        try:
            for index in range(rounds):
                elapsed = time.monotonic() - started
                if elapsed >= max_wall_clock_sec:
                    reason = "BUDGET_EXCEEDED_WALL_CLOCK"
                    break
                query = " ".join((contract, previous_diagnosis.get("failure_code", "") if previous_diagnosis else "",
                                  " ".join(previous_diagnosis.get("mutation_examples", [])) if previous_diagnosis else ""))
                hints = bank.search(query, top_k=5, min_similarity=.05)
                context_packet = {"round": index + 1, "contract": contract,
                                  "expanded_spec": expanded_document,
                                  "impact_context": context, "diagnosis": previous_diagnosis,
                                  "experience_hints": hints, "previous_rounds": history,
                                  "remaining_tokens": max_total_tokens - token_spend,
                                  "remaining_wall_clock_sec": max(0, max_wall_clock_sec - elapsed)}
                raw_candidates = (candidate_provider(context_packet) if candidate_provider is not None
                                  else candidate_rounds[index])
                candidates = [_candidate(item) for item in raw_candidates]
                if not candidates:
                    reason = "no_candidates_for_round"
                    break
                estimate = sum(candidate.token_cost if candidate.token_cost is not None else _patch_tokens(candidate.patch)
                               for candidate in candidates)
                if estimate > max_total_tokens - token_spend:
                    reason = "BUDGET_EXCEEDED_TOKENS"
                    break
                result = tournament.run(candidates, contract=contract)
                spent = sum(item.get("token_cost", 0) for item in result["evaluations"])
                token_spend += spent
                round_record = {"round": index + 1, "candidate_count": len(candidates),
                                "winner": result["winner"], "accepted_count": result["accepted_count"],
                                "pareto_front": result["pareto_front"], "token_spend": spent,
                                "elapsed_sec": round(time.monotonic() - started, 6),
                                "tournament": result, "experience_hints": hints}
                history.append(round_record)
                winner_candidate = next((item for item in candidates if item.candidate_id == result["winner"]), None)
                wall_exceeded = time.monotonic() - started > max_wall_clock_sec
                tokens_exceeded = token_spend > max_total_tokens
                if winner_candidate and not (wall_exceeded or tokens_exceeded):
                    best_result = next(item for item in result["evaluations"] if item["candidate_id"] == result["winner"])
                    terminal, reason, final_winner = "succeeded", "all_candidate_gates_passed", winner_candidate
                    replay_patch = winner_candidate.patch
                    actual = {"status": "succeeded", "baseline_failed": True, "after_passed": True,
                              "passed": True, "gate_failure": None, "cheat_detected": False,
                              "final": _success_receipt(best_result),
                              "changed_files": result["evaluations"][0].get("changed_files", []),
                              "diagnosis": {"kind": "FailureDiagnosis", "failure_code": "RECOVERED"},
                              "winner": result["winner"], "world_model_outcome": "gate_passed"}
                    action = _repair_action(winner_candidate, contract, probe, probe_spec, best_result)
                    self._record(ledger, ticket, lease, action, actual, reward=1.0)
                    break
                previous_diagnosis = _diagnose_tournament(result, index + 1)
                if wall_exceeded or tokens_exceeded:
                    reason = "BUDGET_EXCEEDED_WALL_CLOCK" if wall_exceeded else "BUDGET_EXCEEDED_TOKENS"
                has_next = index + 1 < rounds and reason not in {"BUDGET_EXCEEDED_WALL_CLOCK", "BUDGET_EXCEEDED_TOKENS"}
                best_eval = max(result["evaluations"], key=lambda item: (item.get("score", 0), item["candidate_id"]))
                best_candidate = next((item for item in candidates if item.candidate_id == best_eval["candidate_id"]), candidates[0])
                replay_patch = best_candidate.patch
                actual = {"status": "running" if has_next else "failed", "baseline_failed": True,
                          "after_passed": False, "passed": False,
                          "gate_failure": previous_diagnosis["failure_code"], "cheat_detected": False,
                          "final": _failure_receipt(best_eval),
                          "changed_files": best_eval.get("changed_files", []),
                          "diagnosis": previous_diagnosis, "candidate_id": best_candidate.candidate_id,
                          "world_model_outcome": "assertion_error"}
                action = _repair_action(best_candidate, contract, probe, probe_spec, best_eval)
                self._record(ledger, ticket, lease, action, actual, reward=-1.0)
                if not has_next:
                    terminal = "failed"
                    if reason == "no_candidate_passed":
                        reason = "self_heal_rounds_exhausted"
                    break
            else:
                # The loop can exhaust without a terminal transition only when
                # every round failed and no explicit break was taken.
                state = ledger.get_state(ticket)
                if state["status"] == "running":
                    diagnosis = previous_diagnosis or {"kind": "FailureDiagnosis", "failure_code": "NO_REPAIR"}
                    terminal, reason = "failed", "self_heal_rounds_exhausted"
                    ledger.transition(ticket, lease, "running", "failed", state_s0=state,
                                      action_a0={"op": "autopilot_finish", "reason": reason},
                                      actual_state_s1={"status": "failed", "reason": reason,
                                                       "diagnosis": diagnosis}, reward=0)
        except BaseException:
            current = ledger.get_ticket(ticket)
            if current["status"] == "running":
                state = ledger.get_state(ticket)
                ledger.transition(ticket, lease, "running", "failed", state_s0=state,
                                  action_a0={"op": "autopilot_abort"},
                                  actual_state_s1={"status": "failed", "reason": "execution_error"}, reward=-1)
            raise
        finally:
            if ledger.get_ticket(ticket)["status"] == "running":
                state = ledger.get_state(ticket)
                ledger.transition(ticket, lease, "running", "failed", state_s0=state,
                                  action_a0={"op": "autopilot_finish", "reason": reason},
                                  actual_state_s1={"status": "failed", "reason": reason}, reward=0)
        selected, recoveries, learning_counts = ledger.learning_selection(0, limit=1000)
        memory = bank.ingest_verified_recoveries(recoveries)
        retrieved = bank.search(contract, top_k=10, min_similarity=.05)
        from .probe_synth import render_structured_probe
        from .time_travel import TimeTravelReplay
        probe_path, probe_source, probe_command, _ = render_structured_probe(probe["rule"])
        frozen_command = [sys.executable if item == "{python}" else item for item in probe_command]
        replay_snapshot = TimeTravelReplay(self.root, ledger.path).capture(
            ticket, commit=base, probe=probe_source, probe_path=probe_path, probe_command=frozen_command,
            mutation_seeds=[probe["random_seed"]], candidate_patch=replay_patch,
            world_context={"contract": expanded_document, "impact": context,
                           "failure_diagnosis": previous_diagnosis, "terminal_status": terminal})
        dataset_digest = hashlib.sha256(self.dataset_path.read_bytes()).hexdigest() if self.dataset_path.is_file() else None
        return {"schema_version": "aegis.autopilot.v1", "status": terminal, "reason": reason,
                "ticket_id": ticket, "base_commit": base, "phases": ["contract", "expand", "synth_probe",
                    "tournament", "mutation_probe", "l3_audit", "failure_diagnosis", "multi_round_self_heal",
                    "online_replay_learn"],
                "expanded_spec": expanded_document, "impact_context": context,
                "synthesized_probe": {"case_count": probe["case_count"],
                                      "case_categories": probe["case_categories"],
                                      "baseline_true_red": probe["red"]["failure_code"]},
                "rounds": history, "rounds_attempted": len(history),
                "converged_round": len(history) if terminal == "succeeded" else None,
                "winner": ({"candidate_id": final_winner.candidate_id,
                            "strategy": final_winner.strategy,
                            "patch": final_winner.patch,
                            "patch_sha256": _sha(final_winner.patch)} if final_winner else None),
                "diagnosis": previous_diagnosis, "experience_hints": retrieved,
                "time_travel_snapshot": {key: replay_snapshot[key] for key in
                                         ("ticket_id", "commit_oid", "archive_sha256", "probe_sha256",
                                          "candidate_patch_sha256", "mutation_seeds")},
                "learning": {"verified_recoveries": len(recoveries), "selected_model_traces": len(selected),
                             "selection_counts": learning_counts, "experience_ingest": memory,
                             "experience_bank": bank.status(), "replayed_hints": retrieved,
                             "grpo_dataset_sha256": dataset_digest},
                "budget": {"max_rounds": max_rounds, "max_total_tokens": max_total_tokens,
                           "tokens_used": token_spend, "max_wall_clock_sec": max_wall_clock_sec,
                           "elapsed_sec": round(time.monotonic() - started, 6)},
                "ledger": ledger.stats()}

    def _expand(self, contract: str, answers: dict, experience_db: str):
        from .pm_expander import expand_requirement
        return expand_requirement(contract, self.root, answers=answers, experience_db=experience_db)

    def _record(self, ledger, ticket, lease, action, actual, *, reward):
        previous = ledger.get_state(ticket)
        status = actual["status"]
        ledger.transition(ticket, lease, "running", status, state_s0=previous,
                          action_a0=action, actual_state_s1=actual, reward=reward,
                          token_cost=action["token_cost"])

    def _git(self, *args):
        from .sandbox import run_git
        return run_git(self.root, *args).stdout.decode("utf-8").strip()


def run_autopilot(repo_root: str | Path, **kwargs) -> dict:
    return SelfHealLoop(repo_root, database_path=kwargs.pop("database_path", None),
                        dataset_path=kwargs.pop("dataset_path", None)).run(**kwargs)


def _candidate(item) -> Candidate:
    return item if isinstance(item, Candidate) else Candidate(
        candidate_id=item.get("candidate_id", ""), strategy=item.get("strategy", ""),
        patch=item.get("patch", ""), token_cost=item.get("token_cost"), metadata=item.get("metadata"))


def _patch_tokens(patch: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", patch, re.UNICODE))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repair_action(candidate: Candidate, contract: str, probe: dict,
                   probe_spec: dict, evaluation: dict) -> dict:
    return {"op": "repair_attempt", "candidate_id": candidate.candidate_id,
            "strategy": candidate.strategy, "patch": candidate.patch,
            "patch_sha256": _sha(candidate.patch), "contract": contract,
            "probe_kind": "unit", "probe_script_sha256": probe["hashes"],
            "target_files": [probe_spec["path"]], "token_cost": evaluation.get("token_cost", _patch_tokens(candidate.patch)),
            "tournament_score": evaluation.get("score", 0)}


def _success_receipt(evaluation: dict) -> dict:
    receipts = evaluation.get("dynamic", []) + evaluation.get("mutation", [])
    receipt = next((item for item in receipts if item.get("passed")), None)
    if receipt is None:
        return {"exit_code": 0, "passed": True, "source": "tournament_gate_summary"}
    return {"exit_code": receipt["exit_code"], "passed": receipt["passed"],
            "stdout": receipt.get("stdout", ""), "stderr": receipt.get("stderr", ""),
            "command": receipt.get("command", [])}


def _failure_receipt(evaluation: dict) -> dict:
    for gate in ("static", "dynamic", "mutation"):
        for receipt in evaluation.get(gate, []):
            if not receipt.get("passed"):
                return {"exit_code": receipt.get("exit_code"), "passed": False,
                        "stdout": receipt.get("stdout", ""), "stderr": receipt.get("stderr", ""),
                        "command": receipt.get("command", []), "gate": gate}
    return {"exit_code": 0, "passed": False, "source": "independent_l3_audit",
            "audit": evaluation.get("audit", {}).get("findings", [])}


def _diagnose_tournament(result: dict, attempt: int) -> dict:
    evaluations = result.get("evaluations", [])
    failures = []
    findings = []
    contract_differences = []
    mutation_examples = []
    frames = []
    for evaluation in evaluations:
        if evaluation.get("error"):
            failures.append({"candidate_id": evaluation.get("candidate_id"),
                             "failure_code": "PATCH_OR_WORKTREE_ERROR", "message": evaluation.get("message")})
        for gate in ("static", "dynamic", "mutation"):
            for receipt in evaluation.get(gate, []):
                if not receipt.get("passed"):
                    failures.append({"candidate_id": evaluation.get("candidate_id"), "gate": gate,
                                     "failure_code": "TIMEOUT_KILLED" if receipt.get("timed_out") else gate.upper() + "_PROBE_FAILED",
                                     "exit_code": receipt.get("exit_code"),
                                     "command": receipt.get("command"),
                                     "evidence": (receipt.get("stdout", "") + "\n" + receipt.get("stderr", ""))[-2000:]})
                    excerpt = receipt.get("stdout", "") + "\n" + receipt.get("stderr", "")
                    failed_ids = re.findall(r"(?m)^FAILED\s+[^\n]*?\[([^\]]+)\]", excerpt)
                    mutation_examples.extend(failed_ids)
                    if "AEGIS_ACCEPTANCE" in excerpt:
                        mismatch = next((line.strip() for line in excerpt.splitlines()
                                         if "AEGIS_ACCEPTANCE" in line or "assert " in line), "")
                        if mismatch:
                            contract_differences.append({"code": "ASSERTION_MISMATCH", "path": "tests/test_aegis_probe.py",
                                                         "line": None, "detail": mismatch[:500],
                                                         "candidate_id": evaluation.get("candidate_id"), "gate": gate})
                    frames.extend({"file": match.group(1), "line": int(match.group(2))}
                                  for match in re.finditer(r"([\w./\\-]+\.py):(\d+)", excerpt))
        findings.extend(evaluation.get("audit", {}).get("findings", []))
    failure_code = ("L3_AUDIT_REJECTED" if findings else
                    "STATIC_PROBE_FAILED" if any(item.get("gate") == "static" for item in failures) else
                    "DYNAMIC_PROBE_FAILED" if any(item.get("gate") == "dynamic" for item in failures) else
                    "MUTATION_PROBE_FAILED")
    contract_differences.extend({"code": item.get("code"), "path": item.get("path"),
                                 "line": item.get("line"), "detail": item.get("detail")}
                                for item in findings)
    return {"kind": "FailureDiagnosis", "attempt": attempt, "category": "candidate_rejected",
            "failure_code": failure_code, "frames": _unique_dicts(frames)[:20],
            "contract_differences": contract_differences[:20],
            "mutation_examples": sorted(set(mutation_examples))[:20],
            "candidate_failures": failures[:40], "evidence_excerpt": json.dumps(failures[:8], ensure_ascii=False)[:4000],
            "next_action": "Use the failing probe cases and independent audit findings to target a different repair strategy."}


def _unique_dicts(items):
    output, seen = [], set()
    for item in items:
        key = (item.get("file"), item.get("line"))
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
