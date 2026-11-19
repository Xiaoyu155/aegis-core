"""Role-coordinated repair state machine and conservative semantic three-way merge."""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
import difflib
import hashlib
import re
import threading
import time


ROLES = ("Architect", "Coder", "Adversarial Critic", "Verifier")
_TRANSITIONS = {
    "NEW": {"PLANNED", "ABORTED"}, "PLANNED": {"CODING", "ABORTED"},
    "CODING": {"CRITIQUE", "ABORTED"}, "CRITIQUE": {"VERIFYING", "CODING", "ABORTED"},
    "VERIFYING": {"ACCEPTED", "REJECTED", "CODING", "ABORTED"},
    "ACCEPTED": {"CODING", "ABORTED"}, "REJECTED": {"CODING", "ABORTED"}, "ABORTED": set(),
}


@dataclass
class _Candidate:
    candidate_id: str
    strategy: str
    patch: str
    patch_sha256: str
    critique: list[dict] = field(default_factory=list)
    critiqued: bool = False
    verification: dict | None = None
    accepted: bool = False
    score: float | None = None


class SwarmArbiter:
    """A deterministic state machine that records four explicit engineering roles."""

    def __init__(self, ticket_id: str, base_commit: str):
        if type(ticket_id) is not str or not ticket_id.strip(): raise ValueError("ticket_id is required")
        if type(base_commit) is not str or not re.fullmatch(r"[0-9a-fA-F]{7,64}", base_commit):
            raise ValueError("base_commit must be a hexadecimal Git object id")
        self.ticket_id, self.base_commit = ticket_id, base_commit.lower()
        self.state = "NEW"
        self.task = None
        self.breakdown = []
        self.invariants = []
        self.candidates: dict[str, _Candidate] = {}
        self.events = []
        self._lock = threading.RLock()

    def _move(self, target: str, role: str, payload=None):
        if target not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid swarm transition {self.state} -> {target}")
        if role not in ROLES: raise ValueError(f"unknown swarm role: {role}")
        previous = self.state
        self.state = target
        self.events.append({"index": len(self.events) + 1, "from": previous, "to": target,
                            "role": role, "at_monotonic": round(time.monotonic(), 6),
                            "payload": payload or {}})

    def plan(self, task: str, breakdown, *, invariants=()) -> dict:
        with self._lock:
            if self.state != "NEW": raise RuntimeError("Architect can only plan a new ticket")
            if type(task) is not str or not task.strip() or len(task) > 20000:
                raise ValueError("task must be nonempty and bounded")
            if not isinstance(breakdown, (list, tuple)) or not breakdown or any(
                    type(item) is not str or not item.strip() for item in breakdown):
                raise ValueError("Architect must provide a nonempty list of bounded work items")
            if not isinstance(invariants, (list, tuple)) or any(type(item) is not str or not item.strip() for item in invariants):
                raise ValueError("invariants must be nonempty strings")
            self.task, self.breakdown, self.invariants = task, list(breakdown), sorted(set(invariants))
            self._move("PLANNED", "Architect", {"work_items": self.breakdown, "invariants": self.invariants})
            self._move("CODING", "Coder", {"candidate_generation_open": True})
            return self.status()

    def submit(self, candidate_id: str, strategy: str, patch: str) -> dict:
        with self._lock:
            if self.state not in {"CODING", "REJECTED", "ACCEPTED"}:
                raise RuntimeError("Coder can only submit during the coding phase")
            if self.state in {"REJECTED", "ACCEPTED"}:
                self._move("CODING", "Coder", {"additional_candidate": True})
            if type(candidate_id) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", candidate_id):
                raise ValueError("candidate_id must be a bounded safe identifier")
            if candidate_id in self.candidates: raise ValueError("candidate_id is already registered")
            if type(strategy) is not str or not strategy.strip() or len(strategy) > 240:
                raise ValueError("strategy must be nonempty and bounded")
            if type(patch) is not str or not patch.strip() or len(patch) > 2_000_000:
                raise ValueError("candidate patch must be nonempty and at most 2MB")
            digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
            self.candidates[candidate_id] = _Candidate(candidate_id, strategy, patch, digest)
            self.events.append({"index": len(self.events) + 1, "from": self.state, "to": self.state,
                                "role": "Coder", "at_monotonic": round(time.monotonic(), 6),
                                "payload": {"candidate_id": candidate_id, "strategy": strategy,
                                            "patch_sha256": digest}})
            return {"candidate_id": candidate_id, "patch_sha256": digest, "state": self.state}

    def critique(self, candidate_id: str, findings, *, strategy_note="") -> dict:
        with self._lock:
            if self.state != "CODING": raise RuntimeError("Adversarial Critic requires coding proposals")
            candidate = self._candidate(candidate_id)
            if not isinstance(findings, (list, tuple)):
                raise ValueError("critic findings must be a list")
            normalized = []
            for finding in findings:
                if type(finding) is not dict or type(finding.get("severity")) is not str or type(finding.get("message")) is not str:
                    raise ValueError("each finding needs a severity and message")
                severity = finding["severity"].lower()
                if severity not in {"info", "low", "medium", "high", "critical"}:
                    raise ValueError("unknown finding severity")
                normalized.append({"severity": severity, "message": finding["message"][:2000],
                                   "file": str(finding.get("file", ""))[:512],
                                   "line": finding.get("line"), "mitigated": bool(finding.get("mitigated", False))})
            candidate.critique = normalized
            candidate.critiqued = True
            self.events.append({"index": len(self.events) + 1, "from": self.state, "to": self.state,
                                "role": "Adversarial Critic", "at_monotonic": round(time.monotonic(), 6),
                                "payload": {"candidate_id": candidate_id, "findings": normalized,
                                            "strategy_note": str(strategy_note)[:2000]}})
            return {"candidate_id": candidate_id, "blocking_findings": sum(
                row["severity"] in {"high", "critical"} and not row["mitigated"] for row in normalized)}

    def verify(self, candidate_id: str, *, static_passed: bool, dynamic_passed: bool,
               l3_approved: bool, receipts=None, score: float | None = None) -> dict:
        with self._lock:
            if self.state not in {"CODING", "CRITIQUE", "VERIFYING"}:
                raise RuntimeError("Verifier requires submitted proposals")
            candidate = self._candidate(candidate_id)
            if not candidate.critiqued and self.state == "CODING":
                raise RuntimeError("candidate must receive an adversarial critique before verification")
            if any(type(value) is not bool for value in (static_passed, dynamic_passed, l3_approved)):
                raise ValueError("verification decisions must be boolean receipts")
            if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1):
                raise ValueError("score must be between zero and one")
            if self.state == "CODING": self._move("CRITIQUE", "Adversarial Critic", {"candidate_id": candidate_id})
            if self.state == "CRITIQUE": self._move("VERIFYING", "Verifier", {"candidate_id": candidate_id})
            blocking = any(item["severity"] in {"high", "critical"} and not item["mitigated"]
                           for item in candidate.critique)
            candidate.accepted = static_passed and dynamic_passed and l3_approved and not blocking
            candidate.score = float(score) if score is not None else None
            candidate.verification = {"static_passed": static_passed, "dynamic_passed": dynamic_passed,
                                      "l3_approved": l3_approved, "receipts": receipts or {},
                                      "blocking_findings": blocking, "accepted": candidate.accepted}
            next_state = "ACCEPTED" if candidate.accepted else "REJECTED"
            self._move(next_state, "Verifier", candidate.verification | {"candidate_id": candidate_id})
            return {"candidate_id": candidate_id, "accepted": candidate.accepted,
                    "blocking_findings": blocking, "state": self.state}

    def _candidate(self, candidate_id):
        try: return self.candidates[candidate_id]
        except KeyError: raise KeyError(f"unknown candidate: {candidate_id}") from None

    def select(self) -> dict | None:
        with self._lock:
            passing = [item for item in self.candidates.values() if item.accepted]
            if not passing: return None
            winner = sorted(passing, key=lambda item: (-(item.score or 0), item.candidate_id))[0]
            return {"candidate_id": winner.candidate_id, "strategy": winner.strategy,
                    "patch_sha256": winner.patch_sha256, "score": winner.score}

    def abort(self, reason: str):
        with self._lock:
            self._move("ABORTED", "Architect", {"reason": str(reason)[:2000]})
            return self.status()

    def status(self) -> dict:
        with self._lock:
            return {"schema_version": "aegis.swarm.v1", "ticket_id": self.ticket_id,
                    "base_commit": self.base_commit, "state": self.state, "task": self.task,
                    "breakdown": list(self.breakdown), "invariants": list(self.invariants),
                    "candidate_count": len(self.candidates),
                    "candidates": [{"candidate_id": item.candidate_id, "strategy": item.strategy,
                                    "patch_sha256": item.patch_sha256, "accepted": item.accepted,
                                    "verification": item.verification, "critique": item.critique}
                                   for item in self.candidates.values()],
                    "winner": self.select(), "events": list(self.events), "roles": list(ROLES)}


def semantic_three_way_merge(base: str, left: str, right: str, *, language="python") -> dict:
    """Merge nonoverlapping symbol edits, or report semantic/textual overlap."""
    if any(type(value) is not str for value in (base, left, right)):
        raise TypeError("three-way merge inputs must be source text")
    if not isinstance(language, str) or language.lower() not in {"python", "py", "javascript", "typescript", "js", "ts", "tsx"}:
        raise ValueError("language must be Python, JavaScript or TypeScript")
    if left == right:
        return {"status": "merged", "content": left, "conflicts": [], "strategy": "identical_proposals",
                "left_symbols": [], "right_symbols": []}
    python_mode = language.lower() in {"python", "py"}
    try:
        left_symbols = _changed_symbols(base, left, python_mode)
        right_symbols = _changed_symbols(base, right, python_mode)
    except (SyntaxError, ValueError) as error:
        return {"status": "conflict", "content": None,
                "conflicts": [{"kind": "parse_error", "message": str(error)}],
                "strategy": "parse_before_merge", "left_symbols": [], "right_symbols": []}
    overlap = sorted(left_symbols & right_symbols)
    invariant_conflicts = _shared_invariant_conflicts(base, left, right, left_symbols, right_symbols) if python_mode else []
    if overlap or invariant_conflicts:
        conflicts = ([{"kind": "same_symbol", "symbols": overlap}] if overlap else []) + invariant_conflicts
        return {"status": "conflict", "content": None, "conflicts": conflicts,
                "strategy": "semantic_symbol_overlap", "left_symbols": sorted(left_symbols),
                "right_symbols": sorted(right_symbols)}
    edits_left = _edits(base, left)
    edits_right = _edits(base, right)
    merged_edits = []
    conflicts = []
    for first in edits_left:
        for second in edits_right:
            if _edits_overlap(first, second):
                if first == second:
                    continue
                conflicts.append({"kind": "overlapping_text_ranges", "left_base_range": first[:2],
                                  "right_base_range": second[:2]})
    if conflicts:
        return {"status": "conflict", "content": None, "conflicts": conflicts,
                "strategy": "semantic_and_textual_overlap", "left_symbols": sorted(left_symbols),
                "right_symbols": sorted(right_symbols)}
    for edit in edits_left + edits_right:
        if edit not in merged_edits: merged_edits.append(edit)
    lines = base.splitlines(keepends=True)
    for start, end, replacement in sorted(merged_edits, key=lambda item: (item[0], item[1]), reverse=True):
        lines[start:end] = replacement
    content = "".join(lines)
    try:
        if python_mode: ast.parse(content)
    except SyntaxError as error:
        return {"status": "conflict", "content": None,
                "conflicts": [{"kind": "merged_parse_error", "line": error.lineno, "message": error.msg}],
                "strategy": "parse_after_merge", "left_symbols": sorted(left_symbols),
                "right_symbols": sorted(right_symbols)}
    return {"status": "merged", "content": content, "conflicts": [],
            "strategy": "ast_symbols_plus_disjoint_diff_hunks", "left_symbols": sorted(left_symbols),
            "right_symbols": sorted(right_symbols)}


def _changed_symbols(base: str, variant: str, python_mode: bool) -> set[str]:
    if python_mode:
        before, after = _python_symbols(base), _python_symbols(variant)
        return {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    from .ast_slicer import _js_symbols
    base_map, variant_map = _js_symbols(base), _js_symbols(variant)
    return {name for name in set(base_map) | set(variant_map)
            if base_map.get(name, {}).get("source") != variant_map.get(name, {}).get("source")}


def _python_symbols(source: str):
    tree = ast.parse(source)
    result = {}
    def collect(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.ClassDef):
                result.update({prefix + node.name: "<class>"})
                collect(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    name = target.id if isinstance(target, ast.Name) else None
                    if name: result[prefix + name] = ast.dump(node, include_attributes=False)
    collect(tree.body)
    return result


def _edits(base, variant):
    old = base.splitlines(keepends=True); new = variant.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    return [(i1, i2, new[j1:j2]) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def _edits_overlap(first, second):
    a1, a2, _ = first; b1, b2, _ = second
    if a1 == a2 and b1 == b2: return a1 == b1
    if a1 == a2: return b1 <= a1 < b2
    if b1 == b2: return a1 <= b1 < a2
    return max(a1, b1) < min(a2, b2)


def _shared_invariant_conflicts(base, left, right, left_changed, right_changed):
    tree = ast.parse(base)
    invariant_names = set()
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
                    if re.search(r"(?:^|_)(?:MAX|MIN|LIMIT|CAP|STATE|INVARIANT|THRESHOLD)(?:$|_)", target.id.upper()):
                        invariant_names.add(target.id)
    def referenced(changed, source):
        try: syntax = ast.parse(source)
        except SyntaxError: return set()
        index = {}
        for node in ast.walk(syntax):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                index[node.name] = node
        reads = set()
        for symbol in changed:
            node = index.get(symbol.split(".")[-1])
            if node:
                reads.update(child.id for child in ast.walk(node) if isinstance(child, ast.Name)
                             and isinstance(child.ctx, ast.Load))
        return reads & invariant_names
    shared = sorted(referenced(left_changed, left) & referenced(right_changed, right))
    return [{"kind": "shared_invariant", "symbols": shared}] if shared else []
