"""Expand a short requirement using source evidence and reviewed experience."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

from .ast_slicer import ImportGraph
from .runner import retrieve


@dataclass(frozen=True)
class AcceptanceCriterion:
    kind: str
    expected: str


@dataclass(frozen=True)
class ExpandedSpec:
    intent: str
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    edge_cases: tuple[str, ...]
    engineering_invariants: tuple[str, ...]
    context: dict
    experience: tuple[dict, ...]
    business_parameters: dict
    questions: tuple[dict, ...]

    def to_dict(self):
        return {"kind": "ExpandedSpec",
                "status": "MISSING_CONSTRAINT" if self.questions else "READY",
                **asdict(self)}


def _business_parameters(intent, answers):
    relevant = {}
    explicit = {}
    if re.search(r"drawdown|回撤", intent, re.I):
        relevant["max_drawdown"] = "What drawdown threshold should trigger the policy? Supply a fraction between 0 and 1."
        percent = re.search(r"(?<![\w.-])(-?\d+(?:\.\d+)?)\s*[%％]", intent)
        if percent:
            explicit["max_drawdown"] = float(percent[1]) / 100
    if re.search(r"liquidat|close positions?|平仓", intent, re.I):
        relevant["close_method"] = "Should positions close at market or with limit orders? Supply market or limit."
        market = re.search(r"market order|市价", intent, re.I)
        limit = re.search(r"limit order|限价", intent, re.I)
        if bool(market) != bool(limit):
            explicit["close_method"] = "market" if market else "limit"
    if re.search(r"circuit.break|unlock|熔断|解锁", intent, re.I):
        relevant["unlock_condition"] = "What should unlock the circuit? Supply next_utc_midnight or human."
        midnight = re.search(r"UTC midnight|次日|00:00", intent, re.I)
        human = re.search(r"manual unlock|人工解锁", intent, re.I)
        if bool(midnight) != bool(human):
            explicit["unlock_condition"] = "next_utc_midnight" if midnight else "human"
    for key in sorted(set(re.findall(r"\{\{([a-z][a-z0-9_]{0,63})\}\}", intent))):
        if re.search(r"(?:^|_)(?:password|secret|api_key|access_token|credential)(?:_|$)", key):
            raise ValueError("business answers cannot contain credentials")
        relevant.setdefault(key, f"What business value should {key} use?")
    if not isinstance(answers, dict) or set(answers) - relevant.keys():
        raise ValueError("answers must name business parameters referenced by the requirement")
    parameters = {**explicit, **answers}
    for key, value in parameters.items():
        if type(value) not in (str, int, float, bool) or isinstance(value, str) and not value.strip():
            raise ValueError(f"invalid business parameter {key}")
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"invalid business parameter {key}")
        if key == "max_drawdown" and (type(value) not in (int, float) or not 0 < value <= 1):
            raise ValueError("max_drawdown must be a fraction in (0,1]")
        if key == "close_method" and value not in ("market", "limit"):
            raise ValueError("close_method must be market or limit")
        if key == "unlock_condition" and value not in ("next_utc_midnight", "human"):
            raise ValueError("unlock_condition must be next_utc_midnight or human")
    return parameters, tuple({"parameter": key, "question": question}
                             for key, question in relevant.items() if key not in parameters)


def _existing_drawdown(graph, candidates):
    locations = {}
    for candidate in candidates:
        relative = candidate["file"]
        module = graph.paths.get(relative)
        if module is None or relative in locations:
            continue
        for node in graph.modules[module][2].body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if any(isinstance(target, ast.Name) and target.id.lower() in {"max_drawdown", "drawdown_threshold"}
                   for target in targets):
                value = node.value
                if not (isinstance(value, ast.Constant) and type(value.value) in (int, float)
                        and 0 < value.value <= 1):
                    return None
                locations.setdefault(relative, []).append(value.value)
    values = {value for entries in locations.values() for value in entries}
    if len(values) == 1:
        return values.pop(), sorted(locations)
    return None


def expand_requirement(intent, repo, *, answers=None, experience_db=None, graph=None, retrieval=None):
    if not isinstance(intent, str) or not intent.strip() or len(intent) > 10000:
        raise ValueError("a nonempty requirement of up to 10000 characters is required")
    intent = intent.strip()
    parameters, questions = _business_parameters(intent, {} if answers is None else answers)
    graph = graph or ImportGraph(repo)
    retrieval = retrieve(repo, intent, graph=graph) if retrieval is None else retrieval
    candidates = retrieval["candidates"]
    context = dict(retrieval["context"])
    if (any(question["parameter"] == "max_drawdown" for question in questions)
            and re.search(r"\b(?:existing|current|unchanged|preserve)\b|沿用|现有|保持", intent, re.I)):
        existing = _existing_drawdown(graph, candidates)
        if existing is not None:
            parameters["max_drawdown"], sources = existing
            questions = tuple(question for question in questions if question["parameter"] != "max_drawdown")
            context["business_parameter_sources"] = {"max_drawdown": sources}
    subject = f"{candidates[0]['file']}::{candidates[0]['symbol']}" if candidates else "the affected public entry point"
    experience = ()
    if experience_db is not None and Path(experience_db).is_file():
        from .experience_bank import ExperienceBank
        query = " ".join([intent, *(row["file"] + " " + row["symbol"] for row in candidates)])
        bank = ExperienceBank(experience_db)
        experience = tuple(bank.search(intent, top_k=5, min_similarity=.12)
                           or bank.search(query, top_k=5, min_similarity=.12))
    cases = ["Empty or minimum valid input", "Invalid or malformed input",
             "Repeated operation following a failed attempt"]
    invariants = ["Preserve existing callable signatures and data shapes unless the requirement changes them.",
                  "Validate input before persisting a change and keep state consistent after a failed operation."]
    lowered = intent.lower()
    normal = f"At {subject}, demonstrate the requested result for a representative valid input ({intent})."
    boundary = f"At {subject}, verify empty/minimum input and exercise both sides of each stated or existing limit."
    fallback = f"At {subject}, verify malformed input or a failed dependency produces the expected error or safe recovery."
    cache = re.search(r"cache|缓存|projection|投影", lowered)
    if cache:
        normal = f"At {subject}, repeated valid inputs return the same result as the uncached computation."
        boundary = "Exercise the cache at capacity and immediately before, at and after expiry when TTL applies."
        fallback = "A cache miss, expiry or invalid cache entry recomputes from the source without stale results."
        cases.extend(["Repeated input across consecutive frames", "Capacity and expiry transitions"])
        invariants.append("Cache storage must stay bounded and eviction must keep the uncached result equivalent.")
    if re.search(r"save|storage|persist|存档|持久化", lowered):
        saved = f"At {subject}, saving and loading valid state round-trips all supported fields."
        recovery = "Recover from missing, truncated or invalid state without persisting a partially decoded value."
        normal = normal + " " + saved if cache else saved
        fallback = fallback + " " + recovery if cache else recovery
        cases.extend(["Missing or truncated saved payload", "Legacy and current schema versions"])
    if "max_drawdown" in parameters:
        threshold = parameters["max_drawdown"]
        boundary = f"Exercise drawdown immediately below, exactly at and immediately above {threshold:.12g}."
    elif any(question["parameter"] == "max_drawdown" for question in questions):
        boundary = "Exercise drawdown immediately below, exactly at and immediately above the supplied trigger threshold."
    for record in experience:
        cause = record["cause"]
        if record["resolution"] == "gate_rejected" and cause not in {"unknown", "failed"}:
            case = f"Check recurrence of the observed failure mode {cause} (experience {record['experience_id']})."
            if case not in cases:
                cases.append(case)
    return ExpandedSpec(intent, (AcceptanceCriterion("normal", normal), AcceptanceCriterion("boundary", boundary),
                                 AcceptanceCriterion("fallback", fallback)), tuple(cases),
                        tuple(invariants), context, experience, parameters, questions)
