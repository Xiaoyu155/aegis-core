"""Strict planning protocol, with a portable JSON Schema and stdlib validator."""
import json
import math


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items}


_STRING = {"type": "string", "minLength": 1}
_STATE = {"type": "object"}
_STEP = _object({"step_idx": {"type": "integer", "minimum": 0},
                 "state_s0": _STATE, "action_a0": _STATE,
                 "predicted_next_state": _STATE})
_SUBTICKET = _object({"ticket_id": _STRING, "intent": _STRING,
                      "contract_stub": _STRING, "depends_on": _array(_STRING)})
MissingConstraintQuestion = _object({
    "kind": {"const": "MissingConstraintQuestion"}, "ticket_id": _STRING,
    "question": _STRING, "missing_constraints": _array(_STRING),
    "options": _array(_STRING)})
CausalRolloutPlan = _object({
    "kind": {"const": "CausalRolloutPlan"}, "ticket_id": _STRING,
    "initial_state": _STATE, "steps": _array(_STEP),
    "sub_tickets": _array(_SUBTICKET)})
FaultAttribution = _object({
    "kind": {"const": "FaultAttribution"}, "ticket_id": _STRING,
    "step_idx": {"type": "integer", "minimum": 0},
    "predicted_next_state": _STATE, "actual_state_s1": _STATE,
    "cause": _STRING, "evidence": _array(_STRING),
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "repair_action": _STATE})
SCHEMAS = {"MissingConstraintQuestion": MissingConstraintQuestion,
           "CausalRolloutPlan": CausalRolloutPlan, "FaultAttribution": FaultAttribution}
JSON_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema",
               "title": "Aegis V1 planning protocol", "oneOf": list(SCHEMAS.values())}


def _validate(value, schema, location):
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{location}: unexpected discriminator")
    kind = schema.get("type")
    matches = {"object": type(value) is dict, "array": type(value) is list,
               "string": type(value) is str, "integer": type(value) is int,
               "number": type(value) in (int, float)}
    if kind and not matches[kind]:
        raise ValueError(f"{location}: expected {kind}")
    if kind == "object" and "properties" in schema:
        if set(value) != set(schema["required"]):
            raise ValueError(f"{location}: missing or unknown fields")
        for key, child in schema["properties"].items():
            _validate(value[key], child, location + "." + key)
    if kind == "array":
        for idx, child in enumerate(value):
            _validate(child, schema["items"], f"{location}[{idx}]")
    if kind == "string" and len(value.strip()) < schema.get("minLength", 0):
        raise ValueError(f"{location}: empty string")
    if kind in {"integer", "number"}:
        if not math.isfinite(value) or value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{location}: number out of range")


def _json_value(value):
    """Reject Python-only values even inside arbitrary world-state objects."""
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        for child in value.values():
            _json_value(child)
    elif type(value) is list:
        for child in value:
            _json_value(child)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError("not a JSON value")


def validate_message(payload: dict) -> dict:
    _json_value(payload)
    if type(payload) is not dict or type(payload.get("kind")) is not str or payload["kind"] not in SCHEMAS:
        raise ValueError("unknown protocol kind")
    _validate(payload, SCHEMAS[payload["kind"]], "message")
    if payload["kind"] == "CausalRolloutPlan":
        steps = payload["steps"]
        state = payload["initial_state"]
        for index, step in enumerate(steps):
            if step["step_idx"] != index or step["state_s0"] != state:
                raise ValueError("rollout must have contiguous steps and state continuity")
            state = step["predicted_next_state"]
        tickets = payload["sub_tickets"]
        by_id = {ticket["ticket_id"]: ticket for ticket in tickets}
        if len(by_id) != len(tickets) or payload["ticket_id"] in by_id:
            raise ValueError("duplicate or parent sub-ticket ID")
        for ticket in tickets:
            dependencies = ticket["depends_on"]
            if len(set(dependencies)) != len(dependencies) or any(dep not in by_id for dep in dependencies):
                raise ValueError("duplicate or missing DAG dependency")
        # Kahn traversal avoids recursion limits on long valid DAGs.
        pending = {key: set(value["depends_on"]) for key, value in by_id.items()}
        while pending:
            ready = {key for key, dependencies in pending.items() if not dependencies}
            if not ready:
                raise ValueError("cyclic sub-ticket DAG")
            pending = {key: dependencies - ready for key, dependencies in pending.items() if key not in ready}
    elif payload["kind"] == "MissingConstraintQuestion" and not payload["missing_constraints"]:
        raise ValueError("at least one missing constraint is required")
    return json.loads(json.dumps(payload, allow_nan=False))
