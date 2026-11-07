"""EVREN teacher output validation.

Every teacher-generated task must pass a *strict* gate before it can
enter training:

  1. JSON schema conformance (no unknown fields).
  2. Leakage scan: no solution/hint keys or text anywhere in the
     proposal (the student must discover rules through interaction).
  3. Atom vocabulary check: every atom references known plates, switches,
     objects, and proper boolean values.
  4. Rule-shape check: non-empty clauses, no duplicate atoms.
  5. Solvability: the task must be solvable according to an
     *independent* solver over the environment's hidden dynamics.
  6. Duplicate check: no task identical to an existing pool member.
  7. Evaluation contamination check: the task must be disjoint from the
     frozen validation/test/OOD pools (training must never leak
     evaluation content).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from ..envs.gridworld import (
    ACTIONS, DOORS, PLATES, SWITCHES, TOPOLOGIES, EnvConfig, WS,
    atoms_from_state, make_ws, transition_with_hidden,
)
from .schema import (
    ATOM_ARITY, FORBIDDEN_KEYS, FORBIDDEN_TEXT_TOKENS,
    TASK_PROPOSAL_SCHEMA,
)

STATUS_APPROVED = "approved"
STATUS_INVALID_SCHEMA = "invalid_schema"
STATUS_LEAKED = "leaked"
STATUS_IMPOSSIBLE = "impossible"
STATUS_DUPLICATE = "duplicate"
STATUS_EVAL_CONTAMINATION = "eval_contamination"
STATUS_UNKNOWN = "unknown"

ALL_STATUSES = (
    STATUS_APPROVED, STATUS_INVALID_SCHEMA, STATUS_LEAKED,
    STATUS_IMPOSSIBLE, STATUS_DUPLICATE, STATUS_EVAL_CONTAMINATION,
    STATUS_UNKNOWN,
)


# ---------------------------------------------------------------------------
# Atom grammar helpers
# ---------------------------------------------------------------------------
def _is_bool(v) -> bool:
    return isinstance(v, bool)


def check_atom(atom) -> Optional[str]:
    """Return an error string for an invalid atom, else None."""
    if not isinstance(atom, (list, tuple)):
        return f"atom must be a list, got {atom!r}"
    if not atom or atom[0] not in ("load", "loadany", "empty", "sw"):
        return f"atom must start with one of load/loadany/empty/sw"
    kind = atom[0]
    arity = ATOM_ARITY[kind]
    if len(atom) != arity:
        return f"{kind} atom must have exactly {arity} elements, got {len(atom)}"
    if kind == "sw":
        subject, value = atom[1], atom[2]
        if subject not in SWITCHES:
            return f"sw atom subject must be one of {SWITCHES}"
        if not _is_bool(value):
            return f"sw atom value must be a JSON boolean (true/false)"
    else:
        subject = atom[1]
        if subject not in PLATES:
            return f"plate atom subject must be one of {PLATES}"
        if kind == "load":
            if atom[2] not in ("A", "B"):
                return f"load atom value must be 'A' or 'B'"
    return None


def normalize_rule(rule: list):
    """Convert a JSON rule (list of clauses) to internal representation:
    one clause -> frozenset (conjunctive); many -> tuple of frozensets
    (disjunctive). Returns (internal, error)."""
    if not isinstance(rule, list) or not rule:
        return None, "rule must be a non-empty list of clauses"
    clauses = []
    for clause in rule:
        if not isinstance(clause, list) or not clause:
            return None, "each clause must be a non-empty list of atoms"
        seen = set()
        for atom in clause:
            err = check_atom(atom)
            if err:
                return None, err
            canon = tuple(atom)
            if canon in seen:
                return None, f"duplicate atom in clause: {atom!r}"
            seen.add(canon)
        clauses.append(frozenset(seen))
    if len(clauses) == 1:
        return clauses[0], None
    # canonical form: clause order must not change task identity
    return tuple(sorted(clauses, key=str)), None


def proposal_to_config(raw: dict) -> "EnvConfig":
    """Build an EnvConfig from a validated proposal. Raises ValueError on
    malformed content (callers must validate first)."""
    d1, e1 = normalize_rule(raw["d1_rule"])
    d2, e2 = normalize_rule(raw["d2_rule"])
    if e1 or e2:
        raise ValueError(e1 or e2)
    return EnvConfig(
        topology=raw["topology"],
        d1_rule=d1,
        d2_rule=d2,
        d1_delay=bool(raw["d1_delay"]),
        d2_delay=bool(raw["d2_delay"]),
    )


def config_to_proposal(config: EnvConfig, task_id: Optional[str] = None,
                       difficulty_intent: str = "medium",
                       rationale: str = "") -> dict:
    """Serialize an EnvConfig into the canonical teacher-proposal JSON.

    Encoding: rule -> [clause, ...]; clause -> [atom, ...]; atom ->
    [kind, ...scalars].
    """
    def ser_clause(clause):
        # preserve atom ordering (kind-first), stable clause order
        return [list(a) for a in sorted(clause, key=str)]

    def ser_rule(r):
        if isinstance(r, tuple):          # disjunctive: tuple of clauses
            return [ser_clause(c) for c in sorted(r, key=str)]
        return [ser_clause(r)]            # conjunctive: single clause

    return {
        "task_id": task_id or f"local-{time.time_ns()}",
        "topology": config.topology,
        "d1_rule": ser_rule(config.d1_rule),
        "d2_rule": ser_rule(config.d2_rule),
        "d1_delay": config.d1_delay,
        "d2_delay": config.d2_delay,
        "difficulty_intent": difficulty_intent,
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Independent solver
# ---------------------------------------------------------------------------
def independent_solvable(config: EnvConfig, max_plan: int = 80) -> bool:
    """Independent solvability check for a candidate task.

    This is a *separate* BFS implementation from ``tasks.is_solvable``
    (which is used for the built-in pools). It runs over the
    environment's true hidden dynamics and verifies that a goal-reaching
    plan exists from the initial state within ``max_plan`` steps.
    """
    from collections import deque
    try:
        geom = TOPOLOGIES[config.topology]
    except KeyError:
        return False
    start = make_ws(geom)
    if start.pos == geom.goal:
        return True

    def key(st: WS):
        return (st.pos, st.objs["A"], st.objs["B"],
                (st.switch_state["S1"], st.switch_state["S2"]),
                (st.plate_load["P1"], st.plate_load["P2"]),
                (st.door_open["D1"], st.door_open["D2"]))

    q = deque([(start, 0)])
    seen = {key(start)}
    while q:
        st, depth = q.popleft()
        if depth >= max_plan:
            continue
        for a in ACTIONS:
            nxt, _ = transition_with_hidden(st, a, config, geom)
            if nxt is None:
                continue
            k = key(nxt)
            if k in seen:
                continue
            seen.add(k)
            if nxt.pos == geom.goal:
                return True
            q.append((nxt, depth + 1))
    return False


# ---------------------------------------------------------------------------
# Leakage scan
# ---------------------------------------------------------------------------
_LEAK_TEXT_RE = re.compile(
    r"|".join(re.escape(t.lower()) for t in FORBIDDEN_TEXT_TOKENS))


def scan_for_leakage(obj, path: str = "$") -> list:
    """Recursively scan for forbidden keys and hint-like text."""
    issues = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = str(k).lower()
            if any(f in kp for f in FORBIDDEN_KEYS) or kp in FORBIDDEN_KEYS:
                issues.append(f"{path}.{k}: forbidden key name")
            issues.extend(scan_for_leakage(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            issues.extend(scan_for_leakage(v, f"{path}[{i}]"))
    elif isinstance(obj, str):
        low = obj.lower()
        if _LEAK_TEXT_RE.search(low):
            issues.append(f"{path}: hint-like text")
        elif len(obj) > 2000:
            issues.append(f"{path}: suspiciously long string")
    return issues


# ---------------------------------------------------------------------------
# Schema check (manual, stdlib-only)
# ---------------------------------------------------------------------------
def _type_ok(value, spec) -> bool:
    ts = spec["type"]
    if ts == "string":
        return isinstance(value, str)
    if ts == "boolean":
        return isinstance(value, bool)
    if ts == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if ts == "array":
        return isinstance(value, list)
    if ts == "object":
        return isinstance(value, dict)
    if isinstance(ts, list):
        return any(_type_ok(value, {"type": t}) for t in ts)
    return False


def check_schema(obj, schema) -> list:
    """Minimal JSON-schema checker for TASK_PROPOSAL_SCHEMA."""
    errors = []
    if not isinstance(obj, dict):
        return ["proposal must be a JSON object"]
    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}))
        for k in obj:
            if k not in allowed:
                errors.append(f"unknown field '{k}' (additionalProperties=false)")
    for name, spec in schema.get("properties", {}).items():
        if name not in obj:
            continue
        val = obj[name]
        if not _type_ok(val, spec):
            errors.append(f"field '{name}': expected type {spec['type']}")
            continue
        if "enum" in spec and val not in spec["enum"]:
            errors.append(f"field '{name}': not in enum {spec['enum']}")
        if "minLength" in spec and isinstance(val, str) and len(val) < spec["minLength"]:
            errors.append(f"field '{name}': shorter than minLength")
        if "maxLength" in spec and isinstance(val, str) and len(val) > spec["maxLength"]:
            errors.append(f"field '{name}': longer than maxLength {spec['maxLength']}")
        if "minItems" in spec and isinstance(val, list) and len(val) < spec["minItems"]:
            errors.append(f"field '{name}': fewer items than minItems")
        if name == "d1_rule" or name == "d2_rule":
            for i, clause in enumerate(val):
                if not isinstance(clause, list) or len(clause) < 1:
                    errors.append(f"field '{name}[{i}]': must be a non-empty list")
                    continue
                for j, atom in enumerate(clause):
                    if (not isinstance(atom, list)
                            or len(atom) < 2 or len(atom) > 3):
                        errors.append(f"field '{name}[{i}][{j}]': atom must be "
                                      f"a 2- or 3-element list")
    for name in schema.get("required", []):
        if name not in obj:
            errors.append(f"missing required field '{name}'")
    return errors


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    raw: dict
    status: str = STATUS_UNKNOWN
    reasons: list = field(default_factory=list)
    config: Optional[EnvConfig] = None
    key: Optional[tuple] = None
    validated_at: float = 0.0
    solvable: bool = False

    @property
    def approved(self) -> bool:
        return self.status == STATUS_APPROVED

    def as_dict(self) -> dict:
        return {
            "task_id": (self.raw or {}).get("task_id"),
            "status": self.status,
            "reasons": self.reasons,
            "solvable": self.solvable,
            "key": list(map(str, self.key)) if self.key else None,
            "validated_at": self.validated_at,
        }


class TaskValidator:
    """Strict gate for teacher-generated task proposals."""

    def __init__(self, splits: dict = None, max_plan: int = 80):
        # splits: {"train": [...], "val": [...], "test": [...], "ood": [...]}
        self.splits = splits or {}
        self.max_plan = max_plan
        self._train_keys = {c.key for c in self.splits.get("train", [])}
        self._eval_keys = set()
        for kind in ("val", "test", "ood"):
            self._eval_keys |= {c.key for c in self.splits.get(kind, [])}
        # _seen = training pool + already-approved teacher tasks
        self._seen = set(self._train_keys)

    # -- duplicate bookkeeping ------------------------------------------
    def register(self, config: EnvConfig) -> None:
        """Mark a config as known (used as approved teacher tasks)."""
        self._seen.add(config.key)

    def is_known(self, config: EnvConfig) -> bool:
        return config.key in self._seen

    # -- main entry ------------------------------------------------------
    def validate(self, raw: dict) -> ValidationResult:
        res = ValidationResult(raw=raw, validated_at=time.time())
        if not isinstance(raw, dict):
            res.status = STATUS_INVALID_SCHEMA
            res.reasons.append("proposal is not a JSON object")
            return res

        # 1. leakage scan FIRST (defense-in-depth must run even when the
        # schema would otherwise reject the payload for unknown fields)
        leaked = scan_for_leakage(raw)
        if leaked:
            res.status = STATUS_LEAKED
            res.reasons.extend(leaked[:8])
            return res

        # 2. schema
        schema_errors = check_schema(raw, TASK_PROPOSAL_SCHEMA)
        if schema_errors:
            res.status = STATUS_INVALID_SCHEMA
            res.reasons.extend(schema_errors[:8])
            return res

        # 3-4. atom + rule shape (via conversion)
        try:
            config = proposal_to_config(raw)
        except ValueError as e:
            res.status = STATUS_INVALID_SCHEMA
            res.reasons.append(str(e))
            return res
        res.config = config
        res.key = config.key

        # 5. independent solvability
        res.solvable = independent_solvable(config, max_plan=self.max_plan)
        if not res.solvable:
            res.status = STATUS_IMPOSSIBLE
            res.reasons.append(
                "independent solver found no goal-reaching plan within "
                f"{self.max_plan} steps")
            return res

        # 6. evaluation contamination (checked BEFORE duplicates: eval
        # membership is never acceptable, regardless of training overlap)
        if config.key in self._eval_keys:
            res.status = STATUS_EVAL_CONTAMINATION
            res.reasons.append("task belongs to a frozen evaluation pool "
                               "(val/test/ood); it is rejected")
            return res

        # 7. duplicate
        if config.key in self._seen:
            res.status = STATUS_DUPLICATE
            res.reasons.append("task already exists in a training or "
                               "evaluation pool")
            return res

        res.status = STATUS_APPROVED
        res.reasons.append("schema ok, no leakage, solvable, novel, "
                           "disjoint from evaluation")
        return res


def serialize_result_batch(results: list) -> list:
    return [r.as_dict() for r in results]
