"""EVREN curriculum teacher.

Turns EVREN-hosted LLMs into *curriculum teachers*: given the Academy's
environment grammar, a teacher proposes structured procedural tasks as
strict JSON. Teachers never see student state, never receive evaluation
pools, and are never asked for solutions or hints. Proposals are opaque
JSON that only the validator may consume.

This module is deliberately free of any knowledge about the student
agent's internals.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from .provider import (
    PRIMARY_TEACHER, SECONDARY_TEACHER, EvrenError, EvrenProvider,
    EvrenQuotaExceeded,
)
from .schema import TASK_PROPOSAL_SCHEMA

TeacherModel = str  # "primary" | "secondary"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a curriculum teacher for METARGON Academy, a \
research platform where a student agent learns by interacting with a small \
grid world. Your only job is to propose *new training tasks*: the hidden \
wiring of the world (doors rules). You must NEVER include solutions, hints, \
optimal action sequences, or any content that would hand the student the \
answer. You only propose environment wiring.

Constraints:
- Output ONLY one JSON object, nothing else (no markdown fences, no \
comments, no text).
- The JSON must EXACTLY match the schema specified by the caller.
- Use only the atom vocabulary and topologies the caller lists.
- Prefer tasks that are novel but plausible; propose a difficulty intent."""

_USER_TEMPLATE = """Environment grammar for task proposals:
- Topologies: TP1, TP2 (public world geometry; you do not need to know \
the wall layout).
- Atom vocabulary (each atom is a JSON array):
    ["sw", "S1"|"S2", true|false]        switch state
    ["load", "P1"|"P2", "A"|"B"]          plate loaded with exactly that object
    ["loadany", "P1"|"P2"]                plate loaded with any object
    ["empty", "P1"|"P2"]                  plate unloaded
- A rule is a JSON array of clauses. A clause is an array of atoms.
  One clause = conjunctive rule (all atoms must hold).
  Multiple clauses = disjunctive rule (any clause may hold).
- "d1_rule" and "d2_rule" are required; each must have at least one \
clause; each clause at least one atom.
- "d1_delay" / "d2_delay" are booleans (per-door one-step delay).
- "difficulty_intent" must be one of "easy", "medium", "hard".
- "rationale": a short (<=240 chars) design note about the *task*, NOT \
about how to solve it. Never mention what a learner should do.
- "task_id": a short unique string like "evren-glm-0001".

Required JSON shape (fill values, do not keep the placeholder comments):
{
  "task_id": "evren-glm-0001",
  "topology": "TP1",
  "d1_rule": [[["sw", "S2", true]]],
  "d2_rule": [[["load", "P1", "A"]]],
  "d1_delay": false,
  "d2_delay": false,
  "difficulty_intent": "easy",
  "rationale": "single-switch conjunctive rule"
}

A two-atom conjunctive rule (S2 on AND P1 empty) looks like:
  "d1_rule": [[["sw", "S2", true], ["empty", "P1"]]]

A disjunctive rule (S1 off OR P2 empty) looks like:
  "d2_rule": [[["sw", "S1", false]], [["empty", "P2"]]]

Propose exactly one task now. Output the JSON object only."""


def build_messages(task_slot: int = 1, seed_note: str = "") -> list:
    user = _USER_TEMPLATE
    if seed_note:
        user += f"\n(Generation seed note: {seed_note})"
    user += f"\nThis is candidate slot {task_slot} of the current batch."
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> Optional[dict]:
    """Pull the first balanced JSON object from model output.

    Handles stray prose around the object and markdown code fences.
    Returns None when no parseable object exists.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    # strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except ValueError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Teacher attempt
# ---------------------------------------------------------------------------
@dataclass
class TeacherAttempt:
    slot: int
    status: str                 # "ok" | "quota" | "parse_error" | "error"
    raw: Optional[dict] = None
    error: str = ""
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "slot": self.slot, "status": self.status,
            "raw": self.raw, "error": self.error, "meta": self.meta,
        }


class CurriculumTeacher:
    """A teacher backed by an EVREN-hosted model (or the offline
    generator when the provider is unavailable)."""

    def __init__(self, provider: EvrenProvider, model: TeacherModel = "primary",
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 seed: int = 0):
        self.provider = provider
        self.model = model
        self.display_name = (PRIMARY_TEACHER if model == "primary"
                             else SECONDARY_TEACHER)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    # -- model resolution (once) ----------------------------------------
    def resolve_model_id(self) -> str:
        return self.provider.resolve_model(self.display_name)

    # -- propose ---------------------------------------------------------
    def propose_task(self, slot: int = 1,
                     seed: Optional[int] = None) -> TeacherAttempt:
        """One teacher call for one candidate task.

        The slot (starting at 1) is used in the prompt so proposals are
        contextualized. ``seed`` is passed to the API when supported and
        recorded in provenance.
        """
        seed = self.seed if seed is None else seed
        meta = {
            "model_display": self.display_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
            "slot": slot,
        }
        try:
            model_id = self.resolve_model_id()
            meta["model_id"] = model_id
        except EvrenError as e:
            return TeacherAttempt(slot, "error", error=f"discovery: {e}",
                                  meta=meta)
        messages = build_messages(task_slot=slot, seed_note=f"seed={seed}")
        try:
            resp = self.provider.chat(
                messages, model=model_id,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=seed)
        except EvrenQuotaExceeded as e:
            return TeacherAttempt(slot, "quota", error=str(e), meta=meta)
        except EvrenError as e:
            return TeacherAttempt(slot, "error", error=str(e), meta=meta)
        meta.update({
            "model": resp.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
            "latency_ms": round(resp.latency_ms, 1),
            "finish_reason": resp.finish_reason,
        })
        parsed = extract_json_object(resp.content)
        if parsed is None:
            return TeacherAttempt(
                slot, "parse_error",
                error="response contained no parseable JSON object",
                meta=meta)
        return TeacherAttempt(slot, "ok", raw=parsed, meta=meta)

    def propose_batch(self, n: int, seed: Optional[int] = None) -> list:
        out = []
        for i in range(n):
            out.append(self.propose_task(slot=i + 1, seed=seed))
        return out
