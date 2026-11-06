"""EVREN teacher — JSON schema for structured task proposals.

A teacher proposal describes *environment wiring only* (topology and
hidden door rules). It never contains solutions, hints, policies, or any
other content the student should discover through interaction. The schema
is strict: any unknown field is rejected (``additionalProperties: false``),
and every atom must belong to the environment's known vocabulary.

JSON representation of a rule (each level is a JSON array):
  * rule        = [ clause, clause, ... ]          (1 clause = conjunctive;
                                                 several = disjunctive)
  * clause      = [ atom, atom, ... ]
  * atom        = [ scalar, ... ] with encodings:
      ["sw", switch, bool]           switch state (3 elements)
      ["load", plate, object]        plate loaded with exactly that object
      ["loadany", plate]             plate loaded with any object (2)
      ["empty", plate]               plate unloaded (2)

Example — conjunctive rule "S2 on AND P1 empty":
    "d1_rule": [[["sw", "S2", true], ["empty", "P1"]]]

Example — disjunctive rule "S1 off OR P2 empty":
    "d2_rule": [[["sw", "S1", false]], [["empty", "P2"]]]
"""

from __future__ import annotations

# Atom kinds and their allowed subjects/values.
ATOM_KINDS = ("load", "loadany", "empty", "sw")
PLATES = ("P1", "P2")
SWITCHES = ("S1", "S2")
OBJECTS = ("A", "B")
TOPOLOGIES = ("TP1", "TP2")
DIFFICULTY_INTENTS = ("easy", "medium", "hard")

# Number of elements per atom kind (matching gridworld's internal tuples).
ATOM_ARITY = {"load": 3, "sw": 3, "loadany": 2, "empty": 2}

# Fields forbidden by name anywhere in a proposal (defense in depth:
# these never carry legitimate teacher content).
FORBIDDEN_KEYS = frozenset({
    "solution", "answer", "hint", "hints", "reveal", "groundtruth",
    "ground_truth", "true_rules", "secret", "expert_trace", "policy",
    "action_sequence", "optimal_policy", "cheat", "gold_rules",
    "hidden_rules_answer", "goal_policy", "teacher_trace", "plan",
})

# Forbidden tokens inside string values (heuristic leakage scan).
FORBIDDEN_TEXT_TOKENS = (
    "the answer", "the solution", "hint:", "secret rule", "true rule",
    "door opens when", "rule is", "objectively", "just do", "simply ",
    "expert", "golden", "cheat", "optimal actions", "do this sequence",
    "n,e,s,w", "move", "toggle " "load ", "push ",
)

# The strict JSON schema (draft 2020-12 style, checked manually).
TASK_PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "topology": {"type": "string", "enum": list(TOPOLOGIES)},
        "d1_rule": {"type": "array", "minItems": 1,
                     "items": {"type": "array", "minItems": 1,
                               "items": {
                                   "type": "array", "minItems": 2,
                                   "maxItems": 3,
                                   "items": {"type": ["string", "boolean", "number"]},
                               }}},
        "d2_rule": {"type": "array", "minItems": 1,
                     "items": {"type": "array", "minItems": 1,
                               "items": {
                                   "type": "array", "minItems": 2,
                                   "maxItems": 3,
                                   "items": {"type": ["string", "boolean", "number"]},
                               }}},
        "d1_delay": {"type": "boolean"},
        "d2_delay": {"type": "boolean"},
        "difficulty_intent": {"type": "string",
                              "enum": list(DIFFICULTY_INTENTS)},
        "rationale": {"type": "string", "maxLength": 512},
    },
    "required": ["task_id", "topology", "d1_rule", "d2_rule",
                 "d1_delay", "d2_delay"],
    "additionalProperties": False,
}
