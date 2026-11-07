"""METARGON Academy — EVREN LLM integration.

EVREN-hosted language models are used strictly as *curriculum teachers*:
they may propose tasks, variations and hypotheses, but they never supply
hidden answers to the student, never touch METARGON's decision-making
brain, and are never consulted during evaluation.

Design rules enforced in this package:

  * every teacher output is validated against strict environment schemas
    and then *independently* tested for solvability before approval;
  * impossible tasks, duplicates, invalid actions, and any output that
    leaks hidden solutions are rejected;
  * the API key is only ever read from the environment (or a git-ignored
    ``.env`` file); it is never hardcoded, printed, logged or committed;
  * automatic requests are bounded by a per-run quota (default 10 calls)
    and unlimited usage requires explicit configuration;
  * when EVREN is unavailable, the offline procedural generator keeps the
    Academy fully usable.
"""

__all__ = [
    "provider",
    "schema",
    "validator",
    "offline_generator",
    "curriculum_teacher",
    "task_registry",
    "pilot",
]
