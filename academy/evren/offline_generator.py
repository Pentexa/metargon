"""EVREN teacher — offline procedural task generator.

The Academy remains fully usable when EVREN is unavailable. This
generator produces candidate task proposals of exactly the same JSON
shape as the teacher's, from the environment's documented rule
vocabulary, deterministically seeded. Candidates still pass through the
same strict validator (schema, leakage, solvability, duplicates,
evaluation contamination) — approval is never assumed.

This generator also serves as the *baseline without EVREN-generated
tasks* used in comparisons.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from ..envs.gridworld import EnvConfig
from ..envs.tasks import CORE_RULESETS, OOD_RULESETS, difficulty_of
from .validator import TaskValidator, ValidationResult

_SOURCE = "offline_generator"


def _rule_samples() -> dict:
    """All rule templates available to the offline generator."""
    out = {}
    for tp, (d1, d2) in CORE_RULESETS.items():
        out.setdefault(tp, {"d1": [], "d2": []})
        out[tp]["d1"].extend(d1)
        out[tp]["d2"].extend(d2)
    for tp, (d1, d2) in OOD_RULESETS.items():
        out.setdefault(tp, {"d1": [], "d2": []})
        out[tp]["d1"].extend(d1)
        out[tp]["d2"].extend(d2)
    return out


class OfflineTaskGenerator:
    """Deterministic procedural candidate generator (stdlib only)."""

    def __init__(self, seed: int = 0, splits: dict = None,
                 validator: Optional[TaskValidator] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.splits = splits or {}
        self.validator = validator or TaskValidator(splits=self.splits)
        self._samples = _rule_samples()
        self._topologies = sorted(self._samples)
        self._counter = 0

    # -- candidate generation -------------------------------------------
    def propose_raw(self, task_id: Optional[str] = None) -> dict:
        """One raw candidate proposal (canonical JSON shape)."""
        tp = self.rng.choice(self._topologies)
        pool = self._samples[tp]
        d1 = self.rng.choice(pool["d1"])
        d2 = self.rng.choice(pool["d2"])
        # occasionally build a disjunctive rule from two clauses
        if self.rng.random() < 0.20:
            d1 = (d1, self.rng.choice(pool["d1"]))
        if self.rng.random() < 0.20:
            d2 = (d2, self.rng.choice(pool["d2"]))
        config = EnvConfig(
            topology=tp, d1_rule=d1, d2_rule=d2,
            d1_delay=bool(self.rng.getrandbits(1)),
            d2_delay=bool(self.rng.getrandbits(1)))
        diff = difficulty_of(config)
        self._counter += 1
        from .validator import config_to_proposal
        return config_to_proposal(
            config,
            task_id=task_id or f"offline-{self.seed}-{self._counter}",
            difficulty_intent=diff.bin,
            rationale="offline procedural generator (deterministic seed "
                      f"{self.seed})")

    # -- generate + validate ---------------------------------------------
    def generate(self, n: int, max_attempts: int = 200,
                 source_note: str = _SOURCE) -> list:
        """Attempt to get ``n`` approved candidate tasks.

        Returns a list of ValidationResults (approved ones first). The
        counter is bounded by ``max_attempts`` to avoid infinite loops
        when the space is exhausted or heavily duplicated.
        """
        results = []
        approved = 0
        attempts = 0
        while approved < n and attempts < max_attempts:
            attempts += 1
            raw = self.propose_raw()
            res = self.validator.validate(raw)
            if res.approved:
                approved += 1
                res.reasons.append(f"source={source_note} seed={self.seed}")
                self.validator.register(res.config)
            results.append(res)
        return results


def run_offline_pilot(seed: int, n: int, splits: dict,
                      registry=None, max_attempts: int = 300):
    """Generate, validate and record ``n`` approved offline tasks."""
    gen = OfflineTaskGenerator(seed=seed, splits=splits)
    results = gen.generate(n, max_attempts=max_attempts)
    approved = [r for r in results if r.approved]
    if registry is not None:
        for r in approved:
            meta = {
                "source": _SOURCE,
                "seed": seed,
                "generator": "OfflineTaskGenerator",
            }
            registry.add_approved(_task_from_result(r, meta))
        for r in results:
            if not r.approved:
                registry.add_rejected(r, {"source": _SOURCE, "seed": seed})
    return {"results": results, "approved": approved}


def _task_from_result(res: ValidationResult, meta: dict = None):
    from .task_registry import ApprovedTask
    raw = res.raw
    return ApprovedTask(
        task_id=raw["task_id"],
        proposal=raw,
        config=res.config,
        model="offline",
        model_id="offline_generator",
        temperature=0.0,
        max_tokens=0,
        seed=int(meta.get("seed", 0)),
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,
        difficulty_intent=raw.get("difficulty_intent", "medium"),
        rationale=raw.get("rationale", ""),
        approved_at=time.time(),
        provenance_note=meta.get("source", _SOURCE),
    )
