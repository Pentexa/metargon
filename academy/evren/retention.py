"""EVREN curriculum — retention (anti-forgetting) protocol.

The v0.1 experiment measured "forgetting" with a *single* post-hoc probe
whose result was identical for every condition (delta = +0.000). That is
uninformative: a one-shot probe cannot separate true retention of
previously-learned content from incidental later exposure.

This module implements the standard continual-learning measurement:

  * a **frozen substrate** — the same small task set for every condition,
    which every condition trains on with *matched exposure*;
  * **periodic probes** — the substrate is re-measured at fixed block
    boundaries with fresh evaluation episodes (never inside the training
    stream), giving a retention *curve* rather than one number;
  * **retention deltas** — late mean minus early mean per substrate item
    and in aggregate (positive = improved/stable, negative = forgotten).

The retention suite operates on classic *training-pool* configs. It never
touches val/test/OOD pools, and the probes never feed back into training.
"""

from __future__ import annotations

import random

from ..envs.gridworld import MechanismGrid
from ..envs.tasks import EpisodeSpec, difficulty_of
from ..eval.metrics import summarize_episodes


def sample_substrate(pool: list, n: int, seed: int) -> list:
    """Deterministically sample a frozen substrate subset from a training
    pool (drawn *without* replacement; capped at the pool size)."""
    rng = random.Random(seed)
    items = sorted(pool, key=lambda c: str(c.key))
    rng.shuffle(items)
    return items[:max(0, min(n, len(items)))]


class RetentionSuite:
    """Frozen retention substrate + matched probe runner."""

    def __init__(self, substrate: list, max_steps: int = 80,
                 episodes_per_config: int = 4, seeds=(0, 1),
                 label: str = "substrate"):
        self.substrate = list(substrate)
        self.max_steps = max_steps
        self.episodes_per_config = episodes_per_config
        self.seeds = tuple(seeds)
        self.label = label
        self.snaps = []          # [{block, records, summary}]

    # -- probing ------------------------------------------------------------
    def run_probe(self, agent_state: dict, agent_factory, episode_i: int) -> dict:
        """Measure current performance on the frozen substrate.

        ``agent_factory(seed)`` builds a fresh agent; ``agent_state`` is
        the *current* persistent training state (loaded into each probe
        instance). Probe episodes are evaluation-only: nothing is
        recorded back into training, and matched streams make probes
        comparable across conditions."""
        records = []
        ep = 0
        for cfg in self.substrate:
            for seed_i in self.seeds:
                seed = seed_i * 7 + 911
                for _ in range(self.episodes_per_config):
                    spec = EpisodeSpec(
                        config=cfg, max_steps=self.max_steps, noise_q=0.0,
                        spec_seed=seed * 1000 + ep, pool_tag="retention_probe",
                        difficulty=difficulty_of(cfg))
                    ep += 1
                    agent = agent_factory(seed + ep)
                    if agent_state is not None:
                        agent.load_state(agent_state)
                    env = MechanismGrid(spec.config, spec.max_steps,
                                        spec.noise_q)
                    obs = env.reset(spec.spec_seed)
                    agent.episode_start(spec, env.geom)
                    success = False
                    steps = 0
                    while steps < env.max_steps:
                        a = agent.act(obs)
                        before = obs
                        res = env.step(a)
                        agent.learn(before, a, res)
                        steps += 1
                        obs = res.observation
                        if res.terminated:
                            success = True
                            break
                    agent.finish_episode(success)
                    records.append({
                        "success": success, "steps": steps,
                        "episode": episode_i,
                        "seed": seed,
                        "config": str(spec.config.key),
                    })
        summary = summarize_episodes(records)
        self.snaps.append({"episode_seen": episode_i, "episode_count": None,
                           "summary": {k: v for k, v in summary.items()},
                           "records": records})
        return self.snaps[-1]

    # -- analysis -----------------------------------------------------------
    def retention_delta(self) -> float:
        """success_rate(last snapshot) - success_rate(first snapshot)."""
        if len(self.snaps) < 2:
            return 0.0
        return (self.snaps[-1]["summary"]["success_rate"]
                - self.snaps[0]["summary"]["success_rate"])

    def per_config_delta(self) -> dict:
        """Late minus early success per substrate configuration."""
        if not self.snaps:
            return {}
        def bykey(snap):
            out = {}
            for r in snap["records"]:
                out.setdefault(r["config"], []).append(1.0 if r["success"] else 0.0)
            return {k: sum(v) / len(v) for k, v in out.items()}
        first, last = bykey(self.snaps[0]), bykey(self.snaps[-1])
        out = {}
        for k in sorted(set(first) | set(last)):
            a = first.get(k, 0.0)
            b = last.get(k, 0.0)
            out[k] = {"first": a, "last": b, "delta": b - a}
        return out

    def summary(self) -> dict:
        return {
            "label": self.label,
            "substrate_n": len(self.substrate),
            "substrate_keys": [str(c.key) for c in self.substrate],
            "snapshots": [
                {"episode_seen": s["episode_seen"],
                 "success_rate": s["summary"]["success_rate"],
                 "mean_steps": s["summary"]["mean_steps"],
                 "n": s["summary"]["n"]}
                for s in self.snaps],
            "retention_delta": self.retention_delta(),
            "per_config": self.per_config_delta(),
        }
