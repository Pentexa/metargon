"""METARGON Academy — evaluator (Phase 6).

Independent measurements on frozen task sets. The evaluator refuses to run
if the evaluation pool overlaps the training pool. Each policy is given
the identical episode stream for a seed (the stream is generated once per
seed and reused across policies — matched evaluation).
"""

from __future__ import annotations

import itertools
import random

from ..envs.gridworld import MechanismGrid
from ..envs.tasks import EpisodeSpec, make_episodes
from .metrics import summarize_episodes, paired_difference
from .protocol import spec_for


class Evaluator:
    def __init__(self, splits: dict, seed: int = 0,
                 episodes_per_seed: int = 240,
                 max_steps: int = 80,
                 noise_q: float = 0.0,
                 switch_half: bool = False,
                 pool_tag: str = "test"):
        self.splits = splits
        self.seed = seed
        self.episodes_per_seed = episodes_per_seed
        self.max_steps = max_steps
        self.noise_q = noise_q
        self.switch_half = switch_half
        self.pool_tag = pool_tag
        self._isolation_checked = False

    # -- isolation -------------------------------------------------------
    def check_isolation(self, pool) -> None:
        train_keys = {c.key for c in self.splits["train"]}
        pool_keys = {c.key for c in pool}
        overlap = train_keys & pool_keys
        if overlap:
            raise RuntimeError(
                f"EVALUATION LEAK: {len(overlap)} configuration(s) in the "
                f"evaluation pool also appear in the training pool.")
        self._isolation_checked = True

    # -- streams ---------------------------------------------------------
    def episode_stream(self, pool, seed: int) -> list:
        episodes = make_episodes(
            pool, self.pool_tag, seed, self.episodes_per_seed,
            max_steps=self.max_steps, noise_q=self.noise_q,
            switch_half=self.switch_half)
        return episodes

    # -- runner ----------------------------------------------------------
    def run_policy(self, agent_factory, pool, n_seeds: int = 30) -> dict:
        """Evaluate a policy on `n_seeds` paired seeds. Returns per-seed
        records and summary. Records are produced by the *same* episode
        stream for every policy (matched evaluation)."""
        self.check_isolation(pool)
        all_records = []
        per_seed = []
        for seed_i in range(n_seeds):
            seed = self.seed * 1000 + seed_i
            stream = self.episode_stream(pool, seed)
            agent = agent_factory(seed)
            records = []
            for spec in stream:
                env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
                obs = env.reset(spec.spec_seed)
                agent.episode_start(spec_for(agent, spec), env.geom)
                success = False
                steps = 0
                door_events = 0
                while steps < env.max_steps:
                    action = agent.act(obs)
                    before = obs
                    res = env.step(action)
                    agent.learn(before, action, res)
                    steps += 1
                    door_events += len(res.events)
                    obs = res.observation
                    if res.terminated:
                        success = True
                        break
                agent.finish_episode(success)
                records.append({"success": success, "steps": steps,
                                "door_events": door_events,
                                "seed": seed})
            all_records.extend(records)
            per_seed.append({
                "seed": seed,
                "summary": summarize_episodes(records),
            })
        summary = summarize_episodes(all_records)
        per_seed_success = [p["summary"]["success_rate"] for p in per_seed]
        return {
            "policy": getattr(agent_factory(0), "name", "?"),
            "pool_tag": self.pool_tag,
            "n_seeds": n_seeds,
            "summary": summary,
            "per_seed": per_seed_success,
            "records": all_records,
        }

    # -- paired comparison -----------------------------------------------
    def compare(self, results_a: dict, results_b: dict) -> dict:
        return {
            "policy_a": results_a["policy"],
            "policy_b": results_b["policy"],
            "paired_difference": paired_difference(
                results_a["per_seed"], results_b["per_seed"],
                seed=self.seed),
        }


def compare_all(evaluator: Evaluator, pool, factories: dict,
                n_seeds: int = 30) -> dict:
    """Evaluate all policies; return results plus pairwise comparisons vs a
    reference (first factory)."""
    results = {}
    for name, factory in factories.items():
        results[name] = evaluator.run_policy(factory, pool, n_seeds=n_seeds)
    comparisons = {}
    reference = next(iter(factories))
    for name in factories:
        if name == reference:
            continue
        comparisons[f"{reference}_vs_{name}"] = evaluator.compare(
            results[reference], results[name])
    return {"results": results, "comparisons": comparisons}
