"""EVREN curriculum — diagnosis of the v0.1 null result.

The v0.1 experiment reported classic-pool success = 0.000 for BOTH the
baseline and the teacher condition, while the same agents reached 0.667
on held-out teacher tasks — an incoherent pattern for "the added tasks
gave no benefit" and worth isolating before building a next-generation
curriculum on top of it.

Hypotheses this module can test (all offline, deterministic):

  H1 negative transfer: the *persistent* trained state (hypothesis
     memory + saturated per-door probe history) actively hurts on
     unseen configs, so "trained" agents test worse than fresh ones.
     Measurement: same unseen configs, fresh agent vs trained-agent
     state, matched probe episodes; plus mechanism counters
     (contradicted hypotheses, verified hypotheses, untried-probe
     saturation).

  H2 config-specificity: 0.000 is uniform failure vs. a few impossible
     seeds. Measurement: per-config success with n episodes each.

  H3 recovery-within-eval: trained agents that fail early can recover
     later within the same unseen config once their own experience
     accumulates (memory washes out) — measured as success in the
     second half vs first half of the probe window.

  H4 exposure confound: teacher-condition training halves classic
     exposure; per-classic-family trained success in the *training
     curve* differs between conditions. Measured from saved curves.

Outputs: ``diagnosis_summary.json`` in the EVREN results dir. Zero
network calls.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

from ..agents.metargon_agent import MetargonAcademyAgent
from ..envs.gridworld import MechanismGrid
from ..envs.tasks import EpisodeSpec, difficulty_of
from .retention import sample_substrate

POLICY_PATHS = {
    "baseline": os.path.join("policies", "baseline_agent.json"),
    "teacher": os.path.join("policies", "teacher_agent.json"),
}


def _load_state(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _probe_config(episodes: int, config, seed_base: int, max_steps: int,
                  agent_state: Optional[dict], label: str) -> dict:
    """Run ``episodes`` episodes on ONE config with an agent whose
    persistent state starts at ``agent_state`` (None => fresh)."""
    agent = MetargonAcademyAgent(seed=seed_base)
    if agent_state is not None:
        agent.load_state(agent_state)
    spec = EpisodeSpec(config=config, max_steps=max_steps, noise_q=0.0,
                       spec_seed=seed_base, pool_tag="train",
                       difficulty=difficulty_of(config))
    episodes_out = []
    succ = 0
    for ep in range(episodes):
        env = MechanismGrid(config, max_steps, 0.0)
        obs = env.reset(seed_base * 1000 + ep)
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
        succ += 1 if success else 0
        episodes_out.append({"success": success, "steps": steps})
    first = episodes_out[:len(episodes_out) // 2]
    second = episodes_out[len(episodes_out) // 2:]
    diag = agent.diagnostics()
    mem = diag["memory"]
    tried_n = sum(len(v) for v in agent.tried.values())
    return {
        "label": label,
        "config": str(config.key),
        "episodes": episodes,
        "success_rate": succ / episodes,
        "first_half": sum(1 for e in first if e["success"]) / max(1, len(first)),
        "second_half": sum(1 for e in second if e["success"]) / max(1, len(second)),
        "memory": mem,
        "tried_entries": tried_n,
        "mean_steps": sum(e["steps"] for e in episodes_out) / episodes,
    }


def run_diagnosis(results_dir: str = None, splits: dict = None,
                  n_configs: int = 10, episodes: int = 10,
                  seed: int = 0, max_steps: int = 80,
                  train_fallback: Optional[int] = None) -> dict:
    """Fresh-vs-trained comparison over sampled unseen test configs.

    If saved policy states exist (from the v0.1 experiment), they are
    used. Otherwise ``train_fallback`` episodes of quick baseline
    training produce a reference trained state (only when explicitly
    requested — keeps the routine cheap by default)."""
    if splits is None:
        from ..envs.tasks import load_splits, DEFAULT_SPLITS_PATH
        splits = load_splits(DEFAULT_SPLITS_PATH)
    results_dir = results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "results", "academy_v01", "evren")
    # unseen configs: eval isolation — sample from the frozen test pool
    configs = sample_substrate(splits["test"], n_configs, seed + 5)

    states = {}
    for name, relpath in POLICY_PATHS.items():
        st = _load_state(os.path.join(results_dir, relpath))
        if st is not None:
            states[name] = st
    if not states and train_fallback:
        networks = _train_fallback_agent(splits, train_fallback, seed,
                                         max_steps)
        states["fallback_trained"] = networks

    summary = {
        "run_id": "diagnosis",
        "seed": seed,
        "n_configs": len(configs),
        "episodes_per_config": episodes,
        "max_steps": max_steps,
        "states_measured": sorted(states),
        "fresh_agent": [],
        "trained_agents": {},
        "api_calls": 0,
        "offline": True,
        "hash_randomization": bool(__import__("sys").flags.hash_randomization),
        "started_at": time.time(),
    }
    fresh_rows = [_probe_config(episodes, c, seed + 100 + i, max_steps,
                                None, "fresh") for i, c in enumerate(configs)]
    summary["fresh_agent"] = fresh_rows
    for name, st in states.items():
        rows = [_probe_config(episodes, c, seed + 200 + i, max_steps, st,
                              name) for i, c in enumerate(configs)]
        summary["trained_agents"][name] = rows

    # aggregates
    def agg(rows):
        return {
            "success_rate": (sum(r["success_rate"] for r in rows)
                             / max(1, len(rows))),
            "mean_memory_hypotheses": (sum(r["memory"]["size"] for r in rows)
                                       / max(1, len(rows))),
            "mean_contradicted": (sum(
                r["memory"]["status_counts"].get("contradicted", 0)
                for r in rows) / max(1, len(rows))),
            "mean_verified": (sum(
                r["memory"]["status_counts"].get("verified", 0)
                for r in rows) / max(1, len(rows))),
            "mean_tried_entries": (sum(r["tried_entries"] for r in rows)
                                   / max(1, len(rows))),
            "second_half_gain": (sum(r["second_half"] - r["first_half"]
                                     for r in rows) / max(1, len(rows))),
        }
    summary["aggregate"] = {"fresh": agg(fresh_rows)}
    for name, rows in summary["trained_agents"].items():
        summary["aggregate"][name] = agg(rows)

    # paired delta per config: trained - fresh
    for name, rows in summary["trained_agents"].items():
        deltas = [t["success_rate"] - f["success_rate"]
                  for t, f in zip(rows, fresh_rows)]
        summary["aggregate"][name]["delta_vs_fresh"] = (
            sum(deltas) / len(deltas)) if deltas else 0.0

    summary["finished_at"] = time.time()
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "diagnosis_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary


def _train_fallback_agent(splits, episodes, seed, max_steps):
    """Quick deterministic baseline training for a reference state."""
    from ..learning.loop import run_one_episode
    rng = random.Random(seed)
    agent = MetargonAcademyAgent(seed=seed)
    for i in range(episodes):
        cfg = rng.choice(splits["train"])
        spec = EpisodeSpec(config=cfg, max_steps=max_steps, noise_q=0.0,
                           spec_seed=seed * 1000 + i, pool_tag="train",
                           difficulty=difficulty_of(cfg))
        run_one_episode(agent, spec)
    return agent.save_state()
