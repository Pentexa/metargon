#!/usr/bin/env python3
"""METARGON v0.9 — benchmark harness (Phase E/F).

Fair paired evaluation: for each (condition, seed) all policies play the
same pre-generated worlds with the same per-episode noise streams, the same
action budgets, and the same episode counts. Policy memory persists across
episodes within a run (as in v0.8). The oracle receives the hidden config —
it is privileged and reported separately, never included in deployable
rankings.

Resource instrumentation (per policy run): wall CPU time, peak Python
allocations (tracemalloc), and — for METARGON — planner operations per
decision, probe-plan calls, and memory size.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from lab_env import CONDITIONS, DOORS, LabWorld, generate_worlds
from metargon_v09 import MetargonV09
from baselines_v09 import (
    BayesPlannerPolicy, EpisodicMemoryPolicy, OraclePolicy, QLearningPolicy,
    RandomPolicy,
)

POLICIES = ("random", "episodic", "qlearning", "bayes", "metargon", "oracle")
DEPLOYABLE = ("random", "episodic", "qlearning", "bayes", "metargon")


def make_policy(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed)
    if name == "episodic":
        return EpisodicMemoryPolicy(seed)
    if name == "qlearning":
        return QLearningPolicy(seed)
    if name == "bayes":
        return BayesPlannerPolicy(seed)
    if name == "metargon":
        return MetargonV09(seed)
    if name == "oracle":
        return OraclePolicy(seed)
    raise ValueError(name)


def play_episode(spec, policy) -> dict:
    world = LabWorld(spec.config, ep_seed=spec.seed,
                     noise_q=spec.noise_q, max_steps=spec.max_steps)
    if isinstance(policy, OraclePolicy):
        policy.episode_start(config=spec.config)
    elif isinstance(policy, BayesPlannerPolicy):
        policy.episode_start(noise_q=spec.noise_q)
    else:
        policy.episode_start()
    invalid = 0
    trace = []
    while not world.finished and world.steps < world.max_steps:
        obs = world.observe()
        action = policy.act(obs)
        res = world.step(action)
        invalid += int(not res.valid)
        trace.append(action)
        policy.observe_outcome(obs, action, res)
    return {
        "success": world.finished,
        "steps": world.steps,
        "invalid": invalid,
        "trace": trace,
    }


def score_policy(condition: str, seed: int, name: str, episodes: int) -> dict:
    worlds = generate_worlds(condition, seed, episodes)
    policy = make_policy(name, seed + 1000003)
    successes = 0
    first_half_success = 0
    second_half_success = 0
    total_steps = 0
    total_invalid = 0
    post_mid_success_steps = []
    tracemalloc.start()
    t0 = time.perf_counter()
    for i, spec in enumerate(worlds):
        r = play_episode(spec, policy)
        successes += int(r["success"])
        total_steps += r["steps"]
        total_invalid += r["invalid"]
        if i < episodes // 2:
            first_half_success += int(r["success"])
        else:
            second_half_success += int(r["success"])
            if r["success"]:
                post_mid_success_steps.append(r["steps"])
    cpu_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    diag = {}
    if name == "metargon":
        d = policy.diagnostics()
        diag = {
            "planner_ops": d["planner_ops"],
            "plan_calls": d["plan_calls"],
            "memory_max": d["memory"]["max"],
            "memory_size_end": d["memory"]["size"],
            "evictions": d["memory"]["evictions"],
            "hypotheses_generated": d["memory"]["generated"],
            "verified": d["memory"]["status_counts"]["verified"],
            "contradicted": d["memory"]["status_counts"]["contradicted"],
        }
    return {
        "success_rate": successes / episodes,
        "first_half_success": first_half_success / (episodes // 2),
        "second_half_success": second_half_success / (episodes - episodes // 2),
        "mean_steps": total_steps / episodes,
        "mean_steps_success_second_half": (
            statistics.fmean(post_mid_success_steps) if post_mid_success_steps else None),
        "invalid_actions": total_invalid,
        "cpu_s": cpu_s,
        "peak_mem_bytes": peak,
        **diag,
    }


def paired_bootstrap(diffs: list[float], seed: int = 4301, rounds: int = 1500) -> tuple:
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        values.append(statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs))
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def benchmark(seeds: int = 30, episodes: int = 240) -> dict:
    if seeds < 1 or episodes < 4:
        raise ValueError("seeds >= 1, episodes >= 4")
    scores = {}
    summary = {}
    for condition in CONDITIONS:
        per_seed = []
        for i in range(seeds):
            results = {p: score_policy(condition, i + 11800, p, episodes)
                       for p in POLICIES}
            per_seed.append(results)
        scores[condition] = per_seed
        summary[condition] = {}
        for p in POLICIES:
            summary[condition][p] = {
                k: statistics.fmean(row[p][k] for row in per_seed)
                for k in ("success_rate", "first_half_success", "second_half_success",
                          "mean_steps", "cpu_s")
            }
            summary[condition][p]["peak_mem_mb"] = (
                statistics.fmean(row[p]["peak_mem_bytes"] for row in per_seed) / 1e6)
        for a, b in (("metargon", "bayes"), ("metargon", "qlearning"),
                     ("metargon", "episodic"), ("bayes", "qlearning")):
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.9",
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(CONDITIONS),
            "policies": list(POLICIES),
            "deployable_ranking": list(DEPLOYABLE),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
            "env": "Mechanism Lab (lab_env.py)",
        },
        "summary": summary,
        "raw": scores,
    }


def _score_one(args):
    """Top-level worker for ProcessPoolExecutor (picklable)."""
    condition, seed, episodes = args
    row = {p: score_policy(condition, seed, p, episodes) for p in POLICIES}
    return condition, seed, row


def benchmark_parallel(seeds: int = 30, episodes: int = 240,
                       workers: int | None = None) -> dict:
    """Like benchmark(), but parallelized across (condition, seed) pairs.
    Produces the identical result structure. Windows spawn requires the
    worker to be a top-level function."""
    import concurrent.futures
    tasks = [(cond, i + 11800, episodes) for cond in CONDITIONS
             for i in range(seeds)]
    scores = {cond: [None] * seeds for cond in CONDITIONS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for cond, seed, row in ex.map(_score_one, tasks):
            scores[cond][seed - 11800] = row
    summary = {}
    for condition in CONDITIONS:
        per_seed = scores[condition]
        summary[condition] = {}
        for p in POLICIES:
            summary[condition][p] = {
                k: statistics.fmean(row[p][k] for row in per_seed)
                for k in ("success_rate", "first_half_success", "second_half_success",
                          "mean_steps", "cpu_s")
            }
            summary[condition][p]["peak_mem_mb"] = (
                statistics.fmean(row[p]["peak_mem_bytes"] for row in per_seed) / 1e6)
        for a, b in (("metargon", "bayes"), ("metargon", "qlearning"),
                     ("metargon", "episodic"), ("bayes", "qlearning")):
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.9",
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(CONDITIONS),
            "policies": list(POLICIES),
            "deployable_ranking": list(DEPLOYABLE),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
            "env": "Mechanism Lab (lab_env.py)",
            "parallel": True, "workers": workers,
        },
        "summary": summary,
        "raw": scores,
    }


def _print_result(result, wall_s):
    print(f"METARGON v0.9 — mechanism lab evaluation (wall {wall_s:.1f}s)")
    for scenario, row in result["summary"].items():
        print(f"\n{scenario}:")
        for name in DEPLOYABLE + ("oracle",):
            x = row[name]
            print(f"  {name:>10}: success {x['success_rate']*100:6.2f}% "
                  f"(1st {x['first_half_success']*100:5.2f}% / "
                  f"2nd {x['second_half_success']*100:5.2f}%) steps {x['mean_steps']:5.2f} "
                  f"cpu {x['cpu_s']:6.2f}s mem {x['peak_mem_mb']:6.2f}MB")
        for key, c in row.items():
            if key.startswith("comparison:"):
                print(f"  {key}: delta {c['success_delta']*100:+.2f}pp "
                      f"CI {tuple(round(x*100, 2) for x in c['paired_bootstrap_95pct'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.parallel:
        result = benchmark_parallel(args.seeds, args.episodes, args.workers)
    else:
        result = benchmark(args.seeds, args.episodes)
    _print_result(result, time.perf_counter() - t0)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
