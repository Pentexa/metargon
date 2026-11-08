#!/usr/bin/env python3
"""METARGON v0.11 — paired benchmark harness (v0.10 vs v0.11 + ablations).

Identical worlds/seeds/action budgets as v0.10 (generate_worlds seeds
11800+i). Policies:

  * metargon10 — v0.10 agent (unchanged reference).
  * metargon11 — v0.11 full (confirmed events + certainty weighting +
                  filtered replay).
  * metargon11-cf — v0.11 with confirmed events ablated (belief-diff
                     events, v0.10 style).
  * metargon11-ct — v0.11 with certainty weighting ablated (weight=1).
  * metargon11-rp — v0.11 replay off.
  * metargon11-rw — v0.11 replay weighted (priority×certainty).
  * bayes / oracle — existing comparators.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from lab_env import CONDITIONS, F1_TEMPLATES, F2_TEMPLATES
from lab_env import H1_TEMPLATES, H2_TEMPLATES, LabConfig, LabWorld
from lab_env import generate_worlds, is_solvable
from metargon_v10 import MetargonV10
from metargon_v11 import MetargonV11
from baselines_v09 import BayesPlannerPolicy, OraclePolicy

ABLATIONS = {
    "metargon11": {},
    "metargon11-cf": {"use_confirmed_events": False},
    "metargon11-ct": {"use_certainty": False},
    "metargon11-rp": {"replay_mode": "off"},
    "metargon11-rw": {"replay_mode": "weighted"},
}
ALL_POLICIES = ("metargon10",) + tuple(ABLATIONS) + ("bayes", "oracle")
DEPLOYABLE = ALL_POLICIES


def make_policy(name: str, seed: int):
    if name == "metargon10":
        return MetargonV10(seed)
    if name == "bayes":
        return BayesPlannerPolicy(seed)
    if name == "oracle":
        return OraclePolicy(seed)
    if name in ABLATIONS:
        return MetargonV11(seed, **ABLATIONS[name])
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
    events = 0
    first_event_step = None
    while not world.finished and world.steps < world.max_steps:
        obs = world.observe()
        action = policy.act(obs)
        res = world.step(action)
        invalid += int(not res.valid)
        trace.append(action)
        if res.door_events:
            events += len(res.door_events)
            if first_event_step is None:
                first_event_step = world.steps
        policy.observe_outcome(obs, action, res)
    return {
        "success": world.finished,
        "steps": world.steps,
        "invalid": invalid,
        "events": events,
        "first_event_step": first_event_step,
        "trace": trace,
    }


def score_policy(condition: str, seed: int, name: str, episodes: int,
                 measure_mem: bool = False, pools=None) -> dict:
    if pools is None:
        worlds = generate_worlds(condition, seed, episodes)
    else:
        train, val, test = pools
        worlds = generate_worlds(condition, seed, episodes,
                                 train_pool=train, val_pool=val,
                                 test_pool=test)
    policy = make_policy(name, seed + 1000003)
    successes = 0
    total_steps = 0
    total_invalid = 0
    total_events = 0
    first_events = []
    if measure_mem:
        tracemalloc.start()
    t0 = time.perf_counter()
    for spec in worlds:
        r = play_episode(spec, policy)
        successes += int(r["success"])
        total_steps += r["steps"]
        total_invalid += r["invalid"]
        total_events += r["events"]
        if r["first_event_step"] is not None:
            first_events.append(r["first_event_step"])
    cpu_s = time.perf_counter() - t0
    peak = None
    if measure_mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    diag = {}
    if name in ("metargon10",) or name in ABLATIONS:
        d = policy.diagnostics()
        mem = d.get("memory", {})
        diag = {
            "planner_ops": d.get("planner_ops"),
            "plan_calls": d.get("plan_calls"),
            "hypotheses_generated": mem.get("generated"),
            "verified": mem.get("status_counts", {}).get("verified"),
            "contradicted": mem.get("status_counts", {}).get("contradicted"),
            "evictions": mem.get("evictions"),
            "replay_applied": mem.get("replay_applied"),
            "confirmed_events": d.get("confirmed_events_total"),
            "dropped_candidates": d.get("dropped_candidates_total"),
            "belief_corrections": d.get("belief", {}).get("action_corrections"),
        }
    return {
        "success_rate": successes / episodes,
        "mean_steps": total_steps / episodes,
        "invalid_actions": total_invalid,
        "door_events_total": total_events,
        "mean_first_event_step": (statistics.fmean(first_events)
                                  if first_events else None),
        "cpu_s": cpu_s,
        "peak_mem_bytes": peak,
        **diag,
    }


def family_pools(family: str = "f"):
    f1 = F1_TEMPLATES if family == "f" else H1_TEMPLATES
    f2 = F2_TEMPLATES if family == "f" else H2_TEMPLATES
    prng = __import__("random").Random(20260901 if family == "f" else 20261107)
    all_configs = [
        LabConfig(d1_cond=c1, d2_cond=c2, d1_delay=bool(b1), d2_delay=bool(b2))
        for c1 in f1 for c2 in f2
        for b1 in (0, 1) for b2 in (0, 1)
    ]
    solvable = [c for c in all_configs if is_solvable(c)]
    prng.shuffle(solvable)
    n = len(solvable)
    train_n = int(n * 0.60)
    val_n = int(n * 0.20)
    return (solvable[:train_n], solvable[train_n:train_n + val_n],
            solvable[train_n + val_n:]), n


def paired_bootstrap(diffs, seed=4301, rounds=1500):
    rng = __import__("random").Random(seed)
    values = []
    for _ in range(rounds):
        values.append(statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs))
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def benchmark(seeds=8, episodes=100, measure_mem=False, conditions=None,
              family="f"):
    conds = list(CONDITIONS) if conditions is None else conditions
    pools = None
    if family != "f":
        pools, _ = family_pools(family)
    scores = {}
    summary = {}
    for condition in conds:
        per_seed = []
        for i in range(seeds):
            seed = i + 11800
            results = {p: score_policy(condition, seed, p, episodes,
                                       measure_mem=measure_mem, pools=pools)
                       for p in ALL_POLICIES}
            per_seed.append(results)
        scores[condition] = per_seed
        summary[condition] = {}
        for p in ALL_POLICIES:
            summary[condition][p] = {
                k: statistics.fmean(row[p][k] for row in per_seed)
                for k in ("success_rate", "mean_steps", "invalid_actions",
                          "door_events_total", "mean_first_event_step", "cpu_s")
            }
            if measure_mem:
                summary[condition][p]["peak_mem_mb"] = (
                    statistics.fmean(row[p]["peak_mem_bytes"] for row in per_seed) / 1e6)
            else:
                summary[condition][p]["peak_mem_mb"] = None
        for a, b in (("metargon11", "metargon10"),
                     ("metargon11", "bayes"),
                     ("metargon11", "metargon11-cf"),
                     ("metargon11", "metargon11-ct"),
                     ("metargon11", "metargon11-rp"),
                     ("metargon11", "metargon11-rw")):
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.11",
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(conds), "family": family,
            "policies": list(ALL_POLICIES),
            "deployable_ranking": list(DEPLOYABLE),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
        },
        "summary": summary,
        "raw": scores,
    }


def _score_one(args):
    condition, seed, episodes, measure_mem, family = args
    pools = None
    if family != "f":
        pools, _ = family_pools(family)
    row = {p: score_policy(condition, seed, p, episodes,
                           measure_mem=measure_mem, pools=pools)
           for p in ALL_POLICIES}
    return condition, seed, row


def benchmark_parallel(seeds=8, episodes=100, workers=None, measure_mem=False,
                       conditions=None, family="f"):
    import concurrent.futures
    conds = list(CONDITIONS) if conditions is None else conditions
    tasks = [(cond, i + 11800, episodes, measure_mem, family)
             for cond in conds for i in range(seeds)]
    scores = {cond: [None] * seeds for cond in conds}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for cond, seed, row in ex.map(_score_one, tasks):
            scores[cond][seed - 11800] = row
    summary = {}
    for condition in conds:
        per_seed = scores[condition]
        summary[condition] = {}
        for p in ALL_POLICIES:
            summary[condition][p] = {
                k: statistics.fmean(row[p][k] for row in per_seed)
                for k in ("success_rate", "mean_steps", "invalid_actions",
                          "door_events_total", "mean_first_event_step", "cpu_s")
            }
            if measure_mem:
                summary[condition][p]["peak_mem_mb"] = (
                    statistics.fmean(row[p]["peak_mem_bytes"] for row in per_seed) / 1e6)
            else:
                summary[condition][p]["peak_mem_mb"] = None
        for a, b in (("metargon11", "metargon10"),
                     ("metargon11", "bayes"),
                     ("metargon11", "metargon11-cf"),
                     ("metargon11", "metargon11-ct"),
                     ("metargon11", "metargon11-rp"),
                     ("metargon11", "metargon11-rw")):
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.11",
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(conds), "family": family,
            "policies": list(ALL_POLICIES),
            "deployable_ranking": list(DEPLOYABLE),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
        },
        "summary": summary,
        "raw": scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-mem", action="store_true")
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--family", type=str, default="f", choices=("f", "h"))
    args = parser.parse_args()
    conditions = None if not args.conditions else args.conditions.split(",")
    mem = not args.no_mem
    if args.parallel:
        result = benchmark_parallel(args.seeds, args.episodes, args.workers,
                                    measure_mem=mem, conditions=conditions,
                                    family=args.family)
    else:
        result = benchmark(args.seeds, args.episodes, measure_mem=mem,
                           conditions=conditions, family=args.family)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for c, r in result["summary"].items():
        print(c)
        for p, s in r.items():
            if p.startswith("comparison:"):
                continue
            print(f"  {p:>16} success {s['success_rate']*100:6.2f}% "
                  f"steps {s['mean_steps']:5.2f} invalid {s['invalid_actions']:5.2f} "
                  f"cpu {s['cpu_s']:6.2f}s")


if __name__ == "__main__":
    main()
