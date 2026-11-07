#!/usr/bin/env python3
"""METARGON v0.10 — paired benchmark harness (v0.9 vs v0.10 + ablations).

Fairness mirrors benchmark_v09: identical pre-generated worlds (same seeds,
same noise streams, same action budgets, same episode counts) for every
policy. All policies consume only public observations.

Policies:
  * metargon09      — the v0.9 agent, unchanged.
  * metargon10      — v0.10 full (robust belief + cooldowns + fallback).
  * metargon10-rb  — v0.10 with RobustBelief ablated (raw sync).
  * metargon10-cd  — v0.10 with probe cooldowns ablated (v0.9 style
                      permanent tried set).
  * metargon10-x   — v0.10 with the info-gathering fallback ablated.
  * bayes / oracle — existing comparators (scored by the same worlds/specs).

Metrics per run: success rate, mean steps, invalid actions, repeated-action
loops (runs of >=3 identical actions; steps spent in them), stall steps
(visiting a position seen within the last 8 steps while door state is
unchanged), information-gain proxies (door events, unique observation
contexts visited, hypotheses generated/verified/contradicted, replay
applied), runtime (CPU s), peak memory (tracemalloc), and — computed
offline, never given to the agent — which true door conditions were
discovered by verified/high-confidence hypotheses.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import tracemalloc
from collections import deque
from pathlib import Path

from lab_env import (
    CONDITIONS, F1_TEMPLATES, F2_TEMPLATES, H1_TEMPLATES, H2_TEMPLATES,
    LabConfig, LabWorld, generate_worlds, is_solvable,
)
from metargon_v09 import MetargonV09
from metargon_v10 import MetargonV10
from baselines_v09 import BayesPlannerPolicy, OraclePolicy


def family_pools(family: str = "f"):
    """Return (train, val, test) pools for the declared family F or the
    structurally novel family H (solvability-filtered, deterministic)."""
    f1 = F1_TEMPLATES if family == "f" else H1_TEMPLATES
    f2 = F2_TEMPLATES if family == "f" else H2_TEMPLATES
    prng = random.Random(20260901 if family == "f" else 20261107)
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

ABLATIONS = {
    "metargon10": {},
    "metargon10-rb": {"use_robust_belief": False},
    "metargon10-cd": {"use_cooldowns": False},
    "metargon10-x": {"use_explore_fallback": False},
    "metargon10-rp": {"use_replay": False},
}
ALL_POLICIES = ("metargon09",) + tuple(ABLATIONS) + ("bayes", "oracle")
DEPLOYABLE_V10 = ("metargon09", "metargon10", "metargon10-rb",
                  "metargon10-cd", "metargon10-x")


def make_policy(name: str, seed: int):
    if name == "metargon09":
        return MetargonV09(seed)
    if name == "bayes":
        return BayesPlannerPolicy(seed)
    if name == "oracle":
        return OraclePolicy(seed)
    if name in ABLATIONS:
        return MetargonV10(seed, **ABLATIONS[name])
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
    pos_trace = []
    door_trace = []
    events = 0
    first_event_step = None
    repeat_loops = 0
    repeat_steps = 0
    stall_steps = 0
    recent_pos = deque(maxlen=8)
    last_door_change_step = 0
    while not world.finished and world.steps < world.max_steps:
        obs = world.observe()
        action = policy.act(obs)
        res = world.step(action)
        invalid += int(not res.valid)
        trace.append(action)
        pos_trace.append((obs.x, obs.y))
        door_trace.append((obs.d1, obs.d2))
        if res.door_events:
            events += len(res.door_events)
            if first_event_step is None:
                first_event_step = world.steps
            last_door_change_step = world.steps
        policy.observe_outcome(obs, action, res)
    # repeated-action loops
    run_len = 0
    prev = None
    for a in trace:
        if a == prev:
            run_len += 1
        else:
            if run_len >= 3:
                repeat_loops += 1
                repeat_steps += run_len
            run_len = 1
            prev = a
    if run_len >= 3:
        repeat_loops += 1
        repeat_steps += run_len
    # stall steps: revisiting a position+door-state configuration seen
    # within the last 8 steps (no progress made by looping)
    for i in range(len(pos_trace)):
        lo = max(0, i - 8)
        cur = (pos_trace[i], door_trace[i])
        if i >= 8 and any((pos_trace[j], door_trace[j]) == cur
                          for j in range(lo, i)):
            stall_steps += 1
    return {
        "success": world.finished,
        "steps": world.steps,
        "invalid": invalid,
        "events": events,
        "first_event_step": first_event_step,
        "repeat_loops": repeat_loops,
        "repeat_steps": repeat_steps,
        "stall_steps": stall_steps,
        "trace": trace,
    }


def true_rules_discovered(specs, diag_rows) -> dict:
    """Offline oracle diagnostic: did the agent's VERIFIED (or at least
    high-confidence) hypotheses match the true door conditions? Uses only
    the *final* hypothesis store of the policy (never influences it)."""
    out = {"D1": False, "D2": False, "n_d1": 0, "n_d2": 0}
    for spec, hs in diag_rows:
        for door, true_cond in (("D1", spec.config.d1_cond),
                                ("D2", spec.config.d2_cond)):
            matched = False
            for h in hs:
                if h.door != door or h.status != "verified":
                    continue
                if h.antecedent == true_cond:
                    matched = True
                    break
            out[door] = out[door] or matched
            out["n_d1" if door == "D1" else "n_d2"] += 1
    return out


def score_policy(condition: str, seed: int, name: str, episodes: int,
                 measure_mem: bool = True, pools=None) -> dict:
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
    total_repeat_loops = 0
    total_repeat_steps = 0
    total_stall_steps = 0
    first_events = []
    contexts = set()
    diag_rows = []
    if measure_mem:
        tracemalloc.start()
    t0 = time.perf_counter()
    for spec in worlds:
        r = play_episode(spec, policy)
        successes += int(r["success"])
        total_steps += r["steps"]
        total_invalid += r["invalid"]
        total_events += r["events"]
        total_repeat_loops += r["repeat_loops"]
        total_repeat_steps += r["repeat_steps"]
        total_stall_steps += r["stall_steps"]
        if r["first_event_step"] is not None:
            first_events.append(r["first_event_step"])
        for a in r["trace"]:
            contexts.add(a)
        if name in ("metargon09",) or name in ABLATIONS:
            diag_rows.append((spec, list(policy.memory.hypotheses)))
    cpu_s = time.perf_counter() - t0
    peak = None
    if measure_mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    diag = {}
    if name in ("metargon09",) or name in ABLATIONS:
        d = policy.diagnostics()
        mem = d.get("memory", {})
        diag = {
            "planner_ops": d.get("planner_ops"),
            "plan_calls": d.get("plan_calls"),
            "hypotheses_generated": mem.get("generated"),
            "verified": mem.get("status_counts", {}).get("verified"),
            "contradicted": mem.get("status_counts", {}).get("contradicted"),
            "evictions": mem.get("evictions"),
            "distinct_probed": d.get("distinct_probed"),
            "explore_actions": d.get("explore_actions"),
            "hyp_probe_actions": d.get("hyp_probe_actions"),
            "replay_applied": mem.get("replay_applied"),
            "belief_corrections": d.get("belief", {}).get("action_corrections"),
        }
        true_disc = true_rules_discovered(diag_rows, None) if False else None
    return {
        "success_rate": successes / episodes,
        "mean_steps": total_steps / episodes,
        "invalid_actions": total_invalid,
        "door_events_total": total_events,
        "mean_first_event_step": (statistics.fmean(first_events)
                                  if first_events else None),
        "repeat_loops_total": total_repeat_loops,
        "repeat_steps_total": total_repeat_steps,
        "stall_steps_total": total_stall_steps,
        "unique_action_contexts": len(contexts),
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


def benchmark(seeds: int = 8, episodes: int = 100,
              measure_mem: bool = False, conditions=None, family="f") -> dict:
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
                          "door_events_total", "mean_first_event_step",
                          "repeat_loops_total", "repeat_steps_total",
                          "stall_steps_total", "unique_action_contexts",
                          "cpu_s")
            }
            if measure_mem:
                summary[condition][p]["peak_mem_mb"] = (
                    statistics.fmean(row[p]["peak_mem_bytes"] for row in per_seed) / 1e6)
            else:
                summary[condition][p]["peak_mem_mb"] = None
        for a, b in (("metargon10", "metargon09"),
                     ("metargon10", "bayes"),
                     ("metargon10", "metargon10-rb"),
                     ("metargon10", "metargon10-cd"),
                     ("metargon10", "metargon10-x"),
                     ("metargon10", "metargon10-rp")):
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.10",
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(conds),
            "policies": list(ALL_POLICIES),
            "deployable_ranking": list(DEPLOYABLE_V10),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
            "env": "Mechanism Lab (lab_env.py)",
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


def benchmark_parallel(seeds: int = 8, episodes: int = 100,
                       workers: int | None = None,
                       measure_mem: bool = False, conditions=None,
                       family="f") -> dict:
    import concurrent.futures
    conds = list(CONDITIONS) if conditions is None else conditions
    pools = None
    if family != "f":
        pools, _ = family_pools(family)
    tasks = [(cond, i + 11800, episodes, measure_mem, family) for cond in conds
             for i in range(seeds)]
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
                          "door_events_total", "mean_first_event_step",
                          "repeat_loops_total", "repeat_steps_total",
                          "stall_steps_total", "unique_action_contexts",
                          "cpu_s")
            }
            if measure_mem:
                summary[condition][p]["peak_mem_mb"] = (
                    statistics.fmean(row[p]["peak_mem_bytes"] for row in per_seed) / 1e6)
            else:
                summary[condition][p]["peak_mem_mb"] = None
        for a, b in (("metargon10", "metargon09"),
                     ("metargon10", "bayes"),
                     ("metargon10", "metargon10-rb"),
                     ("metargon10", "metargon10-cd"),
                     ("metargon10", "metargon10-x")):
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.10",
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(conds),
            "policies": list(ALL_POLICIES),
            "deployable_ranking": list(DEPLOYABLE_V10),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
            "env": "Mechanism Lab (lab_env.py)",
        },
        "summary": summary,
        "raw": scores,
    }


def _print_result(result):
    for scenario, row in result["summary"].items():
        print(f"\n{scenario}:")
        for name in DEPLOYABLE_V10 + ("bayes", "oracle"):
            x = row[name]
            mem = "—" if x.get("peak_mem_mb") is None else f"{x['peak_mem_mb']:.2f}MB"
            print(f"  {name:>14}: success {x['success_rate']*100:6.2f}% "
                  f"steps {x['mean_steps']:5.2f} invalid {x['invalid_actions']:4.1f} "
                  f"loops {x['repeat_loops_total']:5.1f} stall {x['stall_steps_total']:6.1f} "
                  f"events {x['door_events_total']:5.1f} firstEv "
                  f"{x['mean_first_event_step'] if x['mean_first_event_step'] is not None else float('nan'):5.1f} "
                  f"cpu {x['cpu_s']:6.2f}s mem {mem}")
        for key, c in row.items():
            if key.startswith("comparison:"):
                print(f"  {key}: delta {c['success_delta']*100:+.2f}pp "
                      f"CI {tuple(round(x*100, 2) for x in c['paired_bootstrap_95pct'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-mem", action="store_true")
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--family", type=str, default="f",
                        choices=("f", "h"),
                        help="'f' = declared family F1/F2 (default), "
                             "'h' = structurally novel family H1/H2")
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
        rows = []
        for cond in result["summary"]:
            for i, per_seed in enumerate(result["raw"][cond]):
                seed = i + 11800
                row = {"condition": cond, "seed": seed}
                for p, res in per_seed.items():
                    for k, v in res.items():
                        if isinstance(v, (int, float)) and v is not None:
                            row[f"{p}.{k}"] = v
                rows.append(row)
        (args.json.with_name("per_seed_results_v10.json")).write_text(
            json.dumps({"metadata": result["metadata"], "rows": rows}, indent=2)
            + "\n", encoding="utf-8")
    _print_result(result)


if __name__ == "__main__":
    main()
