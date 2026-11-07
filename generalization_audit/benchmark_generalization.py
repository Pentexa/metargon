#!/usr/bin/env python3
"""Independent METARGON generalization evaluation benchmark.

Evaluates the unmodified v0.10 student agent against random, episodic,
Q-learning, and the family-F Bayesian planner on *new* hidden-rule
families (R: sustained-hold; L: ordered-sequence), under:

* identical pre-generated worlds per policy (matched seeds, action
  budgets, noise streams, episode counts);
* an additional matched-compute arm (compute-capped variants with a
  shared per-decision expansion budget);
* frozen test configurations (loaded from a committed JSON) so the test
  set is fixed before benchmarking;
* offline discovery / forgetting / adaptation metrics.

Run from the repository root:

    python generalization_audit/benchmark_generalization.py \
        --family r --seeds 6 --episodes 60 \
        --conditions stable,shift,novel,iid \
        --json generalization_audit/results/summary_r.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lab_env import CONDITIONS as LAB_CONDITIONS, generate_worlds, LabEpisode
from lab_env import TRAIN_POOL as F_TRAIN, VAL_POOL as F_VAL, TEST_POOL as F_TEST

from generalization_audit.envs.base import RWorld, LWorld, RConfig, LConfig
from generalization_audit.envs.pools import (
    load_frozen_pools, config_from_json, SOLVABLE,
)
from generalization_audit.policies.factory import (
    ALL_POLICIES, DEPLOYABLE, COMPARISONS, COMPUTE_BUDGET, make_policy,
)
from generalization_audit.metrics.discovery import (
    discovery_from_rows, summarize_discovery, forgetting_between,
    adaptation_summary,
)
from generalization_audit.metrics.shortcuts import assert_all_solvable

WORLD_CLASSES = {"r": RWorld, "l": LWorld}
AUDIT_CONDITIONS = ("stable", "shift", "rapid_shift", "iid", "novel",
                    "noisy_obs", "tight_budget", "cross_family")

_WITH_CAP = [True]


def _roster(condition):
    roster = [p for p in ALL_POLICIES
              if not (condition == "cross_family" and p == "oracle")]
    if not _WITH_CAP[0]:
        roster = [p for p in roster if not p.endswith("-cap")]
    return roster


def crossed_episodes(seed: int, episodes: int, family: str,
                     train_new, val_new, test_new):
    """Mechanism change across families: first half under family F, second
    half under the audited family (midpoint switch to a train config)."""
    rng = random.Random(seed)
    out = []
    f_cfg = rng.choice(list(F_TRAIN))
    n_cfg = rng.choice(list(train_new))
    for e in range(episodes):
        cfg = f_cfg if e < episodes // 2 else n_cfg
        context = "f" if e < episodes // 2 else family
        out.append(LabEpisode(cfg, 80, 0.0, seed=seed * 1000 + e, context=context))
    return out


def make_worlds(condition, seed, episodes, family, pools):
    if condition == "cross_family":
        train, val, test = pools
        return crossed_episodes(seed, episodes, family, train, val, test)
    return generate_worlds(condition, seed, episodes,
                           train_pool=pools[0], val_pool=pools[1],
                           test_pool=pools[2])


def play_episode(spec, policy, family: str) -> dict:
    from lab_env import LabWorld
    world_cls = WORLD_CLASSES[family]
    fam = spec.context if isinstance(spec.context, str) else family
    if fam == "f":
        world_cls = LabWorld
    else:
        world_cls = WORLD_CLASSES.get(fam, world_cls)
    world = world_cls(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                      max_steps=spec.max_steps)
    from generalization_audit.policies.factory import AuditOraclePolicy
    from baselines_v09 import BayesPlannerPolicy
    if isinstance(policy, AuditOraclePolicy):
        policy.episode_start(config=spec.config)
    elif isinstance(policy, BayesPlannerPolicy):
        policy.episode_start(noise_q=spec.noise_q)
    else:
        policy.episode_start()
    invalid = 0
    events = 0
    first_event_step = None
    steps = 0
    while not world.finished and world.steps < world.max_steps:
        obs = world.observe()
        action = policy.act(obs)
        res = world.step(action)
        invalid += int(not res.valid)
        if res.door_events:
            events += len(res.door_events)
            if first_event_step is None:
                first_event_step = world.steps
        policy.observe_outcome(obs, action, res)
        steps += 1
    return {
        "success": world.finished,
        "steps": world.steps,
        "invalid": invalid,
        "events": events,
        "first_event_step": first_event_step,
        "n_actions": steps,
    }


def _hypotheses_of(policy):
    if hasattr(policy, "inner"):
        policy = policy.inner
    mem = getattr(policy, "memory", None)
    if mem is None:
        return None
    hs = getattr(mem, "hypotheses", None)
    if hs is None:
        return None
    return list(hs)


def score_policy(condition: str, seed: int, name: str, episodes: int,
                 family: str, pools, measure_mem: bool = False) -> dict:
    worlds = make_worlds(condition, seed, episodes, family, pools)
    policy = make_policy(name, seed + 1000003, family=family)
    successes = [0] * episodes
    total_steps = 0
    total_invalid = 0
    total_events = 0
    first_events = []
    diag_rows = []
    if measure_mem:
        tracemalloc.start()
    t0 = time.perf_counter()
    for i, spec in enumerate(worlds):
        r = play_episode(spec, policy, family)
        successes[i] = int(r["success"])
        total_steps += r["steps"]
        total_invalid += r["invalid"]
        total_events += r["events"]
        if r["first_event_step"] is not None:
            first_events.append(r["first_event_step"])
        hs = _hypotheses_of(policy)
        if hs is not None:
            diag_rows.append((spec, hs))
    cpu_s = time.perf_counter() - t0
    peak = None
    if measure_mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    result = {
        "success_rate": sum(successes) / episodes,
        "mean_steps": total_steps / episodes,
        "invalid_actions": total_invalid,
        "door_events_total": total_events,
        "mean_first_event_step": (statistics.fmean(first_events)
                                  if first_events else None),
        "cpu_s": cpu_s,
        "peak_mem_bytes": peak,
    }
    if diag_rows:
        disc = discovery_from_rows(diag_rows)
        result["discovery"] = summarize_discovery(disc)
        if condition in ("shift", "cross_family"):
            mid = episodes // 2
            pre_hs = [hs for _, hs in diag_rows[:mid]]
            post_hs = [hs for _, hs in diag_rows[mid:]]
            result["forgetting"] = {
                door: forgetting_between(pre_hs, post_hs, door)
                for door in ("D1", "D2")
            }
            result["adaptation"] = adaptation_summary(
                successes[:mid], successes[mid:], mid)
    return result


def paired_bootstrap(diffs, seed: int = 4301, rounds: int = 1500):
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        values.append(statistics.fmean(diffs[rng.randrange(len(diffs))]
                                       for _ in diffs))
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def benchmark(seeds: int = 6, episodes: int = 60, family: str = "r",
             conditions=None, measure_mem: bool = False, frozen_path=None) -> dict:
    conds = list(AUDIT_CONDITIONS) if conditions is None else list(conditions)
    from generalization_audit.envs.pools import _config_to_json
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    assert frozen["family"] == family, "frozen file family mismatch"
    assert_all_solvable(frozen, SOLVABLE[family])
    pools = load_frozen_pools(frozen)
    scores = {}
    summary = {}
    for condition in conds:
        per_seed = []
        roster = _roster(condition)
        for i in range(seeds):
            seed = i + 11800
            row = {p: score_policy(condition, seed, p, episodes, family,
                                   pools, measure_mem=measure_mem)
                   for p in roster}
            per_seed.append(row)
        scores[condition] = per_seed
        summary[condition] = {}
        for p in roster:
            row0 = per_seed[0][p]
            keys = [k for k in row0 if k in (
                "success_rate", "mean_steps", "invalid_actions",
                "door_events_total", "mean_first_event_step", "cpu_s")]
            entry = {}
            for k in keys:
                vals = [r[p][k] for r in per_seed if r[p][k] is not None]
                entry[k] = statistics.fmean(vals) if vals else None
            summary[condition][p] = entry
            if measure_mem:
                summary[condition][p]["peak_mem_mb"] = (
                    statistics.fmean(r[p]["peak_mem_bytes"] for r in per_seed)
                    / 1e6)
            else:
                summary[condition][p]["peak_mem_mb"] = None
            if "discovery" in row0:
                summary[condition][p]["discovery"] = {
                    door: {k: statistics.fmean(r[p]["discovery"][door][k]
                                               for r in per_seed)
                           for k in ("exact_rate", "covered_rate",
                                     "false_verified_per_episode")}
                    for door in ("D1", "D2")
                }
            if "forgetting" in row0:
                vals = [r[p]["forgetting"] for r in per_seed]
                summary[condition][p]["forgetting"] = {
                    door: (statistics.fmean(v[door] for v in vals
                                            if v[door] is not None)
                           if any(v[door] is not None for v in vals) else None)
                    for door in ("D1", "D2")
                }
            if "adaptation" in row0:
                summary[condition][p]["adaptation"] = {
                    k: (statistics.fmean(r[p]["adaptation"][k]
                                         for r in per_seed
                                         if r[p]["adaptation"][k] is not None)
                        if any(r[p]["adaptation"][k] is not None
                               for r in per_seed) else None)
                    for k in ("post_success_rate", "delta",
                              "first_recovery_episode")
                }
        for a, b in COMPARISONS:
            succ_delta = [r[a]["success_rate"] - r[b]["success_rate"]
                          for r in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON generalization audit",
            "version": "2.0",
            "family": family,
            "frozen_file": str(frozen_path),
            "frozen_sha": _json_sha(frozen),
            "seeds": seeds,
            "episodes_per_seed": episodes,
            "conditions": list(conds),
            "policies": list(ALL_POLICIES),
            "deployable_ranking": list(DEPLOYABLE),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
            "compute": "matched action budgets; matched-compute arm with "
                       f"per-decision expansion cap {COMPUTE_BUDGET}",
        },
        "summary": summary,
        "raw": scores,
    }


def _json_sha(doc) -> str:
    import hashlib
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _score_one(args):
    condition, seed, episodes, family, pools, measure_mem = args
    roster = _roster(condition)
    row = {p: score_policy(condition, seed, p, episodes, family, pools,
                           measure_mem=measure_mem)
           for p in roster}
    return condition, seed, row


def benchmark_parallel(seeds=6, episodes=60, family="r", conditions=None,
                       workers=None, measure_mem=False, frozen_path=None):
    import concurrent.futures
    conds = list(AUDIT_CONDITIONS) if conditions is None else list(conditions)
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    pools = load_frozen_pools(frozen)
    tasks = [(cond, i + 11800, episodes, family, pools, measure_mem)
             for cond in conds for i in range(seeds)]
    scores = {cond: [None] * seeds for cond in conds}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for cond, seed, row in ex.map(_score_one, tasks):
            scores[cond][seed - 11800] = row
    summary = {}
    for condition in conds:
        per_seed = scores[condition]
        summary[condition] = {}
        roster = _roster(condition)
        for p in roster:
            row0 = per_seed[0][p]
            keys = [k for k in row0 if k in (
                "success_rate", "mean_steps", "invalid_actions",
                "door_events_total", "mean_first_event_step", "cpu_s")]
            entry = {}
            for k in keys:
                vals = [r[p][k] for r in per_seed if r[p][k] is not None]
                entry[k] = statistics.fmean(vals) if vals else None
            summary[condition][p] = entry
            summary[condition][p]["peak_mem_mb"] = None
        for a, b in COMPARISONS:
            succ_delta = [r[a]["success_rate"] - r[b]["success_rate"]
                          for r in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
    return {
        "metadata": {
            "name": "METARGON generalization audit", "version": "2.0",
            "family": family, "frozen_file": str(frozen_path),
            "frozen_sha": _json_sha(frozen),
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(conds),
            "policies": list(ALL_POLICIES),
            "deployable_ranking": list(DEPLOYABLE),
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
            "compute": "matched action budgets; matched-compute arm with "
                       f"per-decision expansion cap {COMPUTE_BUDGET}",
        },
        "summary": summary,
        "raw": scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("r", "l"), default="r")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--conditions", default=None,
                        help="comma-separated subset of audit conditions")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-mem", action="store_true")
    parser.add_argument("--no-cap", action="store_true",
                        help="exclude the matched-compute capped arm")
    parser.add_argument("--frozen", type=Path, default=None)
    args = parser.parse_args()
    if args.no_cap:
        _WITH_CAP[0] = False
    frozen_path = args.frozen or Path("generalization_audit/configs") / \
        f"frozen_{args.family}.json"
    conditions = args.conditions.split(",") if args.conditions else None
    mem = not args.no_mem
    if args.parallel:
        result = benchmark_parallel(args.seeds, args.episodes, args.family,
                                    conditions, args.workers, mem, frozen_path)
    else:
        result = benchmark(args.seeds, args.episodes, args.family,
                           conditions, mem, frozen_path)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n",
                             encoding="utf-8")
    # console summary
    for condition, row in result["summary"].items():
        print(f"\n== {condition} ==")
        for p in DEPLOYABLE + ("oracle",):
            if p not in row:
                continue
            x = row[p]
            print(f"  {p:>16}: success {x['success_rate']*100:6.2f}% "
                  f"steps {x['mean_steps']:6.2f} cpu {x['cpu_s']:7.2f}s")
            if "discovery" in x:
                d = x["discovery"]["D1"]
                print(f"      D1 exact={d['exact_rate']*100:5.1f}% "
                      f"false_ver={d['false_verified_per_episode']:.2f}")
        for key, c in row.items():
            if key.startswith("comparison:"):
                print(f"  {key}: delta {c['success_delta']*100:+.2f}pp "
                      f"CI {tuple(round(x*100, 2) for x in c['paired_bootstrap_95pct'])}")
    print("\nAll audit runs complete.")


if __name__ == "__main__":
    main()
