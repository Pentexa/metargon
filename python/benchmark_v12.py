#!/usr/bin/env python3
"""METARGON v0.12 — cross-family paired benchmark harness.

Supports families f (declared), h (structurally novel), r (sustained-hold,
audit), l (ordered-sequence, audit). Family r/l reuse the *independent*
audit infrastructure read-only:
  * frozen config pools: generalization_audit/configs/frozen_{r,l}.json
  * world classes: generalization_audit.envs.base.RWorld / LWorld
  * privileged oracle: generalization_audit.policies.factory.AuditOraclePolicy
No audit file is modified.

Policies:
  * metargon10 / metargon11 — prior versions (unchanged references).
  * metargon12 — full v0.12 (dual context, adaptive confirm, recent credit,
                 optional temporal).
  * metargon12-nodc / -norc / -noad — Mission-A ablations.
  * metargon12-notp — temporal ablated (Mission-B ablation; kept on
    family l, off elsewhere).
  * random / episodic / qlearning / bayes — baselines.
  * oracle — privileged (reported separately).

Metrics: success, steps, invalid actions, door events, CPU (per seed-run),
peak memory (optional sample), and offline exact/covered discovery of the
true hidden rules (per door).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import tracemalloc
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "generalization_audit"))

from lab_env import (  # noqa: E402
    CONDITIONS, LabConfig, LabWorld, generate_worlds, TRAIN_POOL, VAL_POOL,
    TEST_POOL,
)
from metargon_v10 import MetargonV10  # noqa: E402
from metargon_v11 import MetargonV11  # noqa: E402
from metargon_v12 import MetargonV12  # noqa: E402
from baselines_v09 import (  # noqa: E402
    BayesPlannerPolicy, EpisodicMemoryPolicy, QLearningPolicy, RandomPolicy,
    OraclePolicy,
)
from generalization_audit.envs.base import RWorld, LWorld  # noqa: E402
from generalization_audit.envs.pools import load_frozen_pools  # noqa: E402
from generalization_audit.policies.factory import AuditOraclePolicy  # noqa: E402

FAMILIES = ("f", "h", "r", "l")
FROZEN = {"r": ROOT / "generalization_audit" / "configs" / "frozen_r.json",
          "l": ROOT / "generalization_audit" / "configs" / "frozen_l.json"}

# variant factories
ABLATIONS = {
    "metargon12": {},
    "metargon12-legacy": {"dual_context_mode": "dual",
                          "use_recent_credit": True},
    "metargon12-dic": {"dual_context_mode": "dual_if_changed"},
    "metargon12-nodc": {"use_dual_context": False},
    "metargon12-dst": {"dual_context_mode": "dual_stable"},
    "metargon12-norc": {"use_recent_credit": False},
    "metargon12-noad": {"use_adaptive_confirm": False},
    "metargon12-notp": {"use_temporal": False},
    "metargon12-conf": {"adaptive_confirmed_events": False},
}
TEMPORAL_OFF = ("metargon12-nodc", "metargon12-norc", "metargon12-noad",
                "metargon12-notp")


def make_policy(name, seed, family):
    if family == "l" and name == "metargon12":
        return MetargonV12(seed, use_temporal=True)
    if name == "metargon12":
        return MetargonV12(seed, use_temporal=False)
    if name in ABLATIONS:
        return MetargonV12(seed, **ABLATIONS[name])
    if name == "metargon11":
        return MetargonV11(seed)
    if name == "metargon11-cf":
        return MetargonV11(seed, use_confirmed_events=False)
    if name == "metargon11-ct":
        return MetargonV11(seed, use_certainty=False)
    if name == "metargon11-rp":
        return MetargonV11(seed, replay_mode="off")
    if name == "metargon10":
        return MetargonV10(seed)
    if name == "bayes":
        return BayesPlannerPolicy(seed)
    if name == "random":
        return RandomPolicy(seed)
    if name == "episodic":
        return EpisodicMemoryPolicy(seed)
    if name == "qlearning":
        return QLearningPolicy(seed)
    if name == "oracle":
        if family in ("r", "l"):
            return AuditOraclePolicy(seed, family=family)
        return OraclePolicy(seed)
    raise ValueError(name)


def family_pools(family):
    if family in ("r", "l"):
        frozen = json.loads(FROZEN[family].read_text(encoding="utf-8"))
        return load_frozen_pools(frozen), frozen.get("n_solvable")
    if family == "f":
        return (list(TRAIN_POOL), list(VAL_POOL), list(TEST_POOL)), None
    # h: build deterministic H pools with the same convention as v10/v11
    from lab_env import H1_TEMPLATES, H2_TEMPLATES, is_solvable
    prng = random.Random(20261107)
    all_configs = [
        LabConfig(d1_cond=c1, d2_cond=c2, d1_delay=bool(b1), d2_delay=bool(b2))
        for c1 in H1_TEMPLATES for c2 in H2_TEMPLATES
        for b1 in (0, 1) for b2 in (0, 1)
    ]
    solvable = [c for c in all_configs if is_solvable(c)]
    prng.shuffle(solvable)
    n = len(solvable)
    tr = int(n * 0.60)
    va = int(n * 0.20)
    return (solvable[:tr], solvable[tr:tr + va], solvable[tr + va:]), n


def make_worlds(condition, seed, episodes, family, pools):
    train, val, test = pools
    if family in ("r", "l"):
        # mirror lab_env.generate_worlds semantics on frozen pools
        rng = random.Random(seed)
        cfg = rng.choice(train)
        out = []
        for e in range(episodes):
            if condition == "iid":
                cfg = rng.choice(train)
            elif condition in ("shift", "tight_budget") and e == episodes // 2:
                cfg = rng.choice(val)
            elif condition == "novel" and e == episodes // 2:
                cfg = rng.choice(test)
            elif condition == "rapid_shift" and e > 0 and e % 30 == 0:
                cfg = rng.choice(val if rng.getrandbits(1) else train)
            max_steps = 44 if condition == "tight_budget" else 80
            noise_q = 0.08 if condition == "noisy_obs" else 0.0
            from lab_env import LabEpisode
            out.append(LabEpisode(cfg, max_steps, noise_q,
                                  seed=seed * 1000 + e, context=0))
        return out
    return generate_worlds(condition, seed, episodes,
                           train_pool=train, val_pool=val, test_pool=test)


def make_world(spec, family):
    if family in ("r", "l"):
        cls = RWorld if family == "r" else LWorld
        return cls(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                   max_steps=spec.max_steps)
    return LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                    max_steps=spec.max_steps)


def play_episode(spec, policy, family):
    world = make_world(spec, family)
    if isinstance(policy, AuditOraclePolicy):
        policy.episode_start(config=spec.config)
    elif isinstance(policy, OraclePolicy):
        policy.episode_start(config=spec.config)
    elif isinstance(policy, BayesPlannerPolicy):
        policy.episode_start(noise_q=spec.noise_q)
    else:
        policy.episode_start()
    invalid = 0
    events = 0
    first_event_step = None
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
    return {
        "success": world.finished,
        "steps": world.steps,
        "invalid": invalid,
        "events": events,
        "first_event_step": first_event_step,
    }


def _true_rule(spec, family, door):
    if family == "l":
        return (frozenset(spec.config.first_for(door)),
                frozenset(spec.config.second_for(door)))
    if family == "r":
        return frozenset(spec.config.cond_for(door))
    return frozenset(spec.config.cond_for(door))


def discovery_row(spec, policy, family):
    """Offline exact/covered discovery per door (never fed to the agent)."""
    out = {}
    for door in ("D1", "D2"):
        true_rule = _true_rule(spec, family, door)
        exact = False
        covered = False
        # compositional hypotheses
        for h in getattr(getattr(policy, "memory", None), "hypotheses", []):
            if h.door != door or h.status != "verified":
                continue
            if family == "l":
                continue  # conjunctive hyps cannot express order
            if h.antecedent == true_rule:
                exact = True
            if true_rule <= h.antecedent:
                covered = True
        # temporal hypotheses (family l)
        if family == "l":
            tmem = getattr(policy, "temporal", None)
            if tmem is not None:
                for h in tmem.hypotheses:
                    if h.door != door or h.status != "verified":
                        continue
                    if (h.first, h.second) == true_rule:
                        exact = True
                    if h.first <= true_rule[0] and h.second <= true_rule[1]:
                        covered = True
        out[door] = (exact, covered)
    return out


def score_policy(condition, seed, name, episodes, family, pools,
                 measure_mem=False):
    worlds = make_worlds(condition, seed, episodes, family, pools)
    policy = make_policy(name, seed + 1000003, family)
    successes = 0
    total_steps = 0
    total_invalid = 0
    total_events = 0
    first_events = []
    discovery = {"D1": [0, 0], "D2": [0, 0]}
    if measure_mem:
        tracemalloc.start()
    t0 = time.perf_counter()
    for spec in worlds:
        r = play_episode(spec, policy, family)
        successes += int(r["success"])
        total_steps += r["steps"]
        total_invalid += r["invalid"]
        total_events += r["events"]
        if r["first_event_step"] is not None:
            first_events.append(r["first_event_step"])
        dr = discovery_row(spec, policy, family)
        for d in ("D1", "D2"):
            discovery[d][0] += int(dr[d][0])
            discovery[d][1] += int(dr[d][1])
    cpu_s = time.perf_counter() - t0
    peak = None
    if measure_mem:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    result = {
        "success_rate": successes / episodes,
        "mean_steps": total_steps / episodes,
        "invalid_actions": total_invalid,
        "door_events_total": total_events,
        "mean_first_event_step": (statistics.fmean(first_events)
                                  if first_events else None),
        "cpu_s": cpu_s,
        "peak_mem_bytes": peak,
        "exact_discovery_D1": discovery["D1"][0] / episodes,
        "exact_discovery_D2": discovery["D2"][0] / episodes,
        "covered_discovery_D1": discovery["D1"][1] / episodes,
        "covered_discovery_D2": discovery["D2"][1] / episodes,
    }
    if name in ("metargon10", "metargon11") or name in ABLATIONS:
        d = policy.diagnostics()
        mem = d.get("memory", {})
        result.update({
            "verified": mem.get("status_counts", {}).get("verified"),
            "contradicted": mem.get("status_counts", {}).get("contradicted"),
            "replay_applied": mem.get("replay_applied"),
            "confirmed_events": d.get("confirmed_events_total"),
            "temporal_probes": d.get("temporal_probes"),
        })
        if name == "metargon12" and family == "l":
            tm = d.get("temporal", {})
            result["temporal_generated"] = tm.get("generated")
            result["temporal_verified"] = tm.get("status_counts", {}).get("verified")
    return result


def paired_bootstrap(diffs, seed=4301, rounds=1500):
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        values.append(statistics.fmean(diffs[rng.randrange(len(diffs))]
                                       for _ in diffs))
    values.sort()
    return values[int(0.025 * rounds)], values[int(0.975 * rounds)]


def benchmark(seeds=8, episodes=60, family="f", conditions=None,
             measure_mem=False, policies=None):
    conds = list(CONDITIONS) if conditions is None else list(conditions)
    pools, _ = family_pools(family)
    roster = list(policies) if policies else [
        "metargon10", "metargon11", "metargon12", "metargon12-nodc",
        "metargon12-norc", "metargon12-noad", "bayes", "oracle",
    ]
    if family in ("r", "l"):
        if family == "r":
            roster = roster + ["random"]
        else:
            roster = roster + ["random"]
            roster += ["metargon12-notp", "qlearning", "episodic"]
    scores = {}
    summary = {}
    for condition in conds:
        per_seed = []
        for i in range(seeds):
            seed = i + 11800
            row = {p: score_policy(condition, seed, p, episodes, family,
                                   pools, measure_mem=measure_mem)
                   for p in roster}
            per_seed.append(row)
        scores[condition] = per_seed
        summary[condition] = {}
        for p in roster:
            entry = {}
            for k in ("success_rate", "mean_steps", "invalid_actions",
                      "door_events_total", "mean_first_event_step", "cpu_s",
                      "exact_discovery_D1", "exact_discovery_D2",
                      "covered_discovery_D1", "covered_discovery_D2"):
                vals = [r[p][k] for r in per_seed if r[p].get(k) is not None]
                entry[k] = statistics.fmean(vals) if vals else None
            if measure_mem:
                entry["peak_mem_mb"] = (
                    statistics.fmean(r[p]["peak_mem_bytes"] for r in per_seed)
                    / 1e6)
            else:
                entry["peak_mem_mb"] = None
            for k in ("verified", "contradicted", "replay_applied",
                      "confirmed_events", "temporal_probes",
                      "temporal_generated", "temporal_verified"):
                vals = [r[p][k] for r in per_seed if r[p].get(k) is not None]
                entry[k] = statistics.fmean(vals) if vals else None
            summary[condition][p] = entry
        for a, b in (("metargon12", "metargon10"),
                     ("metargon12", "metargon11"),
                     ("metargon12", "bayes"),
                     ("metargon12", "metargon12-nodc"),
                     ("metargon12", "metargon12-norc"),
                     ("metargon12", "metargon12-noad")):
            if a not in roster or b not in roster:
                continue
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
        if family == "l" and "metargon12" in roster and "metargon12-notp" in roster:
            exact_delta = [row["metargon12"]["exact_discovery_D1"]
                           - row["metargon12-notp"]["exact_discovery_D1"]
                           for row in per_seed]
            summary[condition]["comparison:temporal_minus_notp_exact_D1"] = {
                "delta": statistics.fmean(exact_delta),
                "paired_bootstrap_95pct": paired_bootstrap(exact_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.12", "family": family,
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(conds), "policies": roster,
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
        },
        "summary": summary,
        "raw": scores,
    }


def _score_one(args):
    condition, seed, episodes, family, measure_mem, roster = args
    pools, _ = family_pools(family)
    row = {p: score_policy(condition, seed, p, episodes, family, pools,
                           measure_mem=measure_mem)
           for p in roster}
    return condition, seed, row


def benchmark_parallel(seeds=8, episodes=60, family="f", conditions=None,
                       workers=None, measure_mem=False, policies=None):
    import concurrent.futures
    conds = list(CONDITIONS) if conditions is None else list(conditions)
    roster = list(policies) if policies else [
        "metargon10", "metargon11", "metargon12", "metargon12-nodc",
        "metargon12-norc", "metargon12-noad", "bayes", "oracle",
    ]
    if family in ("r", "l"):
        if family == "r":
            roster = roster + ["random"]
        else:
            roster = roster + ["random"]
            roster += ["metargon12-notp", "qlearning", "episodic"]
    tasks = [(cond, i + 11800, episodes, family, measure_mem, tuple(roster))
             for cond in conds for i in range(seeds)]
    scores = {cond: [None] * seeds for cond in conds}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for cond, seed, row in ex.map(_score_one, tasks):
            scores[cond][seed - 11800] = row
    # rebuild summary (same logic as benchmark())
    summary = {}
    for condition in conds:
        per_seed = scores[condition]
        summary[condition] = {}
        for p in roster:
            entry = {}
            for k in ("success_rate", "mean_steps", "invalid_actions",
                      "door_events_total", "mean_first_event_step", "cpu_s",
                      "exact_discovery_D1", "exact_discovery_D2",
                      "covered_discovery_D1", "covered_discovery_D2"):
                vals = [r[p][k] for r in per_seed if r[p].get(k) is not None]
                entry[k] = statistics.fmean(vals) if vals else None
            if measure_mem:
                entry["peak_mem_mb"] = (
                    statistics.fmean(r[p]["peak_mem_bytes"] for r in per_seed)
                    / 1e6)
            else:
                entry["peak_mem_mb"] = None
            for k in ("verified", "contradicted", "replay_applied",
                      "confirmed_events", "temporal_probes",
                      "temporal_generated", "temporal_verified"):
                vals = [r[p][k] for r in per_seed if r[p].get(k) is not None]
                entry[k] = statistics.fmean(vals) if vals else None
            summary[condition][p] = entry
        for a, b in (("metargon12", "metargon10"),
                     ("metargon12", "metargon11"),
                     ("metargon12", "bayes"),
                     ("metargon12", "metargon12-nodc"),
                     ("metargon12", "metargon12-norc"),
                     ("metargon12", "metargon12-noad")):
            if a not in roster or b not in roster:
                continue
            succ_delta = [row[a]["success_rate"] - row[b]["success_rate"]
                          for row in per_seed]
            summary[condition][f"comparison:{a}_minus_{b}"] = {
                "success_delta": statistics.fmean(succ_delta),
                "paired_bootstrap_95pct": paired_bootstrap(succ_delta),
            }
        if family == "l" and "metargon12" in roster and "metargon12-notp" in roster:
            exact_delta = [row["metargon12"]["exact_discovery_D1"]
                           - row["metargon12-notp"]["exact_discovery_D1"]
                           for row in per_seed]
            summary[condition]["comparison:temporal_minus_notp_exact_D1"] = {
                "delta": statistics.fmean(exact_delta),
                "paired_bootstrap_95pct": paired_bootstrap(exact_delta),
            }
    return {
        "metadata": {
            "name": "METARGON", "version": "0.12", "family": family,
            "seeds": seeds, "episodes_per_seed": episodes,
            "conditions": list(conds), "policies": roster,
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
        },
        "summary": summary,
        "raw": scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--family", type=str, choices=FAMILIES, default="f")
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-mem", action="store_true")
    parser.add_argument("--policies", type=str, default=None)
    args = parser.parse_args()
    conditions = None if not args.conditions else args.conditions.split(",")
    policies = None if not args.policies else args.policies.split(",")
    mem = not args.no_mem
    if args.parallel:
        result = benchmark_parallel(args.seeds, args.episodes, args.family,
                                    conditions, args.workers, mem, policies)
    else:
        result = benchmark(args.seeds, args.episodes, args.family, conditions,
                           mem, policies)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n",
                             encoding="utf-8")
    for c, r in result["summary"].items():
        print(c)
        for p, s in r.items():
            if p.startswith("comparison:"):
                if "success_delta" in s:
                    print(f"  {p}: delta {s['success_delta']*100:+.2f} pp "
                          f"CI {tuple(round(x*100,1) for x in s['paired_bootstrap_95pct'])}")
                else:
                    print(f"  {p}: delta {s['delta']*100:+.2f} pp")
                continue
            print(f"  {p:>18} success {s['success_rate']*100:6.2f}% "
                  f"exactD1 {s['exact_discovery_D1']*100:5.1f}% "
                  f"steps {s['mean_steps']:5.2f} "
                  f"invalid {s['invalid_actions']:5.2f} cpu {s['cpu_s']:6.2f}s")


if __name__ == "__main__":
    main()
