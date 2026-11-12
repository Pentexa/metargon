#!/usr/bin/env python3
"""Cognitive Evaluation v0.2 — runner.

Usage (from repo root):

    python -m generalization_audit.cognitive_v02.runner \
        --seeds 0 1 2 3 --out generalization_audit/cognitive_v02/results

    python -m generalization_audit.cognitive_v02.runner --quick
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from .arms import (
    all_arms,
    apply_transform,
    load_candidate_specs,
    make_fresh,
    make_policy,
    trained_arms,
)
from .driver import memory_summary, run_batch
from .metrics import (
    ab_phase_raw,
    eval_pool_raw,
    paired_deltas,
    retention_raw,
    standard_train_raw,
    summarize,
    summarize_proportion,
    truth_rules,
)
from .splits import (
    load_frozen_splits,
    verify_disjoint,
    verify_frozen_splits,
    verify_solvable_sample,
)
from .streams import (
    conflict_pair,
    config_signature,
    eval_stream,
    phase_train_stream,
    retention_eval_stream,
    train_stream,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = ROOT / "generalization_audit" / "cognitive_v02" / "results"


def git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=str(ROOT))
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Per-seed execution
# ---------------------------------------------------------------------------
def run_seed(seed, sizes, phases, oracle_cache=None, with_oracle=True):
    oracle_cache = oracle_cache if oracle_cache is not None else {}
    train_n = sizes["train"]
    eval_n = sizes["eval"]
    ab_n = sizes["ab_train"]
    ab_eval_n = sizes["ab_eval"]

    train = train_stream(seed, train_n)
    eval_pools = {
        name: eval_stream(pool, seed, eval_n, noise_q)
        for name, pool, noise_q in C.EVAL_POOLS
    }
    eval_truths = {name: truth_rules(st) for name, st in eval_pools.items()}
    train_truths = truth_rules(train)

    a_cfg, b_cfg = conflict_pair(seed)
    ab_streams = {
        "A1": phase_train_stream(a_cfg, seed, ab_n, "phaseA1"),
        "A1_eval": retention_eval_stream(a_cfg, seed + 1, ab_eval_n),
        "B": phase_train_stream(b_cfg, seed + 2, ab_n, "phaseB"),
        "B_eval": retention_eval_stream(b_cfg, seed + 3, ab_eval_n),
        "A2": phase_train_stream(a_cfg, seed + 4, ab_n, "phaseA2"),
        "A2_eval": retention_eval_stream(a_cfg, seed + 5, ab_eval_n),
    }
    ab_truths = {k: truth_rules(v) for k, v in ab_streams.items()}

    results = {}

    def arm_results(name):
        return results.setdefault(name, {})

    def eval_one_pool(agent, pool_name, pre_mem):
        recs = run_batch(agent, eval_pools[pool_name],
                         eval_truths[pool_name])
        return {"recs": recs,
                "metrics": eval_pool_raw(recs, pre_mem,
                                         eval_truths[pool_name])}

    trained_state = {}
    for arm in trained_arms():
        agent = make_policy(arm, seed)
        train_recs = run_batch(agent, train, train_truths)
        r = arm_results(arm)
        r["train"] = {"recs": train_recs,
                      "metrics": standard_train_raw(train_recs,
                                                   train_truths)}
        trained_state[arm] = copy.deepcopy(agent.save_state())
        pre_mem = train_recs[-1]["mem"] if train_recs else []
        for pool in eval_pools:
            a2 = make_policy(arm, seed)
            a2.load_state(trained_state[arm])
            r[pool] = eval_one_pool(a2, pool, pre_mem)
        if "ab" in phases:
            r["ab"] = _run_ab(agent, ab_streams, ab_truths, a_cfg)

    for ctrl, spec in C.CONTROL_SPECS.items():
        r = arm_results(ctrl)
        if spec["kind"] == "fresh":
            r["train"] = {"recs": [],
                          "metrics": standard_train_raw([], train_truths)}
            for pool in eval_pools:
                agent = make_fresh(ctrl, seed)
                r[pool] = eval_one_pool(agent, pool, None)
            if "ab" in phases:
                r["ab"] = _run_ab_fresh(seed, ab_streams, ab_truths)
        else:
            base = spec["base"]
            # training metrics are identical to the base arm (transform is
            # applied after training)
            r["train"] = copy.deepcopy(results[base]["train"])
            for pool in eval_pools:
                agent = make_policy(base, seed)
                agent.load_state(trained_state[base])
                apply_transform(agent, spec["transform"], seed)
                pre_mem = memory_summary(agent)  # post-transform view
                r[pool] = eval_one_pool(agent, pool, pre_mem)
            if "ab" in phases:
                r["ab"] = _run_ab_transform(
                    base, spec["transform"], seed, ab_streams, ab_truths,
                    a_cfg, trained_state)

    if with_oracle:
        results["_floors"] = _run_floors(seed, eval_pools, eval_truths,
                                         oracle_cache)
    return results


def _run_ab(agent, streams, truths, a_cfg):
    out = {}
    for ph in ("A1", "B", "A2"):
        recs = run_batch(agent, streams[ph], truths[ph])
        out[ph] = {"recs": recs, "metrics": ab_phase_raw(recs)}
    for ev in ("A1_eval", "B_eval", "A2_eval"):
        recs = run_batch(agent, streams[ev], truths[ev])
        out[ev] = {"recs": recs, "metrics": ab_phase_raw(recs)}
    mem_after = memory_summary(agent)
    out["retention"] = {"metrics": retention_raw(
        out["A1_eval"]["recs"], out["A2_eval"]["recs"],
        out["B_eval"]["recs"], a_cfg, mem_after)}
    out["_final_mem_meta"] = {
        "verified_correct": len([h for h in mem_after
                                 if h["status"] == "verified"]),
        "hypotheses_total": len(mem_after),
    }
    return out


def _run_ab_fresh(seed, streams, truths):
    out = {}
    for ev in ("A1_eval", "B_eval", "A2_eval"):
        agent = make_fresh("fresh", seed)
        recs = run_batch(agent, streams[ev], truths[ev])
        out[ev] = {"recs": recs, "metrics": ab_phase_raw(recs)}
    out["retention"] = {"note": "fresh: no prior knowledge / retention"}
    return out


def _run_ab_transform(base, transform, seed, streams, truths, a_cfg,
                      trained_state):
    out = {}
    for ev in ("A1_eval", "B_eval", "A2_eval"):
        agent = make_policy(base, seed)
        agent.load_state(trained_state[base])
        apply_transform(agent, transform, seed)
        recs = run_batch(agent, streams[ev], truths[ev])
        out[ev] = {"recs": recs, "metrics": ab_phase_raw(recs)}
    out["retention"] = {"note": f"{transform} control: retention n/a"}
    return out


def _run_floors(seed, eval_pools, truths, oracle_cache):
    from academy.agents.baselines import RandomPolicy

    random_res = {}
    for pool, stream in eval_pools.items():
        agent = RandomPolicy(seed=seed)
        recs = run_batch(agent, stream, truths[pool])
        random_res[pool] = {"success_rate": (
            sum(1 for r in recs if r["success"]), len(recs))}

    oracle_res = {}
    for pool, stream in eval_pools.items():
        recs = []
        for spec in stream:
            key = config_signature(spec.config)
            plan = oracle_cache.get(key)
            if plan is None:
                plan = _solve_config(spec)
                oracle_cache[key] = plan
            recs.append(_play_plan(plan, spec))
        oracle_res[pool] = {"success_rate": (
            sum(1 for r in recs if r["success"]), len(recs))}
    return {"random": random_res, "oracle": oracle_res}


def _solve_config(spec):
    from collections import deque
    from academy.envs.gridworld import ACTIONS, TOPOLOGIES, make_ws, \
        transition_with_hidden
    geom = TOPOLOGIES[spec.config.topology]
    start = make_ws(geom)
    goal = geom.goal

    def key(st):
        return (st.pos, st.objs.get("A"), st.objs.get("B"),
                tuple(st.switch_state[s] for s in ("S1", "S2")),
                tuple(st.plate_load[p] for p in ("P1", "P2")),
                tuple(st.door_open[d] for d in ("D1", "D2")),
                tuple(st.armed[d] for d in ("D1", "D2")))

    q = deque([(start, [])])
    seen = {key(start)}
    cap = 400000
    while q and len(seen) < cap:
        st, path = q.popleft()
        if st.pos == goal:
            return path
        for action in ACTIONS:
            nxt, _ = transition_with_hidden(st, action, spec.config, geom)
            if nxt is None:
                continue
            k = key(nxt)
            if k in seen:
                continue
            seen.add(k)
            q.append((nxt, path + [action]))
    return []


def _play_plan(plan, spec):
    from academy.envs.gridworld import MechanismGrid
    env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
    env.reset(spec.spec_seed)
    for action in plan:
        if env.finished or env.steps >= env.max_steps:
            break
        env.step(action)
    return {"success": env.finished, "steps": env.steps}


# ---------------------------------------------------------------------------
# Aggregation -> scorecard
# ---------------------------------------------------------------------------
def _raw_vectors(seed_results, arm, block, key):
    out = []
    for seed in sorted(seed_results):
        node = seed_results[seed].get(arm, {})
        if block:
            for part in block.split("."):
                node = node.get(part, {})
        v = node.get("metrics", {}).get(key) if isinstance(node, dict) else None
        out.append(v)
    return out


_TRAIN_METRICS = (
    ("success_rate", True), ("within_episode_discovery", True),
    ("coverage", True), ("precision", True),
    ("fallback_rate", True), ("invalid_rate", True),
    ("verified_correct", False), ("verified_wrong", False),
    ("hypotheses_total", False), ("planner_ops_per_step", False),
    ("mean_steps_to_success", False),
)

_EVAL_METRICS = (
    ("success_rate", True), ("door_open_rate", True),
    ("within_episode_discovery", True), ("zero_shot_coverage", True),
    ("fallback_rate", True), ("invalid_rate", True),
    ("planner_ops_per_step", False), ("plan_cache_hits_per_step", False),
    ("plans_used_per_step", False), ("mean_steps_to_success", False),
)

_AB_EVAL_METRICS = (
    ("success_rate", True), ("within_episode_discovery", True),
    ("fallback_rate", True),
    ("mean_steps_to_success", False), ("planner_ops_per_step", False),
)

_AB_RETENTION_METRICS = (
    ("a1_success", True), ("a2_success", True), ("b_success", True),
    ("a_rule_retained_final", True),
)


def build_scorecard(seed_results, meta, paired_pairs):
    sc = {"meta": meta, "arms": {}, "paired": {}}

    def add(arm, block, key, is_prop):
        vals = _raw_vectors(seed_results, arm, block, key)
        if is_prop:
            sc["arms"].setdefault(arm, {}).setdefault(block, {})[key] = \
                summarize_proportion(vals)
        else:
            sc["arms"].setdefault(arm, {}).setdefault(block, {})[key] = \
                summarize(vals)

    for arm in all_arms():
        for key, is_prop in _TRAIN_METRICS:
            add(arm, "train", key, is_prop)
        for pool, _, _ in C.EVAL_POOLS:
            for key, is_prop in _EVAL_METRICS:
                add(arm, pool, key, is_prop)
        for ph in ("A1_eval", "B_eval", "A2_eval"):
            for key, is_prop in _AB_EVAL_METRICS:
                add(arm, f"ab.{ph}", key, is_prop)
        for key, is_prop in _AB_RETENTION_METRICS:
            add(arm, "ab.retention", key, is_prop)

    def _prop_to_rate(v):
        if v is None or (isinstance(v, tuple) and v[1] <= 0):
            return None
        if isinstance(v, tuple):
            return v[0] / v[1]
        return v

    # pair tuple: (label, arm_a, block_a, key_a, arm_b, block_b, key_b)
    for label, arm_a, block_a, key_a, arm_b, block_b, key_b in paired_pairs:
        va = _raw_vectors(seed_results, arm_a, block_a, key_a)
        vb = _raw_vectors(seed_results, arm_b, block_b, key_b)
        va = [_prop_to_rate(v) for v in va]
        vb = [_prop_to_rate(v) for v in vb]
        sc["paired"][label] = paired_deltas(va, vb)

    floors = {}
    for seed in sorted(seed_results):
        fl = seed_results[seed].get("_floors")
        if fl is None:
            continue
        for kind in ("random", "oracle"):
            for pool, row in fl[kind].items():
                key = (kind, pool)
                floors.setdefault(key, []).append(row["success_rate"][0] /
                                                  row["success_rate"][1])
    sc["floors"] = {f"{k} {p}": summarize(v)
                    for (k, p), v in floors.items()}
    return sc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cognitive Evaluation v0.2")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(C.DEFAULT_SEEDS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--phases", default="standard,ab")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds = args.seeds[:1]
        C.TRAIN_EPISODES = 4
        C.EVAL_EPISODES = 2
        C.AB_TRAIN_PER_PHASE = 2
        C.AB_EVAL_EPISODES = 2

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    sizes = {"train": C.TRAIN_EPISODES, "eval": C.EVAL_EPISODES,
             "ab_train": C.AB_TRAIN_PER_PHASE,
             "ab_eval": C.AB_EVAL_EPISODES}

    verify_frozen_splits()
    pdata, counts = load_frozen_splits()
    verify_solvable_sample(pdata["train"])
    verify_disjoint(pdata["train"], pdata["val"], pdata["test"],
                    pdata["ood"])
    registered = load_candidate_specs()
    if registered:
        print("registered candidate arms:", registered)

    seed_results = {}
    oracle_cache = {}
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n== seed {seed} ==", flush=True)
        seed_results[seed] = run_seed(
            seed, sizes, phases, oracle_cache=oracle_cache,
            with_oracle=not args.no_oracle)

    paired_pairs = []
    for pool_name, _, _ in C.EVAL_POOLS:
        paired_pairs.append(
            (f"trained-fresh.{pool_name}.success_rate",
             "trained", pool_name, "success_rate",
             "fresh", pool_name, "success_rate"))
        paired_pairs.append(
            (f"muted-trained.{pool_name}.success_rate",
             "muted", pool_name, "success_rate",
             "trained", pool_name, "success_rate"))
        paired_pairs.append(
            (f"shuffled-trained.{pool_name}.success_rate",
             "shuffled", pool_name, "success_rate",
             "trained", pool_name, "success_rate"))
        paired_pairs.append(
            (f"epoch_reset-trained.{pool_name}.success_rate",
             "epoch_reset", pool_name, "success_rate",
             "trained", pool_name, "success_rate"))
        paired_pairs.append(
            (f"explore-trained.{pool_name}.success_rate",
             "explore", pool_name, "success_rate",
             "trained", pool_name, "success_rate"))
        paired_pairs.append(
            (f"explore+epoch-trained.{pool_name}.success_rate",
             "explore_epoch_reset", pool_name, "success_rate",
             "trained", pool_name, "success_rate"))
        paired_pairs.append(
            (f"explore+epoch-epoch.{pool_name}.success_rate",
             "explore_epoch_reset", pool_name, "success_rate",
             "epoch_reset", pool_name, "success_rate"))
    if "ab" in phases:
        paired_pairs.append(
            ("ab.forgetting.trained.a2-a1", "trained",
             "ab.retention", "a2_success",
             "trained", "ab.retention", "a1_success"))

    meta = {
        "benchmark": "METARGON Cognitive Evaluation v0.2",
        "date": datetime.date.today().isoformat(),
        "git_head": git_head(),
        "seeds": list(args.seeds),
        "sizes": sizes,
        "phases": phases,
        "splits": counts,
        "arms": {"trained": C.ARM_SPECS, "controls": C.CONTROL_SPECS},
        "stream_seed_base": C.STREAM_SEED_BASE,
        "questions": [
            "Do cognitive improvements reflect genuine learning, memory "
            "transfer and adaptation, or more search effort?",
            "Does a trained agent outperform a fresh agent with identical "
            "action and computation budgets?",
            "Do apparent gains disappear when learned memory is removed "
            "(muted) or its associations are randomized (shuffled)?",
            "Separating within-episode discovery from cross-episode "
            "knowledge transfer.",
            "Catastrophic forgetting after a conflicting A/B/A task "
            "sequence.",
            "Distinguishing improved planning from improved causal "
            "knowledge.",
        ],
        "methodology": [
            "Frozen train/val/test/OOD splits (snapshotted and verified); "
            "matched deterministic episode streams per seed for every arm.",
            "Arms: trained (default agent), epoch_reset (v0.7 episode-"
            "scoped evidence + per-episode probe reset), explore (Cognitive "
            "Exploration v0.1 on), explore+epoch_reset, fresh, muted "
            "(declarative memory wiped), shuffled (associations randomized).",
            "Bounded probability metrics use pooled 95% Wilson intervals; "
            "unbounded metrics use mean ± 95% t intervals; paired "
            "comparisons use bootstrap 95% CIs.",
            "Zero-shot coverage is measured from pre-evaluation memory "
            "against the true rules of each evaluation pool.",
            "Within-episode discovery marks episodes where a new "
            "verified-correct hypothesis was created inside the episode.",
            "A/B/A: train rule A, eval A', train conflicting rule B, eval "
            "B', retrain A, eval A'' — A'' vs A' quantifies forgetting; "
            "B' quantifies recovery after the mechanism change.",
        ],
        "limitations": [
            "Budgets and seeds are finite; results describe this student "
            "on the Academy gridworld distribution (80-step cap).",
            "Muted/shuffled controls still allow within-episode learning "
            "during evaluation; they remove only cross-episode declarative "
            "knowledge.",
            "Zero-shot coverage is a knowledge-content measure, distinct "
            "from task success (which also involves planning/exploration).",
            "The shuffle control preserves counts/support levels and only "
            "randomizes associations; it does not remove all structure.",
        ],
    }
    scorecard = build_scorecard(seed_results, meta, paired_pairs)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(scorecard, indent=1), encoding="utf-8")
    (args.out / "per_seed.json").write_text(
        json.dumps(_strip_bulk(seed_results), indent=1, default=str),
        encoding="utf-8")
    from .report import render_scorecard
    report_path = args.out.parent / "docs" / "COGNITIVE_EVALUATION_V02.md"
    render_scorecard(scorecard, report_path)
    print(f"\nWrote results to {args.out} ({time.time() - t0:.0f}s)")
    print(f"Report at {report_path}")
    return scorecard


def _strip_bulk(obj):
    """Drop bulky per-episode fields from persisted raw records (mem
    antecedents, event lists); computed metrics are stored separately."""
    if isinstance(obj, dict):
        if "mem" in obj:
            mem = obj["mem"]
            counts = {}
            for h in mem:
                counts[h["status"]] = counts.get(h["status"], 0) + 1
            obj = {k: v for k, v in obj.items() if k != "mem"}
            obj["mem_summary"] = counts
        if "events" in obj and isinstance(obj["events"], list):
            obj["events"] = len(obj["events"])
        return {k: _strip_bulk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_bulk(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
