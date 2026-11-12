#!/usr/bin/env python3
"""METARGON Cognitive Intelligence Benchmark v0.1 — runner.

Usage (from repository root):

    python -m generalization_audit.cognitive_v01.benchmark_cognitive \
        --seeds 4 --out generalization_audit/cognitive_v01/results

    python -m generalization_audit.cognitive_v01.benchmark_cognitive \
        --quick   # tiny preset for CI/tests

Evaluates the unmodified Academy student on six cognitive dimensions using
frozen train/val/test/ood splits, matched episode streams across every
policy, multiple seeds, and 95% confidence intervals.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import config
from .metrics import (
    build_scorecard,
    eval_metrics,
    restart_metrics,
    revisit_metrics,
    switch_metrics,
    truth_rules,
    zero_shot_coverage,
)
from .policies import make_policy, run_episode, wipe_memory
from .report import render_scorecard
from .protocols import (
    config_signature,
    eval_stream,
    revisit_occurrences,
    revisit_stream,
    stream_signature,
    switch_pair,
    switch_split_point,
    switch_stream,
    train_stream,
)
from .splits import (
    load_frozen_splits,
    verify_disjoint,
    verify_frozen_splits,
    verify_solvable,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = ROOT / "generalization_audit" / "cognitive_v01" / "results"


def _strip_bulk(obj):
    """Drop bulky per-step fields (e.g. action lists) from persisted
    per-seed records; they are not needed for scorecard aggregation."""
    if isinstance(obj, dict):
        if "actions" in obj or "plan_cache" in obj:
            obj = {k: v for k, v in obj.items()
                   if k not in ("actions", "plan_cache")}
        if "memory" in obj and isinstance(obj["memory"], list):
            # keep only counts (full hypothesis detail lives in summary.json)
            hyps = obj["memory"]
            status_counts = {}
            for h in hyps:
                status_counts[h["status"]] = status_counts.get(h["status"], 0) + 1
            obj["memory"] = {"n": len(hyps), "status_counts": status_counts}
        if "diagnostics" in obj:
            obj.pop("diagnostics", None)
        return {k: _strip_bulk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_bulk(v) for v in obj]
    return obj


def git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=str(ROOT))
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _run_protocol(policy, stream):
    return [run_episode(policy, spec) for spec in stream]


def _eval_recs(policy, pool_tag, seed, episodes):
    stream = eval_stream(pool_tag, seed, episodes)
    return stream, _run_protocol(policy, stream)


def run_seed(seed, sizes, policies=None, include_restart=True):
    """Run all protocols for one seed. Returns results dict."""
    train_episodes = sizes["train"]
    eval_episodes = sizes["eval"]
    revisit_episodes = sizes["revisit"]
    switch_half = sizes["switch_half"]

    train = train_stream(seed, train_episodes)
    eval_val = eval_stream("val", seed, eval_episodes)
    eval_test = eval_stream("test", seed, eval_episodes)
    eval_ood = eval_stream("ood", seed, eval_episodes)
    revisit = revisit_stream(seed, n_configs=3) if revisit_episodes else []
    switch = switch_stream(seed, switch_half) if switch_half else []
    split = switch_split_point(switch) if switch else None

    results = {}

    def recs(policy, protocol, data):
        results.setdefault(policy, {})[protocol] = data

    # ---- train phase (metargon arms) ------------------------------------
    train_recs = {}
    trained_state = {}
    for pol in ("metargon_trained", "metargon_muted"):
        agent = make_policy(pol, seed)
        recs(pol, "train", {"stream": stream_signature(train),
                             "recs": _run_protocol(agent, train)})
        train_recs[pol] = results[pol]["train"]["recs"]
        trained_state[pol] = copy.deepcopy(agent.save_state())

    # ---- eval protocols --------------------------------------------------
    eval_pools = (("eval_val", eval_val), ("eval_test", eval_test),
                  ("eval_ood", eval_ood))

    def fresh_agent(pol):
        return make_policy(pol, seed)

    # trained arms: each protocol starts from the identical trained memory
    for pol in ("metargon_trained", "metargon_muted"):
        for name, stream in eval_pools:
            agent = fresh_agent(pol)
            agent.load_state(trained_state[pol])
            if pol == "metargon_muted":
                wipe_memory(agent)
            recs(pol, name, {"stream": stream_signature(stream),
                             "recs": _run_protocol(agent, stream)})
        if include_restart:
            restored = fresh_agent(pol)
            restored.load_state(trained_state[pol])
            if pol == "metargon_muted":
                wipe_memory(restored)
            direct = results[pol]["eval_test"]["recs"]
            restored_recs = _run_protocol(restored, eval_test)
            recs(pol, "restart",
                 {"stream": stream_signature(eval_test),
                  "recs": {"direct": direct, "restored": restored_recs}})
        if revisit:
            agent = fresh_agent(pol)
            agent.load_state(trained_state[pol])
            if pol == "metargon_muted":
                wipe_memory(agent)
            recs(pol, "revisit", {"stream": stream_signature(revisit),
                                  "recs": _run_protocol(agent, revisit)})
        if switch:
            agent = fresh_agent(pol)
            agent.load_state(trained_state[pol])
            if pol == "metargon_muted":
                wipe_memory(agent)
            recs(pol, "switch", {"stream": stream_signature(switch),
                                "recs": _run_protocol(agent, switch)})

    # fresh metargon: a fresh agent per pool (no cross-pool contamination)
    for name, stream in eval_pools:
        agent = fresh_agent("metargon_fresh")
        recs("metargon_fresh", name,
             {"stream": stream_signature(stream),
              "recs": _run_protocol(agent, stream)})
    if revisit:
        agent = fresh_agent("metargon_fresh")
        recs("metargon_fresh", "revisit",
             {"stream": stream_signature(revisit),
              "recs": _run_protocol(agent, revisit)})
    if switch:
        agent = fresh_agent("metargon_fresh")
        recs("metargon_fresh", "switch",
             {"stream": stream_signature(switch),
              "recs": _run_protocol(agent, switch)})

    # baselines (model-free comparators; fresh agent per pool)
    for pol in ("random", "episodic", "qlearning", "oracle"):
        for name, stream in eval_pools:
            agent = fresh_agent(pol)
            recs(pol, name, {"stream": stream_signature(stream),
                             "recs": _run_protocol(agent, stream)})

    # ---- zero-shot coverage (pre-eval memory) ----------------------------
    zs = {}
    for pol in ("metargon_trained", "metargon_muted"):
        pre_mem = train_recs[pol][-1].get("memory", []) if train_recs[pol] else []
        for name, stream in eval_pools:
            zs[f"{pol}.{name}.zero_shot_coverage"] = \
                zero_shot_coverage(pre_mem, stream)
    return results, {"zero_shot": zs, "switch_split": split}


def main(argv=None):
    ap = argparse.ArgumentParser(description="METARGON Cognitive Benchmark v0.1")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(config.DEFAULT_SEEDS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true",
                    help="tiny sizes (for tests / smoke)")
    ap.add_argument("--train-episodes", type=int, default=config.TRAIN_EPISODES)
    ap.add_argument("--eval-episodes", type=int, default=config.EVAL_EPISODES)
    ap.add_argument("--revisit-episodes", type=int, default=config.REVISIT_EPISODES)
    ap.add_argument("--switch-half", type=int, default=config.SWITCH_HALF)
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--render-only", action="store_true",
                    help="regenerate the report from an existing summary.json")
    args = ap.parse_args(argv)

    if args.render_only:
        summary = json.loads((args.out / "summary.json").read_text(
            encoding="utf-8"))
        report_path = (args.out.parent / "docs" / "COGNITIVE_REPORT_v01.md")
        render_scorecard(summary, report_path)
        print(f"Report at {report_path}")
        return summary

    if args.quick:
        args.train_episodes = 4
        args.eval_episodes = 2
        args.revisit_episodes = 7
        args.switch_half = 2
        args.seeds = args.seeds[:1]

    sizes = {
        "train": args.train_episodes,
        "eval": args.eval_episodes,
        "revisit": args.revisit_episodes,
        "switch_half": args.switch_half,
    }

    # ---- pre-flight verification -----------------------------------------
    verify_frozen_splits()
    pools_data, counts = load_frozen_splits()
    verify_solvable(pools_data["train"])
    verify_disjoint(pools_data["train"], pools_data["val"],
                    pools_data["test"], pools_data["ood"])

    seed_results = {}
    extra_vectors_sum = {}
    extras = []
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n== seed {seed} ==", flush=True)
        res, extra = run_seed(seed, sizes,
                              include_restart=not args.no_restart)
        seed_results[seed] = res
        extras.append(extra)
        for k, v in extra["zero_shot"].items():
            extra_vectors_sum.setdefault(k, []).append(v)

    # ---- meta ------------------------------------------------------------
    first_seed = args.seeds[0]
    train = train_stream(first_seed, sizes["train"])
    revisit = revisit_stream(first_seed, n_configs=3) if sizes["revisit"] else []
    switch = switch_stream(first_seed, sizes["switch_half"]) if sizes["switch_half"] else []
    meta = {
        "benchmark": "METARGON Cognitive Intelligence Benchmark v0.1",
        "date": datetime.date.today().isoformat(),
        "git_head": git_head(),
        "seeds": list(args.seeds),
        "sizes": sizes,
        "splits": counts,
        "train_truths": len(truth_rules(train)),
        "train_distinct_configs": len(set(config_signature(s.config)
                                          for s in train)),
        "train_stream_signature_len": len(stream_signature(train)),
        "train_stream_first": [config_signature(s.config) for s in train[:3]],
        "switch_pair": [config_signature(s.config) for s in switch[:2]] if switch else None,
        "switch_target_door": "D1",  # switch protocol always swaps D1 rules
        "switch_split": extras[0]["switch_split"] if extras else None,
        "revisit_occurrences": revisit_occurrences(revisit) if revisit else {},
        "protocols": list(config.PROTOCOLS),
    }

    extra_vectors = {}
    for k, vals in extra_vectors_sum.items():
        extra_vectors[k] = vals

    scorecard = build_scorecard(seed_results, meta, extra_vectors,
                                train_stream_specs=train)

    # ---- write outputs ----------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(scorecard, indent=1), encoding="utf-8")
    (args.out / "per_seed.json").write_text(
        json.dumps(_strip_bulk(seed_results), indent=1, default=str),
        encoding="utf-8")
    report_path = (args.out.parent / "docs" / "COGNITIVE_REPORT_v01.md")
    render_scorecard(scorecard, report_path)

    print(f"\nWrote results to {args.out}")
    print(f"Report at {report_path}")
    print(f"Elapsed {time.time() - t0:.1f}s")
    return scorecard


if __name__ == "__main__":
    main()
