#!/usr/bin/env python3
"""Cognitive Integration v0.1 — runner.

Usage (from repo root):

    python -m generalization_audit.cognitive_integration_v01.runner \
        --seeds 101 103 107 109 \
        --out generalization_audit/cognitive_integration_v01/results

    python -m generalization_audit.cognitive_integration_v01.runner --quick

The confirmatory seeds are fixed by the preregistration document; smoke
(`--quick`) uses seed 0 and separate output directories.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from generalization_audit.cognitive_v02 import config as V2
from generalization_audit.cognitive_v02.arms import register_candidate
from generalization_audit.cognitive_v02.runner import (
    build_scorecard,
    git_head as _v2_git_head,
    run_seed,
)
from generalization_audit.cognitive_v02.splits import (
    load_frozen_splits,
    verify_disjoint,
    verify_frozen_splits,
    verify_solvable_sample,
)

from . import config as C
from .interaction import build_analysis

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = ROOT / "generalization_audit" / "cognitive_integration_v01" \
    / "results"


def register_protocol_arms():
    """Add the preregistered control arms to the v0.2 registry."""
    for name, spec in C.EXTRA_CONTROLS.items():
        register_candidate(name, kind=spec["kind"], base=spec["base"],
                           transform=spec["transform"])


def git_head() -> str:
    return _v2_git_head()


def build_pairs():
    pairs = []
    for pool in C.EVAL_POOLS:
        name = pool[0]
        for label, a, b in (
            ("integ-vs-epoch", "explore_epoch_reset", "epoch_reset"),
            ("integ-vs-explore", "explore_epoch_reset", "explore"),
            ("integ-vs-trained", "explore_epoch_reset", "trained"),
            ("epoch-vs-trained", "epoch_reset", "trained"),
            ("explore-vs-trained", "explore", "trained"),
            ("muted-integ-vs-muted-epoch",
             "muted_explore_epoch_reset", "muted_epoch_reset"),
            ("muted-integ-vs-integ",
             "muted_explore_epoch_reset", "explore_epoch_reset"),
            ("shuffled-integ-vs-integ",
             "shuffled_explore_epoch_reset", "explore_epoch_reset"),
        ):
            pairs.append((f"{label}.{name}.success_rate",
                          a, name, "success_rate", b, name, "success_rate"))
            pairs.append((f"{label}.{name}.zero_shot_coverage",
                          a, name, "zero_shot_coverage",
                          b, name, "zero_shot_coverage"))
    return pairs


def make_meta(seeds, sizes, phases, counts):
    return {
        "benchmark": "METARGON Cognitive Integration Evaluation v0.1",
        "date": datetime.date.today().isoformat(),
        "git_head": git_head(),
        "preregistration": "docs/PREREGISTRATION.md (frozen before "
                           "confirmatory execution)",
        "seeds": list(seeds),
        "sizes": sizes,
        "phases": list(phases),
        "splits": counts,
        "arms": {"trained": V2.ARM_SPECS, "controls": V2.CONTROL_SPECS,
                 "extra_controls": C.EXTRA_CONTROLS},
        "questions": [
            "Do memory (epoch_reset) and Cognitive Exploration v0.1 "
            "together improve learning/generalization more than either "
            "alone?",
            "Is any joint benefit carried by retained causal knowledge "
            "rather than planning/search effort?",
        ],
        "methodology": [
            "Reuses the unmodified Cognitive Evaluation v0.2 framework "
            "(merged PR #52); additional arms registered via its public "
            "candidate interface.",
            "Matched deterministic streams per seed across all arms; "
            "equal action/computation budgets.",
            "Preregistered confirmatory seeds 101/103/107/109; smoke uses "
            "seed 0 only.",
            "Bounded probabilities: pooled 95% Wilson intervals; "
            "unbounded scalars: mean ± 95% t; comparisons: paired "
            "bootstrap deltas.",
        ],
        "limitations": [
            "Small fixed seed set; finite budgets; Academy gridworld "
            "distribution only.",
            "Muted/shuffled controls still allow within-episode learning; "
            "they remove only cross-episode declarative knowledge.",
            "Interaction terms depend on the specific arm baselines and "
            "are exploratory beyond the preregistered primary pool.",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cognitive Integration v0.1")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(C.CONFIRMATORY_SEEDS))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args(argv)

    register_protocol_arms()

    if args.quick:
        args.seeds = [C.SMOKE_SEED]
        V2.TRAIN_EPISODES = 4
        V2.EVAL_EPISODES = 2
        V2.AB_TRAIN_PER_PHASE = 2
        V2.AB_EVAL_EPISODES = 2

    sizes = {"train": V2.TRAIN_EPISODES, "eval": V2.EVAL_EPISODES,
             "ab_train": V2.AB_TRAIN_PER_PHASE,
             "ab_eval": V2.AB_EVAL_EPISODES}
    phases = list(C.PHASES)

    verify_frozen_splits()
    pdata, counts = load_frozen_splits()
    verify_solvable_sample(pdata["train"])
    verify_disjoint(pdata["train"], pdata["val"], pdata["test"],
                    pdata["ood"])

    seed_results = {}
    oracle_cache = {}
    t0 = time.time()
    for seed in args.seeds:
        print(f"\n== seed {seed} ==", flush=True)
        seed_results[seed] = run_seed(
            seed, sizes, phases, oracle_cache=oracle_cache,
            with_oracle=not args.no_oracle)

    meta = make_meta(args.seeds, sizes, phases, counts)
    scorecard = build_scorecard(seed_results, meta, build_pairs())
    analysis = build_analysis(seed_results,
                              pools=[p[0] for p in C.EVAL_POOLS])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(scorecard, indent=1), encoding="utf-8")
    (args.out / "analysis.json").write_text(
        json.dumps(analysis, indent=1), encoding="utf-8")
    (args.out / "per_seed.json").write_text(
        json.dumps(_strip_bulk(seed_results), indent=1, default=str),
        encoding="utf-8")

    from .report import render_report
    report_path = args.out.parent / "docs" / "COGNITIVE_INTEGRATION_V01.md"
    render_report(scorecard, analysis, report_path)
    print(f"\nWrote results to {args.out} ({time.time() - t0:.0f}s)")
    print(f"Report at {report_path}")
    return scorecard


def _strip_bulk(obj):
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
