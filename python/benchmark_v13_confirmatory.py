#!/usr/bin/env python3
"""METARGON v0.13 confirmatory runner — bounded, resumable, incremental.

Wraps the benchmark harness primitives with:
  * per-(condition, seed) checkpointing: each completed seed-row is
    appended to a JSONL file immediately, so an interrupted run resumes
    from the last completed row (safe resumption);
  * bounded parallelism (ProcessPoolExecutor, --workers);
  * explicit per-run timeout (--timeout-min) enforced by the caller
    shell; here we also print progress and flush after every row;
  * per-seed raw rows identical to `benchmark_v13.py` semantics, and a
    final summary built with the same `build_summary` function (paired
    bootstrap CIs, candidate-minus-baseline deltas).

Usage:
  python python/benchmark_v13_confirmatory.py \
      --family l --conditions stable,novel \
      --seeds 8 --episodes 100 --seed-base 21800 \
      --policies metargon12,metargon13,metargon13-gen,metargon13-notp \
      --workers 4 --jsonl results/v13_confirmatory/final_l_8x100.jsonl \
      --json results/v13_confirmatory/final_l_8x100.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "generalization_audit"))

from benchmark_v13 import (  # noqa: E402
    _default_roster, build_summary, family_pools, score_policy,
)

def _load_checkpoint(jsonl: Path) -> dict:
    """Existing rows keyed by (condition, seed)."""
    rows = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows[(rec["condition"], rec["seed"])] = rec["row"]
    return rows


def _score_one(args):
    condition, seed, episodes, family, measure_mem, roster = args
    pools, _ = family_pools(family)
    row = {p: score_policy(condition, seed, p, episodes, family, pools,
                           measure_mem=measure_mem)
           for p in roster}
    return condition, seed, row


def run(seeds=8, episodes=100, family="l", conditions=None, workers=4,
        policies=None, seed_base=21800, measure_mem=False, jsonl=None):
    conds = list(conditions) if conditions else ["stable", "novel"]
    roster = _default_roster(family, policies)
    # restrict roster to requested policies for confirmatory lists
    if policies:
        roster = list(policies)
    existing = _load_checkpoint(jsonl) if jsonl else {}
    tasks = []
    for cond in conds:
        for i in range(seeds):
            seed = i + seed_base
            if (cond, seed) in existing:
                continue
            tasks.append((cond, seed, episodes, family, measure_mem,
                          tuple(roster)))
    print(f"[confirmatory] family={family} conds={conds} seeds={seeds} "
          f"episodes={episodes} base={seed_base} policies={roster} "
          f"workers={workers} pending_tasks={len(tasks)}", flush=True)

    scores = {cond: [None] * seeds for cond in conds}
    for cond in conds:
        for i in range(seeds):
            rec = existing.get((cond, i + seed_base))
            if rec is not None:
                scores[cond][i] = rec

    if tasks:
        import concurrent.futures
        t0 = time.perf_counter()
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for cond, seed, row in ex.map(_score_one, tasks):
                idx = seed - seed_base
                scores[cond][idx] = row
                if jsonl is not None:
                    with open(jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(
                            {"condition": cond, "seed": seed, "row": row},
                            separators=(",", ":")) + "\n")
                done += 1
                if done % 8 == 0 or done == len(tasks):
                    el = time.perf_counter() - t0
                    print(f"[confirmatory] completed {done}/{len(tasks)} "
                          f"rows in {el:.0f}s", flush=True)
        print(f"[confirmatory] all tasks done in "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    summary = build_summary(conds, seeds, roster, scores)
    return {
        "metadata": {
            "name": "METARGON", "version": "0.13-confirmatory",
            "family": family,
            "seeds": seeds, "episodes_per_seed": episodes,
            "seed_base": seed_base,
            "conditions": list(conds), "policies": roster,
            "same_pre_generated_worlds": True,
            "oracle_has_hidden_config": True,
            "PYTHONHASHSEED": "0",
            "study": "confirmatory-8x100",
        },
        "summary": summary,
        "raw": scores,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="l")
    ap.add_argument("--conditions", default="stable,novel")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=21800)
    ap.add_argument("--policies", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-mem", action="store_true")
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    policies = None if not args.policies else args.policies.split(",")
    result = run(seeds=args.seeds, episodes=args.episodes, family=args.family,
                 conditions=args.conditions.split(","), workers=args.workers,
                 policies=policies, seed_base=args.seed_base,
                 measure_mem=not args.no_mem, jsonl=args.jsonl)
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
            print(f"  {p:>20} success {s['success_rate']*100:6.2f}% "
                  f"exactD1 {s['exact_discovery_D1']*100:5.1f}% "
                  f"steps {s['mean_steps']:5.2f} "
                  f"invalid {s['invalid_actions']:5.2f} cpu {s['cpu_s']:6.2f}s")


if __name__ == "__main__":
    main()
