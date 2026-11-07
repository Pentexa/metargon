#!/usr/bin/env python3
"""Offline discovery-quality analysis for v0.9 vs v0.10 (evaluation only).

For each configuration actually played, and using the *final* hypothesis
store of the agent (which was never given the truth), check:

  * discovery: does some VERIFIED (or best-usable) hypothesis for a door
    exactly match the true condition?
  * credit coverage: is the true condition a subset of a verified
    hypothesis's antecedent (i.e., a correct but over-specific rule)?
  * noise contamination (noisy_obs): how many verified hypotheses are
    false (do not match the true condition)?

Output: results/v10/discovery_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "python"))

from lab_env import LabWorld, generate_worlds  # noqa: E402
from benchmark_v10 import family_pools  # noqa: E402
from metargon_v09 import MetargonV09  # noqa: E402
from metargon_v10 import MetargonV10  # noqa: E402


def exact_match(h, true_cond):
    return h.antecedent == frozenset(true_cond)


def covers(h, true_cond):
    return frozenset(true_cond) <= h.antecedent


def run(condition, seeds, episodes, family="f", name="metargon10",
        kwargs=None):
    pools = None
    if family == "h":
        pools, _ = family_pools("h")
    totals = {"n": 0, "exact": 0, "covered": 0, "verified_false": 0,
              "verified_total": 0, "best_usable_exact": 0}
    for s in range(seeds):
        if pools is None:
            worlds = generate_worlds(condition, 77000 + s, episodes)
        else:
            worlds = generate_worlds(condition, 77000 + s, episodes,
                                     train_pool=pools[0], val_pool=pools[1],
                                     test_pool=pools[2])
        pol = MetargonV10(s + 500003, **(kwargs or {})) if name.startswith("metargon10") \
            else MetargonV09(s + 500003)
        for spec in worlds:
            pol.episode_start()
            w = LabWorld(spec.config, ep_seed=spec.seed,
                         noise_q=spec.noise_q, max_steps=spec.max_steps)
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = pol.act(obs)
                res = w.step(a)
                pol.observe_outcome(obs, a, res)
            for door, true_cond in (("D1", spec.config.d1_cond),
                                    ("D2", spec.config.d2_cond)):
                totals["n"] += 1
                hyps = [h for h in pol.memory.hypotheses if h.door == door]
                verified = [h for h in hyps if h.status == "verified"]
                totals["verified_total"] += len(verified)
                if any(exact_match(h, true_cond) for h in verified):
                    totals["exact"] += 1
                if any(covers(h, true_cond) for h in verified):
                    totals["covered"] += 1
                if verified and not any(exact_match(h, true_cond)
                                       for h in verified):
                    totals["verified_false"] += len(verified)
                usable = [h for h in hyps if h.status in ("predicted", "verified")]
                if usable and exact_match(usable[0], true_cond):
                    totals["best_usable_exact"] += 1
    return totals


def main():
    out_dir = root / "results" / "v10"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for cond, fam in (("stable", "f"), ("noisy_obs", "f"),
                      ("stable", "h"), ("novel", "h")):
        row = {}
        for name, kw in (("metargon09", None), ("metargon10", None),
                         ("metargon10-rb", {"use_robust_belief": False}),
                         ("metargon10-cd", {"use_cooldowns": False}),
                         ("metargon10-x", {"use_explore_fallback": False})):
            t = run(cond, seeds=3, episodes=20, family=fam, name=name,
                    kwargs=kw)
            row[name] = {
                "doors_evaluated": t["n"],
                "exact_discovered": t["exact"] / max(1, t["n"]),
                "covered_discovered": t["covered"] / max(1, t["n"]),
                "verified_false_ratio": (
                    t["verified_false"] / max(1, t["verified_total"])),
                "verified_total": t["verified_total"],
                "best_usable_exact": (
                    t["best_usable_exact"] / max(1, t["n"])),
            }
        report[f"{cond}_{fam}"] = row
    (out_dir / "discovery_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, row in report.items():
        print(f"\n{key}")
        for name, r in row.items():
            print(f"  {name:>14}: exact {r['exact_discovered']*100:5.1f}% "
                  f"covered {r['covered_discovered']*100:5.1f}% "
                  f"false-verified {r['verified_false_ratio']*100:5.1f}% "
                  f"(n={r['verified_total']}) best-usable "
                  f"{r['best_usable_exact']*100:5.1f}%")


if __name__ == "__main__":
    main()
