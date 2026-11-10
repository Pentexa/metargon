#!/usr/bin/env python3
"""METARGON v0.13 confirmatory failure classification (Mission C).

Consumes the instrumented diagnostics JSON from `diagnose_v13.py` and
classifies each episode (per seed × condition × policy) into the five
preregistered failure classes:

  C1 sequence never observed    — true ordered pair never appears in the
                                  belief-flip stream (floor false)
  C2 candidate not generated    — floor true but correct hypothesis never
                                  created
  C3 generated but not verified — correct candidate created but never
                                  reached `verified` in the episode
  C4 contradicted / evicted     — correct candidate's evidence was
                                  destroyed (contradiction > 0 or
                                  evicted while correct)
  C5 discovered but unused      — verified correct hypothesis exists but
                                  no temporal probe targeted its door in
                                  the episode (planner never used it)

Also reports per-seed counts and the distribution across seeds so the
classification can be inspected for seed-dependence.

Usage:
  python python/classify_v13_failures.py \
      --json results/v13_confirmatory/diag_l_stable_8x40.json \
              results/v13_confirmatory/diag_l_novel_8x40.json \
      --report results/v13_confirmatory/failure_classification.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))


def classify_episode(e):
    """Return the failure class of one instrumented episode (or None when
    the episode did not fail, i.e. discovery or success)."""
    t = e.get("temporal", {})
    floor_d1 = t.get("true_pair_observed", {}).get("D1", False)
    floor_d2 = t.get("true_pair_observed", {}).get("D2", False)
    floor = floor_d1 or floor_d2
    created = t.get("correct_created", 0)
    verified = t.get("correct_verified_max", 0)
    contradicted = t.get("correct_contradicted", 0)
    evicted = t.get("correct_evicted", 0)
    agent = e.get("agent", {})
    probes = agent.get("probe_attempts", 0)
    if not floor:
        return "C1_sequence_never_observed"
    if created == 0:
        return "C2_candidate_not_generated"
    if verified == 0:
        if contradicted > 0 or evicted > 0:
            return "C4_contradicted_or_evicted"
        return "C3_generated_not_verified"
    # verified correct rule exists
    if probes == 0:
        return "C5_discovered_but_unused"
    return None  # success path


def analyze(diag_paths, policy_label=None):
    counts = Counter()
    per_seed = defaultdict(Counter)
    total_episodes = 0
    for p in diag_paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for run in d.get("runs", []):
            seed = run["seed"]
            pol = policy_label or d["metadata"].get("policy", "?")
            for e in run.get("episodes", []):
                total_episodes += 1
                cls = classify_episode(e)
                if cls is not None:
                    counts[cls] += 1
                    per_seed[seed][cls] += 1
                else:
                    counts["success_or_used"] += 1
                    per_seed[seed]["success_or_used"] += 1
    return counts, per_seed, total_episodes


def render(counts, per_seed, total):
    lines = ["# METARGON v0.13 failure classification (Mission C)",
             "", f"total episodes instrumented: {total}", ""]
    lines.append("| class | episodes | fraction |")
    lines.append("|---|---:|---:|")
    for cls, n in counts.most_common():
        lines.append(f"| {cls} | {n} | {n/total:.3f} |")
    lines.append("")
    lines.append("## per-seed distribution")
    for seed in sorted(per_seed):
        lines.append(f"- seed {seed}: "
                     + ", ".join(f"{k}={v}" for k, v in
                                 sorted(per_seed[seed].items())))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--policy", default=None)
    args = ap.parse_args()
    counts, per_seed, total = analyze(args.json, args.policy)
    md = render(counts, per_seed, total)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
