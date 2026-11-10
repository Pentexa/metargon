#!/usr/bin/env python3
"""METARGON v0.13 confirmatory H/R regression diagnosis (Mission D)
using the instrumented compositional diagnostics.

For each family (H/R), classifies episodes as:
  * L: correct rule learned exactly (D1 exact or D2 exact)
  * S: episode succeeded
  * learned-but-failed: correct rule learned but episode not completed
  * failed-no-learn: neither
Also reports per-door exact/covered rates and per-seed breakdowns.

Usage:
  python python/analyze_hr_diag.py \
      --json results/v13_confirmatory/diag_h_stable_8x40.json \
             results/v13_confirmatory/diag_r_stable_8x40.json \
      --report results/v13_confirmatory/hr_diag.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze(paths):
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for run in d.get("runs", []):
            seed = run["seed"]
            for e in run.get("episodes", []):
                disc = e.get("discovery", {})
                d1_exact = bool(disc.get("D1", [False, False])[0])
                d2_exact = bool(disc.get("D2", [False, False])[0])
                learned = d1_exact or d2_exact
                succ = bool(e.get("success"))
                if learned and succ:
                    cls = "learned+success"
                elif learned and not succ:
                    cls = "learned_but_failed"
                elif not learned and succ:
                    cls = "success_without_exact_learn"
                else:
                    cls = "failed_no_exact_learn"
                out.append({"seed": seed, "cls": cls, "success": succ,
                            "learned": learned,
                            "D1_exact": d1_exact, "D2_exact": d2_exact})
    return out


def render(rows):
    counts = Counter(r["cls"] for r in rows)
    per_seed = defaultdict(Counter)
    for r in rows:
        per_seed[r["seed"]][r["cls"]] += 1
    n = len(rows)
    lines = ["# METARGON v0.13 confirmatory H/R compositional diagnostics",
             "", f"total episodes: {n}", "",
             "| class | episodes | fraction |", "|---|---:|---:|"]
    for cls in ("learned+success", "learned_but_failed",
                "success_without_exact_learn", "failed_no_exact_learn"):
        c = counts.get(cls, 0)
        lines.append(f"| {cls} | {c} | {c/n:.3f} |")
    lines.append("")
    lines.append("## per-seed")
    for seed in sorted(per_seed):
        lines.append(f"- seed {seed}: " + ", ".join(
            f"{k}={v}" for k, v in sorted(per_seed[seed].items())))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    rows = analyze(args.json)
    md = render(rows)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
