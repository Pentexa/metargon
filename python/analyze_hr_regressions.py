#!/usr/bin/env python3
"""METARGON v0.13 confirmatory H/R regression diagnosis (Mission D).

Uses the 8x100 confirmatory preservation cells to characterize the
H-stable and R-stable regressions:
  * H stable: success and exact-D2 vs v0.12.1 (known regression from
    v0.11 confirmed-event gating; not caused by v0.13 mechanisms).
  * R stable: success vs v0.12.1 (dual-context cost) and the exploratory
    -nodc recovery (dual-context off).

Usage:
  python python/analyze_hr_regressions.py \
      --artifacts results/v13_confirmatory/final_h_8x100.json \
                   results/v13_confirmatory/final_r_8x100.json \
                   results/v13_confirmatory/final_r_nodc_8x100.json \
      --report results/v13_confirmatory/hr_regressions.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from benchmark_v13 import paired_bootstrap  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def per_seed(art, cond, pol, metric):
    return [r[pol].get(metric) for r in art["raw"][cond] if pol in r]


def report(art, cond, a, b, metric, label):
    va = per_seed(art, cond, a, metric)
    vb = per_seed(art, cond, b, metric)
    diffs = [x - y for x, y in zip(va, vb)]
    d = statistics.fmean(diffs)
    lo, hi = paired_bootstrap(diffs, seed=4301, rounds=1500)
    return {
        "label": label,
        "candidate_mean": statistics.fmean(va),
        "baseline_mean": statistics.fmean(vb),
        "delta": d, "ci": [lo, hi],
        "n_seeds": len(diffs),
        "per_seed_delta": diffs,
    }


def analyze(paths):
    arts = [load(p) for p in paths]
    out = []
    for art in arts:
        fam = art["metadata"]["family"]
        cond = art["metadata"]["conditions"][0]
        if fam == "h":
            out.append(report(art, cond, "metargon13", "metargon12",
                              "success_rate",
                              "H stable success 13-12 (regression)"))
            out.append(report(art, cond, "metargon13", "metargon12",
                              "exact_discovery_D2",
                              "H stable exact-D2 13-12"))
        elif fam == "r":
            pols = art["metadata"]["policies"]
            if any("nodc" in p for p in pols):
                out.append(report(art, cond, "metargon13-nodc",
                                  "metargon12-nodc", "success_rate",
                                  "R stable success 13-nodc minus 12-nodc (exploratory)"))
            else:
                out.append(report(art, cond, "metargon13", "metargon12",
                                  "success_rate",
                                  "R stable success 13-12 (regression)"))
                out.append(report(art, cond, "metargon13", "metargon12",
                                  "exact_discovery_D1",
                                  "R stable exact-D1 13-12"))
    return out


def render(out):
    lines = ["# METARGON v0.13 confirmatory H/R regression diagnosis",
             "", "| endpoint | delta pp | 95% CI | seeds |",
             "|---|---:|---:|---:|"]
    for e in out:
        lines.append(f"| {e['label']} | {e['delta']*100:+.2f} | "
                     f"[{e['ci'][0]*100:.2f}, {e['ci'][1]*100:.2f}] | "
                     f"{e['n_seeds']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", nargs="+", required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()
    out = analyze(args.artifacts)
    md = render(out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
