#!/usr/bin/env python3
"""METARGON v0.13 confirmatory analysis — preregistered endpoints.

Reads the confirmatory raw artifacts (per-seed rows) and computes the
preregistered primary/secondary endpoints with paired bootstrap 95% CIs
(seed 4301, 1500 rounds, candidate-minus-baseline). Prints a markdown
report and (optionally) writes a machine-readable JSON report.

Endpoints (from docs/RESEARCH_V13_CONFIRMATORY.md PART I):
  P1 L stable exact-D1 Δ metargon13 − metargon12
  P2 L novel  exact-D1 Δ metargon13 − metargon12
  P3 L stable/novel exact-D1 Δ metargon13 − metargon13-gen
  P4 L success Δ vs v0.12.1
  S1 F noisy_obs success/D1/D2 metargon13 vs metargon12
  S2 H stable success/D2
  S3 R stable success
Per-seed vectors, nonzero-seed counts, population stddev for every claim.

Usage:
  python python/analysis_v13_confirmatory.py \
      --artifacts results/v13_confirmatory/final_l_8x100.json \
                   results/v13_confirmatory/final_f_8x100.json \
                   results/v13_confirmatory/final_h_8x100.json \
                   results/v13_confirmatory/final_r_8x100.json \
      --report results/v13_confirmatory/analysis_report.md \
      --json results/v13_confirmatory/analysis.json
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

BOOT = {"seed": 4301, "rounds": 1500, "alpha": 0.05}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def per_seed(art, cond, pol, metric):
    return [r[pol].get(metric) for r in art["raw"][cond] if pol in r]


def mean(art, cond, pol, metric):
    v = per_seed(art, cond, pol, metric)
    return statistics.fmean(v) if v else None


def delta_report(a_vec, b_vec, label):
    a = [x for x in a_vec if x is not None]
    b = [x for x in b_vec if x is not None]
    diffs = [x - y for x, y in zip(a, b)]
    d = statistics.fmean(diffs)
    lo, hi = paired_bootstrap(diffs, seed=BOOT["seed"],
                              rounds=BOOT["rounds"])
    nz = sum(1 for v in diffs if v > 0)
    return {
        "label": label,
        "candidate_mean": statistics.fmean(a),
        "baseline_mean": statistics.fmean(b),
        "delta": d,
        "ci": [lo, hi],
        "n_seeds": len(diffs),
        "n_positive_delta_seeds": nz,
        "per_seed_delta": diffs,
    }


def report_endpoint(art, cond, pol_a, pol_b, metric, label):
    va = per_seed(art, cond, pol_a, metric)
    vb = per_seed(art, cond, pol_b, metric)
    return delta_report(va, vb, label)


def analyze(artifacts):
    out = {"endpoints": [], "tables": {}}
    # mapping: artifact family/condition -> (A, B, metric, label)
    for art in artifacts:
        fam = art["metadata"]["family"]
        for cond in art["metadata"]["conditions"]:
            # primary L endpoints
            if fam == "l" and cond in ("stable", "novel"):
                out["endpoints"].append(report_endpoint(
                    art, cond, "metargon13", "metargon12",
                    "exact_discovery_D1",
                    f"P1/P2 L {cond} exact-D1 13-12"))
                out["endpoints"].append(report_endpoint(
                    art, cond, "metargon13", "metargon13-gen",
                    "exact_discovery_D1",
                    f"P3 L {cond} exact-D1 13-gen"))
                out["endpoints"].append(report_endpoint(
                    art, cond, "metargon13", "metargon13-notp",
                    "exact_discovery_D1",
                    f"P3b L {cond} exact-D1 13-notp"))
                out["endpoints"].append(report_endpoint(
                    art, cond, "metargon13", "metargon12",
                    "success_rate",
                    f"P4 L {cond} success 13-12"))
            # secondary preservation cells
            if fam in ("f", "h", "r") and "metargon13" in \
                    art["metadata"]["policies"] and \
                    "metargon12" in art["metadata"]["policies"]:
                for metric, label in (
                        ("success_rate", "success"),
                        ("exact_discovery_D1", "exact-D1"),
                        ("exact_discovery_D2", "exact-D2")):
                    out["endpoints"].append(report_endpoint(
                        art, cond, "metargon13", "metargon12", metric,
                        f"S {fam}/{cond} {label} 13-12"))
            # per-policy per-seed table rows
            for pol in art["metadata"]["policies"]:
                key = f"{fam}/{cond}/{pol}"
                out["tables"][key] = {
                    "success_rate": per_seed(art, cond, pol, "success_rate"),
                    "exact_discovery_D1": per_seed(
                        art, cond, pol, "exact_discovery_D1"),
                    "exact_discovery_D2": per_seed(
                        art, cond, pol, "exact_discovery_D2"),
                }
    return out


def render_md(out):
    lines = ["# METARGON v0.13 confirmatory analysis (preregistered endpoints)",
             "", "| endpoint | delta pp | 95% CI | seeds | +delta seeds |",
             "|---|---:|---:|---:|---:|"]
    for e in out["endpoints"]:
        lines.append(
            f"| {e['label']} | {e['delta']*100:+.2f} | "
            f"[{e['ci'][0]*100:.2f}, {e['ci'][1]*100:.2f}] | "
            f"{e['n_seeds']} | {e['n_positive_delta_seeds']} |")
    lines.append("")
    lines.append("## Per-seed exact-D1 vectors")
    for key, t in sorted(out["tables"].items()):
        lines.append(f"- {key}: {[round(v*100,1) if v is not None else None for v in t['exact_discovery_D1']]}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", nargs="+", required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    artifacts = [load(p) for p in args.artifacts]
    out = analyze(artifacts)
    md = render_md(out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(md, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n",
                             encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
