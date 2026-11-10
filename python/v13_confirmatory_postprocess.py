#!/usr/bin/env python3
"""Render METARGON v0.13 confirmatory summary tables + per-seed vectors.

Usage:
  python python/v13_confirmatory_postprocess.py \
      results/v13_confirmatory/final_l_8x100.json
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

METRIC_KEYS = ("success_rate", "exact_discovery_D1", "exact_discovery_D2",
               "covered_discovery_D1", "covered_discovery_D2",
               "mean_steps", "invalid_actions", "cpu_s",
               "temporal_probes", "temporal_generated", "temporal_verified",
               "temporal_evictions")


def per_seed_vector(artifact, condition, policy, key):
    return [row[policy].get(key) for row in artifact["raw"][condition]]


def summarize(artifact, out=print):
    meta = artifact["metadata"]
    out(f"# METARGON v0.13 confirmatory — family {meta['family']} "
        f"(seeds {meta['seed_base']}+, {meta['episodes_per_seed']} eps)")
    out(f"conditions {meta['conditions']}; policies {meta['policies']}")
    for condition in meta["conditions"]:
        out(f"\n## {condition}")
        out("| policy | " + " | ".join(METRIC_KEYS) + " |")
        out("|---|" + "---|" * len(METRIC_KEYS))
        for p in meta["policies"]:
            vals = []
            for k in METRIC_KEYS:
                v = artifact["summary"][condition][p].get(k)
                if v is None:
                    vals.append("—")
                elif k in ("success_rate", "exact_discovery_D1",
                           "exact_discovery_D2", "covered_discovery_D1",
                           "covered_discovery_D2"):
                    vals.append(f"{v*100:.1f}%")
                elif k == "cpu_s":
                    vals.append(f"{v:.1f}")
                else:
                    vals.append(f"{v:.2f}")
            out(f"| {p} | " + " | ".join(vals) + " |")
        out("\n### per-seed exact D1")
        pols = [p for p in meta["policies"]
                if p.startswith("metargon")]
        out("| seed | " + " | ".join(pols) + " |")
        out("|---|" + "---|" * len(pols))
        for i in range(meta["seeds"]):
            cells = []
            for p in pols:
                v = per_seed_vector(artifact, condition, p,
                                    "exact_discovery_D1")[i]
                cells.append(f"{v*100:.1f}%" if v is not None else "—")
            out(f"| {meta['seed_base']+i} | " + " | ".join(cells) + " |")
        for p in pols:
            vec = [v for v in per_seed_vector(artifact, condition, p,
                                              "exact_discovery_D1")
                   if v is not None]
            if not vec:
                continue
            nz = sum(1 for v in vec if v > 0)
            sd = statistics.pstdev(vec) if len(vec) > 1 else 0.0
            out(f"- {p}: exact D1 nonzero seeds {nz}/{len(vec)}, "
                f"population stddev {sd:.3f}")
        for key, val in artifact["summary"][condition].items():
            if key.startswith("comparison:"):
                if "success_delta" in val:
                    out(f"- {key}: success Δ {val['success_delta']*100:+.2f} "
                        f"pp CI {tuple(round(x*100,1) for x in val['paired_bootstrap_95pct'])}")
                else:
                    out(f"- {key}: Δ {val['delta']*100:+.2f} pp "
                        f"CI {tuple(round(x*100,1) for x in val['paired_bootstrap_95pct'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    artifact = json.loads(args.path.read_text(encoding="utf-8"))
    if args.out:
        lines = []
        summarize(artifact, out=lines.append)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        summarize(artifact)


if __name__ == "__main__":
    main()
