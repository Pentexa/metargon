#!/usr/bin/env python3
"""METARGON v0.13 — render human-readable summary tables and per-seed
vectors from a committed benchmark artifact.

Usage:
  python python/v13_postprocess.py results/v13/summary_l_small.json
  python python/v13_postprocess.py --compare results/v13/baseline_l.json \
      results/v13/final_l.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path = [str(ROOT), str(ROOT / "python"), str(ROOT / "generalization_audit")] \
    + sys.path


METRIC_KEYS = ("success_rate", "exact_discovery_D1", "exact_discovery_D2",
               "covered_discovery_D1", "covered_discovery_D2",
               "mean_steps", "invalid_actions", "cpu_s",
               "temporal_probes", "temporal_generated", "temporal_verified",
               "temporal_evictions")


def per_seed_vector(artifact, condition, policy, key):
    out = []
    for row in artifact["raw"][condition]:
        out.append(row[policy].get(key))
    return out


def summarize(artifact, out=print):
    meta = artifact["metadata"]
    out(f"# {meta['name']} v{meta['version']} — family {meta['family']}")
    out(f"seeds {meta['seeds']} × {meta['episodes_per_seed']} episodes; "
        f"conditions {meta['conditions']}; policies {meta['policies']}")
    for condition in meta["conditions"]:
        out(f"\n## {condition}")
        policies = meta["policies"]
        header = f"| policy | " + " | ".join(METRIC_KEYS) + " |"
        out(header)
        out("|---|" + "---|" * len(METRIC_KEYS))
        for p in policies:
            vals = []
            for k in METRIC_KEYS:
                v = artifact["summary"][condition][p].get(k)
                if v is None:
                    vals.append("—")
                elif k in ("success_rate", "exact_discovery_D1",
                           "exact_discovery_D2", "covered_discovery_D1",
                           "covered_discovery_D2"):
                    vals.append(f"{v*100:.1f}%")
                elif k in ("cpu_s",):
                    vals.append(f"{v:.1f}")
                else:
                    vals.append(f"{v:.2f}")
            out(f"| {p} | " + " | ".join(vals) + " |")
        out("\n### per-seed exact D1 (frozen criteria: seed sensitivity)")
        out("| seed | " + " | ".join(
            p for p in policies if p.startswith("metargon")) + " |")
        out("|---|" + "---|" * sum(
            1 for p in policies if p.startswith("metargon")))
        for i in range(meta["seeds"]):
            cells = []
            for p in policies:
                if not p.startswith("metargon"):
                    continue
                v = per_seed_vector(artifact, condition, p,
                                    "exact_discovery_D1")[i]
                cells.append(f"{v*100:.1f}%" if v is not None else "—")
            out(f"| {i} | " + " | ".join(cells) + " |")
        # nonzero-seed counts + population stddev for temporal claims
        for p in policies:
            if not p.startswith("metargon"):
                continue
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
                    out(f"- {key}: success delta "
                        f"{val['success_delta']*100:+.2f} pp "
                        f"CI {tuple(round(x*100,1) for x in val['paired_bootstrap_95pct'])}")
                else:
                    out(f"- {key}: delta {val['delta']*100:+.2f} pp "
                        f"CI {tuple(round(x*100,1) for x in val['paired_bootstrap_95pct'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="benchmark artifact JSON")
    ap.add_argument("--out", type=Path, default=None,
                    help="write rendered markdown to this file")
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
