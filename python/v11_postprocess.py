#!/usr/bin/env python3
"""Post-process v0.11 benchmark outputs into markdown tables.

  results/v11/summary.json   -> results/v11/benchmark_table_v11.md
  results/v11/summary_h.json -> results/v11/benchmark_table_v11_h.md
Usage: python v11_postprocess.py [summary.json] [out.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "results" / "v11" / "summary.json"

POLICIES = ("metargon10", "metargon11", "metargon11-cf",
            "metargon11-ct", "metargon11-rp", "metargon11-rw",
            "bayes", "oracle")
COMPARISONS = (
    ("metargon11", "metargon10", "v11 vs v10"),
    ("metargon11", "bayes", "v11 vs bayes"),
    ("metargon11", "metargon11-cf", "v11 vs -cf (no confirmed events)"),
    ("metargon11", "metargon11-ct", "v11 vs -ct (no certainty)"),
    ("metargon11", "metargon11-rp", "v11 vs -rp (replay off)"),
    ("metargon11", "metargon11-rw", "v11 vs -rw (replay weighted)"),
)


def main(path: Path, out: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    lines = ["# METARGON v0.11 — benchmark table\n"]
    lines.append(f"Run: {data['metadata']['seeds']} seeds × "
                 f"{data['metadata']['episodes_per_seed']} episodes × "
                 f"{len(data['metadata']['conditions'])} conditions "
                 f"(family {data['metadata'].get('family', 'f')}).\n")
    lines.append("Uncertainty: paired bootstrap 95% intervals (1500 resamples).\n")
    for cond in summary:
        lines.append(f"\n## {cond}\n")
        lines.append("| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for p in POLICIES:
            if p not in summary[cond]:
                continue
            x = summary[cond][p]
            first = x.get("mean_first_event_step")
            first_s = f"{first:.2f}" if first is not None else "—"
            mem = x.get("peak_mem_mb")
            mem_s = f"{mem:.2f}" if mem is not None else "—"
            lines.append(
                f"| {p} | {x['success_rate']*100:.2f}% "
                f"| {x['mean_steps']:.2f} | {x['invalid_actions']:.2f} "
                f"| {x['door_events_total']:.1f} "
                f"| {first_s} | {x['cpu_s']:.2f} | {mem_s} |")
        lines.append("\n### Paired comparisons (success-rate deltas, pp)\n")
        lines.append("| condition | comparison | Δ pp | 95% CI pp |")
        lines.append("|---|---:|---:|---:|")
        for a, b, label in COMPARISONS:
            c = summary[cond].get(f"comparison:{a}_minus_{b}")
            if not c:
                continue
            lo, hi = c["paired_bootstrap_95pct"]
            lines.append(f"| {cond} | {label} | {c['success_delta']*100:+.2f} "
                         f"| [{lo*100:.2f}, {hi*100:.2f}] |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    o = Path(sys.argv[2]) if len(sys.argv) > 2 else p.with_name("benchmark_table_v11.md")
    main(p, o)
