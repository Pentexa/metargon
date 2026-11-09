#!/usr/bin/env python3
"""Post-process v0.12 benchmark JSONs into markdown tables (compact)."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "results" / "v12" / "summary_f.json"

POLICIES = ("metargon10", "metargon11", "metargon12", "metargon12-nodc",
            "metargon12-norc", "metargon12-noad", "metargon12-notp",
            "random", "episodic", "qlearning", "bayes", "oracle")


def main(path: Path, out: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    family = data["metadata"]["family"]
    lines = [f"# METARGON v0.12 — {family} family\n"]
    lines.append(f"Run: {data['metadata']['seeds']} seeds × "
                 f"{data['metadata']['episodes_per_seed']} episodes × "
                 f"{len(data['metadata']['conditions'])} conditions.\n")
    lines.append("Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).\n")
    for cond in summary:
        lines.append(f"\n## {cond}\n")
        lines.append("| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for p in POLICIES:
            if p not in summary[cond]:
                continue
            x = summary[cond][p]
            ex1 = x.get("exact_discovery_D1")
            ex2 = x.get("exact_discovery_D2")
            ex1_s = f"{ex1*100:.1f}%" if ex1 is not None else "—"
            ex2_s = f"{ex2*100:.1f}%" if ex2 is not None else "—"
            lines.append(
                f"| {p} | {x['success_rate']*100:.2f}% "
                f"| {ex1_s} | {ex2_s} "
                f"| {x['mean_steps']:.2f} "
                f"| {x['invalid_actions']:.2f} | {x['cpu_s']:.2f} |")
        # temporal diagnostics on L
        if family == "l" and "metargon12" in summary[cond]:
            x = summary[cond]["metargon12"]
            lines.append("\ntemporal diag (metargon12): "
                         f"generated={x.get('temporal_generated')}, "
                         f"verified={x.get('temporal_verified')}, "
                         f"probes={x.get('temporal_probes')}\n")
        lines.append("\n### Paired comparisons (success-rate deltas, pp)\n")
        lines.append("| comparison | Δ pp | 95% CI pp |")
        lines.append("|---|---:|---:|")
        for key, c in summary[cond].items():
            if not key.startswith("comparison:"):
                continue
            if "success_delta" in c:
                lo, hi = c["paired_bootstrap_95pct"]
                lines.append(f"| {key} | {c['success_delta']*100:+.2f} "
                             f"| [{lo*100:.2f}, {hi*100:.2f}] |")
            elif "delta" in c:
                lo, hi = c["paired_bootstrap_95pct"]
                lines.append(f"| {key} | {c['delta']*100:+.2f} "
                             f"| [{lo*100:.2f}, {hi*100:.2f}] |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    o = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        p.with_name("table_{}.md".format(p.stem.replace("summary_", "v12_")))
    main(p, o)
