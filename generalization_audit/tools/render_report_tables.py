#!/usr/bin/env python3
"""Render audit benchmark tables as Markdown from the result JSONs.

Usage (from repository root):

    python generalization_audit/tools/render_report_tables.py \
        generalization_audit/results/summary_r.json \
        generalization_audit/results/summary_l.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.stdout.encoding and "utf-8" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

DEPLOYABLE = ("metargon10", "random", "episodic", "qlearning", "bayes",
              "metargon10-cap", "qlearning-cap", "bayes-cap")


def render(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    fam = data["metadata"]["family"]
    lines = [f"### Family {fam.upper()} — "
             f"{data['metadata']['seeds']} seeds × "
             f"{data['metadata']['episodes_per_seed']} episodes",
             ""]
    for cond, row in data["summary"].items():
        lines.append(f"**{cond}**")
        lines.append("")
        lines.append("| Policy | Success | Steps | Invalid | Events | CPU s |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for p in DEPLOYABLE + ("oracle",):
            if p not in row:
                continue
            x = row[p]
            lines.append(
                f"| {p} | {x['success_rate']*100:.2f}% | "
                f"{x['mean_steps']:.1f} | {x['invalid_actions']:.1f} | "
                f"{x['door_events_total']:.1f} | {x['cpu_s']:.1f} |")
            if "discovery" in x:
                d1 = x["discovery"]["D1"]
                lines.append(
                    f"| | D1 exact {d1['exact_rate']*100:.1f}% · "
                    f"covered {d1['covered_rate']*100:.1f}% · "
                    f"false-ver {d1['false_verified_per_episode']:.2f}/ep | | | | |")
        lines.append("")
        for key, c in row.items():
            if key.startswith("comparison:"):
                label = key.replace("comparison:", "").replace("_minus_", " − ")
                ci = tuple(round(v * 100, 1) for v in c["paired_bootstrap_95pct"])
                lines.append(
                    f"- {label}: {c['success_delta']*100:+.2f} pp "
                    f"(95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] pp)")
        lines.append("")
    return "\n".join(lines)


def main():
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        print(render(p))
        print()


if __name__ == "__main__":
    main()
