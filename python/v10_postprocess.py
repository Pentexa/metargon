#!/usr/bin/env python3
"""Post-process v0.10 benchmark output into deliverable artifacts.

  results/v10/summary.json            (written by the benchmark)
  results/v10/per_seed_results_v10.json
  results/v10/benchmark_table_v10.md   (readable markdown tables)
  results/v10/ablation_table.md        (v0.9 vs v0.10 vs ablation deltas)

Usage: python v10_postprocess.py [path-to-summary.json]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "results" / "v10" / "summary.json"
DEPLOYABLE = ("metargon09", "metargon10", "metargon10-rb",
              "metargon10-cd", "metargon10-x", "metargon10-rp",
              "bayes", "oracle")
METRICS = (
    ("success_rate", "success", "{:.2%}"),
    ("mean_steps", "steps", "{:.2f}"),
    ("invalid_actions", "invalid", "{:.2f}"),
    ("repeat_loops_total", "loops", "{:.1f}"),
    ("stall_steps_total", "stall", "{:.1f}"),
    ("door_events_total", "events", "{:.1f}"),
    ("mean_first_event_step", "firstEv", "{:.2f}"),
    ("cpu_s", "cpu s", "{:.2f}"),
    ("peak_mem_mb", "peak MB", "{:.2f}"),
)

COMPARISONS = (
    ("metargon10", "metargon09", "v10 vs v09"),
    ("metargon10", "bayes", "v10 vs bayes"),
    ("metargon10", "metargon10-rb", "v10 vs -rb (no robust belief)"),
    ("metargon10", "metargon10-cd", "v10 vs -cd (no cooldowns)"),
    ("metargon10", "metargon10-x", "v10 vs -x (no info fallback)"),
    ("metargon10", "metargon10-rp", "v10 vs -rp (no replay)"),
)


def main(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir = summary_path.parent
    summary = data["summary"]
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# METARGON v0.10 — benchmark table\n"]
    lines.append("Environment: Mechanism Lab (`python/lab_env.py`). Paired "
                 "pre-generated worlds; same seeds per policy; oracle is "
                 "privileged (labelled, excluded from deployable rankings).\n")
    lines.append(f"Run: {data['metadata']['seeds']} seeds × "
                 f"{data['metadata']['episodes_per_seed']} episodes × "
                 f"{len(data['metadata']['conditions'])} conditions.\n")
    lines.append("Uncertainty: paired bootstrap 95% intervals "
                 "(1500 resamples, seed 4301).\n")
    lines.append("Policies: `metargon09` = v0.9 agent; `metargon10` = v0.10 "
                 "full; `-rb` = robust belief ablated; `-cd` = probe "
                 "cooldowns ablated; `-x` = information-gathering fallback "
                 "ablated; `bayes` = model-based particle-filter planner "
                 "(given the declared family F).\n")

    for cond in summary:
        lines.append(f"\n## {cond}\n")
        header = ("| policy | success | steps | invalid | loops | stall | "
                  "events | firstEv | cpu s | peak MB |")
        lines.append(header)
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in DEPLOYABLE:
            if p not in summary[cond]:
                continue
            x = summary[cond][p]
            cells = [p]
            for key, _label, fmt in METRICS:
                v = x.get(key)
                cells.append(fmt.format(v) if v is not None else "—")
            lines.append("| " + " | ".join(cells) + " |")

    # ablations / paired comparisons
    lines.append("\n## Paired comparisons (success-rate deltas, pp)\n")
    lines.append("| condition | comparison | Δ pp | 95% CI pp |")
    lines.append("|---|---:|---:|---:|")
    for cond in summary:
        for a, b, label in COMPARISONS:
            c = summary[cond].get(f"comparison:{a}_minus_{b}")
            if not c:
                continue
            lo, hi = c["paired_bootstrap_95pct"]
            lines.append(f"| {cond} | {label} | {c['success_delta']*100:+.2f} "
                         f"| [{lo*100:.2f}, {hi*100:.2f}] |")

    (out_dir / "benchmark_table_v10.md").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    print(f"wrote {out_dir / 'benchmark_table_v10.md'}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    main(path)
