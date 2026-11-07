#!/usr/bin/env python3
"""Post-process the v0.9 benchmark output into deliverable artifacts:

  results/v09/summary.json           (already written by the benchmark)
  results/v09/per_seed_results.json   (flattened per-seed table)
  results/v09/benchmark_table.md      (readable markdown tables)

Usage: python v09_postprocess.py [path-to-summary.json]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "results" / "v09" / "summary.json"
DEPLOYABLE = ("random", "episodic", "qlearning", "bayes", "metargon")


def main(summary_path: Path):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir = summary_path.parent
    summary = data["summary"]
    raw = data["raw"]

    # ---- per-seed flattened records -------------------------------------
    rows = []
    for cond in summary:
        for i, per_seed in enumerate(raw[cond]):
            seed = i + 11800
            row = {"condition": cond, "seed": seed}
            for p, res in per_seed.items():
                for k, v in res.items():
                    if isinstance(v, float) or isinstance(v, int):
                        row[f"{p}.{k}"] = v
            rows.append(row)
    (out_dir / "per_seed_results.json").write_text(
        json.dumps({"metadata": data["metadata"], "rows": rows}, indent=2) + "\n",
        encoding="utf-8")

    # ---- benchmark tables -----------------------------------------------
    lines = []
    lines.append("# METARGON v0.9 — benchmark table\n")
    lines.append("Environment: Mechanism Lab (`python/lab_env.py`). "
                 "Paired pre-generated worlds; same seeds per policy; "
                 "oracle is privileged (labelled, excluded from deployable "
                 "rankings).\n")
    lines.append(f"Run: {data['metadata']['seeds']} seeds × "
                 f"{data['metadata']['episodes_per_seed']} episodes × "
                 f"{len(data['metadata']['conditions'])} conditions.\n")
    lines.append("Uncertainty: paired bootstrap 95% intervals "
                 "(1500 resamples, seed 4301/4302).\n")

    for cond in summary:
        lines.append(f"\n## {cond}\n")
        header = "| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |"
        lines.append(header)
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for p in DEPLOYABLE + ("oracle",):
            if p not in summary[cond]:
                continue
            x = summary[cond][p]
            mem = "—" if x.get("peak_mem_mb") is None else f"{x['peak_mem_mb']:.2f}"
            lines.append(
                f"| {p} | {x['success_rate']*100:.2f}% | "
                f"{x['first_half_success']*100:.2f}% | "
                f"{x['second_half_success']*100:.2f}% | "
                f"{x['mean_steps']:.2f} | {x['cpu_s']:.2f} | "
                f"{mem} |")
        for key, c in summary[cond].items():
            if key.startswith("comparison:"):
                label = key.replace("comparison:", "").replace("_minus_", " − ")
                ci = tuple(round(x * 100, 2) for x in c["paired_bootstrap_95pct"])
                lines.append(
                    f"\n- {label}: {c['success_delta']*100:+.2f} pp "
                    f"(95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}] pp)")

    (out_dir / "benchmark_table.md").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    print(f"wrote {out_dir / 'per_seed_results.json'}")
    print(f"wrote {out_dir / 'benchmark_table.md'}")
    # quick one-line status
    for cond in summary:
        if "metargon" in summary[cond] and "bayes" in summary[cond]:
            m = summary[cond]["metargon"]
            b = summary[cond]["bayes"]
            d = summary[cond].get("comparison:metargon_minus_bayes")
            d_txt = f"{d['success_delta']*100:+.2f}pp" if d else "n/a"
            print(f"{cond:12s} metargon {m['success_rate']*100:5.2f}% | "
                  f"bayes {b['success_rate']*100:5.2f}% | delta {d_txt}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    main(path)
