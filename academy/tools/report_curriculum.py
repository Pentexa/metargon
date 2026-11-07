"""Render the curriculum v0.2 summary JSON as a compact markdown table.

Usage:
    python -m academy.tools.report_curriculum results/academy_v01/evren/curriculum_v02_summary.json
"""

from __future__ import annotations

import json
import sys


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def report(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    header = ("| Condition | Train | Classic test | Teacher holdout | OOD | "
              "Retention Δ | Substrate | Treatment |")
    sep = "|---|---|---|---|---|---|---|---|"
    rows.append(header)
    rows.append(sep)
    for cond, d in data["conditions_detail"].items():
        tr = d["train"]
        cls = d["classic_test"]
        hold = d.get("teacher_holdout")
        ood = d["ood_probe"]
        ret = d["retention_classic"]
        rows.append(
            f"| {cond} | {tr['success_rate']:.3f} "
            f"| {cls['config_mean_success']:.3f} "
            f"| {_fmt(hold.get('config_mean_success') if hold else None)} "
            f"| {ood['config_mean_success']:.3f} "
            f"| {ret['retention_delta']:+.3f} "
            f"| {tr['substrate_episodes']} "
            f"| {tr['treatment_episodes']} |")
    rows.append("")
    rows.append("Paired bootstrap comparisons (95% CI):")
    for name, cmp_ in data.get("comparisons", {}).items():
        if cmp_:
            lo, hi = cmp_["ci"]
            rows.append(f"- **{name}**: {cmp_['mean']:+.4f} "
                        f"(CI [{lo:+.4f}, {hi:+.4f}])")
    return "\n".join(rows)


if __name__ == "__main__":
    print(report(sys.argv[1] if len(sys.argv) > 1
                 else "results/academy_v01/evren/curriculum_v02_summary.json"))
