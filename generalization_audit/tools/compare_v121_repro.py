#!/usr/bin/env python3
"""Audit v0.4 — compare committed v0.12.1 artifacts with audit reruns.

Reruns of the committed harness (`python/benchmark_v12.py`, pinned
PYTHONHASHSEED=0, 4 seeds x 40 episodes, F-noisy / H-stable) are stored
under `generalization_audit/results/v121_repro/`.  This tool prints the
side-by-side summary so reviewers can verify the reproducibility findings
in `docs/AUDIT_V04.md`:

- the rerun `metargon12` rows reproduce the diagnosis-phase `-norc`
  rows exactly (F-noisy, H-stable);
- the rerun `metargon12-legacy` rows reproduce the diagnosis
  `metargon12` rows and the final `metargon12-legacy` rows exactly;
- the committed FINAL `metargon12` rows (recent-credit off) disagree
  with the rerun on exact-D2 (45.0/41.25 vs 75.62/70.0) and cannot be
  regenerated from the committed harness.

Usage:  python generalization_audit/tools/compare_v121_repro.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REPRO = ROOT / "generalization_audit" / "results" / "v121_repro"

ARTIFACTS = {
    "final_f.json": ("results/v121/final_f.json", "noisy_obs"),
    "diag_f_noisy.json": ("results/v121/diag_f_noisy.json", "noisy_obs"),
    "final_h.json": ("results/v121/final_h.json", "stable"),
    "diag_h_stable.json": ("results/v121/diag_h_stable.json", "stable"),
}
REPRO_FILES = {
    "final_f_repro_20261109.json": ("noisy_obs", "REPRO F-noisy (current main)"),
    "final_h_repro_20261109.json": ("stable", "REPRO H-stable (current main)"),
}
POLICIES = ("metargon10", "metargon11", "metargon12", "metargon12-legacy")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary_rows(artifact, cond):
    d = load(artifact)
    return d["summary"].get(cond, {})


def main():
    rows = []
    for repro_name, (cond, label) in REPRO_FILES.items():
        s = summary_rows(REPRO / repro_name, cond)
        row = {"label": label}
        for pol in POLICIES:
            e = s.get(pol, {})
            row[pol] = (e.get("success_rate"), e.get("exact_discovery_D1"),
                        e.get("exact_discovery_D2"))
        rows.append(row)
    for name, (rel, cond) in ARTIFACTS.items():
        s = summary_rows(ROOT / rel, cond)
        row = {"label": name}
        for pol in POLICIES:
            e = s.get(pol) or {}
            row[pol] = (e.get("success_rate"), e.get("exact_discovery_D1"),
                        e.get("exact_discovery_D2"))
        rows.append(row)

    header = f"{'run':38s}" + "".join(
        f"{p:>16s}" for p in POLICIES)
    print(header)
    for r in rows:
        cells = []
        for pol in POLICIES:
            succ, d1, d2 = r[pol]
            if succ is None:
                cells.append("      n/a")
                continue
            cells.append(f"{succ*100:5.1f}/{d1*100:4.1f}/{d2*100:5.1f}")
        print(f"{r['label']:38s}" + "".join(f"{c:>16s}" for c in cells))
    print()
    print("format: success% / exactD1% / exactD2%")

    # diagnostic identities
    repro_f = summary_rows(REPRO / "final_f_repro_20261109.json", "noisy_obs")
    diag_f = summary_rows(ROOT / "results/v121/diag_f_noisy.json", "noisy_obs")
    final_f = summary_rows(ROOT / "results/v121/final_f.json", "noisy_obs")
    repro_h = summary_rows(REPRO / "final_h_repro_20261109.json", "stable")
    diag_h = summary_rows(ROOT / "results/v121/diag_h_stable.json", "stable")
    final_h = summary_rows(ROOT / "results/v121/final_h.json", "stable")
    print()
    print("1. Rerun metargon12 == diag -norc (F noisy):",
          repro_f["metargon12"]["exact_discovery_D2"] == diag_f["metargon12-norc"]["exact_discovery_D2"])
    print("2. Rerun legacy == diag metargon12 == final legacy (F noisy):",
          repro_f["metargon12-legacy"]["exact_discovery_D2"] == diag_f["metargon12"]["exact_discovery_D2"]
          == final_f["metargon12-legacy"]["exact_discovery_D2"])
    print("3. Rerun metargon12 D2 vs final metargon12 D2 (F noisy):",
          repro_f["metargon12"]["exact_discovery_D2"], "vs",
          final_f["metargon12"]["exact_discovery_D2"])
    print("4. Rerun metargon12 D2 vs final metargon12 D2 (H stable):",
          repro_h["metargon12"]["exact_discovery_D2"], "vs",
          final_h["metargon12"]["exact_discovery_D2"])
    print("5. Rerun metargon12 == diag -norc (H stable):",
          repro_h["metargon12"]["exact_discovery_D2"] == diag_h["metargon12-norc"]["exact_discovery_D2"])


if __name__ == "__main__":
    main()
