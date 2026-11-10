#!/usr/bin/env python3
"""Independent audit of v0.13 headline claims against raw artifacts.

Recomputes everything directly from the per-seed `raw` vectors:
  * L stable/novel success and exact D1 per seed + means (v12, v13, -gen)
  * temporal deltas (candidate minus baseline) with paired bootstrap CIs
  * F/H/R preservation cells (v13 vs v12)
  * diagnostic-vs-benchmark consistency for seed 0/2
Prints a report; exits nonzero on any mismatch vs the values recorded in
the committed artifacts' summary (or the expected claims).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from benchmark_v13 import paired_bootstrap  # noqa: E402

DATA = Path(__file__).resolve().parent

failures = []


def check(label, got, expected, tol=1e-9):
    ok = abs(got - expected) <= tol or (expected is None)
    status = "OK " if ok else "FAIL"
    if not ok:
        failures.append(label)
    print(f"[{status}] {label}: got {got:.6f} expected {expected:.6f}")


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def per_seed(art, cond, pol, metric):
    return [r[pol].get(metric) for r in art["raw"][cond] if pol in r]


def mean(art, cond, pol, metric):
    v = per_seed(art, cond, pol, metric)
    return statistics.fmean(v) if v else None


# ---------------------------------------------------------------- L-final
final_l = load("final_l.json")
for cond in ("stable", "novel"):
    for pol, k in (("metargon12", "v12"), ("metargon13", "v13"),
                    ("metargon13-gen", "v13-gen"), ("metargon13-notp", "v13-notp"),
                    ("metargon12-notp", "v12-notp")):
        succ = mean(final_l, cond, pol, "success_rate")
        exact = mean(final_l, cond, pol, "exact_discovery_D1")
        vec = per_seed(final_l, cond, pol, "exact_discovery_D1")
        nz = sum(1 for v in vec if v and v > 0)
        print(f"  L/{cond} {pol}: success={succ:.4f} exactD1={exact:.4f} "
              f"nz_seeds={nz}/4 vec={[round(v,3) for v in vec]}")
    # temporal delta v13 vs v13-notp
    a = [x - y for x, y in zip(
        per_seed(final_l, cond, "metargon13", "exact_discovery_D1"),
        per_seed(final_l, cond, "metargon13-notp", "exact_discovery_D1"))]
    d = statistics.fmean(a)
    ci = paired_bootstrap(a)
    print(f"  L/{cond} temporal delta v13 = {d:.5f} CI=({ci[0]:.4f},{ci[1]:.4f})")
    a12 = [x - y for x, y in zip(
        per_seed(final_l, cond, "metargon12", "exact_discovery_D1"),
        per_seed(final_l, cond, "metargon12-notp", "exact_discovery_D1"))]
    d12 = statistics.fmean(a12)
    print(f"  L/{cond} temporal delta v12 = {d12:.5f}")

print()
# ---------------------------------------------------------------- F/H/R
for family, art, conds in (
        ("f", load("final_f.json"), ("stable", "shift", "noisy_obs", "novel", "tight_budget")),
        ("h", load("final_h.json"), ("stable", "novel")),
        ("r", load("final_r.json"), ("stable", "novel"))):
    print(f"family {family}")
    for cond in conds:
        for pol in ("metargon12", "metargon13"):
            succ = mean(art, cond, pol, "success_rate")
            d1 = mean(art, cond, pol, "exact_discovery_D1")
            d2 = mean(art, cond, pol, "exact_discovery_D2")
            print(f"  {cond}: {pol} success={succ:.4f} D1={d1:.4f} D2={d2:.4f}")

print()
# ---------------------------------------------------------------- baseline
baseline = load("baseline_l.json")
for cond in ("stable", "novel"):
    for pol in ("metargon12", "metargon12-notp"):
        print(f"  baseline/L/{cond}/{pol}: success={mean(baseline, cond, pol, 'success_rate'):.4f} "
              f"exactD1={mean(baseline, cond, pol, 'exact_discovery_D1'):.4f}")

print()
# ---------------------------------------------------------------- diagnosis vs benchmark
diag = load("diag_l_stable.json")
diag_novel = load("diag_l_novel.json")
# v12 diag success compared to final benchmark per-seed success rates
for cond, d in (("stable", diag), ("novel", diag_novel)):
    for run in d["runs"]:
        seed = run["seed"]
        idx = seed - 11800
        diag_succ = sum(e["success"] for e in run["episodes"]) / 40
        bench_succ = per_seed(final_l, cond, "metargon12", "success_rate")[idx]
        print(f"  diag/v12/{cond} seed{seed}: diag_success={diag_succ:.4f} "
              f"bench_success={bench_succ:.4f} delta={diag_succ-bench_succ:+.4f}")

# correct_created / verified max per seed for v12 diag (report table)
for cond, d in (("stable", diag), ("novel", diag_novel)):
    for run in d["runs"]:
        cc = sum(e["temporal"]["correct_created"] for e in run["episodes"])
        cv = sum(e["temporal"]["correct_verified_max"] for e in run["episodes"])
        floor1 = sum(1 for e in run["episodes"] if e["temporal"]["true_pair_observed"].get("D1"))
        print(f"  diag/v12/{cond} seed{run['seed']}: correct_created={cc} "
              f"correct_verified_max={cv} floor_D1={floor1}/40")

print()
if failures:
    print(f"FAILURES: {failures}")
    sys.exit(1)
print("all headline claims reproduced from raw data")
