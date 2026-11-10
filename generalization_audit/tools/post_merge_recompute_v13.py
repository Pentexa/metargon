"""Independent recomputation of Research v0.13 headline claims.

Reads only committed raw artifacts; recomputes every headline number and
writes a machine-readable verification report (JSON + text).
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path.cwd()  # run from the repository root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "generalization_audit"))

from generalization_audit.verification.checks import harness_paired_bootstrap  # noqa: E402

REPORT = {"checks": [], "claims": []}


def check(name, ok, detail):
    REPORT["checks"].append({"check": name, "status": "PASS" if ok else "FAIL",
                             "detail": detail})
    return ok


def mean(vals):
    return statistics.fmean(vals)


# ---------------------------------------------------------------------------
# L family: temporal exact D1, per-seed, CIs
# ---------------------------------------------------------------------------
final_l = json.loads((ROOT / "results/v13/final_l.json").read_text(encoding="utf-8"))
pilot_l = json.loads((ROOT / "results/v13/pilot_l.json").read_text(encoding="utf-8"))
v12_l = json.loads((ROOT / "results/v12/summary_l_small.json").read_text(encoding="utf-8"))

for cond in ("stable", "novel"):
    rows = final_l["raw"][cond]
    d1_12 = [r["metargon12"]["exact_discovery_D1"] for r in rows]
    d1_13 = [r["metargon13"]["exact_discovery_D1"] for r in rows]
    d1_gen = [r["metargon13-gen"]["exact_discovery_D1"] for r in rows]
    d1_notp = [r["metargon13-notp"]["exact_discovery_D1"] for r in rows]
    d1_12notp = [r["metargon12-notp"]["exact_discovery_D1"] for r in rows]

    m12, m13, mgen = mean(d1_12), mean(d1_13), mean(d1_gen)
    delta13 = m13 - mean(d1_notp)
    delta12 = m12 - mean(d1_12notp)
    ci13 = harness_paired_bootstrap([a - b for a, b in zip(d1_13, d1_notp)], 4301, 1500)
    ci12 = harness_paired_bootstrap([a - b for a, b in zip(d1_12, d1_12notp)], 4301, 1500)

    check(f"L.{cond}.v12_exactD1_16.9_23.1",
          abs(m12 - (0.16875 if cond == "stable" else 0.23125)) < 1e-9,
          [f"recomputed={m12*100:.2f}% per-seed={d1_12}"])
    check(f"L.{cond}.v13_exactD1_39.4_33.1",
          abs(m13 - (0.39375 if cond == "stable" else 0.33125)) < 1e-9,
          [f"recomputed={m13*100:.2f}% per-seed={d1_13}"])
    check(f"L.{cond}.rolling_ablation_equals_v12",
          all(abs(g - v) < 1e-9 for g, v in zip(d1_gen, d1_12)),
          [f"gen per-seed={d1_gen} v12 per-seed={d1_12}"])
    check(f"L.{cond}.notp_zero",
          all(v == 0.0 for v in d1_notp + d1_12notp),
          [f"notp per-seed={d1_notp}"])
    check(f"L.{cond}.temporal_delta_v13_ci",
          abs(ci13[0] - (0.0 if cond == "stable" else 0.06875)) < 1e-9
          and abs(ci13[1] - (0.7875 if cond == "stable" else 0.6625)) < 1e-9,
          [f"delta={delta13*100:.2f}pp CI={tuple(round(x*100,2) for x in ci13)}"])
    check(f"L.{cond}.temporal_delta_v12_ci",
          abs(ci12[0] - 0.0) < 1e-9 and abs(ci12[1] - 0.50625) < 1e-9,
          [f"delta={delta12*100:.2f}pp CI={tuple(round(x*100,2) for x in ci12)}"])
    check(f"L.{cond}.v13_vs_v12_seed_carryover",
          all(d1_13[i] >= d1_12[i] - 1e-12 for i in range(4)),
          [f"v12={d1_12} v13={d1_13}"])

# pilot = final for the exact-D1 cells
for cond in ("stable", "novel"):
    p13 = mean([r["metargon13"]["exact_discovery_D1"] for r in pilot_l["raw"][cond]])
    f13 = mean([r["metargon13"]["exact_discovery_D1"] for r in final_l["raw"][cond]])
    check(f"L.{cond}.pilot_equals_final",
          abs(p13 - f13) < 1e-9, [f"pilot={p13} final={f13}"])

# historical baseline compatibility: metargon12 in final_l == v12 study summary
for cond in ("stable", "novel"):
    study = mean([r["metargon12"]["exact_discovery_D1"]
                  for r in v12_l["raw"][cond]])
    ref = mean([r["metargon12"]["exact_discovery_D1"]
                for r in final_l["raw"][cond]])
    succ_study = mean([r["metargon12"]["success_rate"] for r in v12_l["raw"][cond]])
    succ_ref = mean([r["metargon12"]["success_rate"] for r in final_l["raw"][cond]])
    check(f"L.{cond}.m12_reproduces_v12_study",
          abs(study - ref) < 1e-9,
          [f"exactD1: v12-study={study} final_l={ref}",
           f"success: v12-study={succ_study} final_l={succ_ref}"])

# ---------------------------------------------------------------------------
# F/H/R preservation
# ---------------------------------------------------------------------------
final_f = json.loads((ROOT / "results/v13/final_f.json").read_text(encoding="utf-8"))
final_h = json.loads((ROOT / "results/v13/final_h.json").read_text(encoding="utf-8"))
final_r = json.loads((ROOT / "results/v13/final_r.json").read_text(encoding="utf-8"))

F_EXPECT = {
    "stable": (0.85625, 0.85625, 0.85625, 0.85625),
    "shift": (0.8625, 0.90625, 0.9125, 0.9125),
    "noisy_obs": (0.44375, 0.80625, 0.8125, 0.8125),
    "novel": (0.9125, 0.91875, 0.9125, 0.9125),
    "tight_budget": (0.73125, 0.9, 0.90625, 0.90625),
}
for cond, (v10, v11, v121, v13) in F_EXPECT.items():
    row = final_f["summary"][cond]
    got13 = row["metargon13"]["success_rate"]
    got121 = row["metargon12"]["success_rate"]
    check(f"F.{cond}.v13_success",
          abs(got13 - v13) < 1e-9 and abs(got121 - v121) < 1e-9,
          [f"v13={got13*100:.2f}% v12.1={got121*100:.2f}% expected v13={v13*100:.2f}%"])

for cond in ("stable", "novel"):
    row = final_h["summary"][cond]
    v13 = row["metargon13"]["success_rate"]
    v121 = row["metargon12"]["success_rate"]
    expected = 0.73125 if cond == "stable" else 0.79375
    check(f"H.{cond}.preservation",
          abs(v13 - expected) < 1e-9 and abs(v13 - v121) < 1e-9,
          [f"v13={v13*100:.2f}% v12.1={v121*100:.2f}% expected={expected*100:.2f}%"])
    d2_13 = row["metargon13"]["exact_discovery_D2"]
    d2_121 = row["metargon12"]["exact_discovery_D2"]
    check(f"H.{cond}.d2_preservation",
          abs(d2_13 - d2_121) < 1e-9,
          [f"v13 D2={d2_13*100:.2f}% v12.1 D2={d2_121*100:.2f}%"])

for cond in ("stable", "novel"):
    row = final_r["summary"][cond]
    v13 = row["metargon13"]["success_rate"]
    v121 = row["metargon12"]["success_rate"]
    expected = 0.60625 if cond == "stable" else 0.79375
    check(f"R.{cond}.preservation",
          abs(v13 - expected) < 1e-9 and abs(v13 - v121) < 1e-9,
          [f"v13={v13*100:.2f}% v12.1={v121*100:.2f}% expected={expected*100:.2f}%"])

# R -nodc reproduction
nodc = final_r["summary"].get("stable", {}).get("metargon13-nodc", {})
if nodc:
    check("R.stable.nodc_reproduced",
          abs(nodc["success_rate"] - 0.71875) < 1e-9,
          [f"-nodc stable = {nodc['success_rate']*100:.2f}%"])

# regression artifacts
for name, path, cond, pol, expected in [
        ("regress_f_noisy", "results/v13/regress_f_noisy.json", "noisy_obs",
         "metargon13", 0.8125),
        ("regress_h_stable", "results/v13/regress_h_stable.json", "stable",
         "metargon13", 0.73125),
        ("regress_r_stable", "results/v13/regress_r_stable.json", "stable",
         "metargon13", 0.60625)]:
    d = json.loads((ROOT / path).read_text(encoding="utf-8"))
    got = d["summary"][cond][pol]["success_rate"]
    check(f"{name}.expected",
          abs(got - expected) < 1e-9,
          [f"{pol} = {got*100:.2f}% expected {expected*100:.2f}%"])

# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
fails = [c for c in REPORT["checks"] if c["status"] == "FAIL"]
print(f"RESEARCH v0.13 recomputation: {len(REPORT['checks'])} checks, "
      f"{len(fails)} FAIL")
for c in REPORT["checks"]:
    mark = "OK " if c["status"] == "PASS" else "FAIL"
    print(f"  [{mark}] {c['check']}")
    for d in c["detail"]:
        print(f"        {d}")

out = ROOT / "results" / "verification_v13" / "post_merge_b_recompute.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(REPORT, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {out}")
sys.exit(1 if fails else 0)
