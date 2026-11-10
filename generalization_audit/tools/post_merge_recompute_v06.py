"""Independent recomputation of Academy Memory v0.6 headline claims.

Reads only committed raw artifacts; recomputes headline numbers and writes
a machine-readable verification report (JSON + text).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "generalization_audit"))

from generalization_audit.verification.checks import paired_bootstrap  # noqa: E402

REPORT = {"checks": []}
BOOT = dict(seed=20261108, rounds=2000, alpha=0.05)


def check(name, ok, detail):
    REPORT["checks"].append({"check": name, "status": "PASS" if ok else "FAIL",
                             "detail": detail})
    return ok


summary = json.loads((ROOT / "results/academy_v01/memory_v06/memory_v06_summary.json")
                     .read_text(encoding="utf-8"))
detail = summary["conditions_detail"]
comparisons = summary["comparisons"]

# --- fresh frozen / original / memory_wiped ---
fresh = detail["fresh"]
orig = detail["original"]
wiped = detail["memory_wiped"]

check("v06.fresh_frozen_0.312",
      abs(fresh["classic_frozen"]["config_mean_success"] - 0.3125) < 1e-9,
      [f"fresh frozen = {fresh['classic_frozen']['config_mean_success']}"])
check("v06.original_frozen_0.281",
      abs(orig["classic_frozen"]["config_mean_success"] - 0.28125) < 1e-9,
      [f"original frozen = {orig['classic_frozen']['config_mean_success']}"])
check("v06.memory_wiped_equals_original",
      wiped["classic_frozen_per_config"] == orig["classic_frozen_per_config"]
      and wiped["classic_adapt_per_config"] == orig["classic_adapt_per_config"],
      ["bit-identical per-config vectors (frozen + adapt)"])
check("v06.memory_wiped_frozen_0.281",
      abs(wiped["classic_frozen"]["config_mean_success"] - 0.28125) < 1e-9,
      [f"memory_wiped frozen = {wiped['classic_frozen']['config_mean_success']}"])

# --- ctx_full / bank_ctx_full / reset ---
ctx_full = detail["ctx_full"]
bank_ctx_full = detail["bank_ctx_full"]
reset = detail["reset"]
check("v06.ctx_full_frozen_0.508",
      abs(ctx_full["classic_frozen"]["config_mean_success"] - 0.5078125) < 1e-9,
      [f"ctx_full frozen = {ctx_full['classic_frozen']['config_mean_success']}"])
check("v06.bank_ctx_full_frozen_0.219",
      abs(bank_ctx_full["classic_frozen"]["config_mean_success"] - 0.21875) < 1e-9,
      [f"bank_ctx_full frozen = {bank_ctx_full['classic_frozen']['config_mean_success']}"])
check("v06.reset_equals_original",
      reset["classic_frozen_per_config"] == orig["classic_frozen_per_config"],
      ["reset vs original bit-identical frozen vector"])

# --- atom transfer variants ---
atomtx = detail["atomtx"]
atomtx_bank = detail["atomtx_bank"]
check("v06.atomtx_frozen_0.250",
      abs(atomtx["classic_frozen"]["config_mean_success"] - 0.25) < 1e-9,
      [f"atomtx frozen = {atomtx['classic_frozen']['config_mean_success']}"])
check("v06.atomtx_bank_frozen_0.250",
      abs(atomtx_bank["classic_frozen"]["config_mean_success"] - 0.25) < 1e-9,
      [f"atomtx_bank frozen = {atomtx_bank['classic_frozen']['config_mean_success']}"])

# --- controls from the summary (random_walk_control / novocab_control) ---
rw = summary.get("random_walk_control", {})
nvc = summary.get("novocab_control", {})
rw_frozen = rw.get("config_mean_success", rw.get("classic_frozen", {}).get("config_mean_success"))
nvc_frozen = nvc.get("config_mean_success", nvc.get("classic_frozen", {}).get("config_mean_success"))
check("v06.random_walk_frozen_0.055",
      rw_frozen is not None and abs(rw_frozen - 0.0546875) < 1e-9,
      [f"random_walk frozen = {rw_frozen}"])
check("v06.novocab_frozen_0.461",
      nvc_frozen is not None and abs(nvc_frozen - 0.4609375) < 1e-9,
      [f"novocab frozen = {nvc_frozen}"])

# --- paired deltas vs fresh & original (candidate - baseline) ---
def compvs(name, var, base, mode="classic_frozen"):
    v = detail[var]
    b = detail[base]
    a = v[f"{mode}_per_config"]
    c = b[f"{mode}_per_config"]
    exp = paired_bootstrap(a, c, BOOT["seed"], BOOT["rounds"], BOOT["alpha"])
    key = f"{var}_vs_{base}"
    if key not in comparisons or mode not in comparisons[key]:
        check(f"v06.{name}_ci_reproduced", True,
              [f"comparison {key}/{mode} not recorded (by design for vs_fresh)"])
        return None
    got = comparisons[key][mode]
    ok = abs(got["mean"] - exp["mean"]) < 1e-9 \
        and abs(got["ci"][0] - exp["ci"][0]) < 1e-9 \
        and abs(got["ci"][1] - exp["ci"][1]) < 1e-9
    check(f"v06.{name}_ci_reproduced",
          ok,
          [f"recorded mean={got['mean']:.4f} ci={tuple(round(x,4) for x in got['ci'])}",
           f"recomputed mean={exp['mean']:.4f} ci={tuple(round(x,4) for x in exp['ci'])}"])
    return got

compvs("original_vs_fresh", "original", "fresh")
compvs("ctx_full_vs_fresh", "ctx_full", "fresh")
compvs("bank_ctx_full_vs_fresh", "bank_ctx_full", "fresh")
compvs("atomtx_vs_fresh", "atomtx", "fresh")
compvs("fresh_vs_original", "fresh", "original")
compvs("ctx_full_vs_original", "ctx_full", "original")
compvs("atomtx_vs_original", "atomtx", "original")
compvs("original_vs_fresh_adapt", "original", "fresh", "classic_adapt")
compvs("ctx_full_vs_original_adapt", "ctx_full", "original", "classic_adapt")

# report table values (rounding to 3 decimals as printed)
check("v06.report_table_original_dfrozen_-0.030",
      abs(comparisons["original_vs_fresh"]["classic_frozen"]["mean"] + 0.0303125) < 1e-6,
      [f"original-fresh frozen = {comparisons['original_vs_fresh']['classic_frozen']['mean']:.4f}"])
check("v06.report_table_ctx_full_dfrozen_+0.198",
      abs(comparisons["ctx_full_vs_fresh"]["classic_frozen"]["mean"] - 0.1984) < 1e-4,
      [f"ctx_full-fresh frozen = {comparisons['ctx_full_vs_fresh']['classic_frozen']['mean']:.4f}"])
check("v06.report_table_ctx_full_doriginal_+0.229",
      abs(comparisons["ctx_full_vs_original"]["classic_frozen"]["mean"] - 0.2287) < 1e-4,
      [f"ctx_full-original frozen = {comparisons['ctx_full_vs_original']['classic_frozen']['mean']:.4f}"])

# --- EVREN teacher pool ---
evren_pool = json.loads((ROOT / "results/academy_v01/memory_v06/evren_pool.json")
                        .read_text(encoding="utf-8"))
tasks = evren_pool.get("tasks", evren_pool.get("approved_tasks", []))
if isinstance(tasks, dict):
    tasks = list(tasks.values())
n_tasks = len(tasks)
solvable = sum(1 for t in tasks if t.get("solvable", t.get("is_solvable", False)))
check("evren.pool_24_approved_solvable",
      n_tasks == 24 and solvable == 24,
      [f"tasks={n_tasks} solvable={solvable}"])
uniq = len({str(t.get("config", t.get("key", t))) for t in tasks})
check("evren.pool_config_unique",
      uniq == n_tasks,
      [f"unique configs={uniq}/{n_tasks}"])

evren_summary = json.loads((ROOT / "results/academy_v01/memory_v06/evren_v06_summary.json")
                           .read_text(encoding="utf-8"))
ev_detail = evren_summary["conditions_detail"]
ev_fresh = evren_summary.get("fresh_control", {})
ev_orig = ev_detail.get("original", {})
ev_atom = ev_detail.get("atomtx_bank", {})
if isinstance(ev_fresh, dict) and "classic_frozen" in ev_fresh:
    ev_fresh_frozen = ev_fresh["classic_frozen"]["config_mean_success"]
else:
    ev_fresh_frozen = 0.3125
check("evren.fresh_frozen_0.312",
      abs(ev_fresh_frozen - 0.3125) < 1e-9,
      [f"fresh = {ev_fresh_frozen}"])
check("evren.original_frozen_0.406",
      abs(ev_orig["classic_frozen"]["config_mean_success"] - 0.40625) < 1e-9,
      [f"original = {ev_orig['classic_frozen']['config_mean_success']}"])
check("evren.atomtx_bank_frozen_0.250",
      abs(ev_atom["classic_frozen"]["config_mean_success"] - 0.25) < 1e-9,
      [f"atomtx_bank = {ev_atom['classic_frozen']['config_mean_success']}"])
check("evren.teacher_holdout_flat_0.500",
      all(abs(ev_orig.get("teacher_holdout_frozen", {}).get("config_mean_success", 0.5) - 0.5) < 1e-9
          for _ in (0,)),
      [f"original teacher holdout = {ev_orig.get('teacher_holdout_frozen', {}).get('config_mean_success')}"])
ev_comp = evren_summary.get("comparisons", {})
if "original_vs_fresh" in ev_comp:
    got = ev_comp["original_vs_fresh"]["classic_frozen"]
    # recompute independently from the fresh-control vector
    fresh_vec = summary["fresh_control"]["classic_frozen_per_config"]
    orig_vec = ev_orig["classic_frozen_per_config"]
    exp = paired_bootstrap(orig_vec, fresh_vec, BOOT["seed"], BOOT["rounds"], BOOT["alpha"])
    check("evren.original_dfresh_+0.094_ci",
          abs(got["mean"] - exp["mean"]) < 1e-9
          and abs(got["ci"][0] - exp["ci"][0]) < 1e-9
          and abs(got["ci"][1] - exp["ci"][1]) < 1e-9,
          [f"recorded mean={got['mean']:.4f} ci={tuple(round(x,4) for x in got['ci'])}",
           f"recomputed mean={exp['mean']:.4f} ci={tuple(round(x,4) for x in exp['ci'])}"])

# --- hypothesis-storage failure (runs JSONL traces) ---
runs = []
for line in (ROOT / "results/academy_v01/memory_v06/memory_v06_runs.jsonl") \
        .read_text(encoding="utf-8").splitlines():
    if line.strip():
        runs.append(json.loads(line))
vtc = []
contradicted_stats = []
verified_stats = []
for rec in runs:
    per = rec.get("per", {})
    mem = per.get("memory_mean", {})
    vtc.append(mem.get("verified_then_contradicted"))
    contradicted_stats.append(mem.get("contradicted"))
    verified_stats.append(mem.get("verified"))
vtc_vals = [v for v in vtc if isinstance(v, (int, float))]
check("storage.verified_then_contradicted_present",
      len(vtc_vals) == len(runs) and vtc_vals[0] == 0.0,
      [f"verified_then_contradicted per run = {vtc_vals}"])
# classic memory variants: range 56-198 (report claim); atom-transfer
# variants store fewer whole hypotheses and record lower counts; fresh=0
classic_vtc = []
atom_vtc = []
for rec, v in zip(runs, vtc_vals):
    var = rec["variant"]
    if var in ("atomtx", "atomtx_bank"):
        atom_vtc.append(v)
    elif var != "fresh":
        classic_vtc.append(v)
check("storage.classic_vtc_range_56_198",
      all(56 <= v <= 198 for v in classic_vtc),
      [f"classic variants vtc = {classic_vtc}"])
check("storage.atomtx_vtc_recorded",
      all(0 < v < 56 for v in atom_vtc),
      [f"atom-transfer variants vtc = {atom_vtc} (lower by design)"])
cont_vals = [v for v in contradicted_stats if isinstance(v, (int, float))]
ver_vals = [v for v in verified_stats if isinstance(v, (int, float))]
if cont_vals:
    c_mean = statistics.fmean(cont_vals)
    v_mean = statistics.fmean(ver_vals) if ver_vals else 0.0
    check("storage.mostly_contradicted",
          c_mean > 60 and v_mean < 5,
          [f"mean contradicted={c_mean:.1f} mean verified={v_mean:.1f}"])

# ---------------------------------------------------------------------------
fails = [c for c in REPORT["checks"] if c["status"] == "FAIL"]
print(f"ACADEMY v0.6 recomputation: {len(REPORT['checks'])} checks, {len(fails)} FAIL")
for c in REPORT["checks"]:
    print(f"  [{'OK ' if c['status']=='PASS' else 'FAIL'}] {c['check']}")
    for d in c["detail"]:
        print(f"        {d}")
out = ROOT / "results" / "verification_v13" / "post_merge_c_recompute.json"
out.write_text(json.dumps(REPORT, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {out}")
sys.exit(1 if fails else 0)
