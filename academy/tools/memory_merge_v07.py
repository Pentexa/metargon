"""Merge per-variant JSONL records into the final memory_v07 summary.

Sign convention: candidate - baseline (positive = candidate better),
matching the project-wide convention and the documented
`paired_difference(a, b) = mean(a - b)`.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from academy.tools.memory_experiment_v06 import compare_delta
from academy.tools.memory_merge_v06 import compute_control

ROOT = os.getcwd()
MDIR = os.path.join(ROOT, "results", "academy_v01", "memory_v07")
APPEND = os.path.join(MDIR, "memory_v07_runs.jsonl")
OUT = os.path.join(MDIR, "memory_v07_summary.json")

records = {}
for line in open(APPEND, encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        records[r["variant"]] = r["per"]

summary = {
    "run_id": "memory-v07",
    "seed": 20261107,
    "episodes": 160,
    "variants": list(records),
    "hash_randomization": False,
    "comparison_convention": (
        "candidate - baseline (positive = candidate better)"),
    "conditions_detail": records,
    "comparisons": {},
}
if "original" in records:
    orig = records["original"]
    for name, per in records.items():
        if name == "original":
            continue
        cmp_ = {}
        for mode in ("classic_frozen", "classic_adapt"):
            a = per[f"{mode}_per_config"]
            b = orig[f"{mode}_per_config"]
            if a and b and len(a) == len(b):
                cmp_[mode] = compare_delta(a, b, seed=20261108)
        cmp_["train_delta"] = per["train_success_mean"] - orig["train_success_mean"]
        summary["comparisons"][f"{name}_vs_original"] = cmp_

# fresh control (cached by the experiment runner)
FRESH_CACHE = os.path.join(MDIR, "fresh_control.json")
if os.path.exists(FRESH_CACHE):
    with open(FRESH_CACHE, "r", encoding="utf-8") as f:
        fresh = json.load(f)
    summary["fresh_control"] = fresh
    fresh_vec = fresh.get("classic_frozen_per_config")
    for name, per in records.items():
        if name == "fresh":
            continue
        a = per["classic_frozen_per_config"]
        if a and fresh_vec and len(a) == len(fresh_vec):
            summary["comparisons"][f"{name}_vs_fresh"] = {
                "classic_frozen": compare_delta(a, fresh_vec, seed=20261108),
            }

# shuffled-memory negative control: the same training stream with broken
# antecedent->consequence associations. Fresh delta vs shuffled isolates
# the memory-attributable component directly.
if "shuffled" in records:
    shuff = records["shuffled"]
    shuff_vec = shuff["classic_frozen_per_config"]
    for name, per in records.items():
        if name in ("fresh", "shuffled"):
            continue
        a = per["classic_frozen_per_config"]
        if a and shuff_vec and len(a) == len(shuff_vec):
            summary["comparisons"][f"{name}_vs_shuffled"] = {
                "classic_frozen": compare_delta(a, shuff_vec, seed=20261108),
            }

# exploration-only controls (no memory, no probes), same frozen grid
for _kind in ("random_walk", "novocab"):
    try:
        c = compute_control(_kind)
        summary[f"{_kind}_control"] = c["summary"]
        cvec = c["per_config_list"]
        fresh_vec = summary.get("fresh_control", {}).get(
            "classic_frozen_per_config")
        if fresh_vec and cvec and len(fresh_vec) == len(cvec):
            summary["comparisons"][f"{_kind}_vs_fresh"] = {
                "classic_frozen": compare_delta(cvec, fresh_vec, seed=20261108),
            }
        for name, per in records.items():
            if name == "fresh":
                continue
            a = per["classic_frozen_per_config"]
            if a and cvec and len(a) == len(cvec):
                summary["comparisons"][f"{name}_vs_{_kind}"] = {
                    "classic_frozen": compare_delta(a, cvec, seed=20261108),
                }
    except Exception as e:
        summary[f"{_kind}_control"] = {"error": str(e)}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1, default=str)
print(f"wrote {OUT}")
for name, per in records.items():
    print(f"  {name:12s} train={per['train_success_mean']:.3f} "
          f"frozen={per['classic_frozen']['config_mean_success']:.3f} "
          f"adapt={per['classic_adapt']['config_mean_success']:.3f} "
          f"ood={per['ood_frozen']['config_mean_success']:.3f} "
          f"cont={per['continual_ood']['config_mean_success']:.3f} "
          f"ret={per['retention_delta_mean']:+.3f} "
          f"attr={ {k: round(v,3) if isinstance(v,float) else v for k,v in per['attribution_mean'].items()} }")
print("comparisons (candidate - baseline):")
for k, v in summary["comparisons"].items():
    print(" ", k)
    for mode, d in v.items():
        if isinstance(d, dict) and "mean" in d:
            print(f"    {mode}: {d['mean']:+.4f} CI={tuple(round(x,4) for x in d['ci'])}")
        else:
            print(f"    {mode}: {d}")
