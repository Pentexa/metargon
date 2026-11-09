"""Merge per-variant JSONL records into the final memory_v05 summary."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from academy.eval.metrics import paired_difference

ROOT = os.getcwd()
MDIR = os.path.join(ROOT, "results", "academy_v01", "memory_v05")
APPEND = os.path.join(MDIR, "memory_v05_runs.jsonl")
OUT = os.path.join(MDIR, "memory_v05_summary.json")

records = {}
for line in open(APPEND, encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        records[r["variant"]] = r["per"]

summary = {
    "run_id": "memory-v05",
    "seed": 20261107,
    "episodes": 160,
    "variants": list(records),
    "hash_randomization": False,
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
            a = orig[f"{mode}_per_config"]
            b = per[f"{mode}_per_config"]
            if a and b and len(a) == len(b):
                cmp_[mode] = paired_difference(a, b, seed=20261108)
        cmp_["train_delta"] = per["train_success_mean"] - orig["train_success_mean"]
        summary["comparisons"][f"{name}_vs_original"] = cmp_

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1, default=str)
print(f"wrote {OUT}")
for name, per in records.items():
    print(f"  {name:13s} train={per['train_success_mean']:.3f} "
          f"frozen={per['classic_frozen']['config_mean_success']:.3f} "
          f"adapt={per['classic_adapt']['config_mean_success']:.3f} "
          f"ood={per['ood_frozen']['config_mean_success']:.3f} "
          f"cont={per['continual_ood']['config_mean_success']:.3f} "
          f"ret={per['retention_delta_mean']:+.3f} "
          f"mem={ {k: round(v,1) for k,v in per['memory_mean'].items()} }")
print("comparisons:")
for k, v in summary["comparisons"].items():
    print(" ", k)
    for mode, d in v.items():
        if isinstance(d, dict) and "mean" in d:
            print(f"    {mode}: {d['mean']:+.4f} CI={tuple(round(x,4) for x in d['ci'])}")
        else:
            print(f"    {mode}: {d}")
