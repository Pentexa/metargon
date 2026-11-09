"""Merge per-variant JSONL records into the final memory_v04 summary."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from academy.eval.metrics import paired_difference

ROOT = os.getcwd()
MDIR = os.path.join(ROOT, "results", "academy_v01", "memory_v04")
APPEND = os.path.join(MDIR, "memory_v04_runs.jsonl")
OUT = os.path.join(MDIR, "memory_v04_summary.json")

records = {}
for line in open(APPEND, encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        records[r["variant"]] = r["per"]

summary = {
    "run_id": "memory-v04",
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
        a, b = per["classic_test_per_config"], orig["classic_test_per_config"]
        if a and b and len(a) == len(b):
            # candidate − baseline (positive = candidate improves); see
            # audit v0.2 convention (generalization_audit/docs/AUDIT_V02.md)
            cmp_["classic_test"] = paired_difference(a, b, seed=20261108)
        cmp_["train_delta"] = per["train_success_mean"] - orig["train_success_mean"]
        summary["comparisons"][f"{name}_vs_original"] = cmp_

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1, default=str)
print(f"wrote {OUT}")
for name, per in records.items():
    print(f"  {name:12s} train={per['train_success_mean']:.3f} "
          f"classic={per['classic_test']['config_mean_success']:.3f} "
          f"ood={per['ood']['config_mean_success']:.3f} "
          f"mem={ {k: round(v,1) for k,v in per['memory_status_mean'].items()} } "
          f"ret={per['retention_delta_mean']:+.3f}")
print("comparisons:")
for k, v in summary["comparisons"].items():
    print(" ", k, v.get("classic_test"), "train_delta", v.get("train_delta"))
