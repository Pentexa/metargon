"""Render the final memory v0.6 comparison tables from the merged summary.

Usage: python academy/tools/render_v06_results.py [summary.json]
Prints markdown tables used by docs/MEMORY_V06.md and PR_BODY_V06.md.
"""
import json
import sys
import os

sys.path.insert(0, os.getcwd())

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join("results", "academy_v01", "memory_v06",
                 "memory_v06_summary.json")

with open(PATH, encoding="utf-8") as f:
    s = json.load(f)

print("### Frozen / adapt / continual (mean over 2 seeds, 16-config grid)")
print("| Variant | Frozen | Adapt | OOD frozen | Continual OOD | Train |")
print("|---|---|---|---|---|---|")
for name in s["variants"]:
    d = s["conditions_detail"][name]
    print(f"| {name} | {d['classic_frozen']['config_mean_success']:.3f} "
          f"| {d['classic_adapt']['config_mean_success']:.3f} "
          f"| {d['ood_frozen']['config_mean_success']:.3f} "
          f"| {d['continual_ood']['config_mean_success']:.3f} "
          f"| {d['train_success_mean']:.3f} |")

rw = s.get("random_walk_control") or {}
if rw and "error" not in rw:
    print(f"\nRandom-walk control (no memory, no probes): frozen "
          f"{rw['config_mean_success']:.3f}")

print("\n### Paired frozen deltas vs fresh (candidate − fresh; positive = better)")
print("| Variant | Δ frozen | 95% CI |")
print("|---|---|---|")
for name in s["variants"]:
    key = f"{name}_vs_fresh"
    if key not in s["comparisons"]:
        continue
    d = s["comparisons"][key]["classic_frozen"]
    print(f"| {name} | {d['mean']:+.4f} | {tuple(round(x,4) for x in d['ci'])} |")

print("\n### Paired frozen deltas vs original (candidate − original)")
print("| Variant | Δ frozen | 95% CI | Δ adapt | 95% CI |")
print("|---|---|---|---|---|")
for name in s["variants"]:
    if name == "original":
        continue
    key = f"{name}_vs_original"
    if key not in s["comparisons"]:
        continue
    d = s["comparisons"][key]
    f_ = d.get("classic_frozen") or {}
    a_ = d.get("classic_adapt") or {}
    print(f"| {name} | {f_.get('mean',0):+.4f} | "
          f"{tuple(round(x,4) for x in f_.get('ci',(0,0)))} | "
          f"{a_.get('mean',0):+.4f} | "
          f"{tuple(round(x,4) for x in a_.get('ci',(0,0)))} |")

print("\n### Memory tracking (mean)")
print("| Variant | verified | contradicted | bank size | retrievals | atom profile |")
print("|---|---|---|---|---|---|")
for name in s["variants"]:
    m = s["conditions_detail"][name]["memory_mean"]
    print(f"| {name} | {m.get('verified',0):.1f} "
          f"| {m.get('contradicted',0):.1f} "
          f"| {m.get('bank_size',0):.1f} "
          f"| {m.get('bank_retrievals',0):.1f} "
          f"| {m.get('atom_profile_size',0):.1f} |")

print("\n### Diagnosis (Phase 1)")
print(json.dumps(s.get("diagnosis", {}), indent=1, default=str))
