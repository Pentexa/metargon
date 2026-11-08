"""Compute the fresh-agent control for the memory v0.3 run and inject
it into the merged summary (same grid settings as the variants)."""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from academy.evren.curriculum_experiment import eval_config_grid
from academy.envs.tasks import load_splits, DEFAULT_SPLITS_PATH

MDIR = os.path.join(os.getcwd(), "results", "academy_v01", "memory_v03")
SUMM = os.path.join(MDIR, "memory_v03_summary.json")
splits = load_splits(DEFAULT_SPLITS_PATH)
eval_seed = 20261108

fresh = eval_config_grid(
    None, splits["test"], "test", 80, 16, 4, (0, 1), eval_seed)
fresh_ood = eval_config_grid(
    None, splits["ood"], "ood", 80, 12, 3, (0,), eval_seed + 2)

summary = json.load(open(SUMM, encoding="utf-8"))
summary["fresh_control"] = {
    "classic_test": fresh["summary"],
    "classic_test_per_config": fresh["per_config_list"],
    "ood": fresh_ood["summary"],
}
with open(SUMM, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=1, default=str)
print("fresh classic:", fresh["summary"]["config_mean_success"])
print("fresh ood:", fresh_ood["summary"]["config_mean_success"])
print("injected into", SUMM)
