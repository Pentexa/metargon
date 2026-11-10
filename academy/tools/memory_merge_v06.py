"""Merge per-variant JSONL records into the final memory_v06 summary.

Sign convention (v0.6 correction): all paired deltas are reported as
**candidate − baseline** (positive = candidate improved on baseline),
matching the documented `paired_difference(a, b) = mean(a − b)`
convention. v0.5 passed `(baseline, candidate)` which reported
baseline − candidate (reversed signs); the historical v0.5 raw
evidence is preserved untouched and the correction is documented in
`docs/MEMORY_V06.md`.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from academy.tools.memory_experiment_v06 import compare_delta

ROOT = os.getcwd()
MDIR = os.path.join(ROOT, "results", "academy_v01", "memory_v06")
APPEND = os.path.join(MDIR, "memory_v06_runs.jsonl")
OUT = os.path.join(MDIR, "memory_v06_summary.json")

records = {}
for line in open(APPEND, encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        records[r["variant"]] = r["per"]

summary = {
    "run_id": "memory-v06",
    "seed": 20261107,
    "episodes": 160,
    "variants": list(records),
    "hash_randomization": False,
    "comparison_convention": (
        "candidate - baseline (positive = candidate better); "
        "v0.5 defect corrected"),
    "conditions_detail": records,
    "comparisons": {},
    "fresh_control": {},
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

from academy.tools.memory_experiment_v06 import build_diagnosis  # noqa: E402

FRESH_CACHE = os.path.join(MDIR, "fresh_control.json")
if os.path.exists(FRESH_CACHE):
    with open(FRESH_CACHE, "r", encoding="utf-8") as f:
        fresh = json.load(f)
    fresh_vec = fresh["classic_frozen_per_config"]
    for name, per in records.items():
        if name == "fresh":
            continue
        a = per["classic_frozen_per_config"]
        if a and fresh_vec and len(a) == len(fresh_vec):
            summary["comparisons"][f"{name}_vs_fresh"] = {
                "classic_frozen": compare_delta(a, fresh_vec, seed=20261108),
            }
    summary["fresh_control"] = fresh

# --- exploration controls (no memory, no probes) --------------------------
# The fresh control is itself a specific (weak) explorer. Two non-memory
# controls separate exploration from memory effects:
#   * random_walk — pure random valid actions (no door/goal planning);
#   * novocab — full agent with vocabulary probing disabled but door/goal
#     planning intact (isolates the value of the probe vocabulary).
# Computed here (state=None => no training), mirroring the fresh grid.
def compute_control(kind):
    import sys as _sys
    _sys.path.insert(0, ROOT)
    from academy.agents.metargon_agent import MetargonAcademyAgent
    from academy.envs.tasks import load_splits, DEFAULT_SPLITS_PATH
    from academy.evren.retention import sample_substrate
    from academy.envs.tasks import make_episodes
    from academy.envs.gridworld import MechanismGrid

    class RandomWalkAgent(MetargonAcademyAgent):
        name = "random_walk"

        def _choose_probe(self):
            return None, None

    splits = load_splits(DEFAULT_SPLITS_PATH)
    seed = 20261108
    n_configs, eps_per_cfg, seeds = 16, 4, (0, 1)
    picks = sample_substrate(list(splits["test"]), n_configs, seed)
    rows = []
    for i, cfgc in enumerate(picks):
        for seed_i in seeds:
            base = seed * 100000 + i * 100 + seed_i
            stream = make_episodes([cfgc], "eval", base, eps_per_cfg,
                                   max_steps=80)
            if kind == "random_walk":
                agent = RandomWalkAgent(seed=base)
            else:
                agent = MetargonAcademyAgent(seed=base)
                agent.probe_vocab_disabled = True
            agent.frozen_policy = True
            succ = 0
            for spec in stream:
                env = MechanismGrid(spec.config, 80, 0.0)
                obs = env.reset(spec.spec_seed)
                agent.episode_start(spec, env.geom)
                ok = False
                for _ in range(80):
                    act = agent.act(obs)
                    before = obs
                    res = env.step(act)
                    agent.learn(before, act, res)
                    obs = res.observation
                    if res.terminated:
                        ok = True
                        break
                agent.finish_episode(ok)
                succ += 1 if ok else 0
            rows.append({"config": str(cfgc.key),
                         "success_rate": succ / eps_per_cfg})
    per_config = {}
    for r in rows:
        ent = per_config.setdefault(r["config"], {"succ": 0.0, "n": 0})
        ent["succ"] += r["success_rate"]
        ent["n"] += 1
    per_config = {k: v["succ"] / v["n"] for k, v in per_config.items()}
    vec = list(per_config.values())
    return {"summary": {"config_mean_success": sum(vec) / len(vec) if vec else 0.0,
                        "per_config": {k: round(v, 4) for k, v in per_config.items()}},
            "per_config_list": vec}

for _kind in ("random_walk", "novocab"):
    try:
        c = compute_control(_kind)
        summary[f"{_kind}_control"] = c["summary"]
        cvec = c["per_config_list"]
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
    except Exception as e:  # never block the merge on auxiliary controls
        summary[f"{_kind}_control"] = {"error": str(e)}

# Phase-1 diagnosis (rebuilt from per-variant traces + fresh control)
summary["diagnosis"] = build_diagnosis(summary)

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
print("comparisons (candidate - baseline; positive = candidate better):")
for k, v in summary["comparisons"].items():
    print(" ", k)
    for mode, d in v.items():
        if isinstance(d, dict) and "mean" in d:
            print(f"    {mode}: {d['mean']:+.4f} CI={tuple(round(x,4) for x in d['ci'])}")
        else:
            print(f"    {mode}: {d}")
