"""Memory v0.4 — continual-learning benchmark (multi-seed, matched).

Measures forward transfer, forgetting/retention, OOD robustness, and
whether *retaining vs discarding* learned memory changes performance —
for the original student, the best v0.3 adaptive-probe variant, and
the new v0.4 context-aware variants.

Protocol:
  * train seeds: each seed defines ONE episode stream (matched across
    variants). Each variant trains on the identical stream for that
    seed.
  * During training, at every block boundary, the variant is evaluated
    on a small FROZEN grid (learning curves throughout training).
  * Final evaluation: full frozen grids (16 classic + OOD) + shared
    substrate retention probes.
  * ``memory_wiped`` control: trains exactly like ``original`` but its
    hypothesis memory is discarded before final evaluation (isolates
    whether retaining memory helps or hurts).
  * ``fresh`` control: untrained agent (reference baseline).

All offline; hash seed pinned and recorded.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.tasks import (
    DEFAULT_SPLITS_PATH, EpisodeSpec, difficulty_of, load_splits,
)
from academy.evren.curriculum_experiment import eval_config_grid
from academy.evren.retention import RetentionSuite, sample_substrate
from academy.learning.loop import run_one_episode
from academy.eval.metrics import paired_difference

VARIANTS = {
    "fresh": {},
    "original": {},
    "probe_local": {"probe_adaptive": True, "probe_episode_local": True,
                    "protect_episode_local": True},
    "ctx": {"memory_context_scoped": True},
    "ctx_reset": {"memory_context_scoped": True,
                  "memory_task_change_reset": True},
    "ctx_gate": {"memory_context_scoped": True,
                 "memory_context_gate_transfer": True},
    "ctx_full": {"memory_context_scoped": True,
                 "memory_task_change_reset": True,
                 "memory_context_gate_transfer": True,
                 "probe_adaptive": True, "probe_episode_local": True,
                 "protect_episode_local": True},
    "memory_wiped": {"_wipe_memory": True},
}


def make_stream(splits, episodes, seed, max_steps=80):
    rng = random.Random(seed)
    out = []
    for e in range(episodes):
        cfg = rng.choice(splits["train"])
        out.append(EpisodeSpec(
            config=cfg, max_steps=max_steps, noise_q=0.0,
            spec_seed=seed * 1000 + e, pool_tag="train",
            difficulty=difficulty_of(cfg)))
    return out


class WipeHookAgent(MetargonAcademyAgent):
    """Agent that wipes its hypothesis memory after training (used for
    the memory_wiped control)."""

    def __init__(self, *a, **k):
        self._wipe = bool(k.pop("_wipe_memory", False))
        super().__init__(*a, **k)

    def maybe_wipe(self):
        if self._wipe:
            self.memory.hypotheses = []
            self.memory._continuous_true = {}
            self.memory._dead = {}
            self.memory._sup_timeline = {}
            self.memory._con_timeline = {}


def run_v04_experiment(splits=None, episodes=160, block_size=40,
                       train_seeds=(20261107, 20261108), max_steps=80,
                       variants=None, results_dir=None, verbose=True,
                       append_path=None):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    results_dir = results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "results", "academy_v01", "memory_v04")
    os.makedirs(results_dir, exist_ok=True)
    variants = variants or list(VARIANTS)

    base = {
        "test_grid_n": 16, "eval_eps_per_cfg": 4,
        "ood_probe_n": 12, "ood_eps_per_seed": 3,
        "substrate_n": 8, "retention_eps_per_cfg": 2,
        "mid_grid_n": 8, "mid_eps_per_cfg": 2,
    }
    eval_seed = 20261108
    substrate = sample_substrate(splits["train"], base["substrate_n"],
                                 20261107 + 31)

    summary = {
        "run_id": "memory-v04",
        "episodes": episodes,
        "block_size": block_size,
        "train_seeds": list(train_seeds),
        "variants": variants,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
        "conditions_detail": {},
        "comparisons": {},
        "fresh_control": {},
    }

    # fresh control (same full grids)
    fresh = eval_config_grid(
        None, splits["test"], "test", max_steps,
        base["test_grid_n"], base["eval_eps_per_cfg"], (0, 1), eval_seed)
    fresh_ood = eval_config_grid(
        None, splits["ood"], "ood", max_steps,
        base["ood_probe_n"], base["ood_eps_per_seed"], (0,),
        eval_seed + 2)
    summary["fresh_control"] = {
        "classic_test": fresh["summary"],
        "classic_test_per_config": fresh["per_config_list"],
        "ood": fresh_ood["summary"],
    }
    if verbose:
        print(f"fresh: classic={fresh['summary']['config_mean_success']:.3f} "
              f"ood={fresh_ood['summary']['config_mean_success']:.3f}")

    # per-variant: average over seeds
    for name in variants:
        kwargs = {k: v for k, v in VARIANTS[name].items()
                  if not k.startswith("_")}
        per_seed = []
        mid_curves = []
        for ts in train_seeds:
            agent = WipeHookAgent(seed=0, **kwargs)
            stream = make_stream(splits, episodes, ts, max_steps)
            if name == "fresh":
                stream = []          # untrained control
            t0 = time.perf_counter()
            mid = []
            for ep_i, spec in enumerate(stream):
                run_one_episode(agent, spec)
                if (ep_i + 1) % block_size == 0:
                    mid_res = eval_config_grid(
                        agent.save_state(), splits["test"], "test",
                        max_steps, base["mid_grid_n"],
                        base["mid_eps_per_cfg"], (0,), eval_seed + ts % 100)
                    mid.append({
                        "episode": ep_i + 1,
                        "success": mid_res["summary"]["config_mean_success"],
                    })
            wall = time.perf_counter() - t0
            if getattr(agent, "_wipe", False):
                agent.maybe_wipe()
            state = None if name == "fresh" else agent.save_state()

            classic = eval_config_grid(
                state, splits["test"], "test", max_steps,
                base["test_grid_n"], base["eval_eps_per_cfg"], (0, 1),
                eval_seed)
            ood = eval_config_grid(
                state, splits["ood"], "ood", max_steps,
                base["ood_probe_n"], base["ood_eps_per_seed"], (0,),
                eval_seed + 2)
            retention = RetentionSuite(
                substrate=substrate, max_steps=max_steps,
                episodes_per_config=base["retention_eps_per_cfg"],
                seeds=(0,), label=name)
            retention.run_probe(state,
                                lambda s, _k=kwargs:
                                MetargonAcademyAgent(seed=s, **_k),
                                episode_i=0)
            retention.run_probe(state,
                                lambda s, _k=kwargs:
                                MetargonAcademyAgent(seed=s, **_k),
                                episode_i=1)

            mem = agent.memory
            status = {}
            for h in mem.hypotheses:
                status[h.status] = status.get(h.status, 0) + 1
            per_seed.append({
                "seed": ts,
                "train_success": agent.successes / max(1, agent.episodes),
                "wall_s": round(wall, 2),
                "classic_test": classic["summary"],
                "classic_test_per_config": classic["per_config_list"],
                "ood": ood["summary"],
                "retention": retention.summary(),
                "memory": {
                    "size": len(mem.hypotheses),
                    "status_counts": status,
                    "generated": mem.generated,
                    "evictions": mem.evictions,
                },
                "mid_curve": mid,
            })
            mid_curves.append(mid)

        # aggregate over seeds (mean per-config success across seeds)
        def mean_cfg(key):
            vecs = [p[key] for p in per_seed]
            n = min(len(v) for v in vecs)
            return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]
        agg = {
            "train_success_mean": (sum(p["train_success"] for p in per_seed)
                                   / len(per_seed)),
            "classic_test_per_config": mean_cfg("classic_test_per_config"),
            "classic_test": {
                "config_mean_success": (sum(
                    p["classic_test"]["config_mean_success"]
                    for p in per_seed) / len(per_seed)),
            },
            "ood": {
                "config_mean_success": (sum(
                    p["ood"]["config_mean_success"]
                    for p in per_seed) / len(per_seed)),
            },
            "retention_delta_mean": (sum(
                p["retention"]["retention_delta"] for p in per_seed)
                / len(per_seed)),
            "memory_status_mean": {},
            "mid_curve_mean": [
                {"episode": m["episode"],
                 "success": sum(mid[bi]["success"] for mid in mid_curves)
                            / len(mid_curves)}
                for bi, m in enumerate(mid_curves[0])
            ],
            "per_seed": per_seed,
        }
        for s in set(s for p in per_seed for s in p["memory"]["status_counts"]):
            agg["memory_status_mean"][s] = (sum(
                p["memory"]["status_counts"].get(s, 0) for p in per_seed)
                / len(per_seed))
        summary["conditions_detail"][name] = agg
        if append_path is not None:
            import io
            with io.open(append_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"variant": name, "per": agg},
                                   default=str) + "\n")
        if verbose:
            print(f"  {name:12s} train={agg['train_success_mean']:.3f} "
                  f"classic={agg['classic_test']['config_mean_success']:.3f} "
                  f"ood={agg['ood']['config_mean_success']:.3f} "
                  f"mem={ {k: round(v,1) for k,v in agg['memory_status_mean'].items()} }")

    # paired comparisons vs original (per-config vectors averaged over seeds)
    if "original" in summary["conditions_detail"]:
        orig = summary["conditions_detail"]["original"]
        for name in variants:
            if name == "original":
                continue
            v = summary["conditions_detail"][name]
            cmp_ = {}
            if len(orig["classic_test_per_config"]) == len(
                    v["classic_test_per_config"]):
                cmp_["classic_test"] = paired_difference(
                    orig["classic_test_per_config"],
                    v["classic_test_per_config"], seed=eval_seed)
            cmp_["train_delta"] = (v["train_success_mean"]
                                   - orig["train_success_mean"])
            summary["comparisons"][f"{name}_vs_original"] = cmp_

    summary["finished_at"] = time.time()
    with open(os.path.join(results_dir, "memory_v04_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print("  comparisons:")
        for k, v in summary["comparisons"].items():
            print("   ", k, v)
    return summary


if __name__ == "__main__":
    run_v04_experiment()
