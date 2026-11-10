"""Memory v0.6 — Phase 4: EVREN teacher experiences in the memory
transfer benchmark.

EVREN's teacher workflow (proposal → strict validation → approved task
registry) is used **only to generate validated training tasks** — never
answers, hidden rules, or evaluation configurations. External API usage
is optional and quota-limited (config default 10 calls/run); this
module runs **zero network calls** by default and reuses the
deterministic ``OfflineTaskGenerator`` (same validation pipeline, same
task shape) so the experiment is reproducible without any API key. If a
real teacher registry with approved tasks exists, its tasks can be used
instead via ``registry_dir``.

The experiment trains two memory variants on streams that mix classic
train-pool episodes with validated teacher-pool episodes (matched total
budget), then measures frozen zero-shot transfer on:

- the classic held-out test grid,
- held-out teacher tasks (never trained on),
- the OOD probe,
- within-episode adaptation on the classic grid.

Artifacts: ``results/academy_v01/memory_v06/evren_pool.json`` (approved
task configs + provenance) and ``evren_v06_summary.json``.
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
    make_episodes,
)
from academy.evren.offline_generator import OfflineTaskGenerator
from academy.evren.task_registry import TaskRegistry
from academy.evren.validator import TaskValidator
from academy.learning.loop import run_one_episode
from academy.tools.memory_experiment_v06 import (
    VARIANTS, compare_delta, eval_grid, memory_tracking,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    "results", "academy_v01", "memory_v06")

TEACHER_POOL_PATH = os.path.join(RESULTS_DIR, "evren_pool.json")


def build_teacher_pool(splits, n=24, seed=20261109, max_attempts=1200,
                       path=TEACHER_POOL_PATH):
    """Generate + strictly validate a reproducible teacher-pool artifact.

    Uses the offline generator (zero API calls); every task passes the
    same strict TaskValidator gate (schema, leakage, solvability,
    evaluation contamination, duplicates). Returns configs + record.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    validator = TaskValidator(splits=splits)
    gen = OfflineTaskGenerator(seed=seed, splits=splits, validator=validator)
    results = gen.generate(n, max_attempts=max_attempts,
                           source_note="memory_v06_evren")
    approved = [r for r in results if r.approved]
    pool = [r.config for r in approved]
    payload = {
        "source": "OfflineTaskGenerator (same validation pipeline as the "
                  "EVREN curriculum teacher; zero external API calls)",
        "seed": seed,
        "n_requested": n,
        "n_approved": len(approved),
        "max_attempts": max_attempts,
        "api_calls": 0,
        "quota_policy": "EVREN API calls capped at 10/run in config; "
                        "not used here",
        "tasks": [
            {
                "task_id": r.raw.get("task_id"),
                "proposal": r.raw,
                "config_key": str(r.config.key),
                "solvable": r.solvable,
            }
            for r in approved
        ],
        "rejected_statuses": {},
    }
    for r in results:
        if not r.approved:
            payload["rejected_statuses"][r.status] = \
                payload["rejected_statuses"].get(r.status, 0) + 1
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    os.replace(tmp, path)
    return pool, payload


def load_teacher_pool(path=TEACHER_POOL_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from academy.envs.tasks import _dict_to_config
    return [_dict_to_config(t["proposal"]) for t in data["tasks"]]


def make_mixed_stream(splits, teacher_pool, episodes, seed, teacher_share,
                      max_steps=80):
    """Matched-budget stream: `teacher_share` of episodes are validated
    teacher tasks, the rest classic train-pool episodes."""
    rng = random.Random(seed)
    out = []
    for e in range(episodes):
        if rng.random() < teacher_share and teacher_pool:
            cfg = rng.choice(teacher_pool)
            tag = "train_teacher"
        else:
            cfg = rng.choice(splits["train"])
            tag = "train"
        out.append(EpisodeSpec(
            config=cfg, max_steps=max_steps, noise_q=0.0,
            spec_seed=seed * 1000 + e, pool_tag=tag,
            difficulty=difficulty_of(cfg)))
    return out


def partition_teacher_pool(pool, seed, holdout=6):
    rng = random.Random(seed)
    items = sorted(pool, key=lambda c: str(c.key))
    rng.shuffle(items)
    if len(items) <= holdout:
        return [], items
    return items[holdout:], items[:holdout]


def run_v06_evren_experiment(splits=None, episodes=160,
                             train_seeds=(20261107, 20261108),
                             max_steps=80, teacher_share=0.30,
                             n_teacher=24, teacher_seed=20261109,
                             variants=("atomtx_bank", "original"),
                             eval_seed=20261108, verbose=True,
                             results_dir=RESULTS_DIR):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    os.makedirs(results_dir, exist_ok=True)

    teacher_pool, pool_payload = build_teacher_pool(
        splits, n=n_teacher, seed=teacher_seed)
    train_pool, holdout_pool = partition_teacher_pool(
        teacher_pool, teacher_seed + 1, holdout=6)

    base = {
        "test_grid_n": 16, "eval_eps_per_cfg": 4,
        "ood_probe_n": 12, "ood_eps_per_seed": 3,
        "teacher_holdout_n": min(len(holdout_pool), 6),
        "teacher_eps_per_cfg": 4,
    }
    summary = {
        "run_id": "memory-v06-evren",
        "episodes": episodes,
        "train_seeds": list(train_seeds),
        "teacher_share": teacher_share,
        "n_teacher_approved": len(teacher_pool),
        "n_teacher_train": len(train_pool),
        "n_teacher_holdout": len(holdout_pool),
        "api_calls": 0,
        "quota_policy": pool_payload["quota_policy"],
        "pool": pool_payload,
        "comparison_convention": (
            "candidate - baseline (positive = candidate better)"),
        "conditions_detail": {},
        "comparisons": {},
        "fresh_control": {},
        "finished_at": time.time(),
    }

    fresh_frozen = eval_grid(None, splits["test"], max_steps,
                             base["test_grid_n"], base["eval_eps_per_cfg"],
                             (0, 1), eval_seed, frozen=True)
    fresh_teacher = None
    if holdout_pool:
        fresh_teacher = eval_grid(None, holdout_pool, max_steps,
                                  base["teacher_holdout_n"],
                                  base["teacher_eps_per_cfg"],
                                  (0,), eval_seed + 3, frozen=True)
    summary["fresh_control"] = {
        "classic_frozen": fresh_frozen["summary"],
        "classic_frozen_per_config": fresh_frozen["per_config_list"],
        "teacher_holdout_frozen": (fresh_teacher or {}).get("summary"),
    }

    for name in variants:
        kwargs = {k: v for k, v in VARIANTS[name].items()
                  if not k.startswith("_")}
        per_seed = []
        for ts in train_seeds:
            agent = MetargonAcademyAgent(seed=0, **kwargs)
            stream = make_mixed_stream(splits, train_pool, episodes, ts,
                                       teacher_share, max_steps)
            t0 = time.perf_counter()
            for spec in stream:
                run_one_episode(agent, spec)
            wall = time.perf_counter() - t0
            state = agent.save_state()

            frozen = eval_grid(state, splits["test"], max_steps,
                               base["test_grid_n"], base["eval_eps_per_cfg"],
                               (0, 1), eval_seed, frozen=True,
                               agent_kwargs=kwargs)
            adapt = eval_grid(state, splits["test"], max_steps,
                              base["test_grid_n"], base["eval_eps_per_cfg"],
                              (0, 1), eval_seed, frozen=False,
                              agent_kwargs=kwargs)
            ood_frozen = eval_grid(state, splits["ood"], max_steps,
                                   base["ood_probe_n"],
                                   base["ood_eps_per_seed"], (0,),
                                   eval_seed + 2, frozen=True,
                                   agent_kwargs=kwargs)
            teacher_hold = None
            if holdout_pool:
                teacher_hold = eval_grid(state, holdout_pool, max_steps,
                                         base["teacher_holdout_n"],
                                         base["teacher_eps_per_cfg"],
                                         (0,), eval_seed + 3, frozen=True,
                                         agent_kwargs=kwargs)
            teacher_train_reuse = None
            if train_pool:
                teacher_train_reuse = eval_grid(
                    state, train_pool[:6], max_steps, 6, 2, (0,),
                    eval_seed + 4, frozen=True, agent_kwargs=kwargs)
            per_seed.append({
                "seed": ts,
                "train_success": agent.successes / max(1, agent.episodes),
                "wall_s": round(wall, 2),
                "classic_frozen": frozen["summary"],
                "classic_frozen_per_config": frozen["per_config_list"],
                "classic_adapt": adapt["summary"],
                "classic_adapt_per_config": adapt["per_config_list"],
                "ood_frozen": ood_frozen["summary"],
                "teacher_holdout_frozen": (teacher_hold or {}).get("summary"),
                "teacher_train_reuse_frozen":
                    (teacher_train_reuse or {}).get("summary"),
                "memory": memory_tracking(agent),
            })

        def mean_cfg(key):
            vecs = [p[key] for p in per_seed if p.get(key)]
            if not vecs:
                return []
            n = min(len(v) for v in vecs)
            return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]

        agg = {
            "train_success_mean": (sum(p["train_success"] for p in per_seed)
                                   / len(per_seed)),
            "classic_frozen_per_config": mean_cfg("classic_frozen_per_config"),
            "classic_frozen": {
                "config_mean_success": (sum(
                    p["classic_frozen"]["config_mean_success"]
                    for p in per_seed) / len(per_seed)),
            },
            "classic_adapt_per_config": mean_cfg("classic_adapt_per_config"),
            "classic_adapt": {
                "config_mean_success": (sum(
                    p["classic_adapt"]["config_mean_success"]
                    for p in per_seed) / len(per_seed)),
            },
            "ood_frozen": {
                "config_mean_success": (sum(
                    p["ood_frozen"]["config_mean_success"]
                    for p in per_seed) / len(per_seed)),
            },
            "teacher_holdout_frozen": {
                "config_mean_success": (sum(
                    p["teacher_holdout_frozen"]["config_mean_success"]
                    for p in per_seed if p.get("teacher_holdout_frozen"))
                    / len(per_seed)) if per_seed[0].get(
                        "teacher_holdout_frozen") else 0.0,
            },
            "teacher_train_reuse_frozen": {
                "config_mean_success": (sum(
                    p["teacher_train_reuse_frozen"]["config_mean_success"]
                    for p in per_seed if p.get("teacher_train_reuse_frozen"))
                    / len(per_seed)) if per_seed[0].get(
                        "teacher_train_reuse_frozen") else 0.0,
            },
            "memory_mean": {},
            "per_seed": per_seed,
        }
        keys = set()
        for p in per_seed:
            keys.update(p["memory"]["status_counts"].keys())
        for s in keys:
            agg["memory_mean"][s] = (sum(
                p["memory"]["status_counts"].get(s, 0) for p in per_seed)
                / len(per_seed))
        for k in ("bank_size", "bank_retrievals", "context_specific",
                  "atom_profile_size", "verified_then_contradicted"):
            vals = [p["memory"][k] for p in per_seed
                    if p["memory"].get(k) is not None]
            if vals:
                agg["memory_mean"][k] = sum(vals) / len(vals)
        summary["conditions_detail"][name] = agg
        if verbose:
            print(f"  {name:13s} train={agg['train_success_mean']:.3f} "
                  f"frozen={agg['classic_frozen']['config_mean_success']:.3f} "
                  f"adapt={agg['classic_adapt']['config_mean_success']:.3f} "
                  f"ood={agg['ood_frozen']['config_mean_success']:.3f} "
                  f"teacher_hold={agg['teacher_holdout_frozen']['config_mean_success']:.3f}")

    if "original" in summary["conditions_detail"]:
        orig = summary["conditions_detail"]["original"]
        for name in variants:
            if name == "original":
                continue
            v = summary["conditions_detail"][name]
            cmp_ = {}
            for mode in ("classic_frozen", "classic_adapt"):
                a = v[f"{mode}_per_config"]
                b = orig[f"{mode}_per_config"]
                if a and b and len(a) == len(b):
                    cmp_[mode] = compare_delta(a, b, seed=eval_seed)
            cmp_["train_delta"] = (v["train_success_mean"]
                                   - orig["train_success_mean"])
            summary["comparisons"][f"{name}_vs_original"] = cmp_
        fresh_vec = summary["fresh_control"]["classic_frozen_per_config"]
        for name in variants:
            if name == "fresh":
                continue
            v = summary["conditions_detail"][name]
            a = v["classic_frozen_per_config"]
            if a and fresh_vec and len(a) == len(fresh_vec):
                summary["comparisons"][f"{name}_vs_fresh"] = {
                    "classic_frozen": compare_delta(a, fresh_vec,
                                                    seed=eval_seed),
                }

    path = os.path.join(results_dir, "evren_v06_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print(f"wrote {path}")
    return summary


if __name__ == "__main__":
    if sys.flags.hash_randomization:
        raise SystemExit(
            "Run with a pinned hash seed for reproducibility:\n"
            "  PYTHONHASHSEED=0 python "
            "academy/tools/memory_experiment_v06_evren.py\n"
            "(PowerShell: $env:PYTHONHASHSEED=0; python ...)")
    run_v06_evren_experiment()
