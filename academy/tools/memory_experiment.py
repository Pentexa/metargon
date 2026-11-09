"""Memory v0.3 — matched-budget comparison of student-memory variants.

Conditions (each is an independent ablation of the v0.2 student):

    original  — the Academy student exactly as merged (control)
    recycle   — + contradiction-aware recycling (removes contradicted
                hypotheses, dead-antecedent set prevents churn)
    ratio     — + ratio-gated contradiction (con >= ratio * sup before
                a heavily-supported hypothesis may flip)
    window    — + evidence windowing (status uses recent evidence only)
    probe     — + adaptive probing (composite/priority candidates)
    combined  — recycle + ratio + window + probe

Protocol (mirrors the curriculum v0.2 controls):

  * All conditions train the SAME episode stream (matched budget, same
    seed) — 160 episodes from the classic training pool.
  * Evaluation uses frozen pools with independent per-config grids:
    classic test (12 configs), OOD probe, and a retention suite over a
    shared substrate. Fresh-agent scores provide a control reference.
  * Frozen eval seeds are identical across conditions; paired bootstrap
    CIs compare each variant against `original`.
  * Offline by construction; zero API calls; hash seed pinned and
    recorded.
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
    "original": {},
    "recycle": {"memory_recycle_contradicted": True},
    "ratio": {"memory_contradiction_ratio": 2.0},
    "window": {"memory_evidence_window": 24},
    "probe": {"probe_adaptive": True},
    "protect_local": {"protect_episode_local": True},
    "probe_local": {"probe_adaptive": True,
                    "probe_episode_local": True,
                    "protect_episode_local": True},
    "combined": {"memory_recycle_contradicted": True,
                 "memory_contradiction_ratio": 1.5,
                 "memory_evidence_window": 32,
                 "probe_adaptive": True,
                 "probe_episode_local": True,
                 "protect_episode_local": True},
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


def train_agent(agent, stream):
    for spec in stream:
        run_one_episode(agent, spec)
    return agent


def _run_variant(name, splits, stream, substrate, eval_seed, max_steps,
                 eval_cfg, results_dir):
    """Train + evaluate one variant; returns its detail dict."""
    kwargs = VARIANTS[name]
    agent = MetargonAcademyAgent(seed=0, **kwargs)
    t0 = time.perf_counter()
    train_agent(agent, stream)
    wall = time.perf_counter() - t0
    state = agent.save_state()

    classic = eval_config_grid(
        state, splits["test"], "test", max_steps,
        eval_cfg["test_grid_n"], eval_cfg["eval_eps_per_cfg"], (0, 1),
        eval_seed)
    ood = eval_config_grid(
        state, splits["ood"], "ood", max_steps,
        eval_cfg["ood_probe_n"], eval_cfg["ood_eps_per_seed"], (0,),
        eval_seed + 2)
    retention = RetentionSuite(
        substrate=substrate, max_steps=max_steps,
        episodes_per_config=eval_cfg["retention_eps_per_cfg"],
        seeds=(0,), label=name)
    retention.run_probe(agent.save_state(),
                        lambda s: MetargonAcademyAgent(seed=s, **kwargs),
                        episode_i=0)
    retention.run_probe(agent.save_state(),
                        lambda s: MetargonAcademyAgent(seed=s, **kwargs),
                        episode_i=1)

    mem = agent.memory
    status = {}
    for h in mem.hypotheses:
        status[h.status] = status.get(h.status, 0) + 1

    per = {
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
            "dead_entries": sum(len(dq) for dq in mem._dead.values()),
            "max_hypotheses": mem.max_hypotheses,
        },
    }
    with open(os.path.join(results_dir, f"policy_{name}.json"),
              "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, default=str)
    return per


def run_memory_experiment(splits=None, episodes=160, seed=20261107,
                          max_steps=80, variants=None, results_dir=None,
                          eval_cfg=None, verbose=True, append_path=None):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    results_dir = results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "results", "academy_v01", "memory_v03")
    os.makedirs(results_dir, exist_ok=True)
    eval_cfg = eval_cfg or {
        "test_grid_n": 16, "eval_eps_per_cfg": 4,
        "ood_probe_n": 12, "ood_eps_per_seed": 3,
        "substrate_n": 8, "retention_eps_per_cfg": 2,
    }
    variants = variants or list(VARIANTS)

    stream = make_stream(splits, episodes, seed, max_steps)
    substrate = sample_substrate(splits["train"], eval_cfg["substrate_n"],
                                 seed + 31)
    eval_seed = 20261108

    summary = {
        "run_id": "memory-v03",
        "seed": seed,
        "episodes": episodes,
        "variants": variants,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
        "conditions_detail": {},
        "comparisons": {},
        "fresh_control": {},
    }

    # fresh-agent control (no training) for reference
    fresh = eval_config_grid(
        None, splits["test"], "test", max_steps,
        eval_cfg["test_grid_n"], eval_cfg["eval_eps_per_cfg"], (0, 1),
        eval_seed)
    fresh_ood = eval_config_grid(
        None, splits["ood"], "ood", max_steps,
        eval_cfg["ood_probe_n"], eval_cfg["ood_eps_per_seed"], (0,),
        eval_seed + 2)
    summary["fresh_control"] = {
        "classic_test": fresh["summary"],
        "classic_test_per_config": fresh["per_config_list"],
        "ood": fresh_ood["summary"],
    }

    for name in variants:
        try:
            per = _run_variant(name, splits, stream, substrate, eval_seed,
                               max_steps, eval_cfg, results_dir)
        except Exception as exc:  # noqa: BLE001 — keep whole-run alive
            import traceback
            traceback.print_exc()
            per = {"error": f"{type(exc).__name__}: {exc}"}
        summary["conditions_detail"][name] = per
        if append_path is not None:
            import io
            with io.open(append_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"variant": name, "per": per},
                                   default=str) + "\n")
        if verbose:
            print(f"  {name}: done", flush=True)

    # paired comparisons vs original
    if "original" in summary["conditions_detail"]:
        orig = summary["conditions_detail"]["original"]
        for name in variants:
            if name == "original":
                continue
            v = summary["conditions_detail"][name]
            cmp_ = {}
            a, b = v["classic_test_per_config"], orig["classic_test_per_config"]
            if len(a) == len(b):
                # candidate − baseline (positive = variant improves)
                cmp_["classic_test"] = paired_difference(a, b, seed=eval_seed)
            if v["retention"]["snapshots"] and orig["retention"]["snapshots"]:
                cmp_["retention_delta"] = (
                    v["retention"]["retention_delta"]
                    - orig["retention"]["retention_delta"])
            summary["comparisons"][f"{name}_vs_original"] = cmp_

    summary["finished_at"] = time.time()
    with open(os.path.join(results_dir, "memory_v03_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print("  comparisons:")
        for k, v in summary["comparisons"].items():
            print("   ", k, v)
    return summary


if __name__ == "__main__":
    run_memory_experiment()
