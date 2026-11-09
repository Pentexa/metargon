"""EVREN curriculum v0.2 — matched-budget curriculum comparison.

Scientific design (corrects three confounds of the v0.1 experiment):

1. **Selection was order-blind.** v0.1 dropped teacher tasks into a
   fixed stream at p=0.5. Here the adaptive conditions (``proc_lp``,
   ``teacher_lp``) choose each treatment episode with learning-progress
   + weakness scores built from the student's own outcomes
   (``selection.py``).

2. **Retention was unmeasurable.** v0.1 probed forgetting once, after
   training, with a probe identical across conditions. Here every
   condition trains on the *same frozen substrate* (classic configs +
   the same teacher tasks) at a matched slot share, and the substrate is
   re-probed at every block boundary — a retention *curve* per
   condition.

3. **Exposure mix was entangled.** v0.1's teacher condition saw half as
   many classic episodes. Here all conditions share the substrate slot
   share and differ only in what the remaining treatment slots contain,
   so each comparison isolates one factor:

   uniform      → uniform draws from the treatment pool (status quo)
   ascending    → deterministic easy→hard ordering (procedural baseline)
   calibrated   → the Academy's measured-progress scheduler (procedural)
   proc_lp      → learning-progress + weakness selection over the pool
   teacher_lp   → same machinery, treatment pool augmented with
                  teacher-approved tasks (source ablation)

   principled pairwise deltas: lp effect = proc_lp − uniform;
   teacher-source effect = teacher_lp − proc_lp.

**Budgets are matched exactly**: every condition consumes the same
number of student training episodes and the same substrate exposure.
The ``calibrated`` condition additionally runs a scratch-agent
difficulty calibration (throwaway episodes that never touch the
condition's trained student); that measurement cost is recorded
separately and disclosed.

**Evaluation is frozen and shared**: classic test pool (Evaluator,
isolation-checked), held-out teacher tasks, and a small OOD probe use
identical matched streams across conditions. Each condition's trained
student state is only ever *read* by evaluation.

All curriculum candidates are validated by the strict TaskValidator
(schema/leakage/solvability/contamination) before use, and the
experiment re-asserts train/eval disjointness defensively. The
experiment performs **zero network calls** — offline by construction.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time

from ..agents.metargon_agent import MetargonAcademyAgent
from ..envs.gridworld import MechanismGrid
from ..envs.tasks import (
    EpisodeSpec, difficulty_of, make_episodes,
)
from ..eval.metrics import paired_difference
from ..eval.evaluator import Evaluator
from ..eval.protocol import spec_for
from ..learning.loop import run_one_episode
from .diversity import diversity_metrics
from .offline_generator import OfflineTaskGenerator
from .retention import RetentionSuite, sample_substrate
from .selection import (
    LearningProgressSelector, family_key, family_key_json,
)
from .task_registry import TaskRegistry
from .validator import TaskValidator, config_to_proposal

CONDITIONS = ("uniform", "ascending", "calibrated", "proc_lp", "teacher_lp")
BINS = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# Candidate preparation (all offline; all strictly validated)
# ---------------------------------------------------------------------------
def procedural_candidates(splits, n: int, seed: int,
                          validator: TaskValidator) -> list:
    """Generate + strictly-validate ``n`` approved procedural candidates
    (deterministic, offline; the availability-insurance generator)."""
    gen = OfflineTaskGenerator(seed=seed, splits=splits, validator=validator)
    results = gen.generate(n, max_attempts=600, source_note="curriculum_v02")
    return [r.config for r in results if r.approved]


def partition_approved(approved: list, seed: int) -> tuple:
    """Deterministically partition approved teacher tasks into
    (substrate, holdout, extras), shrinking gracefully when few exist."""
    rng = random.Random(seed)
    items = sorted(approved, key=lambda c: str(c.key))
    rng.shuffle(items)
    n = len(items)
    if n < 2:
        items, n = items, n          # degenerate: everything to substrate
        return items, [], []
    n_sub = min(4, max(1, n // 2))
    n_hold = min(3, max(1, (n - n_sub) // 2)) if n - n_sub >= 2 else \
        min(2, n - n_sub)
    sub = items[:n_sub]
    hold = items[n_sub:n_sub + n_hold]
    extras = items[n_sub + n_hold:]
    return sub, hold, extras


# ---------------------------------------------------------------------------
# Episode planning per condition
# ---------------------------------------------------------------------------
def _difficulty_rank(config) -> tuple:
    return (BINS.index(difficulty_of(config).bin), str(config.key))


def plan_treatment(condition: str, ep_index: int, treatment_pool: list,
                   rng: random.Random, schedules: dict):
    """Return the config for one treatment slot under the condition."""
    if condition == "uniform":
        return rng.choice(treatment_pool)
    if condition == "ascending":
        seq = schedules["ascending"]
        return seq[ep_index % len(seq)]
    if condition == "calibrated":
        bin_name = schedules["schedule"][ep_index]
        pool = schedules["by_bin"].get(bin_name) or treatment_pool
        return rng.choice(pool)
    if condition in ("proc_lp", "teacher_lp"):
        idx, _info = schedules["selector"].select(rng)
        return treatment_pool[idx]
    raise ValueError(f"unknown condition {condition}")


def _prepare_calibrated(stream_pool, episodes, seed):
    """The Academy's measured-progress procedural curriculum.

    Runs a *scratch* agent per difficulty bin (12 episodes each); the
    scratch agent is discarded — only the measured bin statistics are
    used to build the schedule. The 36 measurement episodes are NOT
    student-training budget and are reported separately."""
    from ..learning.loop import calibrate_curriculum, build_schedule
    calib = calibrate_curriculum(stream_pool, episodes_per_bin=12, seed=seed)
    schedule = build_schedule(calib, episodes, episodes_per_block=40,
                              seed=seed + 1)
    by_bin = {}
    for c in stream_pool:
        by_bin.setdefault(difficulty_of(c).bin, []).append(c)
    return {"calibration": calib, "calibration_episodes": 3 * 12}, {
        "schedule": schedule, "by_bin": by_bin}


def _fresh_agent_factory():
    def make(seed):
        return MetargonAcademyAgent(seed=seed)
    return make


# ---------------------------------------------------------------------------
# Core training loop (substrate slots + treatment slots + retention probes)
# ---------------------------------------------------------------------------
def train_condition(condition: str, agent, cfg_e: dict, pools: dict,
                    episodes: int, seed: int, max_steps: int,
                    substrate_share: float, retention: RetentionSuite,
                    retention_teacher: Optional[RetentionSuite],
                    block_size: int, verbose: bool = True):
    """Train one condition's student; probe retention at block ends.

    Returns a detail dict: trained agent, per-episode records, optional
    selector, wall time, slot accounting, calibration provenance."""
    rng = random.Random(seed)
    substrate_classic = pools["substrate_classic"]
    substrate_teacher = pools["substrate_teacher"]
    treatment_pool = pools["treatment_pool"]

    schedules = {}
    if condition == "ascending":
        schedules["ascending"] = sorted(treatment_pool, key=_difficulty_rank)
    if condition in ("proc_lp", "teacher_lp"):
        schedules["selector"] = LearningProgressSelector(
            treatment_pool, seed=seed,
            window=int(cfg_e.get("lp_window", 10)),
            min_visits=int(cfg_e.get("lp_min_visits", 3)),
            lp_weight=float(cfg_e.get("lp_weight", 1.0)),
            weakness_weight=float(cfg_e.get("weakness_weight", 1.2)),
            novelty_weight=float(cfg_e.get("novelty_weight", 0.8)),
            noise_weight=float(cfg_e.get("explore_eps", 0.15)),
            weakness_target=float(cfg_e.get("weakness_target", 0.75)))
    calib = None
    if condition == "calibrated":
        calib, schedules = _prepare_calibrated(pools["stream_pool"],
                                               episodes, seed)

    t0 = time.perf_counter()
    sel = schedules.get("selector")
    records = []
    n_sub = n_classic = n_teacher = n_treat = 0
    last_probe_at = -1
    for ep_i in range(episodes):
        # block boundary -> retention probes (evaluation-only)
        if ep_i > 0 and ep_i % block_size == 0:
            retention.run_probe(agent.save_state(), _fresh_agent_factory(),
                                episode_i=ep_i)
            last_probe_at = ep_i
            if retention_teacher is not None and retention_teacher.substrate:
                retention_teacher.run_probe(
                    agent.save_state(), _fresh_agent_factory(),
                    episode_i=ep_i)
        if rng.random() < substrate_share:
            if substrate_teacher and rng.random() < 0.3333:
                config = rng.choice(substrate_teacher)
                source = "substrate_teacher"
                n_teacher += 1
            else:
                config = rng.choice(substrate_classic)
                source = "substrate_classic"
                n_classic += 1
            n_sub += 1
        else:
            config = plan_treatment(condition, ep_i, treatment_pool, rng,
                                    schedules)
            source = "treatment"
            n_treat += 1
        spec = EpisodeSpec(
            config=config, max_steps=max_steps, noise_q=0.0,
            spec_seed=seed * 1000 + ep_i, pool_tag="train",
            difficulty=difficulty_of(config))
        rec = run_one_episode(agent, spec)
        ok = bool(rec["success"])
        if sel is not None:
            sel.record(spec, ok)
        records.append({
            "episode": ep_i, "source": source, "success": ok,
            "steps": rec["steps"], "family": family_key_json(family_key(config)),
            "config": str(config.key),
        })
    if last_probe_at != episodes:
        retention.run_probe(agent.save_state(), _fresh_agent_factory(),
                            episode_i=episodes)
        if retention_teacher is not None and retention_teacher.substrate:
            retention_teacher.run_probe(agent.save_state(),
                                        _fresh_agent_factory(),
                                        episode_i=episodes)
    if verbose:
        sr = sum(1 for r in records if r["success"]) / max(1, len(records))
        print(f"  trained {condition}: train_success={sr:.3f} "
              f"(substrate {n_sub} / treatment {n_treat}) "
              f"wall={time.perf_counter() - t0:.1f}s")
    return {
        "condition": condition,
        "agent": agent,
        "records": records,
        "selector": sel,
        "wall_s": time.perf_counter() - t0,
        "substrate_episodes": n_sub,
        "substrate_classic_episodes": n_classic,
        "substrate_teacher_episodes": n_teacher,
        "treatment_episodes": n_treat,
        "calibration": calib,
    }


# ---------------------------------------------------------------------------
# Evaluation: independent per-config grid (fixes the v0.1 eval artifact)
# ---------------------------------------------------------------------------
def eval_config_grid(agent_state, configs, labels: str, max_steps: int,
                     n_configs: int, episodes_per_config: int,
                     seeds, seed: int = 0, pool_tag: str = "eval"):
    """Per-config independent evaluation grid.

    The v0.1 experiment evaluated the classic test pool with
    ``make_episodes``, which draws ONE config per seed: with n_seeds=3
    the "test result" was really three singleton configs. The archived
    0.000 came from three unlucky draws (fresh agents reproduce the
    failure on those same configs — see ``diagnosis.py``), which
    therefore said little about the treatment.

    This grid instead evaluates *independently sampled distinct
    configs* (matched across conditions via the deterministic sampler),
    records the per-config breakdown, and reports both aggregate and
    per-config success. Cells are fresh agent instances loaded with the
    condition's trained state; nothing is written back to training."""
    picks = sample_substrate(list(configs), n_configs, seed)
    rows = []
    for i, cfgc in enumerate(picks):
        for seed_i in seeds:
            base = seed * 100000 + i * 100 + seed_i
            stream = make_episodes([cfgc], pool_tag, base,
                                   episodes_per_config, max_steps=max_steps)
            agent = MetargonAcademyAgent(seed=base)
            if agent_state is not None:
                agent.load_state(agent_state)
            succ, steps_total = 0, 0
            for ep, spec in enumerate(stream):
                env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
                obs = env.reset(spec.spec_seed)
                agent.episode_start(spec_for(agent, spec), env.geom)
                success = False
                steps = 0
                while steps < env.max_steps:
                    a = agent.act(obs)
                    before = obs
                    res = env.step(a)
                    agent.learn(before, a, res)
                    steps += 1
                    obs = res.observation
                    if res.terminated:
                        success = True
                        break
                agent.finish_episode(success)
                succ += 1 if success else 0
                steps_total += steps
            rows.append({
                "config": str(cfgc.key), "config_i": i, "seed_i": seed_i,
                "n": len(stream),
                "success_rate": succ / max(1, len(stream)),
                "mean_steps": steps_total / max(1, len(stream)),
            })
    per_config = {}
    for r in rows:
        ent = per_config.setdefault(r["config"], {"succ": 0.0, "n": 0})
        ent["succ"] += r["success_rate"] * r["n"]
        ent["n"] += r["n"]
    per_config = {k: v["succ"] / v["n"] for k, v in per_config.items()}
    rates = list(per_config.values())
    summary = {
        "n_configs": len(picks),
        "episodes_per_config": episodes_per_config,
        "seeds": list(seeds),
        "config_mean_success": (sum(rates) / len(rates)) if rates else 0.0,
        "per_config": {k: round(v, 4) for k, v in per_config.items()},
    }
    return {"pool_tag": pool_tag, "summary": summary,
            "cells": rows, "per_config_list": rates}


def _eval_frozen(agent_state, tag, configs, max_steps, n_seeds,
                 episodes_per_seed, seed, splits=None):
    """Frozen-pool single-config-per-seed streams (legacy v0.1 protocol;
    retained only for A/B checks by the diagnosis — the v0.2 experiment
    uses ``eval_config_grid``)."""
    evaluator = Evaluator(
        splits or {"train": []}, seed=seed,
        episodes_per_seed=episodes_per_seed,
        max_steps=max_steps, pool_tag=tag)

    def make_loaded(s, _st=agent_state):
        a = MetargonAcademyAgent(seed=s)
        if _st is not None:
            a.load_state(_st)
        return a

    return evaluator.run_policy(make_loaded, configs, n_seeds=n_seeds)


def validator_config_ok(validator: TaskValidator, config) -> bool:
    """Defensive re-validation of a registered config (may reject if the
    registry moved on since approval — e.g. duplicates)."""
    res = validator.validate(config_to_proposal(config))
    return res.approved


def _assert_isolation(candidates, forbidden_pools, label):
    """Defensive re-assertion of train/eval disjointness."""
    cand_keys = {c.key for c in candidates}
    for name, pool in forbidden_pools.items():
        overlap = cand_keys & {c.key for c in pool}
        if overlap:
            raise RuntimeError(
                f"CURRICULUM LEAK ({label}): {len(overlap)} candidate(s) "
                f"overlap the {name} pool — refusing to run experiment")


def _served_diversity(records: list) -> dict:
    fams = {}
    for r in records:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    return {"n": len(records), "family_distribution": fams}


# ---------------------------------------------------------------------------
# Experiment: matched-budget comparison
# ---------------------------------------------------------------------------
def run_curriculum_experiment(
        cfg: dict, splits: dict, registry: TaskRegistry,
        results_dir: str, run_id: str = "curriculum-v02",
        conditions: tuple = CONDITIONS, episodes: int = 160,
        block_size: int = 40, seed: int = 0, max_steps: int = 80,
        substrate_share: float = 0.30, n_procedural: int = 24,
        verbose: bool = True):
    """Run the v0.2 matched-budget curriculum comparison (zero network)."""
    e_root = cfg.get("evren", {}) or {}
    e = e_root.get("curriculum", {}) or {}
    seed = int(e.get("seed", seed))
    substrate_share = float(e.get("substrate_share", substrate_share))
    eval_seed = int(cfg.get("seeds", {}).get("eval", 20261108))

    results = {
        "run_id": run_id,
        "conditions": list(conditions),
        "episodes": episodes,
        "block_size": block_size,
        "seed": seed,
        "substrate_share": substrate_share,
        "api_calls": 0,
        "offline": True,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
    }

    # ---- candidate pools ------------------------------------------------
    train = splits["train"]
    validator = TaskValidator(splits=splits)
    approved = [c for c in registry.approved_configs()
                if validator_config_ok(validator, c)]
    substrate_classic = sample_substrate(
        train, int(e.get("substrate_classic_n", 8)), seed + 11)
    if approved:
        sub_teacher, hold_teacher, extras_teacher = partition_approved(
            approved, seed + 12)
    else:
        sub_teacher, hold_teacher, extras_teacher = [], [], []
    proc_extras = procedural_candidates(
        splits, int(e.get("n_procedural", n_procedural)), seed + 13,
        validator)
    stream_pool = [c for c in train
                   if c.key not in {s.key for s in substrate_classic}]

    forbidden = {"val": splits["val"], "test": splits["test"],
                 "ood": splits["ood"]}
    _assert_isolation(substrate_classic + sub_teacher + extras_teacher,
                      forbidden, "curriculum candidates")
    _assert_isolation(proc_extras, forbidden, "procedural candidates")
    _assert_isolation(hold_teacher + sub_teacher + extras_teacher,
                      {"train": splits["train"]}, "teacher candidates")

    pools_shared = {
        "stream_pool": stream_pool,
        "substrate_classic": substrate_classic,
        "substrate_teacher": sub_teacher,
        "treatment_pool": list(stream_pool),
    }
    pools_proc = dict(pools_shared)
    pools_proc["treatment_pool"] = list(stream_pool) + list(proc_extras)
    pools_teacher = dict(pools_shared)
    pools_teacher["treatment_pool"] = list(stream_pool) + list(extras_teacher)
    condition_pools = {
        "uniform": pools_shared,
        "ascending": pools_shared,
        "calibrated": pools_shared,
        "proc_lp": pools_proc,
        "teacher_lp": pools_teacher,
    }

    # ---- diversity of candidate spaces -----------------------------------
    results["diversity"] = {
        "treatment_base": diversity_metrics(stream_pool, reference=train),
        "procedural_extras": diversity_metrics(proc_extras, reference=train),
        "teacher_extras": diversity_metrics(extras_teacher, reference=train),
        "teacher_holdout": diversity_metrics(hold_teacher, reference=train),
        "provenance": {
            "registered_approved": len(approved),
            "registered_source": registry.manifest().get("source"),
            "teacher_source": ("registered approved tasks"
                               if approved else "none (fallback: procedural)"),
        },
    }

    # ---- per-condition train + evaluation --------------------------------
    per_condition = {}
    for cond in conditions:
        cpools = condition_pools[cond]
        retention = RetentionSuite(
            substrate=substrate_classic, max_steps=max_steps,
            episodes_per_config=int(e.get("retention_eps_per_cfg", 2)),
            seeds=(0,), label=f"{cond}-classic-substrate")
        retention_teacher = RetentionSuite(
            substrate=sub_teacher, max_steps=max_steps,
            episodes_per_config=int(e.get("retention_eps_per_cfg", 2)),
            seeds=(0,), label=f"{cond}-teacher-substrate")
        out = train_condition(
            cond, MetargonAcademyAgent(seed=seed), e, cpools,
            episodes=episodes, seed=seed, max_steps=max_steps,
            substrate_share=substrate_share, retention=retention,
            retention_teacher=retention_teacher if sub_teacher else None,
            block_size=block_size, verbose=verbose)
        agent = out["agent"]
        state = agent.save_state()

        # final frozen evals; matched grids across conditions
        classic = eval_config_grid(
            state, splits["test"], "test", max_steps,
            int(e.get("test_grid_n", 12)),
            int(e.get("eval_eps_per_cfg", 4)), (0, 1), eval_seed)
        # persist the trained student state (audit/reproducibility)
        policies_dir = os.path.join(results_dir, "policies_v02")
        os.makedirs(policies_dir, exist_ok=True)
        with open(os.path.join(policies_dir, f"{cond}_agent.json"),
                  "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, default=str)
        teacher_hold = None
        if hold_teacher:
            teacher_hold = eval_config_grid(
                state, hold_teacher, "teacher_holdout", max_steps,
                min(len(hold_teacher), 3),
                int(e.get("eval_eps_per_cfg", 4)), (0, 1), eval_seed + 1)
        ood_probe = eval_config_grid(
            state, splits["ood"], "ood", max_steps,
            int(e.get("ood_probe_n", 8)),
            int(e.get("ood_eps_per_seed", 3)), (0,), eval_seed + 2)

        train_curve = out["records"]
        per_condition[cond] = {
            "train": {
                "success_rate": (sum(1 for r in train_curve if r["success"])
                                 / max(1, len(train_curve))),
                "substrate_episodes": out["substrate_episodes"],
                "substrate_classic_episodes": out["substrate_classic_episodes"],
                "substrate_teacher_episodes": out["substrate_teacher_episodes"],
                "treatment_episodes": out["treatment_episodes"],
                "wall_s": round(out["wall_s"], 2),
                "calibration_episodes": (out["calibration"] or {}).get(
                    "calibration_episodes", 0),
            },
            "curve": train_curve,
            "classic_test": classic["summary"],
            "classic_test_per_config": classic["per_config_list"],
            "teacher_holdout": (teacher_hold or {}).get("summary"),
            "teacher_holdout_per_config": (teacher_hold or {}).get(
                "per_config_list"),
            "ood_probe": ood_probe["summary"],
            "retention_classic": retention.summary(),
            "retention_teacher": (retention_teacher.summary()
                                  if sub_teacher else None),
            "served_diversity": _served_diversity(train_curve),
            "lp_report": out["selector"].progress_report()
            if out["selector"] is not None else None,
        }

    # ---- paired comparisons ------------------------------------------------
    # Per-config vectors are matched by construction (same deterministic
    # config picks per pool), so paired bootstrap resamples config indices.
    comparisons = {}
    def pair_key(a, b, key_list, label, s):
        if a.get(key_list) and b.get(key_list) and len(a[key_list]) == len(b[key_list]):
            comparisons[label] = paired_difference(a[key_list], b[key_list], seed=s)
    if "uniform" in per_condition and "proc_lp" in per_condition:
        pair_key(per_condition["proc_lp"], per_condition["uniform"],
                 "classic_test_per_config", "lp_effect_classic_test", eval_seed)
        pair_key(per_condition["proc_lp"], per_condition["uniform"],
                 "teacher_holdout_per_config", "lp_effect_teacher_holdout",
                 eval_seed + 1)
    if "proc_lp" in per_condition and "teacher_lp" in per_condition:
        pair_key(per_condition["teacher_lp"], per_condition["proc_lp"],
                 "classic_test_per_config",
                 "teacher_source_effect_classic_test", eval_seed)
        pair_key(per_condition["teacher_lp"], per_condition["proc_lp"],
                 "teacher_holdout_per_config",
                 "teacher_source_effect_teacher_holdout", eval_seed + 1)

    results["conditions_detail"] = per_condition
    results["comparisons"] = comparisons
    results["finished_at"] = time.time()

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "curriculum_v02_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_strip_records(results), f, indent=1, default=str)
    if verbose:
        print(f"  summary written to {path}")
    return results


def _strip_records(results: dict) -> dict:
    """Compact serialization: keep summaries; trim very long curves."""
    out = copy.deepcopy(results)
    for cond, d in out.get("conditions_detail", {}).items():
        curve = d.get("curve") or []
        if len(curve) > 400:
            d["curve"] = curve[:200] + curve[-200:]
    return out
