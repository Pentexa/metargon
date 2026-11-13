"""Persistent Learning Diagnosis v0.1 — memory retention/reuse attribution.

Investigates *why* METARGON sometimes fails to retain and reuse correct
causal knowledge across tasks. This is a diagnosis-only harness: no
memory mechanism, planner or exploration agent is modified. It exercises
the existing composition (Memory v0.7 episode-scoped evidence and the
agent as shipped on main) under controlled, matched, seeded protocols.

Questions addressed
-------------------
Q1. Why do verified hypotheses become unreliable later?
     -> per-phase memory snapshots: exact-correct hypotheses by status,
        false contradictions (true rules contradicted), survival across
        a task switch.
Q2. Why does retained memory sometimes reduce success?
     -> negative-transfer control: trained agent vs memory-muted control
        and shuffled-content control on the same eval stream.
Q3. Do successful decisions actually depend on previous knowledge?
     -> reuse attribution: hypothesis-sourced plans, and the fraction of
        those plans whose hypothesis predates the eval phase (pre-
        existing knowledge) vs hypotheses created during eval.
Q4. Is A/B/A recovery retention or rediscovery?
     -> repeated-learning protocol (A/B/A) with per-phase hypothesis
        bookkeeping: in phase A2, successful episode explained by
        hypotheses created in A1 and surviving B (retained) vs created
        in A2 (rediscovered).
Q5. When does epoch_reset protect learning and when does it fail?
     -> variant x family matrix (classic / conflict / noisy / OOD) x
        regime (frozen, adaptive): success, correct-hypothesis survival,
        false contradictions, reuse.

Controls
--------
* fresh        — untrained (floor).
* original     — default memory (global counts), trained.
* epoch        — v0.7 episode-scoped evidence, trained.
* epoch_reset  — epoch + per-episode probe reset (current best config),
                 trained.
* muted        — original trained, memory wiped before eval (memory-
                 muted control; same policy, no retained knowledge).
* shuffled     — original trained, antecedent->consequence associations
                 shuffled before eval (content-broken control).

Determinism
-----------
Seeds: diagnostic base 20270201 (distinct from all published/accredited
runs). Canonical config ordering (tuple-sorted clause keys) for hash-
seed independence; PYTHONHASHSEED=0 recommended; frozen-eval integrity
checks assert no learning leaks into frozen cells.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from academy.agents.memory import (  # noqa: E402
    CONTRADICTED, VERIFIED,
)
from academy.agents.metargon_agent import MetargonAcademyAgent  # noqa: E402
from academy.envs.gridworld import (  # noqa: E402
    EnvConfig, MechanismGrid, is_solvable,
)
from academy.envs.tasks import (  # noqa: E402
    CORE_RULESETS, EpisodeSpec, difficulty_of, load_splits, make_episodes,
)
from academy.tools.memory_experiment_v06 import (  # noqa: E402
    check_frozen_integrity,
)


# ---------------------------------------------------------------------------
# Canonical config sampling (hash-seed independent)
# ---------------------------------------------------------------------------
def _canonical_key(config) -> tuple:
    def clauses(r):
        if isinstance(r, tuple):
            return tuple(sorted(tuple(sorted(clause)) for clause in r))
        return (tuple(sorted(r)),)
    return (config.topology, clauses(config.d1_rule), clauses(config.d2_rule),
            config.d1_delay, config.d2_delay)


def pick_configs(pool, n, seed):
    """Deterministic sample of `n` configs: canonical sort, then seeded
    shuffle, then take the head. Independent of hash seed."""
    items = sorted(pool, key=_canonical_key)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:max(0, min(n, len(items)))]


# ---------------------------------------------------------------------------
# Conflicting task family (flipped polarity; same observable contexts)
# ---------------------------------------------------------------------------
def _flip_rule(rule):
    def flip_clause(clause):
        out = set()
        for a in clause:
            if a[0] == "sw":
                out.add((a[0], a[1], not a[2]))
            else:
                out.add(a)
        return frozenset(out)
    if isinstance(rule, tuple):
        return tuple(flip_clause(c) for c in rule)
    return flip_clause(rule)


def _flipped_door_rules(topology, door):
    d1r, d2r = CORE_RULESETS[topology]
    rules = d1r if door == "D1" else d2r
    out = set()
    for r in rules:
        fr = _flip_rule(r)
        if fr != r:
            out.add(fr)
    return out


def build_conflict_pool(splits, seed=20270213, target=10, max_tries=800):
    """Configs where at least one door uses flipped polarity of a core
    rule, on the same topologies/delays; excluded from train/val/test/
    ood; solvability-filtered; easy configs preferred."""
    train_keys = {c.key for c in splits["train"]}
    seen_keys = set(train_keys)
    for k in ("val", "test", "ood"):
        seen_keys |= {c.key for c in splits[k]}
    rng = random.Random(seed)
    cands = set()
    for tp in CORE_RULESETS:
        d1f = _flipped_door_rules(tp, "D1")
        d2f = _flipped_door_rules(tp, "D2")
        d1f = {r for r in d1f if sum(len(c) for c in
                                      (r if isinstance(r, tuple) else (r,))) <= 2}
        d2f = {r for r in d2f if sum(len(c) for c in
                                      (r if isinstance(r, tuple) else (r,))) <= 2}
        d1c, d2c = CORE_RULESETS[tp]
        d1_opt = [r for r in d1c if len(r) <= 2] + sorted(
            d1f, key=lambda r: tuple(sorted(r)))
        d2_opt = [r for r in d2c if len(r) <= 2] + sorted(
            d2f, key=lambda r: tuple(sorted(r)))
        if not d1f and not d2f:
            continue
        for _ in range(max_tries):
            r1 = rng.choice(d1_opt)
            r2 = rng.choice(d2_opt)
            if r1 not in d1f and r2 not in d2f:
                continue
            b1 = rng.choice((False, True))
            b2 = rng.choice((False, True))
            c = EnvConfig(topology=tp, d1_rule=r1, d2_rule=r2,
                          d1_delay=b1, d2_delay=b2)
            if c.key in seen_keys:
                continue
            if not is_solvable(c):
                continue
            cands.add(c)
            if len(cands) >= max(4, target):
                break
        if len(cands) >= max(4, target):
            break
    order = {"easy": 0, "medium": 1, "hard": 2}
    pool = sorted(cands, key=lambda c: (order[difficulty_of(c).bin],
                                        c.has_delay(), _canonical_key(c)))
    return pool[:max(4, target)]

# ---------------------------------------------------------------------------
# Diagnostic constants (independent from auditor/accredited seeds)
# ---------------------------------------------------------------------------
DIAG_SEED_BASE = 20270201
TRAIN_SEEDS = (DIAG_SEED_BASE, DIAG_SEED_BASE + 1)
EVAL_SEED = DIAG_SEED_BASE + 10
CONFLICT_SEED = DIAG_SEED_BASE + 12
NOISY_SEED = DIAG_SEED_BASE + 13
OOD_SEED = DIAG_SEED_BASE + 14
RETENTION_SEED = DIAG_SEED_BASE + 31
FOCUS_SEED = DIAG_SEED_BASE + 47

TRAIN_EPISODES = 120
GRID = {
    "classic": {"n_configs": 8, "eps_per_cfg": 3, "noise_q": 0.0},
    "conflict": {"n_configs": 8, "eps_per_cfg": 3, "noise_q": 0.0},
    "noisy": {"n_configs": 6, "eps_per_cfg": 3, "noise_q": 0.15},
    "ood": {"n_configs": 8, "eps_per_cfg": 3, "noise_q": 0.0},
}

# ---------------------------------------------------------------------------
# Variants (matched kwargs)
# ---------------------------------------------------------------------------
VARIANTS = {
    "fresh": {},
    "original": {},
    "epoch": {"memory_episode_scoped": True,
              "memory_min_verified_epochs_for_transfer": 2},
    "epoch_reset": {"memory_episode_scoped": True,
                    "memory_min_verified_epochs_for_transfer": 2,
                    "probe_reset_per_episode": True},
    "muted": {"_wipe_memory": True},
    "shuffled": {"_shuffle_memory": True},
}

# label -> (flags, needs_training)
VARIANT_SPEC = {
    name: (flags, name not in ("fresh",))
    for name, flags in VARIANTS.items()
}


class DiagnosticAgent(MetargonAcademyAgent):
    """Wrapper implementing the control hooks (memory wipe / shuffle)."""

    def __init__(self, *a, **k):
        self._wipe = bool(k.pop("_wipe_memory", False))
        self._shuffle = bool(k.pop("_shuffle_memory", False))
        self._shuffle_seed = int(k.pop("_shuffle_seed", 0))
        super().__init__(*a, **k)

    def maybe_wipe(self):
        if not self._wipe:
            return
        m = self.memory
        m.hypotheses = []
        m._continuous_true = {}
        m._dead = {}
        m._sup_timeline = {}
        m._con_timeline = {}
        m.evidence_bank = []
        m._atom_stats = {}
        m._atom_ctxs = {}
        m._pair_stats = {}
        m._door_event_count = {}
        m._atom_event_support = {}
        m._atom_nonevent_support = {}
        m._door_nonevent_count = {}

    def maybe_shuffle(self):
        if not self._shuffle:
            return
        hyps = [h for h in self.memory.hypotheses if h.usable]
        rng = random.Random(self._shuffle_seed)
        antes = [h.antecedent for h in hyps]
        rng.shuffle(antes)
        for h, ante in zip(hyps, antes):
            h.antecedent = ante


# ---------------------------------------------------------------------------
# Ground-truth classification of stored hypotheses
# ---------------------------------------------------------------------------
def rule_clauses(rule):
    return rule if isinstance(rule, tuple) else (rule,)


def classify_hypothesis(h, config):
    """Classify a hypothesis against the true rule of its door.

    Returns one of: exact (antecedent == true clause), subset (nonempty
    proper subset of a true clause), incorrect (anything else)."""
    clause = config.rule_for(h.door)
    clauses = rule_clauses(clause)
    if any(h.antecedent == c for c in clauses):
        return "exact"
    if any(h.antecedent < c for c in clauses):
        return "subset"
    return "incorrect"


def memory_snapshot(agent, config):
    """Per-config memory snapshot: counts by class x status."""
    out = {"size": len(agent.memory.hypotheses),
           "by_class": {"exact": {"total": 0, "verified": 0,
                                   "contradicted": 0, "usable": 0},
                        "subset": {"total": 0, "verified": 0,
                                   "contradicted": 0, "usable": 0},
                        "incorrect": {"total": 0, "verified": 0,
                                      "contradicted": 0, "usable": 0}},
           "verified_total": 0, "contradicted_total": 0}
    for h in agent.memory.hypotheses:
        cls = classify_hypothesis(h, config)
        e = out["by_class"][cls]
        e["total"] += 1
        if h.status == VERIFIED:
            e["verified"] += 1
            out["verified_total"] += 1
        if h.status == CONTRADICTED:
            e["contradicted"] += 1
            out["contradicted_total"] += 1
        if h.usable:
            e["usable"] += 1
    out["false_contradictions"] = out["by_class"]["exact"]["contradicted"]
    out["retained_exact_usable"] = out["by_class"]["exact"]["usable"]
    return out


# ---------------------------------------------------------------------------
# Instrumented episode runner (reuse + snapshot per episode)
# ---------------------------------------------------------------------------
def run_episode_instrumented(agent, spec, eval_start_step=None):
    """Run one episode; record reuse attribution and door outcomes.

    ``eval_start_step`` is the memory step at eval-phase start: plans
    whose hypothesis has ``created_step < eval_start_step`` are
    attributed to pre-existing knowledge (retained).
    """
    env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
    obs = env.reset(spec.spec_seed)
    prev = agent.probe_trace()
    prev_plan_len = len(prev.get("planned_probes", []))
    agent.episode_start(spec, env.geom)
    snapshot = memory_snapshot(agent, spec.config)
    steps = 0
    success = False
    while steps < env.max_steps:
        action = agent.act(obs)
        before = obs
        res = env.step(action)
        agent.learn(before, action, res)
        steps += 1
        obs = res.observation
        if res.terminated:
            success = True
            break
    agent.finish_episode(success)
    tr = agent.probe_trace()
    new_plans = tr.get("planned_probes", [])[prev_plan_len:]
    hyp_plans = 0
    reuse_plans = 0
    for plan in new_plans:
        door, ante, src = plan[0], frozenset(plan[1]), plan[2]
        if src != "hypothesis":
            continue
        hyp_plans += 1
        if eval_start_step is not None:
            for h in agent.memory.hypotheses:
                if (h.door == door and h.antecedent == ante
                        and h.created_step < eval_start_step):
                    reuse_plans += 1
                    break
    delta = {}
    for k in ("source_hypothesis", "memory_actions", "memory_correct",
              "memory_incorrect", "door_open_after_hypothesis",
              "cross_epoch_transfer_retrieved", "source_novel_single",
              "plans_used"):
        delta[k] = tr.get(k, 0) - prev.get(k, 0)
    return {
        "success": success,
        "steps": steps,
        "snapshot": snapshot,
        "hypothesis_plans": hyp_plans,
        "reuse_plans": reuse_plans,
        "trace_delta": delta,
    }


# ---------------------------------------------------------------------------
# Evaluation grids (frozen / adaptive) with per-episode diagnostics
# ---------------------------------------------------------------------------
def eval_grid_diagnostic(agent_state, configs, max_steps, n_configs,
                         episodes_per_config, seeds, seed, frozen=False,
                         agent_kwargs=None, noise_q=0.0, eval_start_step=None):
    picks = pick_configs(list(configs), n_configs, seed)
    rows = []
    for i, cfgc in enumerate(picks):
        for seed_i in seeds:
            base = seed * 100000 + i * 100 + seed_i
            stream = make_episodes([cfgc], "eval", base, episodes_per_config,
                                   max_steps=max_steps, noise_q=noise_q)
            agent = DiagnosticAgent(seed=base, **dict(agent_kwargs or {}))
            pre_generated = pre_step = pre_steps = None
            if agent_state is not None:
                agent.load_state(agent_state)
                pre_generated = agent.memory.generated
                pre_step = agent.memory.step
                pre_steps = agent.steps
            agent.maybe_wipe()
            agent.maybe_shuffle()
            agent._collect_probe_log = True
            if frozen:
                agent.frozen_policy = True
            succ = 0
            eps_rows = []
            for spec in stream:
                rec = run_episode_instrumented(
                    agent, spec, eval_start_step=eval_start_step)
                succ += 1 if rec["success"] else 0
                eps_rows.append(rec)
            if frozen and pre_generated is not None:
                check_frozen_integrity(agent, pre_generated, pre_step,
                                       pre_steps, f"grid@{base}")
            row = {
                "config": str(cfgc.key), "config_i": i, "seed_i": seed_i,
                "success_rate": succ / max(1, len(stream)),
                "n": len(stream),
            }
            row.update(_aggregate_episodes(eps_rows))
            rows.append(row)
    return _summarize_grid(rows, picks)


def _aggregate_episodes(eps_rows):
    out = {"successes": 0, "mem_snapshot": None, "hypothesis_plans": 0,
           "reuse_plans": 0, "trace": {}}
    for r in eps_rows:
        out["successes"] += 1 if r["success"] else 0
        out["hypothesis_plans"] += r["hypothesis_plans"]
        out["reuse_plans"] += r["reuse_plans"]
        if out["mem_snapshot"] is None:
            out["mem_snapshot"] = {k: (dict(v) if isinstance(v, dict) else v)
                                   for k, v in r["snapshot"].items()}
        else:
            out["mem_snapshot"] = _add_nested(out["mem_snapshot"],
                                              r["snapshot"])
        for k, v in r["trace_delta"].items():
            out["trace"][k] = out["trace"].get(k, 0) + v
    return out


def _add_nested(a, b):
    """Recursively add numeric leaves of two same-shaped dicts."""
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = _add_nested(a.get(k, {}), v)
        else:
            a[k] = a.get(k, 0) + v
    return a


def _summarize_grid(rows, picks):
    per_config = {}
    for r in rows:
        e = per_config.setdefault(r["config"], {"succ": 0.0, "n": 0})
        e["succ"] += r["success_rate"] * r["n"]
        e["n"] += r["n"]
    per_config = {k: v["succ"] / v["n"] for k, v in per_config.items()}
    rates = list(per_config.values())
    summary = {
        "n_configs": len(picks),
        "config_mean_success": (sum(rates) / len(rates)) if rates else 0.0,
    }
    agg = {"hypothesis_plans": 0, "reuse_plans": 0, "trace": {}}
    snap = None
    for r in rows:
        agg["hypothesis_plans"] += r["hypothesis_plans"]
        agg["reuse_plans"] += r["reuse_plans"]
        for k, v in r["trace"].items():
            agg["trace"][k] = agg["trace"].get(k, 0) + v
        if r.get("mem_snapshot"):
            if snap is None:
                snap = {k: (dict(v) if isinstance(v, dict) else v)
                        for k, v in r["mem_snapshot"].items()}
            else:
                snap = _add_nested(snap, r["mem_snapshot"])
    return {"summary": summary, "per_config_list": rates, "rows": rows,
            "agg": agg, "snapshot_agg": snap}


def spec_for_public(spec):
    from types import SimpleNamespace
    return SimpleNamespace(max_steps=spec.max_steps, noise_q=spec.noise_q,
                           spec_seed=spec.spec_seed,
                           pool_tag=getattr(spec, "pool_tag", None),
                           difficulty=getattr(spec, "difficulty", None))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_agent(agent, splits, episodes, seed):
    """Train on a stream drawn deterministically from a pool (dict with
    'train' key, or a bare list of configs)."""
    pool = splits["train"] if isinstance(splits, dict) else splits
    rng = random.Random(seed)
    stream = []
    for e in range(episodes):
        cfg = rng.choice(pool)
        stream.append(EpisodeSpec(
            config=cfg, max_steps=80, noise_q=0.0,
            spec_seed=seed * 1000 + e, pool_tag="train",
            difficulty=difficulty_of(cfg)))
    succ = 0
    for spec in stream:
        env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
        obs = env.reset(spec.spec_seed)
        agent.episode_start(spec_for_public(spec), env.geom)
        ok = False
        while env.steps < env.max_steps:
            a = agent.act(obs)
            res = env.step(a)
            agent.learn(obs, a, res)
            obs = res.observation
            if res.terminated:
                ok = True
                break
        agent.finish_episode(ok)
        succ += 1 if ok else 0
    return succ / max(1, len(stream))


# ---------------------------------------------------------------------------
# Main matrix
# ---------------------------------------------------------------------------
def run_main_matrix(splits, seeds=TRAIN_SEEDS, train_eps=TRAIN_EPISODES,
                    results_dir=None):
    pools = {
        "classic": splits["test"],
        "conflict": build_conflict_pool(splits, seed=CONFLICT_SEED,
                                        target=10),
        "noisy": splits["test"],
        "ood": splits["ood"],
    }
    out = {"run_id": "memory-diagnosis-attribution-v01-main",
           "seed_base": DIAG_SEED_BASE, "train_episodes": train_eps,
           "variants": list(VARIANT_SPEC), "cells": {}}
    for cell in ("classic", "conflict", "noisy", "ood"):
        out["cells"][cell] = {"variants": {}}
        cfg = GRID[cell]
        for name, (flags, needs_train) in VARIANT_SPEC.items():
            seed_results = []
            for seed in seeds:
                agent = DiagnosticAgent(seed=seed, **flags)
                train_succ = None
                if needs_train:
                    train_succ = train_agent(agent, splits, train_eps, seed)
                agent_state = agent.save_state()
                start_step = agent.memory.step if needs_train else None
                frozen = eval_grid_diagnostic(
                    agent_state, pools[cell], 80, cfg["n_configs"],
                    cfg["eps_per_cfg"], (seed,), EVAL_SEED, frozen=True,
                    agent_kwargs=flags, noise_q=cfg["noise_q"],
                    eval_start_step=start_step)
                adaptive = eval_grid_diagnostic(
                    agent_state, pools[cell], 80, cfg["n_configs"],
                    cfg["eps_per_cfg"], (seed,), EVAL_SEED, frozen=False,
                    agent_kwargs=flags, noise_q=cfg["noise_q"],
                    eval_start_step=start_step)
                seed_results.append({
                    "seed": seed, "train_success": train_succ,
                    "frozen": frozen, "adaptive": adaptive,
                })
            out["cells"][cell]["variants"][name] = seed_results
            fs = [r["frozen"]["summary"]["config_mean_success"]
                  for r in seed_results]
            ad = [r["adaptive"]["summary"]["config_mean_success"]
                  for r in seed_results]
            print(f"[{cell}] {name}: frozen {sum(fs)/len(fs):.3f} "
                  f"adaptive {sum(ad)/len(ad):.3f}")
    _save(out, "main_summary.json", results_dir)
    return out


def _save(obj, fname, results_dir):
    results_dir = results_dir or os.path.join(
        ROOT, "results", "academy_v01", "memory_diagnosis_attribution_v01")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, fname)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Repeated learning A/B/A with retention-vs-rediscovery attribution
# ---------------------------------------------------------------------------
def build_focus_pools(splits, seed=FOCUS_SEED):
    ra1 = frozenset({("sw", "S2", True)})
    ra2 = frozenset({("sw", "S1", False)})
    rb1 = frozenset({("sw", "S2", False)})
    rb2 = frozenset({("sw", "S1", True)})
    train_keys = {c.key for c in splits["train"]}
    compat, conflict = [], []
    for tp in ("TP1", "TP2"):
        for b1 in (False, True):
            for b2 in (False, True):
                ca = EnvConfig(topology=tp, d1_rule=ra1, d2_rule=ra2,
                               d1_delay=b1, d2_delay=b2)
                if ca.key in train_keys and is_solvable(ca):
                    compat.append(ca)
                cb = EnvConfig(topology=tp, d1_rule=rb1, d2_rule=rb2,
                               d1_delay=b1, d2_delay=b2)
                if cb.key not in train_keys and is_solvable(cb):
                    conflict.append(cb)
    rng = random.Random(seed)
    rng.shuffle(compat)
    rng.shuffle(conflict)
    return compat[:8], conflict[:8]


def run_repeated_learning(splits, variants=("original", "epoch_reset"),
                          seed=TRAIN_SEEDS[0], results_dir=None,
                          phase_episodes=None):
    compat, conflict = build_focus_pools(splits)
    phase_episodes = phase_episodes or {"A1": 40, "B": 30, "A2": 30}
    out = {"run_id": "memory-diagnosis-attribution-v01-rlearn",
           "seed": seed, "phase_episodes": phase_episodes,
           "variants": {}}
    for name in variants:
        flags = VARIANTS[name]
        agent = DiagnosticAgent(seed=seed, **flags)
        agent._collect_probe_log = True

        def train(pool, n):
            return train_agent(agent, {"train": pool}, n, seed)

        def phase_snapshot():
            return {"step": agent.memory.step,
                    "hypotheses": [{"h_id": h.h_id,
                                    "created_step": h.created_step,
                                    "status": h.status,
                                    "door": h.door,
                                    "antecedent": sorted(h.antecedent)}
                                   for h in agent.memory.hypotheses]}

        # phase A1
        a1 = train(compat, phase_episodes["A1"])
        b_step = None
        a1_snap = phase_snapshot()
        # frozen eval A'
        eval_a1 = eval_grid_diagnostic(
            agent.save_state(), compat, 80, 4, 2, (0,), EVAL_SEED,
            frozen=True, agent_kwargs=flags, eval_start_step=agent.memory.step)
        # phase B (conflict)
        b = train(conflict, phase_episodes["B"])
        b_snap = phase_snapshot()
        b_step = b_snap["step"]
        eval_b = eval_grid_diagnostic(
            agent.save_state(), conflict, 80, 4, 2, (0,), EVAL_SEED,
            frozen=True, agent_kwargs=flags, eval_start_step=b_step)
        # phase A2
        a2 = train(compat, phase_episodes["A2"])
        a2_snap = phase_snapshot()
        eval_a2 = eval_grid_diagnostic(
            agent.save_state(), compat, 80, 4, 2, (0,), EVAL_SEED,
            frozen=True, agent_kwargs=flags, eval_start_step=b_step)
        out["variants"][name] = {
            "success": {"A1": a1, "B": b, "A2": a2},
            "snapshots": {"A1": a1_snap, "B": b_snap, "A2": a2_snap},
            "eval": {"A1_frozen": eval_a1["summary"],
                     "B_frozen": eval_b["summary"],
                     "A2_frozen": eval_a2["summary"]},
            "attribution": _rlearn_attribution(a1_snap, b_snap, a2_snap,
                                                eval_a2),
        }
        print(f"[rlearn] {name}: A1 {a1:.3f} B {b:.3f} A2 {a2:.3f}")
    _save(out, "rlearn_summary.json", results_dir)
    return out


def _rlearn_attribution(a1_snap, b_snap, a2_snap, eval_a2):
    """Classify A2 hypothesis-plans by origin: retained from A1
    (created_step < B-start) vs rediscovered (created >= B-start).

    The A2 evaluation grid was run with ``eval_start_step=B-start`` so
    per-row ``reuse_plans`` counts plans whose hypothesis was created
    before the conflicting phase B.
    """
    a1_ids = {h["h_id"] for h in a1_snap["hypotheses"]}
    # status at end of A1 for A1 hypotheses (for transition counting)
    a1_status = {h["h_id"]: h["status"] for h in a1_snap["hypotheses"]}
    retained = 0
    rediscovered = 0
    for r in eval_a2.get("rows", []):
        retained += r.get("reuse_plans", 0)
        rediscovered += max(0, r.get("hypothesis_plans", 0)
                            - r.get("reuse_plans", 0))
    hyp_plans = retained + rediscovered
    a1_usable_after_B = 0
    a1_exact_after_B = 0
    status_transitions = {}
    for h in b_snap["hypotheses"]:
        if h["h_id"] not in a1_ids:
            continue
        before = a1_status.get(h["h_id"])
        after = h["status"]
        key = (before, after)
        status_transitions[key] = status_transitions.get(key, 0) + 1
        if h["status"] != CONTRADICTED:
            a1_usable_after_B += 1
        if h["status"] == VERIFIED:
            a1_exact_after_B += 1
    return {
        "a1_hypotheses_present_after_B": sum(
            1 for h in b_snap["hypotheses"] if h["h_id"] in a1_ids),
        "a1_hypotheses_usable_after_B": a1_usable_after_B,
        "a1_hypotheses_verified_after_B": a1_exact_after_B,
        "a1_status_transitions_during_B": {
            f"{a}->{b}": c for (a, b), c in sorted(
                status_transitions.items(), key=lambda kv: str(kv[0]))},
        "a2_hypothesis_plans": hyp_plans,
        "a2_retained_plans": retained,
        "a2_rediscovered_plans": rediscovered,
        "a2_retention_share": (retained / hyp_plans) if hyp_plans else None,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _cell_table(cell_data):
    """Rows: variant | frozen | adaptive | retained-exact(usable) | """
    """false-contradictions | reuse share |"""
    lines = []
    lines.append("| variant | frozen | adaptive | exact-usable | "
                 "false-contradictions | reuse share |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, recs in cell_data.items():
        n = len(recs)
        fz = [r["frozen"]["summary"]["config_mean_success"] for r in recs]
        ad = [r["adaptive"]["summary"]["config_mean_success"] for r in recs]
        f_mean = sum(fz) / n
        a_mean = sum(ad) / n
        # aggregate snapshot across seeds
        snap = None
        for r in recs:
            s = r["frozen"].get("snapshot_agg")
            if s:
                snap = s if snap is None else _add_nested(snap, s)
        reuse = 0.0
        hyp = 0.0
        for r in recs:
            agg = r["frozen"].get("agg", {})
            hyp += agg.get("hypothesis_plans", 0)
            reuse += agg.get("reuse_plans", 0)
        reuse_share = (reuse / hyp) if hyp else None
        if snap:
            exact_usable = snap["by_class"]["exact"]["usable"]
            false_c = snap["false_contradictions"]
        else:
            exact_usable, false_c = 0, 0
        lines.append(
            f"| {name} | {f_mean:.3f} | {a_mean:.3f} | {exact_usable} | "
            f"{false_c} | "
            f"{(f'{reuse_share:.2f}' if reuse_share is not None else '—')} |")
    return "\n".join(lines)


def render_report(results_dir=None):
    results_dir = results_dir or os.path.join(
        ROOT, "results", "academy_v01", "memory_diagnosis_attribution_v01")
    with open(os.path.join(results_dir, "main_summary.json"),
              encoding="utf-8") as f:
        main = json.load(f)
    rlearn_path = os.path.join(results_dir, "rlearn_summary.json")
    rlearn = None
    if os.path.exists(rlearn_path):
        with open(rlearn_path, encoding="utf-8") as f:
            rlearn = json.load(f)
    lines = []
    lines.append("# Persistent Learning Diagnosis v0.1 — root-cause report")
    lines.append("")
    lines.append(f"Run: `{main['run_id']}`; seeds {main['seed_base']}; "
                 f"train {main['train_episodes']} eps/seed; "
                 "PYTHONHASHSEED=0.")
    lines.append("")
    for cell in ("classic", "conflict", "noisy", "ood"):
        if cell not in main["cells"]:
            continue
        lines.append(f"## Cell: {cell}")
        lines.append("")
        lines.append(_cell_table(main["cells"][cell]["variants"]))
        lines.append("")
    if rlearn:
        lines.append("## A/B/A repeated learning")
        lines.append("")
        lines.append("| variant | A1 | B | A2 | A2−A1 | retained share | "
                     "A1-verified-after-B |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, d in rlearn["variants"].items():
            s = d["success"]
            attr = d["attribution"]
            share = attr.get("a2_retention_share")
            lines.append(
                f"| {name} | {s['A1']:.3f} | {s['B']:.3f} | "
                f"{s['A2']:.3f} | {s['A2']-s['A1']:+.3f} | "
                f"{(f'{share:.2f}' if share is not None else '—')} | "
                f"{attr.get('a1_hypotheses_verified_after_B', 0)} |")
        lines.append("")
        lines.append("A1→B status transitions (A1 hypotheses that persist "
                     "into B):")
        lines.append("")
        lines.append("| variant | transition | count |")
        lines.append("|---|---:|---:|")
        for name, d in rlearn["variants"].items():
            for tr, c in d["attribution"].get(
                    "a1_status_transitions_during_B", {}).items():
                lines.append(f"| {name} | {tr} | {c} |")
        lines.append("")
    path = os.path.join(results_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Persistent Learning Diagnosis")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(TRAIN_SEEDS))
    ap.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    ap.add_argument("--cells", nargs="*", default=("classic", "conflict",
                                                   "noisy", "ood"))
    ap.add_argument("--rlearn", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    if args.report:
        p = render_report(args.results_dir)
        print(f"report written: {p}")
        return
    splits = load_splits()
    if args.rlearn:
        run_repeated_learning(splits, results_dir=args.results_dir)
    else:
        run_main_matrix(splits, seeds=tuple(args.seeds),
                        train_eps=args.train_episodes,
                        results_dir=args.results_dir)


if __name__ == "__main__":
    main()
