"""Cognitive Memory v0.2 (CM v0.2) — protect-exploration controlled evaluation.

Background
----------
PR #50 (CM v0.1) is treated as a negative scientific finding: globally
re-ranking hypotheses by reliability reduced exact novel-single probes and
broke A/B/A repeated learning. v0.2 therefore keeps the ``epoch_reset``
retrieval order unchanged and adds only a minimal, context-sensitive
*eligibility filter + cap* on hypothesis reuse (see memory.py).

Intervention (all off by default; ``CognitiveMemoryV02`` wires them on)
------------------------------------------------------------------------
- ``protect_exploration``: hypothesis reuse requires >=
  ``protection_min_active_support`` support events in the active evidence
  basis (epoch > context > cumulative); hypotheses with more than
  ``protection_max_failures`` repeated failure windows are excluded from
  reuse (evidence kept); at most ``protection_max_reuse_per_door``
  hypotheses are returned per door for probing (after baseline sort).

Evaluation (offline, deterministic, matched seeds; canonical ordering)
---------------------------------------------------------------------
Controls: fresh, original, memory_wiped (memory-muted), epoch (v0.7),
epoch_reset (primary baseline), v02 (proposed), plus mechanism ablations
v02_gate_only (cap off) and v02_cap_only (gate off).

Regimes measured separately: classic (compatible) frozen/adaptive;
conflicting (flipped-polarity) frozen/adaptive; noisy (q=0.2)
frozen/adaptive; OOD (held-out OOD pool) frozen/adaptive; retention
(mid-training vs final); repeated learning A/B/A.

Exact novel-rule discovery is measured directly: novel-single probe counts
and their exact-relevance counts per variant (classic frozen).

The environment, agent, and baseline behavior are untouched.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from academy.agents.memory import (  # noqa: E402
    CognitiveMemoryV02, CompositionalMemory,
)
from academy.agents.metargon_agent import MetargonAcademyAgent  # noqa: E402
from academy.envs.gridworld import (  # noqa: E402
    EnvConfig, MechanismGrid, is_solvable,
)
from academy.envs.tasks import (  # noqa: E402
    CORE_RULESETS, DEFAULT_SPLITS_PATH, EpisodeSpec, difficulty_of,
    load_splits, make_episodes,
)
from academy.eval.metrics import summarize_episodes  # noqa: E402
from academy.evren.retention import RetentionSuite  # noqa: E402
from academy.learning.loop import run_one_episode  # noqa: E402
from academy.tools.memory_experiment_v06 import (  # noqa: E402
    check_frozen_integrity, compare_delta, summarize_probe_relevance,
)
from academy.tools.memory_experiment_v07 import (  # noqa: E402
    attribution_metrics, memory_tracking_v07,
)

# ---------------------------------------------------------------------------
# Variants. Underscore-prefixed keys are hooks for the agent wrapper.
# ---------------------------------------------------------------------------
VARIANTS_V02 = {
    "fresh": {},
    "original": {},
    "memory_wiped": {"_wipe_memory": True},
    "epoch": {"memory_episode_scoped": True,
              "memory_min_verified_epochs_for_transfer": 2},
    "epoch_reset": {"memory_episode_scoped": True,
                    "memory_min_verified_epochs_for_transfer": 2,
                    "probe_reset_per_episode": True},
    "v02": {"memory_factory": CognitiveMemoryV02,
            "memory_episode_scoped": True,
            "memory_min_verified_epochs_for_transfer": 2,
            "probe_reset_per_episode": True},
    # mechanism ablations
    "v02_gate_only": {"memory_factory": CognitiveMemoryV02,
                      "memory_episode_scoped": True,
                      "memory_min_verified_epochs_for_transfer": 2,
                      "probe_reset_per_episode": True,
                      "protection_allow_trusted_transfer": False},
    "v02_nocap": {"memory_factory": CognitiveMemoryV02,
                  "memory_episode_scoped": True,
                  "memory_min_verified_epochs_for_transfer": 2,
                  "probe_reset_per_episode": True,
                  "protection_max_reuse_per_door": 10},
    "v02_nogate": {"memory_factory": CognitiveMemoryV02,
                   "memory_episode_scoped": True,
                   "memory_min_verified_epochs_for_transfer": 2,
                   "probe_reset_per_episode": True,
                   "protection_min_active_support": 0},
}

DEFAULT_BASE = {
    "train_episodes": 160,
    "test_grid_n": 16, "test_eps_per_cfg": 4,
    "conflict_grid_n": 16, "conflict_eps_per_cfg": 4,
    "noisy_grid_n": 12, "noisy_eps_per_cfg": 3,
    "ood_grid_n": 12, "ood_eps_per_cfg": 3,
    "substrate_n": 8, "retention_eps_per_cfg": 2,
    "focus_n": 8, "focus_eps_per_cfg": 3,
    "conflict_n": 16,
    "rlearn_phase_a": 60, "rlearn_phase_b": 40,
    "rlearn_phase_a2": 40,
}

EVAL_SEED = 20261108           # same base as v0.6/v0.7/v0.1 (comparable)
CONFLICT_SEED = EVAL_SEED + 2
NOISY_SEED = EVAL_SEED + 3
OOD_SEED = EVAL_SEED + 6
RETENTION_SEED = 20261107 + 31
FOCUS_SEED = 20261107 + 47


MEMORY_KWARGS = ("protection_min_active_support",
                 "protection_max_failures",
                 "protection_max_reuse_per_door",
                 "protection_allow_trusted_transfer")


def split_variant_raw(raw):
    """Separate agent kwargs from hook kwargs and memory-level kwargs.

    Returns (agent_kwargs, hooks) where hooks may include
    ``_memory_kwargs`` (forwarded to the memory constructor via the
    factory) and ``_wipe_memory``.
    """
    raw = dict(raw)
    agent_kwargs = {}
    hooks = {}
    mem_kwargs = {}
    for k, v in raw.items():
        if k.startswith("_"):
            hooks[k] = v
        elif k in MEMORY_KWARGS:
            mem_kwargs[k] = v
        else:
            agent_kwargs[k] = v
    mf = hooks.pop("_memory_factory", None)
    if mf is not None:
        agent_kwargs["memory_factory"] = mf
    if mem_kwargs:
        hooks["_memory_kwargs"] = mem_kwargs
    return agent_kwargs, hooks


class V02VariantAgent(MetargonAcademyAgent):
    """Agent wrapper implementing the variant hooks."""

    def __init__(self, *a, **k):
        self._wipe = bool(k.pop("_wipe_memory", False))
        mk = k.pop("_memory_kwargs", None)
        mf = k.pop("memory_factory", None)
        if mf is not None:
            def _factory(mf=mf, mk=mk):
                def _make(*aa, **kk):
                    if mk:
                        kk = {**kk, **mk}
                    return mf(*aa, **kk)
                return _make
            k["memory_factory"] = _factory()
        super().__init__(*a, **k)

    def maybe_wipe(self):
        if not self._wipe:
            return
        self.memory.hypotheses = []
        self.memory._continuous_true = {}
        self.memory._dead = {}
        self.memory._sup_timeline = {}
        self.memory._con_timeline = {}
        self.memory.evidence_bank = []
        self.memory._atom_stats = {}
        self.memory._atom_ctxs = {}
        self.memory._pair_stats = {}
        self.memory._door_event_count = {}
        self.memory._atom_event_support = {}
        self.memory._atom_nonevent_support = {}
        self.memory._door_nonevent_count = {}


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


# ---------------------------------------------------------------------------
# Canonical ordering (hash-seed independent)
# ---------------------------------------------------------------------------
def _canonical_key(config) -> tuple:
    def clauses(r):
        if isinstance(r, tuple):
            return tuple(sorted(tuple(sorted(clause)) for clause in r))
        return (tuple(sorted(r)),)
    return (config.topology, clauses(config.d1_rule), clauses(config.d2_rule),
            config.d1_delay, config.d2_delay)


def pick_configs(pool, n, seed):
    items = sorted(pool, key=_canonical_key)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:max(0, min(n, len(items)))]


# ---------------------------------------------------------------------------
# Conflicting task family (same as v0.1: flipped polarity, isolated)
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


def build_conflict_pool(splits, seed=CONFLICT_SEED, target=16,
                        max_tries=800):
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


# ---------------------------------------------------------------------------
# Evaluation grid (canonical pick; integrity checks identical to v0.6)
# ---------------------------------------------------------------------------
def eval_grid(agent_state, configs, max_steps, n_configs, episodes_per_config,
              seeds, seed, frozen=False, agent_kwargs=None,
              collect_trace=False, noise_q=0.0):
    agent_kwargs = dict(agent_kwargs or {})
    picks = pick_configs(list(configs), n_configs, seed)
    rows = []
    for i, cfgc in enumerate(picks):
        for seed_i in seeds:
            base = seed * 100000 + i * 100 + seed_i
            stream = make_episodes([cfgc], "eval", base, episodes_per_config,
                                   max_steps=max_steps, noise_q=noise_q)
            agent = V02VariantAgent(seed=base, **agent_kwargs)
            pre_generated = pre_step = pre_steps = None
            if agent_state is not None:
                agent.load_state(agent_state)
                pre_generated = agent.memory.generated
                pre_step = agent.memory.step
                pre_steps = agent.steps
            if frozen:
                agent.frozen_policy = True
            if collect_trace:
                agent._collect_probe_log = True
            succ, steps_total = 0, 0
            planned_all = []
            for spec in stream:
                env = MechanismGrid(spec.config, spec.max_steps,
                                    spec.noise_q)
                obs = env.reset(spec.spec_seed)
                agent.episode_start(spec, env.geom)
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
                if collect_trace:
                    planned_all.extend(agent.trace.get("planned_probes", []))
                    agent.trace["planned_probes"] = []
            if frozen and pre_generated is not None:
                check_frozen_integrity(agent, pre_generated, pre_step,
                                       pre_steps, f"grid@{base}")
            row = {
                "config": str(cfgc.key), "config_i": i, "seed_i": seed_i,
                "success_rate": succ / max(1, len(stream)),
                "n": len(stream),
            }
            if collect_trace:
                row["trace"] = dict(agent.trace)
                row["relevance"] = summarize_probe_relevance(planned_all, cfgc)
            rows.append(row)
    per_config = {}
    for r in rows:
        ent = per_config.setdefault(r["config"], {"succ": 0.0, "n": 0})
        ent["succ"] += r["success_rate"] * r["n"]
        ent["n"] += r["n"]
    per_config = {k: v["succ"] / v["n"] for k, v in per_config.items()}
    rates = list(per_config.values())
    summary = {
        "n_configs": len(picks),
        "config_mean_success": (sum(rates) / len(rates)) if rates else 0.0,
        "per_config": {k: round(v, 4) for k, v in per_config.items()},
    }
    out = {"summary": summary, "per_config_list": rates, "rows": rows}
    if collect_trace:
        out["trace_agg"] = _aggregate_trace(rows)
        out["relevance_agg"] = _aggregate_relevance(rows)
    return out


def _aggregate_trace(rows):
    keys = (
        "source_hypothesis", "source_bank", "source_atom_single",
        "source_atom_pair", "source_novel_single",
        "source_conjunctive_probe", "source_conjunction", "source_other",
        "skipped_tried", "skipped_infeasible", "probes_evaluated",
        "plans_used", "random_fallback_steps", "door_open_after_probe",
        "door_open_after_hypothesis", "door_open_after_bank",
        "door_open_after_atom_single", "door_open_after_atom_pair",
        "door_open_after_novel_single", "door_open_after_conjunction",
        "door_open_after_other", "query_calls", "query_useful",
        "hypothesis_candidates", "hypothesis_skipped_lowconf",
        "hypothesis_retrieved", "cross_epoch_transfer_retrieved",
        "memory_actions", "memory_correct", "memory_incorrect",
    )
    return {
        k: sum(r.get("trace", {}).get(k, 0) for r in rows)
        for k in keys
    }


def _aggregate_relevance(rows):
    out = {"n": 0, "exact": 0, "partial": 0, "irrelevant": 0,
           "by_source": {}}
    for r in rows:
        rel = r.get("relevance", {})
        if not rel:
            continue
        for k in ("n", "exact", "partial", "irrelevant"):
            out[k] += rel.get(k, 0)
        for src, s in rel.get("by_source", {}).items():
            bs = out["by_source"].setdefault(src, {"n": 0, "exact": 0,
                                                   "partial": 0,
                                                   "irrelevant": 0})
            for k in ("n", "exact", "partial", "irrelevant"):
                bs[k] += s.get(k, 0)
    return out


def exact_novel_discovery(grid):
    """(novel_single probes, novel_single exact hits) from a trace grid."""
    trace = grid.get("trace_agg", {}) if isinstance(grid, dict) else {}
    rel = grid.get("relevance_agg", {}) if isinstance(grid, dict) else {}
    ns = trace.get("source_novel_single", 0)
    exact = rel.get("by_source", {}).get("novel_single", {}).get("exact", 0)
    return ns, exact


# ---------------------------------------------------------------------------
# Repeated learning (A/B/A)
# ---------------------------------------------------------------------------
def run_repeated_learning(agent, compat_pool, conflict_pool, seed,
                          base, max_steps=80):
    def train(pool, n, tag):
        stream = make_episodes(pool, tag, seed, n, max_steps=max_steps)
        succ = 0
        for spec in stream:
            rec = run_one_episode(agent, spec)
            succ += 1 if rec["success"] else 0
        return succ / max(1, len(stream))

    def agent_kwargs_of(agent):
        kw = {}
        if isinstance(agent.memory, CognitiveMemoryV02):
            kw["memory_factory"] = CognitiveMemoryV02
            kw["_memory_kwargs"] = {
                "protection_min_active_support":
                    agent.memory.protection_min_active_support,
                "protection_max_failures":
                    agent.memory.protection_max_failures,
                "protection_max_reuse_per_door":
                    agent.memory.protection_max_reuse_per_door,
                "protection_allow_trusted_transfer":
                    agent.memory.protection_allow_trusted_transfer,
            }
        if agent.memory.episode_scoped:
            kw["memory_episode_scoped"] = True
            kw["memory_min_verified_epochs_for_transfer"] = \
                agent.memory.min_verified_epochs_for_transfer
        if agent.probe_reset_per_episode:
            kw["probe_reset_per_episode"] = True
        return kw

    def eval_frozen(pool, n_configs, seed_i):
        g = eval_grid(agent.save_state(), pool, max_steps, n_configs,
                      base["focus_eps_per_cfg"], (seed_i,), seed,
                      frozen=True, agent_kwargs=agent_kwargs_of(agent))
        return g["summary"]

    results = {}
    train(compat_pool, base["rlearn_phase_a"], "compat")
    results["stage_A1"] = eval_frozen(compat_pool, base["focus_n"], 0)
    train(conflict_pool, base["rlearn_phase_b"], "conflict")
    results["stage_B"] = eval_frozen(conflict_pool, base["conflict_n"], 0)
    train(compat_pool, base["rlearn_phase_a2"], "compat")
    results["stage_A2"] = eval_frozen(compat_pool, base["focus_n"], 0)
    results["train_episodes"] = (base["rlearn_phase_a"]
                                 + base["rlearn_phase_b"]
                                 + base["rlearn_phase_a2"])
    results["memory"] = memory_tracking_v07(agent)
    return results


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_v02_experiment(splits=None, base=None, variants=None,
                       results_dir=None, verbose=True, max_steps=80,
                       collect_trace=True, train_seeds=(20261107, 20261108),
                       fresh_cache_path=None, run_repeated=True):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    base = dict(DEFAULT_BASE if base is None else base)
    results_dir = results_dir or os.path.join(
        ROOT, "results", "academy_v01", "memory_cognitive_v02")
    os.makedirs(results_dir, exist_ok=True)
    variants = variants or list(VARIANTS_V02)

    summary = {
        "run_id": "memory-cognitive-v02",
        "episodes": base["train_episodes"],
        "train_seeds": list(train_seeds),
        "eval_seed": EVAL_SEED,
        "conflict_seed": CONFLICT_SEED,
        "noisy_seed": NOISY_SEED,
        "ood_seed": OOD_SEED,
        "base": base,
        "variants": variants,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
        "comparison_convention": (
            "candidate - baseline (positive = candidate better)"),
        "preregistration": {
            "note": "Primary baseline is epoch_reset (v0.7 + probe "
                    "reset). Eval seeds reuse the v0.6/v0.7 base "
                    "20261108 for comparability; OOD uses +6; "
                    "repeated-learning focus 20261107+47; retention "
                    "substrate 20261107+31. Mechanism ablations: "
                    "v02_gate_only (cap off), v02_cap_only (gate off).",
            "controls": ["fresh", "original", "memory_wiped", "epoch",
                         "epoch_reset", "v02", "v02_gate_only",
                         "v02_cap_only"],
        },
        "conditions_detail": {},
        "repeated_learning": {},
        "comparisons": {},
        "fresh_control": {},
    }

    conflict_pool = build_conflict_pool(splits, CONFLICT_SEED,
                                        base["conflict_n"])
    compat_pool = splits["test"]
    ood_pool = splits["ood"]
    compat_focus, conflict_focus = build_focus_pools(splits, FOCUS_SEED)
    substrate = pick_configs(splits["train"], base["substrate_n"],
                             RETENTION_SEED)

    train_keys = {c.key for c in splits["train"]}
    assert not ({c.key for c in conflict_pool} & train_keys)
    assert not ({c.key for c in conflict_focus} & train_keys)

    def compute_fresh():
        def cell(pool, n, eps, seeds, seed, frozen, q=0.0, tg="eval"):
            return eval_grid(None, pool, max_steps, n, eps, seeds, seed,
                             frozen=frozen, collect_trace=collect_trace,
                             noise_q=q)
        ff = cell(compat_pool, base["test_grid_n"], base["test_eps_per_cfg"],
                  (0, 1), EVAL_SEED, True)
        fa = cell(compat_pool, base["test_grid_n"], base["test_eps_per_cfg"],
                  (0, 1), EVAL_SEED, False)
        cf = cell(conflict_pool, base["conflict_grid_n"],
                  base["conflict_eps_per_cfg"], (0, 1), CONFLICT_SEED, True)
        ca = cell(conflict_pool, base["conflict_grid_n"],
                  base["conflict_eps_per_cfg"], (0, 1), CONFLICT_SEED, False)
        nf = cell(compat_pool, base["noisy_grid_n"],
                  base["noisy_eps_per_cfg"], (0,), NOISY_SEED, True, q=0.2)
        na = cell(compat_pool, base["noisy_grid_n"],
                  base["noisy_eps_per_cfg"], (0,), NOISY_SEED, False, q=0.2)
        of_ = cell(ood_pool, base["ood_grid_n"], base["ood_eps_per_cfg"],
                   (0,), OOD_SEED, True)
        oa = cell(ood_pool, base["ood_grid_n"], base["ood_eps_per_cfg"],
                  (0,), OOD_SEED, False)
        return {
            "classic_frozen": ff["summary"],
            "classic_frozen_per_config": ff["per_config_list"],
            "classic_frozen_trace": ff.get("trace_agg"),
            "classic_frozen_relevance": ff.get("relevance_agg"),
            "classic_adapt": fa["summary"],
            "classic_adapt_per_config": fa["per_config_list"],
            "conflict_frozen": cf["summary"],
            "conflict_frozen_per_config": cf["per_config_list"],
            "conflict_adapt": ca["summary"],
            "conflict_adapt_per_config": ca["per_config_list"],
            "noisy_frozen": nf["summary"],
            "noisy_frozen_per_config": nf["per_config_list"],
            "noisy_adapt": na["summary"],
            "noisy_adapt_per_config": na["per_config_list"],
            "ood_frozen": of_["summary"],
            "ood_frozen_per_config": of_["per_config_list"],
            "ood_adapt": oa["summary"],
            "ood_adapt_per_config": oa["per_config_list"],
        }

    fresh_fp = (len(conflict_pool),
                tuple(sorted(tuple(_canonical_key(c)) for c in conflict_pool)),
                base["conflict_grid_n"], base["conflict_eps_per_cfg"],
                base["test_grid_n"], base["test_eps_per_cfg"],
                base["ood_grid_n"], base["ood_eps_per_cfg"])
    fresh_control = None
    if fresh_cache_path and os.path.exists(fresh_cache_path):
        with open(fresh_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("pool_fingerprint") == fresh_fp:
            fresh_control = cached
    if fresh_control is None:
        fresh_control = compute_fresh()
        fresh_control["pool_fingerprint"] = fresh_fp
        if fresh_cache_path:
            os.makedirs(os.path.dirname(fresh_cache_path), exist_ok=True)
            tmp = fresh_cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(fresh_control, f, indent=1, default=str)
            os.replace(tmp, fresh_cache_path)
    summary["fresh_control"] = fresh_control
    if verbose:
        f_ = fresh_control
        print("fresh: classic=%.3f conflict=%.3f noisy=%.3f ood=%.3f" % (
            f_["classic_frozen"]["config_mean_success"],
            f_["conflict_frozen"]["config_mean_success"],
            f_["noisy_frozen"]["config_mean_success"],
            f_["ood_frozen"]["config_mean_success"]))

    for name in variants:
        raw = VARIANTS_V02[name]
        per_seed = []
        eval_kwargs = {}
        for ts in train_seeds:
            agent_kwargs, hooks = split_variant_raw(raw)
            eval_kwargs = dict(agent_kwargs)
            if "_memory_kwargs" in hooks:
                eval_kwargs["_memory_kwargs"] = hooks["_memory_kwargs"]
            agent = V02VariantAgent(seed=0, **agent_kwargs, **hooks)
            stream = make_stream(splits, base["train_episodes"], ts,
                                 max_steps)
            if name == "fresh":
                stream = []
            stream = list(stream)
            half = len(stream) // 2
            t0 = time.perf_counter()
            for spec in stream[:half]:
                run_one_episode(agent, spec)
            mid_state = None if name == "fresh" else agent.save_state()
            for spec in stream[half:]:
                run_one_episode(agent, spec)
            wall = time.perf_counter() - t0
            if getattr(agent, "_wipe", False):
                agent.maybe_wipe()
                assert len(agent.memory.hypotheses) == 0
            state = None if name == "fresh" else agent.save_state()

            classic_frozen = eval_grid(state, compat_pool, max_steps,
                                       base["test_grid_n"],
                                       base["test_eps_per_cfg"],
                                       (0, 1), EVAL_SEED, frozen=True,
                                       agent_kwargs=eval_kwargs,
                                       collect_trace=collect_trace)
            classic_adapt = eval_grid(state, compat_pool, max_steps,
                                      base["test_grid_n"],
                                      base["test_eps_per_cfg"],
                                      (0, 1), EVAL_SEED, frozen=False,
                                      agent_kwargs=eval_kwargs,
                                      collect_trace=False)
            conflict_frozen = eval_grid(state, conflict_pool, max_steps,
                                        base["conflict_grid_n"],
                                        base["conflict_eps_per_cfg"],
                                        (0, 1), CONFLICT_SEED, frozen=True,
                                        agent_kwargs=eval_kwargs,
                                        collect_trace=collect_trace)
            conflict_adapt = eval_grid(state, conflict_pool, max_steps,
                                       base["conflict_grid_n"],
                                       base["conflict_eps_per_cfg"],
                                       (0, 1), CONFLICT_SEED, frozen=False,
                                       agent_kwargs=eval_kwargs,
                                       collect_trace=False)
            noisy_frozen = eval_grid(state, compat_pool, max_steps,
                                     base["noisy_grid_n"],
                                     base["noisy_eps_per_cfg"],
                                     (0,), NOISY_SEED, frozen=True,
                                     agent_kwargs=eval_kwargs,
                                     collect_trace=collect_trace,
                                     noise_q=0.2)
            noisy_adapt = eval_grid(state, compat_pool, max_steps,
                                    base["noisy_grid_n"],
                                    base["noisy_eps_per_cfg"],
                                    (0,), NOISY_SEED, frozen=False,
                                    agent_kwargs=eval_kwargs,
                                    collect_trace=False, noise_q=0.2)
            ood_frozen = eval_grid(state, ood_pool, max_steps,
                                   base["ood_grid_n"],
                                   base["ood_eps_per_cfg"],
                                   (0,), OOD_SEED, frozen=True,
                                   agent_kwargs=eval_kwargs,
                                   collect_trace=collect_trace)
            ood_adapt = eval_grid(state, ood_pool, max_steps,
                                  base["ood_grid_n"],
                                  base["ood_eps_per_cfg"],
                                  (0,), OOD_SEED, frozen=False,
                                  agent_kwargs=eval_kwargs,
                                  collect_trace=False)

            retention = RetentionSuite(
                substrate=substrate, max_steps=max_steps,
                episodes_per_config=base["retention_eps_per_cfg"],
                seeds=(0,), label=name)
            _rf = lambda s, _k=eval_kwargs: V02VariantAgent(seed=s, **_k)
            retention.run_probe(mid_state, _rf, episode_i=0)
            retention.run_probe(state, _rf, episode_i=1)

            rlearn = None
            if run_repeated and name not in ("fresh", "memory_wiped"):
                a2 = V02VariantAgent(seed=0, **agent_kwargs, **hooks)
                rlearn = run_repeated_learning(
                    a2, compat_focus, conflict_focus, ts, base,
                    max_steps=max_steps)

            ns, ns_exact = exact_novel_discovery(classic_frozen)
            cf_ns, cf_ns_exact = exact_novel_discovery(conflict_frozen)

            per_seed.append({
                "seed": ts,
                "train_success": agent.successes / max(1, agent.episodes),
                "wall_s": round(wall, 2),
                "classic_frozen": classic_frozen["summary"],
                "classic_frozen_per_config": classic_frozen["per_config_list"],
                "classic_frozen_trace": classic_frozen.get("trace_agg"),
                "classic_frozen_relevance": classic_frozen.get("relevance_agg"),
                "classic_frozen_attribution": attribution_metrics(
                    classic_frozen["rows"]),
                "classic_adapt": classic_adapt["summary"],
                "conflict_frozen": conflict_frozen["summary"],
                "conflict_frozen_per_config": conflict_frozen["per_config_list"],
                "conflict_frozen_attribution": attribution_metrics(
                    conflict_frozen["rows"]),
                "conflict_adapt": conflict_adapt["summary"],
                "noisy_frozen": noisy_frozen["summary"],
                "noisy_frozen_per_config": noisy_frozen["per_config_list"],
                "noisy_adapt": noisy_adapt["summary"],
                "ood_frozen": ood_frozen["summary"],
                "ood_frozen_per_config": ood_frozen["per_config_list"],
                "ood_adapt": ood_adapt["summary"],
                "retention": retention.summary(),
                "memory": memory_tracking_v07(agent),
                "exact_novel_classic": [ns, ns_exact],
                "exact_novel_conflict": [cf_ns, cf_ns_exact],
                "repeated_learning": rlearn,
            })

        def mean_cfg(key):
            vecs = [p[key] for p in per_seed if p.get(key)]
            if not vecs:
                return []
            n = min(len(v) for v in vecs)
            return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]

        def mean_attr(rows):
            if not rows:
                return {}
            keys = set()
            for r in rows:
                keys.update(r)
            out = {}
            for k in keys:
                vals = [r[k] for r in rows if r.get(k) is not None]
                out[k] = (sum(vals) / len(vals)) if vals else None
            return out

        agg = {
            "train_success_mean": (sum(p["train_success"]
                                       for p in per_seed) / len(per_seed)),
            "classic_frozen_per_config": mean_cfg("classic_frozen_per_config"),
            "classic_frozen": {"config_mean_success": (sum(
                p["classic_frozen"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "classic_adapt": {"config_mean_success": (sum(
                p["classic_adapt"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "conflict_frozen_per_config": mean_cfg("conflict_frozen_per_config"),
            "conflict_frozen": {"config_mean_success": (sum(
                p["conflict_frozen"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "conflict_adapt": {"config_mean_success": (sum(
                p["conflict_adapt"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "noisy_frozen_per_config": mean_cfg("noisy_frozen_per_config"),
            "noisy_frozen": {"config_mean_success": (sum(
                p["noisy_frozen"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "noisy_adapt": {"config_mean_success": (sum(
                p["noisy_adapt"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "ood_frozen_per_config": mean_cfg("ood_frozen_per_config"),
            "ood_frozen": {"config_mean_success": (sum(
                p["ood_frozen"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "ood_adapt": {"config_mean_success": (sum(
                p["ood_adapt"]["config_mean_success"]
                for p in per_seed) / len(per_seed))},
            "retention_delta_mean": (sum(
                p["retention"]["retention_delta"] for p in per_seed)
                / len(per_seed)),
            "attribution_mean_classic": mean_attr(
                [p["classic_frozen_attribution"] for p in per_seed]),
            "exact_novel_classic": [
                sum(p["exact_novel_classic"][0] for p in per_seed),
                sum(p["exact_novel_classic"][1] for p in per_seed)],
            "exact_novel_conflict": [
                sum(p["exact_novel_conflict"][0] for p in per_seed),
                sum(p["exact_novel_conflict"][1] for p in per_seed)],
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
                  "atom_profile_size", "verified_then_contradicted",
                  "cross_epoch_verified", "verified_epochs_total",
                  "reuse_eligible", "reuse_filtered"):
            vals = [p["memory"][k] for p in per_seed
                    if p["memory"].get(k) is not None]
            if vals:
                agg["memory_mean"][k] = sum(vals) / len(vals)
        summary["conditions_detail"][name] = agg
        if run_repeated:
            rl = [p["repeated_learning"] for p in per_seed
                  if p["repeated_learning"]]
            if rl:
                stage = {}
                for s in ("stage_A1", "stage_B", "stage_A2"):
                    stage[s] = (sum(r[s]["config_mean_success"] for r in rl)
                                / len(rl))
                summary["repeated_learning"][name] = stage
        if verbose:
            print("  %-16s train=%.3f c=%.3f/%.3f x=%.3f/%.3f "
                  "n=%.3f/%.3f o=%.3f/%.3f ret=%.3f novel=%s" % (
                      name, agg["train_success_mean"],
                      agg["classic_frozen"]["config_mean_success"],
                      agg["classic_adapt"]["config_mean_success"],
                      agg["conflict_frozen"]["config_mean_success"],
                      agg["conflict_adapt"]["config_mean_success"],
                      agg["noisy_frozen"]["config_mean_success"],
                      agg["noisy_adapt"]["config_mean_success"],
                      agg["ood_frozen"]["config_mean_success"],
                      agg["ood_adapt"]["config_mean_success"],
                      agg["retention_delta_mean"],
                      agg["exact_novel_classic"]))

    if "epoch_reset" in summary["conditions_detail"]:
        ref = summary["conditions_detail"]["epoch_reset"]
        ref_label = "epoch_reset"
        for name in variants:
            if name == ref_label:
                continue
            v = summary["conditions_detail"][name]
            cmp_ = {}
            for mode in ("classic_frozen", "classic_adapt",
                         "conflict_frozen", "conflict_adapt",
                         "noisy_frozen", "noisy_adapt",
                         "ood_frozen", "ood_adapt"):
                a = v.get(f"{mode}_per_config")
                b = ref.get(f"{mode}_per_config")
                if a and b and len(a) == len(b):
                    cmp_[mode] = compare_delta(a, b, seed=EVAL_SEED)
            cmp_["train_delta"] = (v["train_success_mean"]
                                    - ref["train_success_mean"])
            summary["comparisons"][f"{name}_vs_{ref_label}"] = cmp_
        fresh_c = summary["fresh_control"]
        for mode, fkey in (("classic_frozen", "classic_frozen_per_config"),
                           ("conflict_frozen", "conflict_frozen_per_config"),
                           ("noisy_frozen", "noisy_frozen_per_config"),
                           ("ood_frozen", "ood_frozen_per_config")):
            a = ref.get(f"{mode}_per_config")
            b = fresh_c.get(fkey)
            if a and b and len(a) == len(b):
                summary["comparisons"].setdefault(
                    f"{ref_label}_vs_fresh", {})[mode] = compare_delta(
                        a, b, seed=EVAL_SEED)

    summary["finished_at"] = time.time()
    with open(os.path.join(results_dir, "memory_cognitive_v02_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print("comparisons vs epoch_reset:")
        for k, v in summary["comparisons"].items():
            if k.endswith("_vs_epoch_reset"):
                out = {}
                for m, d in v.items():
                    if isinstance(d, dict):
                        out[m] = {kk: (round(vv, 4)
                                        if isinstance(vv, float) else vv)
                                   for kk, vv in d.items()}
                    else:
                        out[m] = round(d, 4) if isinstance(d, float) else d
                print("  ", k, out)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--no-repeated", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train-seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    fresh_cache = os.path.join(ROOT, "results", "academy_v01",
                               "memory_cognitive_v02", "fresh_control.json")
    if args.quick:
        base = {
            "train_episodes": 40,
            "test_grid_n": 6, "test_eps_per_cfg": 2,
            "conflict_grid_n": 6, "conflict_eps_per_cfg": 2,
            "noisy_grid_n": 4, "noisy_eps_per_cfg": 2,
            "ood_grid_n": 4, "ood_eps_per_cfg": 2,
            "substrate_n": 4, "retention_eps_per_cfg": 1,
            "focus_n": 4, "focus_eps_per_cfg": 2,
            "conflict_n": 8,
            "rlearn_phase_a": 20, "rlearn_phase_b": 16,
            "rlearn_phase_a2": 16,
        }
        run_v02_experiment(
            variants=args.variants or ["fresh", "epoch_reset", "v02",
                                       "v02_gate_only", "v02_nocap",
                                       "v02_nogate"],
            base=base, verbose=True, train_seeds=(20261107,),
            fresh_cache_path=fresh_cache,
            run_repeated=not args.no_repeated)
    else:
        run_v02_experiment(
            variants=args.variants, verbose=True,
            train_seeds=args.train_seeds or (20261107, 20261108),
            fresh_cache_path=fresh_cache,
            run_repeated=not args.no_repeated)
