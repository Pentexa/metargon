"""Cognitive Memory v0.1 (CM v0.1) — controlled evaluation.

Background
----------
Memory v0.7 (episode-scoped evidence) improved frozen transfer vs the v0.6
baseline, but attribution still showed *negative* memory-to-action
improvement: rows where hypothesis-driven probes occurred had lower success
than rows without them. The open problem is that previously verified
hypotheses can dominate exploration in a new task (negative transfer).

CM v0.1 intervention (bounded, interpretable, off by default)
-------------------------------------------------------------
- ``cognitive_reliability``: retrieval priority is split into
  *current-context applicability* (Laplace-smoothed support proportion
  over the active evidence basis — primary) and *historical verification*
  (bounded bonus). Task boundaries reset current applicability;
  evidence is never erased.
- Repeated prediction failures (completed contradiction windows) demote
  a hypothesis below neutral exploration; fresh support re-promotes it.
- ``CognitiveMemoryV01`` wires the layer onto episode-scoped evidence.
  The agent is unchanged (used via ``memory_factory``).

Evaluation (offline, deterministic, matched seeds)
--------------------------------------------------
Controls: fresh, original, memory_wiped (memory-muted), epoch (v0.7
baseline), epoch_reset (isolates per-episode probe reset), cognitive
(proposed), cognitive_shuffled (negative control for the proposed
mechanism). Regimes measured separately: frozen zero-shot transfer,
adaptive transfer, retention, and repeated learning (A/B/A interference
protocol). Task families: compatible (held-out classic test pool),
conflicting (flipped-polarity pool built from the same observable
contexts), and noisy (same configs under observational noise q=0.2).

The environment, the student agent and the memory are unchanged; only
this harness and the memory layer are new.
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
    CONTRADICTED, VERIFIED, CognitiveMemoryV01, CompositionalMemory,
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
from academy.evren.retention import RetentionSuite, sample_substrate  # noqa: E402
from academy.learning.loop import run_one_episode  # noqa: E402
from academy.tools.memory_experiment_v06 import (  # noqa: E402
    check_frozen_integrity, compare_delta, eval_grid, memory_tracking,
)
from academy.tools.memory_experiment_v07 import (  # noqa: E402
    attribution_metrics, memory_tracking_v07,
)

# ---------------------------------------------------------------------------
# Variants (matched kwargs; underscore-prefixed keys are hooks, not agent
# kwargs).
# ---------------------------------------------------------------------------
VARIANTS_COGNITIVE = {
    "fresh": {},
    "original": {},
    "memory_wiped": {"_wipe_memory": True},
    "epoch": {"memory_episode_scoped": True,
              "memory_min_verified_epochs_for_transfer": 2},
    "epoch_reset": {"memory_episode_scoped": True,
                    "memory_min_verified_epochs_for_transfer": 2,
                    "probe_reset_per_episode": True},
    "cognitive": {"memory_episode_scoped": True,
                  "memory_min_verified_epochs_for_transfer": 2,
                  "probe_reset_per_episode": True,
                  "_memory_factory": CognitiveMemoryV01},
    "cognitive_shuffled": {"memory_episode_scoped": True,
                           "memory_min_verified_epochs_for_transfer": 2,
                           "probe_reset_per_episode": True,
                           "_memory_factory": CognitiveMemoryV01,
                           "_shuffle_memory": True},
}

DEFAULT_BASE = {
    "train_episodes": 160,
    "test_grid_n": 16, "test_eps_per_cfg": 4,
    "conflict_grid_n": 12, "conflict_eps_per_cfg": 3,
    "noisy_grid_n": 10, "noisy_eps_per_cfg": 3,
    "substrate_n": 8, "retention_eps_per_cfg": 2,
    "focus_n": 8, "focus_eps_per_cfg": 3,
    "conflict_n": 12,
    "rlearn_phase_a": 60, "rlearn_phase_b": 40,
    "rlearn_phase_a2": 40,
}

EVAL_SEED = 20261108          # same base seed as v0.6/v0.7 (comparable)
CONFLICT_SEED = EVAL_SEED + 2
NOISY_SEED = EVAL_SEED + 3
RETENTION_SEED = 20261107 + 31
FOCUS_SEED = 20261107 + 47


class CognitiveVariantAgent(MetargonAcademyAgent):
    """Agent wrapper that implements the variant hooks (memory wipe,
    antecedent shuffle, memory factory)."""

    def __init__(self, *a, **k):
        self._wipe = bool(k.pop("_wipe_memory", False))
        self._shuffle = bool(k.pop("_shuffle_memory", False))
        self._shuffle_seed = int(k.pop("_shuffle_seed", 0))
        mf = k.pop("_memory_factory", None)
        if mf is not None:
            k["memory_factory"] = mf
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

    def maybe_shuffle(self):
        """Negative control: shuffle antecedent->consequence associations
        after training, keeping counts, doors and history on the
        hypotheses. If the content of memory determines transfer, this
        must change frozen behavior; if the ranking is content-blind,
        shuffled == cognitive."""
        if not self._shuffle:
            return
        hyps = [h for h in self.memory.hypotheses if h.usable]
        rng = random.Random(self._shuffle_seed)
        antes = [h.antecedent for h in hyps]
        rng.shuffle(antes)
        for h, ante in zip(hyps, antes):
            h.antecedent = ante


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
# Conflicting task family: flipped-polarity configs in the same observable
# contexts (the negative-transfer scenario).
# ---------------------------------------------------------------------------
def _flip_rule(rule):
    """Negate every switch atom in a conjunctive rule (or disjunctive
    clause tuple). Load/empty atoms are unchanged."""
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
    """Flipped variants of the core rules for a door on a topology."""
    d1r, d2r = CORE_RULESETS[topology]
    rules = d1r if door == "D1" else d2r
    out = set()
    for r in rules:
        fr = _flip_rule(r)
        if fr != r:
            out.add(fr)
    return out


def build_conflict_pool(splits, seed=CONFLICT_SEED, target=14,
                        max_tries=800):
    """Configs where at least one door uses the flipped polarity of a
    core rule, on the same topology and with the same delay options.
    Excluded: any config present in train (isolation) or in val/test/ood.

    Candidate generation is bounded (sampled, solvability-filtered) and
    easy configs are preferred so that the family mainly tests *conflict*
    rather than difficulty.
    """
    train_keys = {c.key for c in splits["train"]}
    seen_keys = set(train_keys)
    for k in ("val", "test", "ood"):
        seen_keys |= {c.key for c in splits[k]}
    rng = random.Random(seed)
    cands = set()
    tps = list(CORE_RULESETS)
    for tp in CORE_RULESETS:
        d1f = _flipped_door_rules(tp, "D1")
        d2f = _flipped_door_rules(tp, "D2")
        # restrict to compact rules (<= 2 atoms) to keep the family
        # solvable within the episode budget
        d1f = {r for r in d1f if sum(len(c) for c in
                                      (r if isinstance(r, tuple) else (r,))) <= 2}
        d2f = {r for r in d2f if sum(len(c) for c in
                                      (r if isinstance(r, tuple) else (r,))) <= 2}
        d1c, d2c = CORE_RULESETS[tp]
        d1_opt = [r for r in d1c if len(r) <= 2] + sorted(d1f, key=str)
        d2_opt = [r for r in d2c if len(r) <= 2] + sorted(d2f, key=str)
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
    # prefer easy configs (few atoms, no delay) for a clean conflict signal
    order = {"easy": 0, "medium": 1, "hard": 2}
    pool = sorted(cands, key=lambda c: (order[difficulty_of(c).bin],
                                        c.has_delay(), str(c.key)))
    return pool[:max(4, target)]


def build_focus_pools(splits, seed=FOCUS_SEED):
    """Focused rule pair for the repeated-learning (A/B/A) protocol:
    pattern A is the two single-switch rules (S2-on opens D1, S2-off
    opens D2); pattern B is the literal polarity flip of both (S2-off
    opens D1, S2-on opens D2) on the same topologies/delays."""
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
# Noisy evaluation grid (fork of memory_experiment_v06.eval_grid with a
# noise_q parameter; identical integrity checks).
# ---------------------------------------------------------------------------
def eval_grid_noisy(agent_state, configs, max_steps, n_configs,
                    episodes_per_config, seeds, seed, frozen=False,
                    agent_kwargs=None, collect_trace=False, noise_q=0.0):
    agent_kwargs = dict(agent_kwargs or {})
    picks = sample_substrate(list(configs), n_configs, seed)
    rows = []
    for i, cfgc in enumerate(picks):
        for seed_i in seeds:
            base = seed * 100000 + i * 100 + seed_i
            stream = make_episodes([cfgc], "eval", base, episodes_per_config,
                                   max_steps=max_steps, noise_q=noise_q)
            agent = MetargonAcademyAgent(seed=base, **agent_kwargs)
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


# ---------------------------------------------------------------------------
# Repeated-learning protocol (A/B/A) — retention + recovery under
# interference.
# ---------------------------------------------------------------------------
def run_repeated_learning(agent, compat_pool, conflict_pool, seed,
                          base, max_steps=80, collect_trace=False):
    """Train on A (60 eps) -> frozen-eval A' -> train on B (40 eps) ->
    frozen-eval B -> train on A again (40 eps) -> frozen-eval A''."""
    def train(pool, n, tag):
        stream = make_episodes(pool, tag, seed, n,
                               max_steps=max_steps)
        succ = 0
        for spec in stream:
            rec = run_one_episode(agent, spec)
            succ += 1 if rec["success"] else 0
        return succ / max(1, len(stream))

    def eval_frozen(pool, n_configs, tag, seed_i):
        g = eval_grid(agent.save_state(), pool, max_steps, n_configs,
                      base["focus_eps_per_cfg"], (seed_i,), seed,
                      frozen=True, agent_kwargs=agent_kwargs_of(agent),
                      collect_trace=collect_trace)
        return g

    def agent_kwargs_of(agent):
        # reconstruct the variant kwargs for eval cells
        kw = {}
        if isinstance(agent.memory, CognitiveMemoryV01):
            kw["memory_factory"] = CognitiveMemoryV01
        if agent.memory.episode_scoped:
            kw["memory_episode_scoped"] = True
            kw["memory_min_verified_epochs_for_transfer"] = \
                agent.memory.min_verified_epochs_for_transfer
        if agent.probe_reset_per_episode:
            kw["probe_reset_per_episode"] = True
        return kw

    results = {}
    train(compat_pool, base["rlearn_phase_a"], "compat")
    results["stage_A1"] = eval_frozen(compat_pool, base["focus_n"],
                                      "compat", 0)["summary"]
    train(conflict_pool, base["rlearn_phase_b"], "conflict")
    results["stage_B"] = eval_frozen(conflict_pool, base["conflict_n"],
                                     "conflict", 0)["summary"]
    train(compat_pool, base["rlearn_phase_a2"], "compat")
    results["stage_A2"] = eval_frozen(compat_pool, base["focus_n"],
                                      "compat", 0)["summary"]
    results["train_episodes"] = (base["rlearn_phase_a"]
                                 + base["rlearn_phase_b"]
                                 + base["rlearn_phase_a2"])
    results["memory"] = memory_tracking_v07(agent)
    results["memory_cognitive"] = _cognitive_tracking(agent)
    return results


def _cognitive_tracking(agent):
    mem = agent.memory
    if not mem.cognitive_reliability:
        return {}
    rels = [mem._cognitive_reliability(h) for h in mem.hypotheses]
    return {
        "reliability_mean": (sum(rels) / len(rels)) if rels else None,
        "under_attack": sum(1 for h in mem.hypotheses
                            if h.consecutive_failures > 0),
        "streak_max": max((h.consecutive_failures
                           for h in mem.hypotheses), default=0),
        "verified_history_avg": (sum(h.verified_epoch_count
                                     for h in mem.hypotheses)
                                 / max(1, len(mem.hypotheses))),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_cognitive_experiment(splits=None, base=None, variants=None,
                             results_dir=None, verbose=True, max_steps=80,
                             collect_trace=True, train_seeds=(20261107, 20261108),
                             fresh_cache_path=None, run_repeated=True):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    base = dict(DEFAULT_BASE if base is None else base)
    results_dir = results_dir or os.path.join(
        ROOT, "results", "academy_v01", "memory_cognitive_v01")
    os.makedirs(results_dir, exist_ok=True)
    variants = variants or list(VARIANTS_COGNITIVE)

    summary = {
        "run_id": "memory-cognitive-v01",
        "episodes": base["train_episodes"],
        "train_seeds": list(train_seeds),
        "eval_seed": EVAL_SEED,
        "conflict_seed": CONFLICT_SEED,
        "noisy_seed": NOISY_SEED,
        "base": base,
        "variants": variants,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
        "comparison_convention": (
            "candidate - baseline (positive = candidate better)"),
        "preregistration": {
            "note": "Classic/conflict grids reuse the v0.6/v0.7 base "
                    "eval seed 20261108 for comparability; noisy grid uses "
                    "20261108+3; repeated-learning uses 20261107+47. "
                    "Controls: fresh, original, memory_wiped, epoch, "
                    "epoch_reset, cognitive, cognitive_shuffled.",
            "controls": ["fresh", "original", "memory_wiped", "epoch",
                         "epoch_reset", "cognitive", "cognitive_shuffled"],
        },
        "conditions_detail": {},
        "repeated_learning": {},
        "comparisons": {},
        "fresh_control": {},
    }

    conflict_pool = build_conflict_pool(splits, CONFLICT_SEED,
                                        base["conflict_n"])
    compat_pool = splits["test"]
    compat_focus, conflict_focus = build_focus_pools(splits, FOCUS_SEED)

    # isolation audit: eval pools must be disjoint from the training pool.
    # (compat_focus is intentionally drawn from training tasks — it is the
    # repeated-learning family A, and evaluation on the same rule family is
    # the standard retention probe.)
    train_keys = {c.key for c in splits["train"]}
    assert not ({c.key for c in conflict_pool} & train_keys), \
        "conflict pool leaks into train"
    assert not ({c.key for c in conflict_focus} & train_keys), \
        "conflict focus leaks into train"
    substrate = sample_substrate(splits["train"], base["substrate_n"],
                                 RETENTION_SEED)

    def compute_fresh():
        ff = eval_grid(None, compat_pool, max_steps,
                       base["test_grid_n"], base["test_eps_per_cfg"],
                       (0, 1), EVAL_SEED, frozen=True,
                       collect_trace=collect_trace)
        fa = eval_grid(None, compat_pool, max_steps,
                       base["test_grid_n"], base["test_eps_per_cfg"],
                       (0, 1), EVAL_SEED, frozen=False,
                       collect_trace=collect_trace)
        cf = eval_grid(None, conflict_pool, max_steps,
                       base["conflict_grid_n"], base["conflict_eps_per_cfg"],
                       (0, 1), CONFLICT_SEED, frozen=True,
                       collect_trace=collect_trace)
        ca = eval_grid(None, conflict_pool, max_steps,
                       base["conflict_grid_n"], base["conflict_eps_per_cfg"],
                       (0, 1), CONFLICT_SEED, frozen=False,
                       collect_trace=collect_trace)
        nf = eval_grid_noisy(None, compat_pool, max_steps,
                             base["noisy_grid_n"], base["noisy_eps_per_cfg"],
                             (0,), NOISY_SEED, frozen=True,
                             collect_trace=collect_trace, noise_q=0.2)
        na = eval_grid_noisy(None, compat_pool, max_steps,
                             base["noisy_grid_n"], base["noisy_eps_per_cfg"],
                             (0,), NOISY_SEED, frozen=False,
                             collect_trace=collect_trace, noise_q=0.2)
        return {
            "classic_frozen": ff["summary"],
            "classic_frozen_per_config": ff["per_config_list"],
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
        }

    fresh_control = None
    if fresh_cache_path and os.path.exists(fresh_cache_path):
        with open(fresh_cache_path, "r", encoding="utf-8") as f:
            fresh_control = json.load(f)
    if fresh_control is None:
        fresh_control = compute_fresh()
        if fresh_cache_path:
            os.makedirs(os.path.dirname(fresh_cache_path), exist_ok=True)
            tmp = fresh_cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(fresh_control, f, indent=1, default=str)
            os.replace(tmp, fresh_cache_path)
    summary["fresh_control"] = fresh_control
    if verbose:
        print("fresh: classic frozen=%.3f conflict frozen=%.3f "
              "noisy frozen=%.3f" % (
                  fresh_control["classic_frozen"]["config_mean_success"],
                  fresh_control["conflict_frozen"]["config_mean_success"],
                  fresh_control["noisy_frozen"]["config_mean_success"]))

    for name in variants:
        raw = VARIANTS_COGNITIVE[name]
        kwargs = {k: v for k, v in raw.items()
                  if not k.startswith("_") and k != "_memory_factory"}
        factory = raw.get("_memory_factory")
        if factory is not None:
            kwargs["memory_factory"] = factory
        per_seed = []
        for ts in train_seeds:
            # pass the FULL variant dict (including underscore hooks) so
            # the wipe/shuffle controls actually fire
            agent = CognitiveVariantAgent(seed=0, _shuffle_seed=ts, **raw)
            stream = make_stream(splits, base["train_episodes"], ts,
                                 max_steps)
            if name == "fresh":
                stream = []
            t0 = time.perf_counter()
            for spec in stream:
                run_one_episode(agent, spec)
            wall = time.perf_counter() - t0
            if getattr(agent, "_wipe", False):
                agent.maybe_wipe()
                assert len(agent.memory.hypotheses) == 0, \
                    "memory wipe control did not fire"
            if getattr(agent, "_shuffle", False):
                pre_antes = [h.antecedent for h in agent.memory.hypotheses]
                agent.maybe_shuffle()
                # guard: the shuffle control must have permuted at least
                # one antecedent (proves the control actually ran)
                assert any(
                    h.antecedent != pre
                    for h, pre in zip(agent.memory.hypotheses, pre_antes)), \
                    "shuffle control did not fire"
            state = None if name == "fresh" else agent.save_state()

            classic_frozen = eval_grid(state, compat_pool, max_steps,
                                       base["test_grid_n"],
                                       base["test_eps_per_cfg"],
                                       (0, 1), EVAL_SEED, frozen=True,
                                       agent_kwargs=kwargs,
                                       collect_trace=collect_trace)
            classic_adapt = eval_grid(state, compat_pool, max_steps,
                                      base["test_grid_n"],
                                      base["test_eps_per_cfg"],
                                      (0, 1), EVAL_SEED, frozen=False,
                                      agent_kwargs=kwargs,
                                      collect_trace=False)
            conflict_frozen = eval_grid(state, conflict_pool, max_steps,
                                        base["conflict_grid_n"],
                                        base["conflict_eps_per_cfg"],
                                        (0, 1), CONFLICT_SEED, frozen=True,
                                        agent_kwargs=kwargs,
                                        collect_trace=collect_trace)
            conflict_adapt = eval_grid(state, conflict_pool, max_steps,
                                       base["conflict_grid_n"],
                                       base["conflict_eps_per_cfg"],
                                       (0, 1), CONFLICT_SEED, frozen=False,
                                       agent_kwargs=kwargs,
                                       collect_trace=False)
            noisy_frozen = eval_grid_noisy(state, compat_pool, max_steps,
                                           base["noisy_grid_n"],
                                           base["noisy_eps_per_cfg"],
                                           (0,), NOISY_SEED, frozen=True,
                                           agent_kwargs=kwargs,
                                           collect_trace=collect_trace,
                                           noise_q=0.2)
            noisy_adapt = eval_grid_noisy(state, compat_pool, max_steps,
                                          base["noisy_grid_n"],
                                          base["noisy_eps_per_cfg"],
                                          (0,), NOISY_SEED, frozen=False,
                                          agent_kwargs=kwargs,
                                          collect_trace=False,
                                          noise_q=0.2)

            retention = RetentionSuite(
                substrate=substrate, max_steps=max_steps,
                episodes_per_config=base["retention_eps_per_cfg"],
                seeds=(0,), label=name)
            if state is not None:
                retention.run_probe(state, lambda s, _k=kwargs:
                                    MetargonAcademyAgent(seed=s, **_k),
                                    episode_i=0)
                retention.run_probe(state, lambda s, _k=kwargs:
                                    MetargonAcademyAgent(seed=s, **_k),
                                    episode_i=1)
            else:
                retention.run_probe(None, lambda s, _k=kwargs:
                                    MetargonAcademyAgent(seed=s, **_k),
                                    episode_i=0)
                retention.run_probe(None, lambda s, _k=kwargs:
                                    MetargonAcademyAgent(seed=s, **_k),
                                    episode_i=1)

            rlearn = None
            if run_repeated and name not in ("fresh", "memory_wiped",
                                             "cognitive_shuffled"):
                a2 = CognitiveVariantAgent(seed=0, _shuffle_seed=ts, **kwargs)
                rlearn = run_repeated_learning(
                    a2, compat_focus, conflict_focus, ts, base,
                    max_steps=max_steps, collect_trace=False)

            per_seed.append({
                "seed": ts,
                "train_success": agent.successes / max(1, agent.episodes),
                "wall_s": round(wall, 2),
                "classic_frozen": classic_frozen["summary"],
                "classic_frozen_per_config": classic_frozen["per_config_list"],
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
                "retention": retention.summary(),
                "memory": memory_tracking_v07(agent),
                "memory_cognitive": _cognitive_tracking(agent),
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
            "retention_delta_mean": (sum(
                p["retention"]["retention_delta"] for p in per_seed)
                / len(per_seed)),
            "attribution_mean_classic": mean_attr(
                [p["classic_frozen_attribution"] for p in per_seed]),
            "attribution_mean_conflict": mean_attr(
                [p["conflict_frozen_attribution"] for p in per_seed]),
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
                  "cross_epoch_verified", "verified_epochs_total"):
            vals = [p["memory"][k] for p in per_seed
                    if p["memory"].get(k) is not None]
            if vals:
                agg["memory_mean"][k] = sum(vals) / len(vals)
        cog_vals = [p["memory_cognitive"] for p in per_seed
                    if p["memory_cognitive"]]
        if cog_vals:
            agg["memory_cognitive_mean"] = {
                k: (sum(v[k] for v in cog_vals if v.get(k) is not None)
                    / sum(1 for v in cog_vals if v.get(k) is not None))
                for k in cog_vals[0]
            }
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
            print("  %-17s train=%.3f classic=%.3f/%.3f "
                  "conflict=%.3f/%.3f noisy=%.3f/%.3f ret=%.3f %s" % (
                      name, agg["train_success_mean"],
                      agg["classic_frozen"]["config_mean_success"],
                      agg["classic_adapt"]["config_mean_success"],
                      agg["conflict_frozen"]["config_mean_success"],
                      agg["conflict_adapt"]["config_mean_success"],
                      agg["noisy_frozen"]["config_mean_success"],
                      agg["noisy_adapt"]["config_mean_success"],
                      agg["retention_delta_mean"],
                      {k: round(v, 3) for k, v in
                       agg.get("memory_cognitive_mean", {}).items()}))

    # comparisons (candidate - baseline), paired per-config vectors
    if "original" in summary["conditions_detail"]:
        orig = summary["conditions_detail"]["original"]
        for name in variants:
            if name == "original":
                continue
            v = summary["conditions_detail"][name]
            cmp_ = {}
            for mode in ("classic_frozen", "classic_adapt",
                         "conflict_frozen", "conflict_adapt",
                         "noisy_frozen", "noisy_adapt"):
                a = v.get(f"{mode}_per_config")
                b = orig.get(f"{mode}_per_config")
                if a and b and len(a) == len(b):
                    cmp_[mode] = compare_delta(a, b, seed=EVAL_SEED)
            cmp_["train_delta"] = (v["train_success_mean"]
                                    - orig["train_success_mean"])
            summary["comparisons"][f"{name}_vs_original"] = cmp_
        fresh_c = summary["fresh_control"]
        for name in variants:
            if name == "fresh":
                continue
            v = summary["conditions_detail"][name]
            cmp_ = {}
            for mode, fkey in (("classic_frozen", "classic_frozen_per_config"),
                               ("conflict_frozen", "conflict_frozen_per_config"),
                               ("noisy_frozen", "noisy_frozen_per_config")):
                a = v.get(f"{mode}_per_config")
                b = fresh_c.get(fkey)
                if a and b and len(a) == len(b):
                    cmp_[mode] = compare_delta(a, b, seed=EVAL_SEED)
            summary["comparisons"][f"{name}_vs_fresh"] = cmp_

    summary["finished_at"] = time.time()
    with open(os.path.join(results_dir, "memory_cognitive_v01_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print("comparisons vs original:")
        for k, v in summary["comparisons"].items():
            if k.endswith("_vs_original"):
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
                               "memory_cognitive_v01", "fresh_control.json")
    if args.quick:
        base = {
            "train_episodes": 40,
            "test_grid_n": 6, "test_eps_per_cfg": 2,
            "conflict_grid_n": 6, "conflict_eps_per_cfg": 2,
            "noisy_grid_n": 4, "noisy_eps_per_cfg": 2,
            "substrate_n": 4, "retention_eps_per_cfg": 1,
            "focus_n": 4, "focus_eps_per_cfg": 2,
            "conflict_n": 8,
            "rlearn_phase_a": 20, "rlearn_phase_b": 16,
            "rlearn_phase_a2": 16,
        }
        run_cognitive_experiment(
            variants=args.variants or ["fresh", "original", "epoch_reset",
                                       "cognitive"],
            base=base, verbose=True, train_seeds=(20261107,),
            fresh_cache_path=fresh_cache,
            run_repeated=not args.no_repeated)
    else:
        run_cognitive_experiment(
            variants=args.variants, verbose=True,
            train_seeds=args.train_seeds or (20261107, 20261108),
            fresh_cache_path=fresh_cache,
            run_repeated=not args.no_repeated)
