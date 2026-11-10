"""Memory v0.6 — truthful knowledge transfer under an evaluation-safe protocol.

v0.5 established the 3-regime protocol and produced a critical negative
result: frozen (zero-shot) transfer was absent, `memory_wiped ≡ original`
bit-identical, and the evidence bank did not help. v0.6 does two things:

1. **Diagnoses** the full memory chain (observation → hypothesis →
   verification → storage → retrieval → planning → action) with fine-
   grained per-run traces, and answers the Phase-1 questions:
   - are verified memories actually retrieved during evaluation?
   - does retrieved knowledge change selected actions?
   - is reused knowledge relevant to the new task?
   - do cross-task contradictions destroy useful knowledge?
   - does the planning system ignore retained evidence?
   - do context signatures prevent legitimate transfer?

2. **Intervenes** with small, interpretable mechanisms (each off by
   default, each independently ablatable):

   - `probe_reset_per_episode`: probe bookkeeping (`tried`,
     `pending_probe`) is per-episode. v0.5 carried the tried set from
     training into evaluation (checkpointed, never reset under the
     frozen policy), which silently blocked re-exploration of the probe
     vocabulary in a new task and prevented hypothesis-backed reuse
     (candidates in `tried` are skipped).
   - `memory_atom_transfer`: a bounded per-door atom profile is built
     from verified / contradicted hypotheses (causal structure matching
     across tasks). Probe candidates are singles and compact
     conjunctions of the most cross-task-robust atoms instead of
     verbatim (fragile) antecedent replay.
   - `probe_min_effective_confidence`: context-conditioned confidence
     gate for verbatim hypothesis reuse — in a new context with no
     local evidence, a hypothesis is not trusted verbatim.
   - `memory_bank_refresh_on_contradiction`: adaptive forgetting — a
     bank entry is removed as soon as the hypothesis that produced it
     is contradicted (the bank keeps only currently verified entries).

**Evaluation integrity (v0.6):**

- The three v0.5 regimes are retained: frozen zero-shot, within-episode
  adaptation, continual-learning retention. Fresh, original, memory-
  wiped and the v0.5 controls are mandatory.
- Evaluation cells recreate the agent with the variant's full kwargs
  (v0.5 dropped agent-level behaviour flags in eval cells) and then load
  the trained checkpoint; frozen cells disable learning *and* assert
  that the memory state (generated count, step counter) did not move.
- **Comparison-sign defect corrected**: v0.5 reported Δ = baseline −
  candidate (it called `paired_difference(orig, candidate)` while the
  function documents candidate − baseline). All v0.6 comparisons use the
  documented candidate − baseline convention: positive Δ means the
  candidate improved on the baseline. The historical v0.5 raw evidence
  is not modified; the correction is documented in `docs/MEMORY_V06.md`.
- Per-seed reporting, matched streams, pinned hash, paired bootstrap CIs
  vs fresh and vs original on per-config vectors.

All offline; deterministic given seeds.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from academy.agents.memory import CONTRADICTED, VERIFIED, CompositionalMemory
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import MechanismGrid
from academy.envs.tasks import (
    DEFAULT_SPLITS_PATH, EpisodeSpec, difficulty_of, load_splits,
    make_episodes,
)
from academy.evren.retention import RetentionSuite, sample_substrate
from academy.learning.loop import run_one_episode
from academy.eval.metrics import paired_difference

# ---------------------------------------------------------------------------
# Variant definitions.
# v0.5 controls are reproduced exactly (same kwargs). v0.6 interventions are
# additive and each is independent (off by default in the controls).
# ---------------------------------------------------------------------------
VARIANTS = {
    "fresh": {},
    "original": {},
    "memory_wiped": {"_wipe_memory": True},
    "ctx": {"memory_context_scoped": True},
    "ctx_full": {"memory_context_scoped": True,
                 "memory_task_change_reset": True,
                 "memory_context_gate_transfer": True,
                 "probe_adaptive": True, "probe_episode_local": True,
                 "protect_episode_local": True},
    "bank": {"memory_evidence_bank_cap": 16},
    "bank_ctx": {"memory_evidence_bank_cap": 16,
                 "memory_context_scoped": True},
    "bank_ctx_full": {"memory_evidence_bank_cap": 16,
                      "memory_context_scoped": True,
                      "memory_task_change_reset": True,
                      "memory_context_gate_transfer": True,
                      "probe_adaptive": True,
                      "probe_episode_local": True,
                      "protect_episode_local": True},
    # v0.6 interventions
    "reset": {"probe_reset_per_episode": True},
    "atomtx": {"probe_reset_per_episode": True,
               "memory_atom_transfer": True,
               "probe_min_effective_confidence": 0.5},
    "atomtx_bank": {"probe_reset_per_episode": True,
                    "memory_atom_transfer": True,
                    "probe_min_effective_confidence": 0.5,
                    "memory_evidence_bank_cap": 16,
                    "memory_bank_refresh_on_contradiction": True,
                    "probe_bank_reuse": True},
}

V05_CONTROLS = ("fresh", "original", "memory_wiped", "ctx", "ctx_full",
                "bank", "bank_ctx", "bank_ctx_full")

# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------
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
    def __init__(self, *a, **k):
        self._wipe = bool(k.pop("_wipe_memory", False))
        mc = k.pop("_memory_class", None)
        super().__init__(*a, **k)
        if mc is not None:
            # build the transition-log memory in place (diagnosis mode)
            self.memory.__class__ = mc
            self.memory.transitions = []
            self.memory.verified_then_contradicted = 0

    def maybe_wipe(self):
        if self._wipe:
            self.memory.hypotheses = []
            self.memory._continuous_true = {}
            self.memory._dead = {}
            self.memory._sup_timeline = {}
            self.memory._con_timeline = {}
            self.memory.evidence_bank = []
            self.memory._atom_stats = {}
            self.memory._atom_ctxs = {}


# ---------------------------------------------------------------------------
# Memory transition log (cross-task contradiction measurement)
# ---------------------------------------------------------------------------
class TransitionLogMemory(CompositionalMemory):
    """Records hypothesis status transitions (bounded) for diagnosis."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.transitions = []          # [h_id, before, after, step]
        self.verified_then_contradicted = 0

    def _recompute_status(self, h):
        before = h.status
        super()._recompute_status(h)
        if h.status != before:
            self.transitions.append([h.h_id, before, h.status, self.step])
            if len(self.transitions) > 4000:
                self.transitions = self.transitions[-2000:]
            if before == VERIFIED and h.status == CONTRADICTED:
                self.verified_then_contradicted += 1


def _transition_memory_factory(kwargs):
    """Build a memory with the transition log (diagnosis mode)."""
    def make(seed, **extra):
        agent = MetargonAcademyAgent(seed=seed, **kwargs, **extra)
        agent.memory.__class__ = TransitionLogMemory
        return agent
    return make


# ---------------------------------------------------------------------------
# Frozen-integrity check
# ---------------------------------------------------------------------------
def check_frozen_integrity(agent, pre_generated, pre_step, pre_steps, label):
    """Assert that no learning/memory update happened during a frozen cell."""
    if agent.memory.generated != pre_generated:
        raise AssertionError(
            f"{label}: frozen policy leaked learning "
            f"(generated {pre_generated} -> {agent.memory.generated})")
    if agent.memory.step != pre_step:
        raise AssertionError(
            f"{label}: frozen policy advanced memory step "
            f"({pre_step} -> {agent.memory.step})")
    if agent.steps != pre_steps:
        raise AssertionError(
            f"{label}: frozen policy advanced agent.steps "
            f"({pre_steps} -> {agent.steps})")


# ---------------------------------------------------------------------------
# Probe relevance (diagnosis): does reused knowledge match the true rule?
# ---------------------------------------------------------------------------
def rule_atoms(rule) -> frozenset:
    """All atoms appearing in a rule (union over clauses)."""
    if isinstance(rule, tuple):
        atoms = set()
        for c in rule:
            atoms |= set(c)
        return frozenset(atoms)
    return frozenset(rule)


def rule_clauses(rule) -> list:
    if isinstance(rule, tuple):
        return list(rule)
    return [rule]


def probe_relevance(ante, rule) -> dict:
    """(exact, partial, irrelevant) for one planned probe vs the true rule."""
    clauses = rule_clauses(rule)
    if any(ante == c for c in clauses):
        return {"exact": 1, "partial": 0, "irrelevant": 0}
    atoms = rule_atoms(rule)
    if (ante & atoms):
        return {"exact": 0, "partial": 1, "irrelevant": 0}
    return {"exact": 0, "partial": 0, "irrelevant": 1}


def summarize_probe_relevance(planned, config) -> dict:
    """planned: list of [door, ante_list, src]; config: EnvConfig."""
    out = {"n": 0, "exact": 0, "partial": 0, "irrelevant": 0,
           "by_source": {}}
    for door, ante_list, src in planned:
        ante = frozenset(map(tuple, ante_list))
        rule = config.rule_for(door)
        r = probe_relevance(ante, rule)
        out["n"] += 1
        for k in ("exact", "partial", "irrelevant"):
            out[k] += r[k]
        bs = out["by_source"].setdefault(src, {"n": 0, "exact": 0,
                                               "partial": 0,
                                               "irrelevant": 0})
        bs["n"] += 1
        for k in ("exact", "partial", "irrelevant"):
            bs[k] += r[k]
    return out


# ---------------------------------------------------------------------------
# Evaluation grid (v0.6)
# ---------------------------------------------------------------------------
def eval_grid(agent_state, configs, max_steps, n_configs, episodes_per_config,
              seeds, seed, frozen=False, wiped=False, agent_kwargs=None,
              collect_trace=False):
    """Independent per-config grid; optionally frozen (no learning).

    ``agent_kwargs`` are passed to every eval agent constructor AND the
    trained checkpoint is loaded afterwards — the variant's full
    behaviour is preserved in evaluation (v0.5 dropped agent-level flags
    here). When ``frozen``, memory updates are blocked by the frozen
    policy and the memory state is asserted unchanged.
    """
    agent_kwargs = dict(agent_kwargs or {})
    picks = sample_substrate(list(configs), n_configs, seed)
    rows = []
    for i, cfgc in enumerate(picks):
        for seed_i in seeds:
            base = seed * 100000 + i * 100 + seed_i
            stream = make_episodes([cfgc], "eval", base, episodes_per_config,
                                   max_steps=max_steps)
            agent = MetargonAcademyAgent(seed=base, **agent_kwargs)
            pre_generated = pre_step = pre_steps = None
            if agent_state is not None:
                agent.load_state(agent_state)
                pre_generated = agent.memory.generated
                pre_step = agent.memory.step
                pre_steps = agent.steps
            if frozen:
                agent.frozen_policy = True
            if wiped:
                agent.memory.hypotheses = []
                agent.memory.evidence_bank = []
            if collect_trace:
                agent._collect_probe_log = True
            succ, steps_total = 0, 0
            planned_all = []
            for spec in stream:
                env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
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
                    planned_all.extend(
                        agent.trace.get("planned_probes", []))
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
                row["relevance"] = summarize_probe_relevance(
                    planned_all, cfgc)
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
    out = {
        "summary": summary,
        "per_config_list": rates,
        "rows": rows,
    }
    if collect_trace:
        out["trace_agg"] = aggregate_trace(rows)
        out["relevance_agg"] = aggregate_relevance(rows)
    return out


def aggregate_trace(rows):
    keys = (
        "source_hypothesis", "source_bank", "source_atom_single",
        "source_atom_pair", "source_novel_single",
        "source_conjunctive_probe", "source_conjunction", "source_other",
        "skipped_tried", "skipped_infeasible",
        "probes_evaluated", "plans_used",
        "random_fallback_steps", "door_open_after_probe",
        "door_open_after_hypothesis", "door_open_after_bank",
        "door_open_after_atom_single", "door_open_after_atom_pair",
        "door_open_after_novel_single", "door_open_after_conjunction",
        "door_open_after_other",
        "query_calls", "query_useful",
        "hypothesis_candidates", "hypothesis_skipped_lowconf",
    )
    return {
        k: sum(r.get("trace", {}).get(k, 0) for r in rows)
        for k in keys
    }


def aggregate_relevance(rows):
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


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------
def memory_tracking(agent):
    mem = agent.memory
    status = {}
    for h in mem.hypotheses:
        status[h.status] = status.get(h.status, 0) + 1
    ctx_specific = sum(
        1 for h in mem.hypotheses if h.sup_by_ctx or h.con_by_ctx)
    return {
        "size": len(mem.hypotheses),
        "status_counts": status,
        "generated": mem.generated,
        "evictions": mem.evictions,
        "context_specific": ctx_specific,
        "bank_size": len(mem.evidence_bank),
        "bank_retrievals": mem.bank_retrievals,
        "atom_profile_size": sum(len(s) for s in mem._atom_stats.values()),
        "verified_then_contradicted": getattr(
            mem, "verified_then_contradicted", None),
    }


# ---------------------------------------------------------------------------
# Paired comparisons (corrected sign convention)
# ---------------------------------------------------------------------------
def compare_delta(candidate_vec, baseline_vec, seed):
    """candidate − baseline on matched per-config vectors.

    Positive mean ⇒ candidate improved on baseline. This is the
    documented `paired_difference` convention; v0.5 passed
    (baseline, candidate) which flipped the sign.
    """
    return paired_difference(candidate_vec, baseline_vec, seed=seed)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def _default_base():
    return {
        "test_grid_n": 16, "eval_eps_per_cfg": 4,
        "ood_probe_n": 12, "ood_eps_per_seed": 3,
        "substrate_n": 8, "retention_eps_per_cfg": 2,
        "continual_n": 6, "continual_eps_per_cfg": 3,
    }


def run_v06_experiment(splits=None, episodes=160, train_seeds=(20261107, 20261108),
                       max_steps=80, variants=None, results_dir=None,
                       verbose=True, append_path=None, collect_trace=True,
                       base=None, eval_seed=20261108, fresh_cache_path=None):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    base = dict(_default_base() if base is None else base)
    results_dir = results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "results", "academy_v01", "memory_v06")
    os.makedirs(results_dir, exist_ok=True)
    variants = variants or list(VARIANTS)

    summary = {
        "run_id": "memory-v06",
        "episodes": episodes,
        "train_seeds": list(train_seeds),
        "eval_seed": eval_seed,
        "variants": variants,
        "base": base,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
        "comparison_convention": (
            "candidate - baseline (positive = candidate better); "
            "v0.5 defect corrected"),
        "conditions_detail": {},
        "comparisons": {},
        "fresh_control": {},
        "diagnosis": {},
    }

    def compute_fresh():
        ff = eval_grid(None, splits["test"], max_steps,
                       base["test_grid_n"], base["eval_eps_per_cfg"],
                       (0, 1), eval_seed, frozen=True,
                       collect_trace=collect_trace)
        fa = eval_grid(None, splits["test"], max_steps,
                       base["test_grid_n"], base["eval_eps_per_cfg"],
                       (0, 1), eval_seed, frozen=False,
                       collect_trace=collect_trace)
        return {
            "classic_frozen": ff["summary"],
            "classic_frozen_per_config": ff["per_config_list"],
            "classic_adapt": fa["summary"],
            "classic_adapt_per_config": fa["per_config_list"],
            "classic_frozen_trace": ff.get("trace_agg"),
            "classic_frozen_relevance": ff.get("relevance_agg"),
            "classic_adapt_trace": fa.get("trace_agg"),
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
    fresh_frozen = {"summary": fresh_control["classic_frozen"],
                    "per_config_list": fresh_control["classic_frozen_per_config"]}
    fresh_adapt = {"summary": fresh_control["classic_adapt"],
                   "per_config_list": fresh_control["classic_adapt_per_config"]}
    if verbose:
        print(f"fresh: frozen={fresh_frozen['summary']['config_mean_success']:.3f} "
              f"adapt={fresh_adapt['summary']['config_mean_success']:.3f}")

    substrate = sample_substrate(splits["train"], base["substrate_n"],
                                 20261107 + 31)

    for name in variants:
        kwargs = {k: v for k, v in VARIANTS[name].items()
                  if not k.startswith("_")}
        per_seed = []
        for ts in train_seeds:
            agent = WipeHookAgent(seed=0, _memory_class=TransitionLogMemory,
                                  **kwargs)
            stream = make_stream(splits, episodes, ts, max_steps)
            if name == "fresh":
                stream = []
            t0 = time.perf_counter()
            for spec in stream:
                run_one_episode(agent, spec)
            wall = time.perf_counter() - t0
            if getattr(agent, "_wipe", False):
                agent.maybe_wipe()
            state = None if name == "fresh" else agent.save_state()

            # 1) frozen zero-shot
            frozen = eval_grid(state, splits["test"], max_steps,
                               base["test_grid_n"], base["eval_eps_per_cfg"],
                               (0, 1), eval_seed, frozen=True,
                               agent_kwargs=kwargs,
                               collect_trace=collect_trace)
            # 2) within-episode adaptation
            adapt = eval_grid(state, splits["test"], max_steps,
                              base["test_grid_n"], base["eval_eps_per_cfg"],
                              (0, 1), eval_seed, frozen=False,
                              agent_kwargs=kwargs,
                              collect_trace=collect_trace)
            # OOD frozen probe
            ood_frozen = eval_grid(state, splits["ood"], max_steps,
                                   base["ood_probe_n"],
                                   base["ood_eps_per_seed"], (0,),
                                   eval_seed + 2, frozen=True,
                                   agent_kwargs=kwargs,
                                   collect_trace=collect_trace)
            # 3) continual: keep learning on eval configs then measure
            cont = eval_grid(state, splits["ood"][:6], max_steps,
                             base["continual_n"],
                             base["continual_eps_per_cfg"],
                             (0,), eval_seed + 5, frozen=False,
                             agent_kwargs=kwargs,
                             collect_trace=collect_trace)

            retention = RetentionSuite(
                substrate=substrate, max_steps=max_steps,
                episodes_per_config=base["retention_eps_per_cfg"],
                seeds=(0,), label=name)
            retention.run_probe(
                state,
                lambda s, _k=kwargs: MetargonAcademyAgent(seed=s, **_k),
                episode_i=0)
            retention.run_probe(
                state,
                lambda s, _k=kwargs: MetargonAcademyAgent(seed=s, **_k),
                episode_i=1)

            per_seed.append({
                "seed": ts,
                "train_success": agent.successes / max(1, agent.episodes),
                "wall_s": round(wall, 2),
                "classic_frozen": frozen["summary"],
                "classic_frozen_per_config": frozen["per_config_list"],
                "classic_frozen_trace": frozen.get("trace_agg"),
                "classic_frozen_relevance": frozen.get("relevance_agg"),
                "classic_adapt": adapt["summary"],
                "classic_adapt_per_config": adapt["per_config_list"],
                "ood_frozen": ood_frozen["summary"],
                "continual_ood": cont["summary"],
                "retention": retention.summary(),
                "memory": memory_tracking(agent),
            })

        def mean_cfg(key):
            vecs = [p[key] for p in per_seed if p.get(key)]
            if not vecs:
                return []
            n = min(len(v) for v in vecs)
            return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]

        def mean_trace(key):
            rows = [p.get(key) for p in per_seed if p.get(key)]
            if not rows:
                return {}

            def deep_mean(acc, n):
                out = {}
                for k, v in acc.items():
                    if isinstance(v, dict):
                        out[k] = deep_mean(v, n)
                    else:
                        out[k] = v / n
                return out

            acc = {}
            for r in rows:
                for k, v in r.items():
                    if k not in acc:
                        acc[k] = dict(v) if isinstance(v, dict) else 0
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            if k2 not in acc[k]:
                                acc[k][k2] = dict(v2) if isinstance(v2, dict) else 0
                            if isinstance(v2, dict):
                                for k3, v3 in v2.items():
                                    acc[k][k2][k3] = acc[k][k2].get(k3, 0) + v3
                            else:
                                acc[k][k2] += v2
                    else:
                        acc[k] += v
            return deep_mean(acc, len(rows))

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
            "continual_ood": {
                "config_mean_success": (sum(
                    p["continual_ood"]["config_mean_success"]
                    for p in per_seed) / len(per_seed)),
            },
            "retention_delta_mean": (sum(
                p["retention"]["retention_delta"] for p in per_seed)
                / len(per_seed)),
            "memory_mean": {},
            "frozen_trace_mean": mean_trace("classic_frozen_trace"),
            "frozen_relevance_mean": mean_trace("classic_frozen_relevance"),
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
        if append_path is not None:
            import io
            with io.open(append_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"variant": name, "per": agg},
                                   default=str) + "\n")
        if verbose:
            print(f"  {name:13s} train={agg['train_success_mean']:.3f} "
                  f"frozen={agg['classic_frozen']['config_mean_success']:.3f} "
                  f"adapt={agg['classic_adapt']['config_mean_success']:.3f} "
                  f"ood={agg['ood_frozen']['config_mean_success']:.3f} "
                  f"cont={agg['continual_ood']['config_mean_success']:.3f} "
                  f"mem={ {k: round(v,1) for k,v in agg['memory_mean'].items()} }")

    # ---- paired comparisons (candidate − baseline convention) ----------
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
        # vs fresh control (frozen only)
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

    # ---- diagnosis summary (Phase 1) ----------------------------------
    summary["diagnosis"] = build_diagnosis(summary)
    summary["finished_at"] = time.time()
    with open(os.path.join(results_dir, "memory_v06_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print("  comparisons:")
        for k, v in summary["comparisons"].items():
            print("   ", k, v)
    return summary


# ---------------------------------------------------------------------------
# Phase-1 diagnosis synthesis
# ---------------------------------------------------------------------------
def build_diagnosis(summary):
    """Answer the Phase-1 questions from the measured traces."""
    fresh = summary["fresh_control"]
    fresh_trace = fresh.get("classic_frozen_trace") or {}
    out = {
        "A_retrieval": {},
        "B_memory_influences_actions": {},
        "C_relevance": {},
        "D_cross_task_contradiction": {},
        "E_planning_ignores_evidence": {},
        "F_context_signatures_block_transfer": {},
    }
    orig = summary["conditions_detail"].get("original", {})
    orig_trace = orig.get("frozen_trace_mean") or {}
    # A. retrieval
    out["A_retrieval"] = {
        "fresh_query_calls": fresh_trace.get("query_calls", 0),
        "original_query_calls": orig_trace.get("query_calls", 0),
        "original_hypothesis_candidates":
            orig_trace.get("hypothesis_candidates", 0),
        "original_hypothesis_probes_planned":
            orig_trace.get("source_hypothesis", 0),
        "original_bank_retrievals":
            orig.get("memory_mean", {}).get("bank_retrievals", 0),
        "original_hypothesis_skipped_lowconf":
            orig_trace.get("hypothesis_skipped_lowconf", 0),
    }
    # B. action influence
    out["B_memory_influences_actions"] = {
        "fresh_plans_used": fresh_trace.get("plans_used", 0),
        "original_plans_used": orig_trace.get("plans_used", 0),
        "fresh_random_fallback": fresh_trace.get("random_fallback_steps", 0),
        "original_random_fallback":
            orig_trace.get("random_fallback_steps", 0),
        "original_source_distribution": {
            k: orig_trace.get(k, 0)
            for k in ("source_hypothesis", "source_bank",
                      "source_atom_single", "source_atom_pair",
                      "source_novel_single", "source_conjunctive_probe",
                      "source_other")
        },
    }
    # C. relevance (probes actually planned during frozen eval)
    orig_rel = orig.get("frozen_relevance_mean") or {}
    fresh_rel = fresh.get("classic_frozen_relevance") or {}
    out["C_relevance"] = {
        "fresh_probes": fresh_rel.get("n", 0),
        "fresh_exact": fresh_rel.get("exact", 0),
        "fresh_partial": fresh_rel.get("partial", 0),
        "fresh_irrelevant": fresh_rel.get("irrelevant", 0),
        "original_probes": orig_rel.get("n", 0),
        "original_exact": orig_rel.get("exact", 0),
        "original_partial": orig_rel.get("partial", 0),
        "original_irrelevant": orig_rel.get("irrelevant", 0),
        "original_by_source": orig_rel.get("by_source", {}),
    }
    # D. cross-task contradictions
    mem = orig.get("memory_mean", {})
    out["D_cross_task_contradiction"] = {
        "verified": mem.get(VERIFIED, 0),
        "contradicted": mem.get(CONTRADICTED, 0),
        "verified_then_contradicted":
            mem.get("verified_then_contradicted", 0),
    }
    # E. planning ignores retained evidence?
    out["E_planning_ignores_evidence"] = {
        "skipped_tried_frozen": orig_trace.get("skipped_tried", 0),
        "skipped_infeasible_frozen":
            orig_trace.get("skipped_infeasible", 0),
        "plans_from_hypothesis":
            orig_trace.get("source_hypothesis", 0),
        "door_open_after_hypothesis":
            orig_trace.get("door_open_after_hypothesis", 0),
    }
    # F. context signatures
    ctx = summary["conditions_detail"].get("ctx", {})
    ctx_full = summary["conditions_detail"].get("ctx_full", {})
    out["F_context_signatures_block_transfer"] = {
        "original_frozen": orig.get("classic_frozen", {}).get(
            "config_mean_success", 0),
        "ctx_frozen": ctx.get("classic_frozen", {}).get(
            "config_mean_success", 0),
        "ctx_full_frozen": ctx_full.get("classic_frozen", {}).get(
            "config_mean_success", 0),
        "ctx_hypothesis_probes": (ctx.get("frozen_trace_mean") or {}).get(
            "source_hypothesis", 0),
        "ctx_full_hypothesis_probes":
            (ctx_full.get("frozen_trace_mean") or {}).get(
                "source_hypothesis", 0),
    }
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="small development run (few configs/episodes)")
    args = ap.parse_args()
    if args.quick:
        base = {
            "test_grid_n": 6, "eval_eps_per_cfg": 2,
            "ood_probe_n": 4, "ood_eps_per_seed": 2,
            "substrate_n": 4, "retention_eps_per_cfg": 1,
            "continual_n": 3, "continual_eps_per_cfg": 2,
        }
        run_v06_experiment(
            variants=args.variants or ["fresh", "original", "reset",
                                       "atomtx", "atomtx_bank"],
            episodes=60, base=base, verbose=True,
            fresh_cache_path=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "results", "academy_v01", "memory_v06",
                "fresh_control.json"))
    else:
        run_v06_experiment(
            variants=args.variants, verbose=True,
            fresh_cache_path=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                "results", "academy_v01", "memory_v06",
                "fresh_control.json"))
