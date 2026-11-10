"""Memory v0.7 — persistent causal memory: episode-scoped evidence.

Background (Audit v0.6 confirmed)
---------------------------------
- Retained memory does not currently demonstrate genuine zero-shot
  knowledge transfer.
- Most stored hypotheses become contradicted after mixed-task training;
  cross-task negative evidence destroys previously verified hypotheses.
- Verbatim memory reuse often transfers irrelevant information; the
  `novocab` planning-only baseline can outperform the trained agent.

v0.7 hypothesis
---------------
The destructive mechanism is the *global, cross-task contradiction
accumulator*: a hypothesis verified in task A is contradicted in task B
(which shares the observable context but has a different hidden causal
mechanism), and the contradiction is cumulative globally, permanently
marking the hypothesis unusable. Because the agent cannot observe task
identity, the only sound policy is to treat each episode as an evidence
epoch, scope evidence per epoch, and demote (not destroy) hypotheses in
epochs where they are contradicted while retaining the record of past
verifications for uncertainty-aware reuse in future epochs.

Intervention (bounded, interpretable, off by default)
-----------------------------------------------------
- `episode_scoped`: per-episode support/contradiction records; status is
  computed relative to the active epoch only; the consecutive-nonevent
  window resets at every task boundary (a task switch can never be read
  as latent contradiction).
- `min_verified_epochs_for_transfer`: hypotheses verified in >= K past
  epochs and untested in the active epoch are retrieved as cross-epoch
  transfer candidates (uncertainty-aware reuse).

Evaluation (preregistered, no new evaluation seeds)
---------------------------------------------------
Controls: fresh, original (v0.6 baseline), memory_wiped, atomtx
(v0.6 atom-transfer baseline), shuffled-memory (negative control), epoch
(v0.7 mechanism). Regimes: frozen zero-shot, within-episode adaptation,
continual retention, OOD frozen. Diagnostic attribution counters
(memory_actions, memory_correct, memory_incorrect, cross-epoch
retrievals) are recorded per run.

All offline; deterministic given seeds; pinned PYTHONHASHSEED.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from academy.agents.memory import CONTRADICTED, VERIFIED
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import MechanismGrid
from academy.envs.tasks import (
    DEFAULT_SPLITS_PATH, EpisodeSpec, difficulty_of, load_splits,
    make_episodes,
)
from academy.evren.retention import RetentionSuite, sample_substrate
from academy.learning.loop import run_one_episode
from academy.tools.memory_experiment_v06 import (
    check_frozen_integrity, compare_delta, eval_grid, memory_tracking,
)
from academy.eval.metrics import paired_difference

# ---------------------------------------------------------------------------
# v0.7 variants.
# v0.6 controls are reproduced with identical kwargs.
# ---------------------------------------------------------------------------
VARIANTS_V07 = {
    "fresh": {},
    "original": {},
    "memory_wiped": {"_wipe_memory": True},
    "atomtx": {"probe_reset_per_episode": True,
               "memory_atom_transfer": True,
               "probe_min_effective_confidence": 0.5},
    "shuffled": {"_shuffle_memory": True},
    "epoch": {"memory_episode_scoped": True,
              "memory_min_verified_epochs_for_transfer": 2},
}

DEFAULT_BASE = {
    "test_grid_n": 16, "eval_eps_per_cfg": 4,
    "ood_probe_n": 12, "ood_eps_per_seed": 3,
    "substrate_n": 8, "retention_eps_per_cfg": 2,
    "continual_n": 6, "continual_eps_per_cfg": 3,
}

EVAL_SEED = 20261108


class WipeHookAgent(MetargonAcademyAgent):
    def __init__(self, *a, **k):
        self._wipe = bool(k.pop("_wipe_memory", False))
        self._shuffle = bool(k.pop("_shuffle_memory", False))
        self._shuffle_seed = int(k.pop("_shuffle_seed", 0))
        super().__init__(*a, **k)

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
            self.memory._pair_stats = {}
            self.memory._door_event_count = {}
            self.memory._atom_event_support = {}
            self.memory._atom_nonevent_support = {}
            self.memory._door_nonevent_count = {}

    def maybe_shuffle(self):
        """Negative control: shuffle antecedent->consequence associations
        after training, keeping the same hypothesis count, doors, and
        support counts. If memory determines transfer, this must change
        frozen behavior; if memory is irrelevant, shuffled == original."""
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
# Memory-to-action attribution (Mission C / F)
# ---------------------------------------------------------------------------
def attribution_metrics(rows):
    """From frozen-evaluation rows (each carrying a trace), compute
    memory-to-action attribution:
    - memory_actions: plans sourced from hypotheses
    - memory_correct / memory_incorrect: door opened / not opened after a
      memory-driven probe
    - cross_epoch_transfer_retrieved
    - memory_attributable_improvement: mean success over rows where a
      memory-driven action occurred, minus mean success over rows without
    """
    with_mem = [r for r in rows
                if r.get("trace", {}).get("source_hypothesis", 0) > 0]
    without_mem = [r for r in rows
                   if r.get("trace", {}).get("source_hypothesis", 0) == 0]
    tot = {"memory_actions": 0, "memory_correct": 0,
           "memory_incorrect": 0, "cross_epoch_transfer_retrieved": 0}
    for r in rows:
        t = r.get("trace", {})
        for k in tot:
            tot[k] += t.get(k, 0)
    def mean_succ(rs):
        return (sum(r["success_rate"] for r in rs) / len(rs)) if rs else None
    return {
        **tot,
        "rows_with_memory": len(with_mem),
        "rows_without_memory": len(without_mem),
        "success_with_memory": mean_succ(with_mem),
        "success_without_memory": mean_succ(without_mem),
        "memory_attributable_improvement": (
            (mean_succ(with_mem) or 0.0) - (mean_succ(without_mem) or 0.0)
            if with_mem and without_mem else None),
    }


# ---------------------------------------------------------------------------
# Memory tracking (v0.7 additions)
# ---------------------------------------------------------------------------
def memory_tracking_v07(agent):
    d = memory_tracking(agent)
    d["episode_scoped"] = agent.memory.episode_scoped
    d["current_epoch"] = agent.memory._current_epoch
    d["cross_epoch_verified"] = sum(
        1 for h in agent.memory.hypotheses
        if h.verified_epoch_count
        >= agent.memory.min_verified_epochs_for_transfer)
    d["verified_epochs_total"] = sum(
        h.verified_epoch_count for h in agent.memory.hypotheses)
    return d


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_v07_experiment(splits=None, episodes=160,
                       train_seeds=(20261107, 20261108), max_steps=80,
                       variants=None, results_dir=None, verbose=True,
                       append_path=None, collect_trace=True, base=None,
                       eval_seed=EVAL_SEED, fresh_cache_path=None):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    base = dict(DEFAULT_BASE if base is None else base)
    results_dir = results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "results", "academy_v01", "memory_v07")
    os.makedirs(results_dir, exist_ok=True)
    variants = variants or list(VARIANTS_V07)

    summary = {
        "run_id": "memory-v07",
        "episodes": episodes,
        "train_seeds": list(train_seeds),
        "eval_seed": eval_seed,
        "base": base,
        "variants": variants,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
        "comparison_convention": (
            "candidate - baseline (positive = candidate better)"),
        "preregistration": {
            "note": "Evaluation seeds and grids are those preregistered in "
                    "Memory v0.6 (eval seed 20261108, 16-config classic "
                    "grid, 12-config OOD, 8-config retention substrate); "
                    "no new evaluation seeds were introduced for v0.7.",
            "controls": ["fresh", "original", "memory_wiped", "atomtx",
                         "shuffled", "epoch"],
        },
        "conditions_detail": {},
        "comparisons": {},
        "fresh_control": {},
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
            "classic_frozen_trace": ff.get("trace_agg"),
            "classic_frozen_attribution": attribution_metrics(ff["rows"]),
            "classic_adapt": fa["summary"],
            "classic_adapt_per_config": fa["per_config_list"],
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
        print("fresh: frozen=%.3f adapt=%.3f" % (
            fresh_control["classic_frozen"]["config_mean_success"],
            fresh_control["classic_adapt"]["config_mean_success"]))

    substrate = sample_substrate(splits["train"], base["substrate_n"],
                                 20261107 + 31)

    for name in variants:
        kwargs = {k: v for k, v in VARIANTS_V07[name].items()
                  if not k.startswith("_")}
        per_seed = []
        for ts in train_seeds:
            agent = WipeHookAgent(seed=0, _shuffle_seed=ts, **kwargs)
            stream = make_stream(splits, episodes, ts, max_steps)
            if name == "fresh":
                stream = []
            t0 = time.perf_counter()
            for spec in stream:
                run_one_episode(agent, spec)
            wall = time.perf_counter() - t0
            if getattr(agent, "_wipe", False):
                agent.maybe_wipe()
            if getattr(agent, "_shuffle", False):
                agent.maybe_shuffle()
            state = None if name == "fresh" else agent.save_state()

            frozen = eval_grid(state, splits["test"], max_steps,
                               base["test_grid_n"], base["eval_eps_per_cfg"],
                               (0, 1), eval_seed, frozen=True,
                               agent_kwargs=kwargs,
                               collect_trace=collect_trace)
            adapt = eval_grid(state, splits["test"], max_steps,
                              base["test_grid_n"], base["eval_eps_per_cfg"],
                              (0, 1), eval_seed, frozen=False,
                              agent_kwargs=kwargs,
                              collect_trace=collect_trace)
            ood_frozen = eval_grid(state, splits["ood"], max_steps,
                                   base["ood_probe_n"],
                                   base["ood_eps_per_seed"], (0,),
                                   eval_seed + 2, frozen=True,
                                   agent_kwargs=kwargs,
                                   collect_trace=collect_trace)
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
                "classic_frozen_attribution": attribution_metrics(
                    frozen["rows"]),
                "classic_adapt": adapt["summary"],
                "classic_adapt_per_config": adapt["per_config_list"],
                "ood_frozen": ood_frozen["summary"],
                "continual_ood": cont["summary"],
                "retention": retention.summary(),
                "memory": memory_tracking_v07(agent),
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
                if vals:
                    out[k] = sum(vals) / len(vals)
                else:
                    out[k] = None
            return out

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
            "attribution_mean": mean_attr(
                [p["classic_frozen_attribution"] for p in per_seed]),
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
        summary["conditions_detail"][name] = agg
        if append_path is not None:
            import io
            with io.open(append_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"variant": name, "per": agg},
                                   default=str) + "\n")
        if verbose:
            print("  %-12s train=%.3f frozen=%.3f adapt=%.3f ood=%.3f "
                  "cont=%.3f mem=%s attr=%s" % (
                      name, agg["train_success_mean"],
                      agg["classic_frozen"]["config_mean_success"],
                      agg["classic_adapt"]["config_mean_success"],
                      agg["ood_frozen"]["config_mean_success"],
                      agg["continual_ood"]["config_mean_success"],
                      {k: round(v, 1) for k, v in agg["memory_mean"].items()},
                      {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in agg["attribution_mean"].items()}))

    # comparisons (candidate - baseline)
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

    summary["finished_at"] = time.time()
    with open(os.path.join(results_dir, "memory_v07_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print("  comparisons:")
        for k, v in summary["comparisons"].items():
            print("   ", k, v)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="small development pilot (few configs/episodes)")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    fresh_cache = os.path.join(root, "results", "academy_v01",
                               "memory_v07", "fresh_control.json")
    if args.quick:
        base = {
            "test_grid_n": 6, "eval_eps_per_cfg": 2,
            "ood_probe_n": 4, "ood_eps_per_seed": 2,
            "substrate_n": 4, "retention_eps_per_cfg": 1,
            "continual_n": 3, "continual_eps_per_cfg": 2,
        }
        run_v07_experiment(
            variants=args.variants or ["fresh", "original", "shuffled",
                                       "epoch"],
            episodes=40, base=base, verbose=True,
            fresh_cache_path=fresh_cache)
    else:
        run_v07_experiment(variants=args.variants, verbose=True,
                           fresh_cache_path=fresh_cache)
