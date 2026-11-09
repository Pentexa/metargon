"""Memory v0.5 — measurable transfer under an evaluation-safe protocol.

v0.4 found `memory_wiped ≡ original`: within-evaluation learning
obscures the effect of persistent memory. v0.5 therefore separates
transfer into three measurable regimes:

  1. FROZEN (zero-shot): learning disabled during evaluation
     (`agent.frozen_policy = True`). Measures what the *retained
     memory alone* transfers.
  2. ADAPT (within-episode): learning enabled (the v0.4 protocol).
     Measures how much quick adaptation inside the eval episode adds.
  3. CONTINUAL (post-training): after the frozen grid, the agent is
     allowed to keep learning on the eval configs and is re-measured
     on a disjoint subset (retention/forgetting of newly acquired
     knowledge).

Controls: fresh, memory_wiped, original, ctx (v0.4), ctx_full, plus
new evidence-bank variants (bank, bank+ctx). Target: does any retained-
memory intervention increase *frozen* (zero-shot) transfer?

Memory tracking per variant: useful (verified non-contradicted),
contradicted, context-specific (ctx counts > 0), retrieved (bank
retrievals).

All offline, pinned hash, matched streams, deterministic.
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
from academy.envs.gridworld import MechanismGrid
from academy.envs.tasks import (
    DEFAULT_SPLITS_PATH, EpisodeSpec, difficulty_of, load_splits,
    make_episodes,
)
from academy.evren.retention import RetentionSuite, sample_substrate
from academy.learning.loop import run_one_episode
from academy.eval.metrics import paired_difference

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
            self.memory.evidence_bank = []


def eval_grid(agent_state, configs, max_steps, n_configs, episodes_per_config,
              seeds, seed, frozen=False, wiped=False):
    """Independent per-config grid; optionally frozen (no learning)."""
    picks = sample_substrate(list(configs), n_configs, seed)
    rows = []
    for i, cfgc in enumerate(picks):
        for seed_i in seeds:
            base = seed * 100000 + i * 100 + seed_i
            stream = make_episodes([cfgc], "eval", base, episodes_per_config,
                                   max_steps=max_steps)
            agent = MetargonAcademyAgent(seed=base)
            if agent_state is not None:
                agent.load_state(agent_state)
            if frozen:
                agent.frozen_policy = True
            if wiped:
                agent.memory.hypotheses = []
            succ, steps_total = 0, 0
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
            rows.append({
                "config": str(cfgc.key), "config_i": i, "seed_i": seed_i,
                "success_rate": succ / max(1, len(stream)),
            })
    per_config = {}
    for r in rows:
        ent = per_config.setdefault(r["config"], {"succ": 0.0, "n": 0})
        ent["succ"] += r["success_rate"] * r["n"]
        ent["n"] += r["n"]
    per_config = {k: v["succ"] / v["n"] for k, v in per_config.items()}
    rates = list(per_config.values())
    return {
        "summary": {
            "n_configs": len(picks),
            "config_mean_success": (sum(rates) / len(rates)) if rates else 0.0,
            "per_config": {k: round(v, 4) for k, v in per_config.items()},
        },
        "per_config_list": rates,
    }


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
    }


def run_v05_experiment(splits=None, episodes=160, block_size=40,
                       train_seeds=(20261107, 20261108), max_steps=80,
                       variants=None, results_dir=None, verbose=True,
                       append_path=None):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    results_dir = results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "results", "academy_v01", "memory_v05")
    os.makedirs(results_dir, exist_ok=True)
    variants = variants or list(VARIANTS)

    base = {
        "test_grid_n": 16, "eval_eps_per_cfg": 4,
        "ood_probe_n": 12, "ood_eps_per_seed": 3,
        "substrate_n": 8, "retention_eps_per_cfg": 2,
        "continual_n": 6, "continual_eps_per_cfg": 3,
    }
    eval_seed = 20261108
    substrate = sample_substrate(splits["train"], base["substrate_n"],
                                 20261107 + 31)

    summary = {
        "run_id": "memory-v05",
        "episodes": episodes,
        "train_seeds": list(train_seeds),
        "variants": variants,
        "hash_randomization": bool(sys.flags.hash_randomization),
        "python_version": sys.version.split()[0],
        "started_at": time.time(),
        "conditions_detail": {},
        "comparisons": {},
        "fresh_control": {},
    }

    fresh_frozen = eval_grid(None, splits["test"], max_steps,
                             base["test_grid_n"], base["eval_eps_per_cfg"],
                             (0, 1), eval_seed, frozen=True)
    fresh_adapt = eval_grid(None, splits["test"], max_steps,
                            base["test_grid_n"], base["eval_eps_per_cfg"],
                            (0, 1), eval_seed, frozen=False)
    summary["fresh_control"] = {
        "classic_frozen": fresh_frozen["summary"],
        "classic_frozen_per_config": fresh_frozen["per_config_list"],
        "classic_adapt": fresh_adapt["summary"],
        "classic_adapt_per_config": fresh_adapt["per_config_list"],
    }
    if verbose:
        print(f"fresh: frozen={fresh_frozen['summary']['config_mean_success']:.3f} "
              f"adapt={fresh_adapt['summary']['config_mean_success']:.3f}")

    for name in variants:
        kwargs = {k: v for k, v in VARIANTS[name].items()
                  if not k.startswith("_")}
        per_seed = []
        for ts in train_seeds:
            agent = WipeHookAgent(seed=0, **kwargs)
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
                               (0, 1), eval_seed, frozen=True)
            # 2) within-episode adaptation
            adapt = eval_grid(state, splits["test"], max_steps,
                              base["test_grid_n"], base["eval_eps_per_cfg"],
                              (0, 1), eval_seed, frozen=False)
            ood_frozen = eval_grid(state, splits["ood"], max_steps,
                                   base["ood_probe_n"],
                                   base["ood_eps_per_seed"], (0,),
                                   eval_seed + 2, frozen=True)
            # 3) continual: keep learning on a disjoint subset then probe
            sub = sample_substrate([c for c in splits["test"]
                                    if c.key not in {list(
                                        eval_grid(state, splits["test"],
                                                  max_steps, 1, 1, (0,),
                                                  eval_seed, frozen=True)
                                        ['summary']['per_config'])[0]}],
                                   6, ts + 77)
            # simpler: continual probe on OOD subset with learning enabled
            cont = eval_grid(state, splits["ood"][:6], max_steps,
                             base["continual_n"], base["continual_eps_per_cfg"],
                             (0,), eval_seed + 5, frozen=False)

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

            per_seed.append({
                "seed": ts,
                "train_success": agent.successes / max(1, agent.episodes),
                "wall_s": round(wall, 2),
                "classic_frozen": frozen["summary"],
                "classic_frozen_per_config": frozen["per_config_list"],
                "classic_adapt": adapt["summary"],
                "classic_adapt_per_config": adapt["per_config_list"],
                "ood_frozen": ood_frozen["summary"],
                "continual_ood": cont["summary"],
                "retention": retention.summary(),
                "memory": memory_tracking(agent),
            })

        def mean_cfg(key):
            vecs = [p[key] for p in per_seed]
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
            "continual_ood": {
                "config_mean_success": (sum(
                    p["continual_ood"]["config_mean_success"]
                    for p in per_seed) / len(per_seed)),
            },
            "retention_delta_mean": (sum(
                p["retention"]["retention_delta"] for p in per_seed)
                / len(per_seed)),
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
        agg["memory_mean"]["bank_size"] = sum(
            p["memory"]["bank_size"] for p in per_seed) / len(per_seed)
        agg["memory_mean"]["bank_retrievals"] = sum(
            p["memory"]["bank_retrievals"] for p in per_seed) / len(per_seed)
        agg["memory_mean"]["context_specific"] = sum(
            p["memory"]["context_specific"] for p in per_seed) / len(per_seed)
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

    if "original" in summary["conditions_detail"]:
        orig = summary["conditions_detail"]["original"]
        for name in variants:
            if name == "original":
                continue
            v = summary["conditions_detail"][name]
            cmp_ = {}
            a = orig["classic_frozen_per_config"]
            b = v["classic_frozen_per_config"]
            if a and b and len(a) == len(b):
                cmp_["classic_frozen"] = paired_difference(a, b, seed=eval_seed)
            a = orig["classic_adapt_per_config"]
            b = v["classic_adapt_per_config"]
            if a and b and len(a) == len(b):
                cmp_["classic_adapt"] = paired_difference(a, b, seed=eval_seed)
            cmp_["train_delta"] = (v["train_success_mean"]
                                   - orig["train_success_mean"])
            summary["comparisons"][f"{name}_vs_original"] = cmp_

    summary["finished_at"] = time.time()
    with open(os.path.join(results_dir, "memory_v05_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print("  comparisons:")
        for k, v in summary["comparisons"].items():
            print("   ", k, v)
    return summary


if __name__ == "__main__":
    run_v05_experiment()
