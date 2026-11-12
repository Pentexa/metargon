"""METARGON Academy — cognitive exploration v0.2 experiment (dedicated).

Robust-intelligence study: does the noise-robust measurement-confirmation
mechanism (``cog_noise_robust``) recover the noisy-condition regression of
v0.1 while preserving its OOD/clean gains?

Agents (all opt-in flags; baseline unchanged):
  * baseline            — all flags off
  * v01                — cognitive_exploration=True (v0.1)
  * v02                — cognitive_exploration=True + cog_noise_robust=True

Conditions (matched streams per seed):
  * classic_clean, classic_q04, classic_q08, classic_q15
  * ood_clean (unseen causal mechanisms), conflict (rule flip), tight (44)

Endpoints (per cell, per agent):
  * success rate (paired bootstrap vs baseline and vs v01)
  * useful/exact ground-truth evidence (post-hoc, never fed to policy)
  * false contradictions — a hypothesis matching a true clause that ends
    the episode with status CONTRADICTED
  * repeated probes, confirmed vs unconfirmed flips, recovery steps,
    planner cost, first-open latency

Independent validation seed base 20261201 (v0.1 used 20261112).
"""

from __future__ import annotations

import json
import os
import time
import unittest

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import MechanismGrid
from academy.envs.tasks import get_splits, make_episodes
from academy.eval.protocol import spec_for
from academy.tests.helpers import cached_splits
from academy.tests.test_cognitive_exploration_experiment import (
    TRACE_COUNTERS, hypothesis_matches, paired_bootstrap,
)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "academy_v01",
    "cognitive_exploration_v02")

SEED_BASE = 20261201          # independent validation seeds
N_SEEDS_MAIN = 6
EPISODES_MAIN = 40
N_SEEDS_CONFIRMATORY = 8
EPISODES_CONFIRMATORY = 60
N_SEEDS_ABLATION = 6
EPISODES_ABLATION = 40

AGENTS = {
    "baseline": {},
    "v01": {"cognitive_exploration": True},
    "v02": {"cognitive_exploration": True, "cog_noise_robust": True},
}

ABLATIONS = {
    "v02_noiseoff": {"cognitive_exploration": True,
                     "cog_noise_robust": False},
    "v02_norecov": {"cognitive_exploration": True,
                    "cog_noise_robust": True, "cog_recovery": False},
    "v02_nohold": {"cognitive_exploration": True,
                   "cog_noise_robust": True, "cog_probe_hold": False},
    "v02_nogap": {"cognitive_exploration": True,
                  "cog_noise_robust": True, "cog_knowledge_gaps": False},
}

CONDITIONS = {
    "classic_clean": dict(pool="test", noise=0.00, max_steps=80,
                          switch_half=False),
    "classic_q04": dict(pool="test", noise=0.04, max_steps=80,
                        switch_half=False),
    "classic_q08": dict(pool="test", noise=0.08, max_steps=80,
                        switch_half=False),
    "classic_q15": dict(pool="test", noise=0.15, max_steps=80,
                        switch_half=False),
    "ood_clean": dict(pool="ood", noise=0.00, max_steps=80,
                      switch_half=False),
    "ood_q08": dict(pool="ood", noise=0.08, max_steps=80,
                    switch_half=False),
    "conflict": dict(pool="train", noise=0.00, max_steps=80,
                     switch_half=True),
    "tight": dict(pool="test", noise=0.00, max_steps=44,
                  switch_half=False),
}

# cells used in the main comparison
MAIN_CELLS = ("classic_clean", "classic_q04", "classic_q08",
              "classic_q15", "ood_clean", "conflict", "tight")
# confirmatory cells (higher power on the key comparisons)
CONFIRMATORY_CELLS = ("classic_q08", "classic_q15", "ood_clean",
                      "classic_clean")
ABLATION_CELLS = ("classic_q08", "classic_q15", "ood_clean")


def true_clause_match(h, config, door):
    """Match regardless of status (used for false-contradiction counting:
    a contradicted hypothesis that matches a true clause is a false
    contradiction)."""
    if h.door != door or h.predicted_opens is not True:
        return False
    for clause in (config.rule_for(door)
                   if not isinstance(config.rule_for(door), tuple)
                   else config.rule_for(door)):
        if isinstance(clause, frozenset) and h.antecedent == clause:
            return True
        if isinstance(clause, frozenset) and h.antecedent <= clause \
                and h.antecedent:
            return True
    return False


def false_contradictions(agent, config):
    """Count hypotheses whose antecedent matches a true clause for their
    door but ended the episode with status CONTRADICTED (a false
    contradiction — true knowledge destroyed or demoted)."""
    from academy.agents.memory import CONTRADICTED
    count = 0
    for h in agent.memory.hypotheses:
        if h.status == CONTRADICTED and any(
                true_clause_match(h, config, door)
                for door in ("D1", "D2")):
            count += 1
    return count


def run_episode(agent, spec):
    env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
    obs = env.reset(spec.spec_seed)
    prev = agent.probe_trace()
    agent.episode_start(spec_for(agent, spec), env.geom)
    steps = 0
    opened = 0
    first_open_step = None
    success = False
    while steps < env.max_steps:
        action = agent.act(obs)
        before = obs
        res = env.step(action)
        agent.learn(before, action, res)
        steps += 1
        if res.events and first_open_step is None:
            first_open_step = steps
        opened += len(res.events)
        obs = res.observation
        if res.terminated:
            success = True
            break
    agent.finish_episode(success)
    tr = agent.probe_trace()
    counters = TRACE_COUNTERS + ("confirmed_events", "unconfirmed_flips")
    delta = {k: tr.get(k, 0) - prev.get(k, 0) for k in counters}
    match = _match_summary(agent, spec.config)
    rec = {
        "success": success,
        "steps": steps,
        "door_events": opened,
        "first_open_step": first_open_step,
        "probes": delta["probes_new"],
        "repeated_probes": delta["probes_repeat"],
        "loop_steps": delta["loop_steps"],
        "stalls": delta["stalls_detected"],
        "recovery_steps": delta["recovery_steps"],
        "random_fallback": delta["random_fallback_steps"],
        "plans_used": delta["plans_used"],
        "memory_correct": delta["memory_correct"],
        "memory_incorrect": delta["memory_incorrect"],
        "confirmed_events": delta.get("confirmed_events", 0),
        "unconfirmed_flips": delta.get("unconfirmed_flips", 0),
        "false_contradictions": false_contradictions(agent, spec.config),
        "match": match,
    }
    rec["useful_evidence"] = any(v is not None for v in match.values())
    rec["exact_evidence"] = any(v == "exact" for v in match.values())
    return rec


def _match_summary(agent, config):
    out = {}
    for door in ("D1", "D2"):
        best = None
        for h in agent.memory.hypotheses:
            m = hypothesis_matches(h, config, door)
            if m is not None and (best is None
                                  or (m == "exact" and best != "exact")):
                best = m
        out[door] = best
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def run_cell(flags: dict, splits, cell: str, seed_base: int,
             n_seeds: int, episodes: int, tag: str):
    cond = CONDITIONS[cell]
    pool = splits[cond["pool"]]
    per_seed = []
    all_records = []
    t0 = time.perf_counter()
    for si in range(n_seeds):
        seed = seed_base * 1000 + si
        stream = make_episodes(pool, tag, seed, episodes,
                               max_steps=cond["max_steps"],
                               noise_q=cond["noise"],
                               switch_half=cond["switch_half"])
        agent = MetargonAcademyAgent(seed=seed, **flags)
        agent._collect_probe_log = True
        records = [run_episode(agent, spec) for spec in stream]
        all_records.extend(records)
        succ = [1.0 if r["success"] else 0.0 for r in records]
        per_seed.append(sum(succ) / len(succ) if succ else 0.0)
    wall = time.perf_counter() - t0
    n = len(all_records)
    succ = [1.0 if r["success"] else 0.0 for r in all_records]
    steps_ok = [r["steps"] for r in all_records if r["success"]]
    reps = [r["repeated_probes"] for r in all_records]
    probes = [r["probes"] for r in all_records]
    loop = [r["loop_steps"] / max(1, r["steps"]) for r in all_records]
    fb = [r["random_fallback"] / max(1, r["steps"]) for r in all_records]
    useful = [1.0 if r["useful_evidence"] else 0.0 for r in all_records]
    exact = [1.0 if r["exact_evidence"] else 0.0 for r in all_records]
    fc = [r["false_contradictions"] for r in all_records]
    conf = [r["confirmed_events"] for r in all_records]
    unc = [r["unconfirmed_flips"] for r in all_records]
    open_steps = [r["first_open_step"] for r in all_records
                  if r["first_open_step"] is not None]
    return {
        "n_episodes": n,
        "wall_s": round(wall, 1),
        "success_rate": _mean(succ),
        "mean_steps_ok": _mean(steps_ok),
        "mean_probes": _mean(probes),
        "mean_repeated_probes": _mean(reps),
        "loop_fraction": _mean(loop),
        "fallback_fraction": _mean(fb),
        "useful_evidence_rate": _mean(useful),
        "exact_evidence_rate": _mean(exact),
        "mean_false_contradictions": _mean(fc),
        "false_contradiction_rate": _mean([1.0 if x > 0 else 0.0
                                           for x in fc]),
        "mean_confirmed_events": _mean(conf),
        "mean_unconfirmed_flips": _mean(unc),
        "mean_first_open": _mean(open_steps),
        "per_seed_success": list(per_seed),
    }


def run_main(seed_base=SEED_BASE, n_seeds=N_SEEDS_MAIN,
             episodes=EPISODES_MAIN, results_dir=RESULTS_DIR,
             cells=MAIN_CELLS, agents=None):
    splits = cached_splits()
    agents = agents or AGENTS
    out = {"run_id": "cognitive-exploration-v02-main",
           "seed_base": seed_base, "n_seeds": n_seeds,
           "episodes": episodes, "conditions": {}}
    for ci, cell in enumerate(cells):
        out["conditions"][cell] = {"agents": {}}
        for name, flags in agents.items():
            out["conditions"][cell]["agents"][name] = run_cell(
                flags, splits, cell, seed_base, n_seeds, episodes, name)
        b = out["conditions"][cell]["agents"]["baseline"]
        for name in ("v01", "v02"):
            if name not in out["conditions"][cell]["agents"]:
                continue
            a = out["conditions"][cell]["agents"][name]
            cmp_ = paired_bootstrap(a["per_seed_success"],
                                    b["per_seed_success"],
                                    seed=seed_base + ci)
            out["conditions"][cell].setdefault("vs_baseline", {})[name] = {
                "mean": round((a["success_rate"] - b["success_rate"]) * 100, 2),
                "ci": (round(cmp_["ci"][0] * 100, 2),
                       round(cmp_["ci"][1] * 100, 2)),
            }
        if "v01" in out["conditions"][cell]["agents"] and \
                "v02" in out["conditions"][cell]["agents"]:
            v1 = out["conditions"][cell]["agents"]["v01"]
            v2 = out["conditions"][cell]["agents"]["v02"]
            cmp_ = paired_bootstrap(v2["per_seed_success"],
                                    v1["per_seed_success"],
                                    seed=seed_base + 777 + ci)
            out["conditions"][cell]["v02_vs_v01"] = {
                "mean": round(cmp_["mean"] * 100, 2),
                "ci": (round(cmp_["ci"][0] * 100, 2),
                       round(cmp_["ci"][1] * 100, 2)),
            }
        print(f"[{cell}] " + " | ".join(
            f"{name} {out['conditions'][cell]['agents'][name]['success_rate']*100:.1f}%"
            for name in out["conditions"][cell]["agents"]) +
            f" | v02-vs-v01 {out['conditions'][cell].get('v02_vs_v01', {})}")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "main_summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return out


def run_ablations(seed_base=SEED_BASE + 500, n_seeds=N_SEEDS_ABLATION,
                  episodes=EPISODES_ABLATION, results_dir=RESULTS_DIR,
                  cells=ABLATION_CELLS):
    splits = cached_splits()
    out = {"run_id": "cognitive-exploration-v02-ablations",
           "seed_base": seed_base, "n_seeds": n_seeds,
           "episodes": episodes, "conditions": {}}
    for cell in cells:
        out["conditions"][cell] = {"v02": {}, "ablations": {}}
        v2 = run_cell(AGENTS["v02"], splits, cell, seed_base,
                      n_seeds, episodes, "v02")
        out["conditions"][cell]["v02"] = v2
        for ai, (name, flags) in enumerate(ABLATIONS.items()):
            res = run_cell(flags, splits, cell, seed_base,
                           n_seeds, episodes, name)
            cmp_ = paired_bootstrap(res["per_seed_success"],
                                    v2["per_seed_success"],
                                    seed=seed_base + ai)
            out["conditions"][cell]["ablations"][name] = {
                "result": res,
                "delta_vs_v02": {
                    "mean": round(cmp_["mean"] * 100, 2),
                    "ci": (round(cmp_["ci"][0] * 100, 2),
                           round(cmp_["ci"][1] * 100, 2)),
                },
            }
            print(f"[{cell}] {name} {res['success_rate']*100:.1f}% "
                  f"(vs v02 {v2['success_rate']*100:.1f}%, "
                  f"delta {cmp_['mean']*100:.1f} pp)")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "ablations_summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return out


def run_confirmatory(seed_base=SEED_BASE + 900,
                     n_seeds=N_SEEDS_CONFIRMATORY,
                     episodes=EPISODES_CONFIRMATORY,
                     results_dir=RESULTS_DIR,
                     cells=CONFIRMATORY_CELLS):
    return run_main(seed_base=seed_base, n_seeds=n_seeds,
                    episodes=episodes, results_dir=results_dir,
                    cells=cells)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cognitive exploration v0.2")
    ap.add_argument("--seeds", type=int, default=N_SEEDS_MAIN)
    ap.add_argument("--episodes", type=int, default=EPISODES_MAIN)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()
    if args.ablations:
        run_ablations(seed_base=args.seed_base,
                      results_dir=args.results_dir)
    elif args.confirmatory:
        run_confirmatory(seed_base=args.seed_base,
                         results_dir=args.results_dir)
    else:
        cells = args.conditions or list(MAIN_CELLS)
        cells = [c for c in cells if c in CONDITIONS]
        run_main(seed_base=args.seed_base, n_seeds=args.seeds,
                 episodes=args.episodes, results_dir=args.results_dir,
                 cells=cells)


class TestCognitiveExplorationV02Smoke(unittest.TestCase):
    def test_main_smoke(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = run_main(seed_base=888000, n_seeds=2, episodes=6,
                           results_dir=tmp,
                           cells=("classic_clean", "classic_q15",
                                  "ood_clean"))
            cell = out["conditions"]["classic_q15"]
            self.assertIn("v02", cell["agents"])
            self.assertEqual(cell["agents"]["v02"]["n_episodes"], 12)
            self.assertIn("v02_vs_v01", cell)
            for name in cell["agents"]:
                self.assertGreaterEqual(cell["agents"][name]["success_rate"],
                                        0.0)

    def test_ablation_smoke(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = run_ablations(seed_base=888500, n_seeds=2, episodes=4,
                                results_dir=tmp,
                                cells=("classic_q15",))
            self.assertIn("v02_noiseoff",
                          out["conditions"]["classic_q15"]["ablations"])


if __name__ == "__main__":
    main()
