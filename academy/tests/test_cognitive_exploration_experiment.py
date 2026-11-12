"""METARGON Academy — cognitive exploration v0.1 experiment (dedicated).

Compares the baseline student (``MetargonAcademyAgent`` with every
cognitive flag off) against the proposed student (``cognitive_exploration
=True``) on five condition cells:

  * classic_test — unseen conjunctive mechanisms (held-out core split);
  * ood        — unseen causal mechanisms (absence-atom OOD family);
  * noisy_test  — classic test with noisy observations (noise_q=0.08);
  * tight_test  — classic test under a tight action budget (44 steps);
  * conflict    — training pool with a mid-run rule switch (conflicting
                   rules, adaptation protocol).

Measured endpoints (per policy-cell):

  * success rate (per-seed, paired bootstrap vs baseline);
  * action efficiency — mean steps on successful episodes (and mean
    steps over all episodes);
  * repeated experiments — duplicate probe antecedents within an episode
    (normalized by probes executed);
  * useful new evidence — episodes where the agent's memory ends up with
    a usable hypothesis matching a true hidden clause (exact or subset);
  * recovery time — steps until the first door-opening event, and the
    share of steps spent in detected loop/stall states;
  * random fallback share (wasted budget proxy).

Ablation cells run a subset of mechanisms so each independent flag's
contribution can be attributed.

The hidden ``EnvConfig`` is only ever used *after* an episode, to score
whether the agent's hypotheses match ground truth — never given to the
agent (the public protocol boundary still holds).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import unittest

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import MechanismGrid
from academy.envs.tasks import get_splits, make_episodes
from academy.eval.protocol import spec_for
from academy.tests.helpers import cached_splits

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "academy_v01",
    "cognitive_exploration")

SEED_BASE = 20261112
N_SEEDS_MAIN = 12
EPISODES_MAIN = 100
N_SEEDS_ABLATION = 8
EPISODES_ABLATION = 60

CONDITIONS = {
    "classic_test": dict(pool="test", noise=0.0, max_steps=80,
                         switch_half=False),
    "ood": dict(pool="ood", noise=0.0, max_steps=80, switch_half=False),
    "noisy_test": dict(pool="test", noise=0.08, max_steps=80,
                       switch_half=False),
    "tight_test": dict(pool="test", noise=0.0, max_steps=44,
                       switch_half=False),
    "conflict": dict(pool="train", noise=0.0, max_steps=80,
                     switch_half=True),
}

ABLATIONS = {
    "proposed_nogap": dict(cognitive_exploration=True,
                           cog_knowledge_gaps=False),
    "proposed_noinfo": dict(cognitive_exploration=True, cog_info_gain=False),
    "proposed_norecov": dict(cognitive_exploration=True, cog_recovery=False),
    "proposed_nohold": dict(cognitive_exploration=True, cog_probe_hold=False),
    "proposed_nodecomp": dict(cognitive_exploration=True, cog_decompose=False),
}

# ---------------------------------------------------------------------------
# Ground-truth scoring (post-hoc only; never fed to the agent).
# ---------------------------------------------------------------------------
def true_clauses(config, door: str):
    rule = config.rule_for(door)
    if isinstance(rule, tuple):
        return list(rule)
    return [rule]


def hypothesis_matches(h, config, door: str):
    """Return "exact" / "subset" / None: usable hypothesis with predicted
    opens whose antecedent matches a true hidden clause (exactly or as a
    subset) for ``door``."""
    if h.door != door or h.predicted_opens is not True or not h.usable:
        return None
    for clause in true_clauses(config, door):
        if h.antecedent == clause:
            return "exact"
    for clause in true_clauses(config, door):
        if h.antecedent <= clause and h.antecedent:
            return "subset"
    return None


def match_summary(agent, config):
    """Per-door ground-truth match for the agent's memory at episode end."""
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


# ---------------------------------------------------------------------------
# Episode runner with instrumentation.
# ---------------------------------------------------------------------------
TRACE_COUNTERS = (
    "probes_new", "probes_repeat", "loop_steps", "stalls_detected",
    "recovery_steps", "random_fallback_steps", "plans_used",
    "memory_correct", "memory_incorrect", "gap_guided_candidates",
    "decomposed_candidates", "holds_started", "holds_resolved_open",
    "holds_resolved_fail", "holds_interrupted",
)


def run_episode(agent, spec):
    env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
    obs = env.reset(spec.spec_seed)
    # trace counters are cumulative across a run: snapshot before the
    # episode and measure deltas (per-episode values)
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
    delta = {k: tr.get(k, 0) - prev.get(k, 0) for k in TRACE_COUNTERS}
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
        "gap_guided": delta["gap_guided_candidates"],
        "decomposed": delta["decomposed_candidates"],
        "holds": delta["holds_started"],
        "match": match_summary(agent, spec.config),
    }
    rec["useful_evidence"] = any(v is not None
                                 for v in rec["match"].values())
    rec["exact_evidence"] = any(v == "exact"
                                for v in rec["match"].values())
    return rec


def policy_factory(flags: dict):
    def make(seed: int):
        return MetargonAcademyAgent(seed=seed, **flags)
    return make


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
        agent = policy_factory(flags)(seed)
        agent._collect_probe_log = True
        records = []
        for spec in stream:
            records.append(run_episode(agent, spec))
        all_records.extend(records)
        succ = [1.0 if r["success"] else 0.0 for r in records]
        per_seed.append({
            "seed": seed,
            "success_rate": sum(succ) / len(succ) if succ else 0.0,
            "records": records,
        })
    wall = time.perf_counter() - t0
    return _summarize(per_seed, all_records, wall)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _summarize(per_seed, all_records, wall):
    n = len(all_records)
    succ = [1.0 if r["success"] else 0.0 for r in all_records]
    steps_ok = [r["steps"] for r in all_records if r["success"]]
    steps_all = [r["steps"] for r in all_records]
    probes = [r["probes"] for r in all_records]
    reps = [r["repeated_probes"] for r in all_records]
    loop = [r["loop_steps"] / max(1, r["steps"]) for r in all_records]
    fb = [r["random_fallback"] / max(1, r["steps"]) for r in all_records]
    useful = [1.0 if r["useful_evidence"] else 0.0 for r in all_records]
    exact = [1.0 if r["exact_evidence"] else 0.0 for r in all_records]
    first_open = [r["first_open_step"] for r in all_records
                  if r["first_open_step"] is not None]
    stalls = [r["stalls"] for r in all_records]
    recovery = [r["recovery_steps"] for r in all_records]
    holds = [r["holds"] for r in all_records]
    decomp = [r["decomposed"] for r in all_records]
    return {
        "n_episodes": n,
        "wall_s": round(wall, 1),
        "success_rate": _mean(succ),
        "mean_steps_ok": _mean(steps_ok),
        "mean_steps_all": _mean(steps_all),
        "mean_probes": _mean(probes),
        "repeat_probe_rate": _mean(reps) / max(1e-9, _mean(probes))
        if _mean(probes) else 0.0,
        "mean_repeated_probes": _mean(reps),
        "loop_fraction": _mean(loop),
        "fallback_fraction": _mean(fb),
        "useful_evidence_rate": _mean(useful),
        "exact_evidence_rate": _mean(exact),
        "mean_first_open": _mean(first_open),
        "mean_stalls": _mean(stalls),
        "mean_recovery_steps": _mean(recovery),
        "mean_holds": _mean(holds),
        "mean_decomposed": _mean(decomp),
        "per_seed_success": [p["success_rate"] for p in per_seed],
    }


def paired_bootstrap(a_succ, b_succ, seed: int = 0, n_boot: int = 2000):
    """Paired bootstrap on (a - b) mean per-seed success. Positive mean
    means a improved over b."""
    assert len(a_succ) == len(b_succ)
    n = len(a_succ)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a_succ[i] for i in idx) / n
        db = sum(b_succ[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    return {"mean": sum(diffs) / n_boot, "ci": (lo, hi)}


# ---------------------------------------------------------------------------
# Experiment drivers.
# ---------------------------------------------------------------------------
def run_main(seed_base=SEED_BASE, n_seeds=N_SEEDS_MAIN,
             episodes=EPISODES_MAIN, results_dir=RESULTS_DIR,
             conditions=None):
    splits = cached_splits()
    conditions = conditions or CONDITIONS
    baseline_flags = {}
    proposed_flags = {"cognitive_exploration": True}
    out = {"run_id": "cognitive-exploration-v01-main",
           "seed_base": seed_base, "n_seeds": n_seeds,
           "episodes": episodes, "conditions": {}}
    for cell in conditions:
        base = run_cell(baseline_flags, splits, cell, seed_base,
                        n_seeds, episodes, "baseline")
        prop = run_cell(proposed_flags, splits, cell, seed_base,
                        n_seeds, episodes, "proposed")
        cmp = paired_bootstrap(prop["per_seed_success"],
                               base["per_seed_success"],
                               seed=seed_base + len(out["conditions"]))
        out["conditions"][cell] = {
            "baseline": base, "proposed": prop,
            "paired_success_delta": {
                "mean": round(cmp["mean"] * 100, 2),
                "ci": (round(cmp["ci"][0] * 100, 2),
                       round(cmp["ci"][1] * 100, 2)),
                "sign": "positive" if cmp["ci"][0] > 0
                else ("negative" if cmp["ci"][1] < 0 else "inclusive"),
            },
        }
        print(f"[{cell}] baseline {base['success_rate']:.3f} -> "
              f"proposed {prop['success_rate']:.3f} "
              f"(delta {cmp['mean']*100:.1f} pp, CI "
              f"[{cmp['ci'][0]*100:.1f}, {cmp['ci'][1]*100:.1f}])")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "main_summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return out


def run_ablations(seed_base=SEED_BASE + 500, n_seeds=N_SEEDS_ABLATION,
                  episodes=EPISODES_ABLATION, results_dir=RESULTS_DIR,
                  conditions=("classic_test", "ood")):
    splits = cached_splits()
    proposed_flags = {"cognitive_exploration": True}
    out = {"run_id": "cognitive-exploration-v01-ablations",
           "seed_base": seed_base, "n_seeds": n_seeds,
           "episodes": episodes, "conditions": {}}
    for cell in conditions:
        out["conditions"][cell] = {"proposed": {}, "ablations": {}}
        prop = run_cell(proposed_flags, splits, cell, seed_base,
                        n_seeds, episodes, "proposed")
        out["conditions"][cell]["proposed"] = prop
        for name, flags in ABLATIONS.items():
            res = run_cell(flags, splits, cell, seed_base,
                           n_seeds, episodes, name)
            cmp = paired_bootstrap(res["per_seed_success"],
                                   prop["per_seed_success"],
                                   seed=seed_base + len(out["conditions"]))
            out["conditions"][cell]["ablations"][name] = {
                "result": res,
                "paired_success_delta_vs_proposed": {
                    "mean": round(cmp["mean"] * 100, 2),
                    "ci": (round(cmp["ci"][0] * 100, 2),
                           round(cmp["ci"][1] * 100, 2)),
                },
            }
            print(f"[{cell}] {name} {res['success_rate']:.3f} "
                  f"(vs proposed {prop['success_rate']:.3f}, "
                  f"delta {cmp['mean']*100:.1f} pp)")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "ablations_summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return out


# ---------------------------------------------------------------------------
# CLI (also importable; unittest runs a small smoke).
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cognitive exploration v0.1")
    ap.add_argument("--seeds", type=int, default=N_SEEDS_MAIN)
    ap.add_argument("--episodes", type=int, default=EPISODES_MAIN)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()
    conditions = args.conditions or None
    if conditions is not None:
        conditions = {c: CONDITIONS[c] for c in conditions if c in CONDITIONS}
    if args.ablations:
        run_ablations(seed_base=args.seed_base, results_dir=args.results_dir)
    else:
        run_main(seed_base=args.seed_base, n_seeds=args.seeds,
                 episodes=args.episodes, results_dir=args.results_dir,
                 conditions=conditions)


class TestCognitiveExplorationExperimentSmoke(unittest.TestCase):
    """CI-safe smoke: tiny config of the main comparison, asserting the
    harness runs end-to-end and the proposed agent is not pathological.
    Always writes to a temp dir (never the real results)."""

    def test_main_smoke_default(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = run_main(
                seed_base=999000, n_seeds=2, episodes=8,
                results_dir=tmp,
                conditions={"classic_test": CONDITIONS["classic_test"]})
            cell = out["conditions"]["classic_test"]
            self.assertGreater(cell["proposed"]["n_episodes"], 0)
            self.assertGreaterEqual(cell["baseline"]["success_rate"], 0.0)
            self.assertLessEqual(cell["proposed"]["success_rate"], 1.0)

    def test_ablation_smoke(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = run_ablations(seed_base=999500, n_seeds=2, episodes=6,
                                results_dir=tmp,
                                conditions=("classic_test",))
            self.assertIn("proposed_nogap",
                          out["conditions"]["classic_test"]["ablations"])


if __name__ == "__main__":
    main()
