"""METARGON Academy — exploration noise regression diagnosis (v0.1).

Mission: identify the underlying causes of Cognitive Exploration's noisy-
observation regression (PR #49 finding: noisy_test q=0.08 −12.1 pp; the
independent auditor replicated it: explore Δ −0.21 [−0.29, −0.17]).

This harness is a *diagnosis* tool. It implements no new exploration
mechanism; it only observes the existing student on matched diagnostic
seeds and records fine-grained perception/learning/execution counters to
separate error classes: perception (phantom events, corrupted atoms),
causal-learning (false contradictions, wrong hypotheses), and planning/
execution (invalid plans through phantom-open doors, premature recovery,
hold budget burn, missed transient openings).

Seeds: DIAG_SEED_BASE = 20261118 is disjoint from all previous runs
(v0.1 20261112, v0.2 20261201, independent auditor 20261121); the
auditor's confirmatory seeds are intentionally NOT used.

Ground-truth access: the harness may read ``MechanismGrid._state`` and
``spec.config`` *after* each episode/step purely for scoring (the same
post-hoc convention the existing v0.1 harness uses for match scoring);
this information never reaches the agent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import unittest

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import MechanismGrid, atoms_from_state
from academy.envs.tasks import make_episodes
from academy.eval.protocol import spec_for
from academy.agents.common import atoms_from_observation
from academy.tests.helpers import cached_splits

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "academy_v01",
    "exploration_noise_diagnosis")

DIAG_SEED_BASE = 20261118
N_SEEDS = 6
EPISODES = 40

CONDITIONS = {
    "clean": dict(pool="test", noise=0.0, max_steps=80),
    "noisy": dict(pool="test", noise=0.08, max_steps=80),
    "ood": dict(pool="ood", noise=0.0, max_steps=80),
}

ARMS = {
    "baseline": {},
    "v01": {"cognitive_exploration": True},
    "v01_norecov": {"cognitive_exploration": True, "cog_recovery": False},
    "v01_nohold": {"cognitive_exploration": True, "cog_probe_hold": False},
    "v01_nogap": {"cognitive_exploration": True, "cog_knowledge_gaps": False},
    "v01_noinfo": {"cognitive_exploration": True, "cog_info_gain": False},
}

TRACE_COUNTERS = (
    "probes_new", "probes_repeat", "loop_steps", "stalls_detected",
    "recovery_steps", "random_fallback_steps", "plans_used",
    "memory_correct", "memory_incorrect", "holds_started",
    "holds_resolved_open", "holds_resolved_fail",
    "holds_interrupted", "hold_preserve_steps",
)


# ---------------------------------------------------------------------------
# Perception/execution counters (harness-side, observation-only).
# ---------------------------------------------------------------------------
def step_perception_counters(before, after, res, env):
    """Compare what the student's ``learn()`` will derive from the two raw
    observations against the environment's exact event stream, plus belief
    vs reality divergence and corrupted ground atoms.

    Returns a dict of per-step counters.
    """
    c = {
        "claimed_open": 0, "claimed_close": 0,
        "true_open": 0, "true_close": 0,
        "phantom_open": 0, "phantom_close": 0,
        "missed_true_open": 0, "missed_true_close": 0,
        "belief_reality_divergence": 0,
        "ground_atom_mismatch": 0,
        "invalid_action": 0,
    }
    if not res.valid:
        c["invalid_action"] = 1
    for door, key in (("D1", "d1"), ("D2", "d2")):
        before_open = getattr(before, key)
        after_open = getattr(after, key)
        claimed_open = (not before_open) and after_open
        claimed_close = before_open and (not after_open)
        true_open = ("D1" if door == "D1" else "D2", "open") in res.events
        true_close = (door, "close") in res.events
        if claimed_open:
            c["claimed_open"] += 1
        if claimed_close:
            c["claimed_close"] += 1
        if true_open:
            c["true_open"] += 1
        if true_close:
            c["true_close"] += 1
        if claimed_open and not true_open:
            c["phantom_open"] += 1
        if claimed_close and not true_close:
            c["phantom_close"] += 1
        if true_open and not claimed_open:
            c["missed_true_open"] += 1
        if true_close and not claimed_close:
            c["missed_true_close"] += 1
    # belief (raw readings) vs reality (exact hidden state)
    state = env._state
    for door in ("D1", "D2"):
        b_open = getattr(after, "d1" if door == "D1" else "d2")
        if b_open != state.door_open[door]:
            c["belief_reality_divergence"] += 1
    # ground-truth atom mismatch (plate/switch readings vs exact state)
    obs_atoms = atoms_from_observation(after)
    true_atoms = atoms_from_state(state.plate_load, state.switch_state)
    if obs_atoms != true_atoms:
        c["ground_atom_mismatch"] = 1
    return c


def run_episode(agent, spec, collect):
    env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
    obs = env.reset(spec.spec_seed)
    prev = agent.probe_trace()
    agent.episode_start(spec_for(agent, spec), env.geom)
    steps = 0
    success = False
    true_events_total = 0
    opened_not_used = 0
    used_open = False
    while steps < env.max_steps:
        action = agent.act(obs)
        before = obs
        res = env.step(action)
        agent.learn(before, action, res)
        c = step_perception_counters(before, res.observation, res, env)
        for k, v in c.items():
            collect[k] += v
        steps += 1
        true_events_total += len(res.events)
        if res.events:
            opened_not_used += 1
        obs = res.observation
        if res.terminated:
            success = True
            break
    # did the agent ever exploit a real opening? (passed door = used)
    if success or any(agent.passed.get(d, False) for d in ("D1", "D2")):
        used_open = True
    agent.finish_episode(success)
    tr = agent.probe_trace()
    delta = {k: tr.get(k, 0) - prev.get(k, 0) for k in TRACE_COUNTERS}
    rec = {
        "success": success,
        "steps": steps,
        "true_events": true_events_total,
        "opened_not_used": opened_not_used if not used_open else 0,
        "perception": dict(collect),
        "trace": delta,
    }
    return rec


def run_cell(flags, splits, cell, seed_base, n_seeds, episodes, tag):
    cond = CONDITIONS[cell]
    pool = splits[cond["pool"]]
    per_seed = []
    all_records = []
    t0 = time.perf_counter()
    for si in range(n_seeds):
        seed = seed_base * 1000 + si
        stream = make_episodes(pool, tag, seed, episodes,
                               max_steps=cond["max_steps"],
                               noise_q=cond["noise"])
        agent = MetargonAcademyAgent(seed=seed, **flags)
        records = []
        for spec in stream:
            collect = {k: 0 for k in (
                "claimed_open", "claimed_close", "true_open", "true_close",
                "phantom_open", "phantom_close", "missed_true_open",
                "missed_true_close", "belief_reality_divergence",
                "ground_atom_mismatch", "invalid_action")}
            records.append(run_episode(agent, spec, collect))
        all_records.extend(records)
        succ = [1.0 if r["success"] else 0.0 for r in records]
        per_seed.append({
            "seed": seed,
            "success_rate": sum(succ) / len(succ) if succ else 0.0,
        })
    wall = time.perf_counter() - t0
    return _summarize(per_seed, all_records, wall)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _summarize(per_seed, all_records, wall):
    n = len(all_records)
    succ = [1.0 if r["success"] else 0.0 for r in all_records]
    steps_all = [r["steps"] for r in all_records]
    perc_keys = ("claimed_open", "claimed_close", "true_open", "true_close",
                 "phantom_open", "phantom_close", "missed_true_open",
                 "missed_true_close", "belief_reality_divergence",
                 "ground_atom_mismatch", "invalid_action")
    trace_keys = ("probes_new", "loop_steps", "stalls_detected",
                  "recovery_steps", "random_fallback_steps",
                  "holds_started", "holds_resolved_open",
                  "holds_resolved_fail", "holds_interrupted",
                  "hold_preserve_steps", "memory_correct",
                  "memory_incorrect")
    perc = {k: _mean(r["perception"][k] / max(1, r["steps"])
                      for r in all_records) for k in perc_keys}
    tr = {k: _mean(r["trace"][k] for r in all_records) for k in trace_keys}
    return {
        "n_episodes": n,
        "wall_s": round(wall, 1),
        "success_rate": _mean(succ),
        "mean_steps_all": _mean(steps_all),
        "opened_not_used_rate": _mean(
            1.0 if r["opened_not_used"] > 0 else 0.0 for r in all_records),
        "perception_per_step": perc,
        "trace_per_episode": tr,
        "per_seed_success": [p["success_rate"] for p in per_seed],
    }


def paired_bootstrap(a_succ, b_succ, seed=0, n_boot=2000):
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
    return {"mean": sum(diffs) / n_boot,
            "ci": (diffs[int(0.025 * n_boot)],
                   diffs[int(0.975 * n_boot) - 1])}


def run_diagnosis(seed_base=DIAG_SEED_BASE, n_seeds=N_SEEDS,
                  episodes=EPISODES, results_dir=RESULTS_DIR,
                  conditions=None, arms=None):
    splits = cached_splits()
    conditions = conditions or CONDITIONS
    arms = arms or ARMS
    out = {"run_id": "exploration-noise-diagnosis-v01",
           "seed_base": seed_base, "n_seeds": n_seeds,
           "episodes": episodes, "conditions": {}}
    for cell in conditions:
        out["conditions"][cell] = {"arms": {}}
        res = {}
        for name, flags in arms.items():
            r = run_cell(flags, splits, cell, seed_base, n_seeds,
                         episodes, name)
            res[name] = r
            out["conditions"][cell]["arms"][name] = r
            print(f"[{cell}] {name:14s} succ={r['success_rate']:.3f} "
                  f"phantom_open={r['perception_per_step']['phantom_open']:.3f} "
                  f"phantom_close={r['perception_per_step']['phantom_close']:.3f} "
                  f"diverg={r['perception_per_step']['belief_reality_divergence']:.3f} "
                  f"stalls={r['trace_per_episode']['stalls_detected']:.2f} "
                  f"recovery={r['trace_per_episode']['recovery_steps']:.2f} "
                  f"holds={r['trace_per_episode']['holds_started']:.2f} "
                  f"missed={r['perception_per_step']['missed_true_open']:.3f}")
        # paired comparisons vs baseline and vs v01
        base = res["baseline"]["per_seed_success"]
        v01 = res["v01"]["per_seed_success"]
        cmp = {}
        for name in arms:
            if name == "baseline":
                continue
            cmp[f"{name}_vs_baseline"] = paired_bootstrap(
                res[name]["per_seed_success"], base,
                seed=seed_base + 1)
            if name != "v01":
                cmp[f"{name}_vs_v01"] = paired_bootstrap(
                    res[name]["per_seed_success"], v01,
                    seed=seed_base + 2)
        out["conditions"][cell]["comparisons"] = cmp
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "diagnosis_summary.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return out


def main():
    ap = argparse.ArgumentParser(description="Exploration noise diagnosis")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--seed-base", type=int, default=DIAG_SEED_BASE)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()
    conditions = {c: CONDITIONS[c] for c in (args.conditions or CONDITIONS)
                  if c in CONDITIONS}
    arms = {a: ARMS[a] for a in (args.arms or ARMS) if a in ARMS}
    run_diagnosis(seed_base=args.seed_base, n_seeds=args.seeds,
                  episodes=args.episodes, results_dir=args.results_dir,
                  conditions=conditions, arms=arms)


class TestNoiseDiagnosisSmoke(unittest.TestCase):
    def test_smoke(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = run_diagnosis(seed_base=999100, n_seeds=1, episodes=6,
                                results_dir=tmp,
                                conditions={"noisy": CONDITIONS["noisy"]},
                                arms={"baseline": {}, "v01": ARMS["v01"]})
            cell = out["conditions"]["noisy"]
            self.assertIn("baseline", cell["arms"])
            self.assertIn("v01", cell["arms"])
            self.assertIn("v01_vs_baseline", cell["comparisons"])
            self.assertGreater(cell["arms"]["v01"]["n_episodes"], 0)


if __name__ == "__main__":
    main()
