"""NEST — post-training regression diagnosis (isolated mechanisms).

Objective: reproduce fresh-vs-trained failure on IDENTICAL evaluation
episodes, then isolate the cause across independent mechanisms:

- memory CONTENT (hypotheses/evidence/atom stats cleared)
- memory FUNCTION (query/bank/transfer paths muted, content kept)
- exploration state (tried/plan_cache/pending_probe reset)
- counters (episodes/successes/epoch reset)
- RNG state (restored to pre-training state)
- transfer threshold (min_verified_epochs_for_transfer 1 vs 2)

The student remains the only action authority. Every arm consumes the same
evaluation episodes and the same per-decision budgets. Results are
reported as measured; no success ordering is forced.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (ROOT, ROOT / "python", ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from academy.envs.tasks import get_splits, make_episodes
from nest.agent_session import NestSession
from nest.storage import NestStorage

TRAIN_EPS = 4
EVAL_EPS = 3
MAX_STEPS = 80
SEED_BASE = 90000
SEED_STRIDE = 37


# ---------------------------------------------------------------------------
# controlled per-arm mutations
# ---------------------------------------------------------------------------
def mutation_clear_memory(agent):
    mem = agent.memory
    mem.hypotheses = []
    mem.evidence_bank = []
    mem._atom_stats = {}
    mem._pair_stats = {}
    mem._atom_event_support = {}
    mem._atom_nonevent_support = {}
    mem._door_event_count = {}
    mem._door_nonevent_count = {}
    mem._continuous_true = {}
    mem._dead = {d: __import__("collections").deque() for d in ("D1", "D2")}


def mutation_mute_memory(agent):
    """Functional ablation: memory content is KEPT but all read paths used
    by act() are disconnected. This is the genuinely matched ablation:
    same state, no memory influence."""
    mem = agent.memory
    mem.query = lambda *a, **k: []
    mem.bank_entries = lambda: []
    mem.transfer_atoms = lambda *a, **k: []
    mem.transfer_pairs = lambda *a, **k: []
    # protect path also reads query()
    agent._protected_atoms = lambda: frozenset()


def mutation_reset_explore_state(agent):
    agent.tried = {d: set() for d in agent.geom.door_cell}
    agent.pending_probe = {}
    agent.plan_cache = {}
    agent._goal_infeasible = {}
    agent.passed = {d: False for d in agent.geom.door_cell}


def mutation_reset_counters(agent):
    agent.episodes = 0
    agent.successes = 0
    agent.steps = 0
    agent.memory._current_epoch = 0


def mutation_reset_rng(agent, rng0):
    agent.rng.setstate(copy.deepcopy(rng0))


MUTATIONS = {
    "clear_memory": lambda agent, ctx: mutation_clear_memory(agent),
    "mute_memory": lambda agent, ctx: mutation_mute_memory(agent),
    "reset_explore_state": lambda agent, ctx: mutation_reset_explore_state(agent),
    "reset_counters": lambda agent, ctx: mutation_reset_counters(agent),
    "reset_rng": lambda agent, ctx: mutation_reset_rng(agent, ctx["rng0"]),
}


def make_session(seed, thr):
    kw = dict(memory_episode_scoped=True,
              memory_min_verified_epochs_for_transfer=thr,
              memory_atom_transfer=True,
              probe_bank_reuse=True,
              probe_reset_per_episode=True)
    return NestSession(storage=NestStorage(Path(tempfile.mkdtemp())),
                       seed=seed, agent_kwargs=kw)


def run_episodes(agent, episodes, collect_probe=True):
    agent._collect_probe_log = collect_probe
    rows = []
    for spec in episodes:
        from academy.envs.gridworld import MechanismGrid
        env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
        obs = env.reset(spec.spec_seed)
        agent.episode_start(spec, env.geom)
        success = False
        steps = 0
        invalid = 0
        probes = []
        actions_seq = []
        for _ in range(MAX_STEPS):
            if env.finished or env.steps >= env.max_steps:
                break
            action = agent.act(obs)
            actions_seq.append(action)
            before = obs
            res = env.step(action)
            agent.learn(before, action, res)
            invalid += int(not res.valid)
            steps += 1
            probes.extend(agent.trace.get("planned_probes", []))
            agent.trace["planned_probes"] = []
            obs = res.observation
            if res.terminated:
                success = True
                break
        agent.finish_episode(success)
        rows.append({"success": success, "steps": steps, "invalid": invalid,
                     "probes": probes, "actions": actions_seq})
    return rows


def arm(seed, thr, train_eps, eval_eps, mutations=()):
    sess = make_session(seed, thr)
    agent = sess.agent
    ctx = {"rng0": agent.rng.getstate()}
    if train_eps:
        run_episodes(agent, train_eps)
    for name in mutations:
        MUTATIONS[name](agent, ctx)
    ev = run_episodes(agent, eval_eps)
    mem = sess.memory_snapshot()
    tr = agent.trace
    return {
        "success": statistics.fmean([r["success"] for r in ev]),
        "steps_used": [r["steps"] for r in ev],
        "invalid": sum(r["invalid"] for r in ev),
        "first_divergence_actions": [a for r in ev for a in r["actions"]],
        "probe_sources": _source_hist(ev),
        "memory": {k: mem["stats"].get(k, 0)
                   for k in ("verified", "contradicted", "hypothesis", "predicted")},
        "memory_generated": mem["generated"],
        "trace": {"plans_used": tr.get("plans_used", 0),
                  "memory_actions": tr.get("memory_actions", 0),
                  "hypothesis_retrieved": tr.get("hypothesis_retrieved", 0),
                  "random_fallback_steps": tr.get("random_fallback_steps", 0)},
    }


def _actions(row):
    return row["actions"]


def _source_hist(ev):
    h = {}
    for r in ev:
        for _, _, src in r["probes"]:
            h[src] = h.get(src, 0) + 1
    return h


def first_divergence(steps_a, steps_b):
    n = min(len(steps_a), len(steps_b))
    for i in range(n):
        if steps_a[i] != steps_b[i]:
            return i
    return n  # equal prefix; returns length when equal


def run_seed(seed, pairs):
    out = {"seed": seed, "arms": {}}
    for tag, (T, E) in pairs.items():
        train_eps = make_episodes([T], "train", seed, TRAIN_EPS, MAX_STEPS)
        eval_eps = make_episodes([E], "eval", seed, EVAL_EPS, MAX_STEPS)
        arm_specs = {
            "fresh": (1, (), ()),
            "trained_full_thr1": (1, train_eps, ()),
            "trained_no_memory_content": (1, train_eps, ("clear_memory",)),
            "trained_memory_muted": (1, train_eps, ("mute_memory",)),
            "trained_explore_state_reset": (1, train_eps, ("reset_explore_state",)),
            "trained_counters_reset": (1, train_eps, ("reset_counters",)),
            "trained_rng_reset": (1, train_eps, ("reset_rng",)),
            "trained_full_thr2": (2, train_eps, ()),
        }
        for name, (thr, tr, muts) in arm_specs.items():
            key = f"{tag}/{name}"
            out["arms"][key] = arm(seed, thr, tr, eval_eps, mutations=muts)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    splits = get_splits()
    # two independent training/eval pairs; each pair records the actual
    # topology of each side (they need not match).
    trains = [c for c in splits["train"] if c.n_atoms() <= 2][:2]
    tests = []
    # T1E1: same topology as train[0], different rule
    same = [c for c in splits["test"]
            if c.topology == trains[0].topology and c.key != trains[0].key]
    tests.append(same[0] if same else trains[0])
    # T2E2: deliberately CROSS-topology — train TP2, eval TP1
    cross = [c for c in splits["test"] if c.topology != trains[1].topology]
    tests.append(cross[0] if cross else tests[0])
    pairs = {"T1E1": (trains[0], tests[0]),
             "T2E2": (trains[1], tests[1])}
    pair_topology = {tag: {"train": T.topology, "eval": E.topology}
                     for tag, (T, E) in pairs.items()}

    per_seed = {}
    for i in range(args.seeds):
        seed = SEED_BASE + i * SEED_STRIDE
        per_seed[seed] = run_seed(seed, pairs)

    # aggregate arms across seeds per pair
    report = {"trains": [str(t.key) for t in trains],
              "tests": [str(t.key) for t in tests],
              "pair_topology": pair_topology,
              "seeds": sorted(per_seed),
              "pairs": {}}
    for tag in pairs:
        report["pairs"][tag] = {}
        for name in ("fresh", "trained_full_thr1", "trained_no_memory_content",
                     "trained_memory_muted", "trained_explore_state_reset",
                     "trained_counters_reset", "trained_rng_reset",
                     "trained_full_thr2"):
            key = f"{tag}/{name}"
            succ = [per_seed[s]["arms"][key]["success"] for s in per_seed]
            mem_actions = [per_seed[s]["arms"][key]["trace"]["memory_actions"]
                           for s in per_seed]
            retrieved = [per_seed[s]["arms"][key]["trace"]["hypothesis_retrieved"]
                         for s in per_seed]
            report["pairs"][tag][name] = {
                "success_mean": round(statistics.fmean(succ), 3),
                "success_values": [round(v, 2) for v in succ],
                "steps_used": [per_seed[s]["arms"][key]["steps_used"]
                               for s in per_seed],
                "memory_actions_mean": round(statistics.fmean(mem_actions), 1),
                "retrieved_mean": round(statistics.fmean(retrieved), 1),
                "probe_sources": _merge_sources(
                    [per_seed[s]["arms"][key]["probe_sources"] for s in per_seed]),
                "memory": {k: round(statistics.fmean(
                    [per_seed[s]["arms"][key]["memory"][k] for s in per_seed]), 1)
                    for k in ("verified", "contradicted", "hypothesis", "predicted")},
            }

    # first-divergence trace: fresh vs each trained arm, first seed
    s0 = per_seed[SEED_BASE]
    for tag in pairs:
        fresh = s0["arms"][f"{tag}/fresh"]["first_divergence_actions"]
        report["pairs"][tag]["fresh_actions"] = fresh
        for name in ("trained_full_thr1", "trained_no_memory_content",
                     "trained_memory_muted", "trained_full_thr2"):
            other = s0["arms"][f"{tag}/{name}"]["first_divergence_actions"]
            div = first_divergence(fresh, other)
            report["pairs"][tag][f"first_divergence_{name}"] = div

    _print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2),
                                   encoding="utf-8")
        print(f"\njson report -> {args.json}")
    return 0


def _merge_sources(hist_list):
    out = {}
    for h in hist_list:
        for k, v in h.items():
            out[k] = round(statistics.fmean([h.get(k, 0) for h in hist_list]), 1)
    return out


def _print_report(report):
    print("=" * 86)
    print("POST-TRAINING REGRESSION DIAGNOSIS (matched budgets; identical eval episodes)")
    print("=" * 86)
    for tag, arms in report["pairs"].items():
        topo = report["pair_topology"][tag]
        print(f"\n-- pair {tag}: train {topo['train']} -> eval {topo['eval']} "
              f"(train {report['trains'][0] if tag=='T1E1' else report['trains'][1]}, "
              f"eval {report['tests'][0] if tag=='T1E1' else report['tests'][1]}) --")
        print(f"{'arm':26s} {'succ':>6s} {'memAct':>7s} {'retr':>6s} {'ver':>4s} {'con':>4s} {'unc':>4s}")
        for name in ("fresh", "trained_full_thr1", "trained_no_memory_content",
                     "trained_memory_muted", "trained_explore_state_reset",
                     "trained_counters_reset", "trained_rng_reset",
                     "trained_full_thr2"):
            a = arms[name]
            print(f"{name:26s} {a['success_mean']:6.2f} {a['memory_actions_mean']:7.1f} "
                  f"{a['retrieved_mean']:6.1f} {a['memory']['verified']:4.0f} "
                  f"{a['memory']['contradicted']:4.0f} {a['memory']['hypothesis']+a['memory']['predicted']:4.0f}")
        if "fresh_actions" in arms:
            print("  probe-source hist (fresh):", arms["fresh_actions"])
        for name in ("trained_full_thr1", "trained_no_memory_content",
                     "trained_memory_muted", "trained_full_thr2"):
            key = f"first_divergence_{name}"
            if key in arms:
                print(f"  first ACTION divergence (true environment actions) fresh vs {name}: {arms[key]}")
    print()


if __name__ == "__main__":
    sys.exit(main())
