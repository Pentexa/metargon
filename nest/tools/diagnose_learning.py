"""METARGON NEST — controlled learning-validation experiment (CLI).

Compares, under matched conditions:

1. fresh (no training)
2. trained without guidance
3. trained with exploration guidance
4. trained with an unverified causal claim
5. trained with misleading guidance
6. trained then memory-wiped

on the same evaluation task/seed streams, and reports success, action
efficiency, memory usage and uncertainty. Evaluation is separated from
training (different pool configs, same planner budgets).

No result is forced; the report states exactly what was measured.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (ROOT, ROOT / "python", ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from academy.envs.tasks import get_splits, make_episodes
from nest.agent_session import Challenge, NestSession
from nest.storage import NestStorage
from nest.teaching import parse_teaching_text

TRAIN_EPS = 4
EVAL_EPS = 3
MAX_STEPS = 80


def _run(session, episodes, budget=MAX_STEPS):
    out = []
    for spec in episodes:
        from academy.envs.gridworld import MechanismGrid
        env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
        obs = env.reset(spec.spec_seed)
        session.agent.episode_start(spec, env.geom)
        session.teaching.begin_episode_snapshot()
        success = False
        invalid = 0
        steps = 0
        for _ in range(budget):
            if env.finished or env.steps >= env.max_steps:
                break
            action = session.agent.act(obs)
            before = obs
            res = env.step(action)
            session.agent.learn(before, action, res)
            session.teaching.record_step(action, res)
            invalid += int(not res.valid)
            steps += 1
            obs = res.observation
            if res.terminated:
                success = True
                break
        session.agent.finish_episode(success)
        out.append({"success": success, "steps": steps, "invalid": invalid})
    return out


def _arm(session, seed, train_eps, eval_eps, teaching=None, wipe=False):
    session.teaching.clear_all()
    if teaching:
        session.teaching.apply(teaching)
    if train_eps:
        _run(session, train_eps)
    if wipe:
        session.agent.memory.hypotheses = []
        session.agent.memory.evidence_bank = []
        session.agent.trace["memory_actions"] = 0
        session.agent.trace["hypothesis_retrieved"] = 0
    ev = _run(session, eval_eps)
    mem = session.memory_snapshot()
    tr = session.agent.trace
    return {
        "success": statistics.fmean([r["success"] for r in ev]),
        "mean_steps": statistics.fmean([r["steps"] for r in ev]),
        "invalid_total": sum(r["invalid"] for r in ev),
        "memory": {
            "verified": mem["stats"].get("verified", 0),
            "contradicted": mem["stats"].get("contradicted", 0),
            "uncertain": mem["stats"].get("hypothesis", 0)
            + mem["stats"].get("predicted", 0),
            "generated": mem["generated"],
        },
        "trace": {
            "plans_used": tr.get("plans_used", 0),
            "memory_actions": tr.get("memory_actions", 0),
            "hypothesis_retrieved": tr.get("hypothesis_retrieved", 0),
            "random_fallback_steps": tr.get("random_fallback_steps", 0),
        },
        "teaching": {
            "guidance_applied": session.teaching.stats["guidance_applied"],
            "claims_added": session.teaching.stats["claims_added"],
            "claim_probes": session.teaching.stats["teacher_claim_probes"],
        },
    }


def run_seed(seed, splits, T, E, storage_root):
    train_eps = make_episodes([T], "train", seed, TRAIN_EPS, MAX_STEPS)
    eval_eps = make_episodes([E], "eval", seed, EVAL_EPS, MAX_STEPS)
    arms = {}
    arms["fresh"] = _arm(NestSession(storage=NestStorage(storage_root / f"s{seed}"),
                                     seed=seed), seed, [], eval_eps)
    arms["train_no_guidance"] = _arm(
        NestSession(storage=NestStorage(storage_root / f"s{seed}b"), seed=seed),
        seed, train_eps, eval_eps)
    arms["train_guidance"] = _arm(
        NestSession(storage=NestStorage(storage_root / f"s{seed}c"), seed=seed),
        seed, train_eps, eval_eps,
        teaching=parse_teaching_text("try switch S1"))
    arms["train_claim"] = _arm(
        NestSession(storage=NestStorage(storage_root / f"s{seed}d"), seed=seed),
        seed, train_eps, eval_eps,
        teaching=parse_teaching_text("I think switch S1 opens door D1"))
    arms["train_misleading"] = _arm(
        NestSession(storage=NestStorage(storage_root / f"s{seed}e"), seed=seed),
        seed, train_eps, eval_eps,
        teaching=parse_teaching_text("try empty P1"))
    arms["train_wiped"] = _arm(
        NestSession(storage=NestStorage(storage_root / f"s{seed}f"), seed=seed),
        seed, train_eps, eval_eps, wipe=True)
    return arms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    splits = get_splits()
    T = next(c for c in splits["train"] if c.n_atoms() <= 2)
    E = next(c for c in splits["test"]
             if c.topology == T.topology and c.key != T.key)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        agg = {}
        for i in range(args.seeds):
            seed = 90000 + i * 37
            agg[seed] = run_seed(seed, splits, T, E, root)
        # aggregate across seeds
        report = {"T_topology": T.topology, "E_topology": E.topology,
                  "T_key": str(T.key), "E_key": str(E.key),
                  "seeds": sorted(agg),
                  "arms": {}}
        for arm in agg[list(agg)[0]]:
            report["arms"][arm] = {}
            for metric in ("success", "mean_steps", "invalid_total"):
                vals = [agg[s][arm][metric] for s in agg]
                report["arms"][arm][metric] = {
                    "mean": round(statistics.fmean(vals), 3),
                    "values": [round(v, 3) for v in vals],
                }
            report["arms"][arm]["memory"] = {
                k: round(statistics.fmean([agg[s][arm]["memory"][k]
                                           for s in agg]), 1)
                for k in ("verified", "contradicted", "uncertain", "generated")
            }
            report["arms"][arm]["trace"] = {
                k: round(statistics.fmean([agg[s][arm]["trace"][k]
                                           for s in agg]), 1)
                for k in ("plans_used", "memory_actions",
                          "hypothesis_retrieved", "random_fallback_steps")
            }
        # print table
        print("arm                    success  steps  invalid  verified  contrad  uncert  generated")
        for arm, r in report["arms"].items():
            print(f"{arm:22s} "
                  f"{r['success']['mean']:6.2f}  "
                  f"{r['mean_steps']['mean']:5.1f}  "
                  f"{r['invalid_total']['mean']:7.1f}  "
                  f"{r['memory']['verified']:8.0f}  "
                  f"{r['memory']['contradicted']:7.0f}  "
                  f"{r['memory']['uncertain']:6.0f}  "
                  f"{r['memory']['generated']:9.0f}")
        print("\ntrace: plans_used / memory_actions / retrieved / fallback")
        for arm, r in report["arms"].items():
            t = r["trace"]
            print(f"  {arm:22s} {t['plans_used']:7.1f} "
                  f"{t['memory_actions']:8.1f} {t['hypothesis_retrieved']:10.1f} "
                  f"{t['random_fallback_steps']:8.1f}")
        if args.json:
            out = Path(args.json)
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\njson report -> {out}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
