#!/usr/bin/env python3
"""v0.10 noise diagnostics (analysis only; no privileged info to the agent).

Quantifies, per episode on noisy_obs:
  * true door events (offline, from the environment's own event list),
  * events the agent *believed* happened (belief door flips),
  * precision/recall of believed vs true events (per door),
  * invalid-action rate,
  * end-of-episode false-verified hypotheses (offline oracle vs true rules),
  * replay amplification proxy: replay_applied count vs true events.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "python"))

from lab_env import DOORS, LabWorld, generate_worlds  # noqa: E402
from metargon_v10 import MetargonV10  # noqa: E402


def run(condition="noisy_obs", seeds=6, episodes=50, make=None):
    out = []
    for s in range(seeds):
        worlds = generate_worlds(condition, 11800 + s, episodes)
        pol = make(s + 1000003)
        for spec in worlds:
            pol.episode_start()
            w = LabWorld(spec.config, ep_seed=spec.seed,
                         noise_q=spec.noise_q, max_steps=spec.max_steps)
            true_events = {"D1": [], "D2": []}
            believed_events = {"D1": [], "D2": []}
            invalid = 0
            prev_door = {"D1": False, "D2": False}
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = pol.act(obs)
                res = w.step(a)
                invalid += int(not res.valid)
                for door, kind in res.door_events:
                    true_events[door].append((w.steps, kind))
                b = pol.belief.belief
                for door in DOORS:
                    cur = b.door_open[door]
                    if cur != prev_door[door]:
                        believed_events[door].append(
                            (w.steps, "open" if cur else "close"))
                    prev_door[door] = cur
                pol.observe_outcome(obs, a, res)
            # offline oracle: false-verified hypotheses
            false_verified = 0
            verified_total = 0
            for h in pol.memory.hypotheses:
                if h.status != "verified":
                    continue
                verified_total += 1
                true_cond = spec.config.d1_cond if h.door == "D1" \
                    else spec.config.d2_cond
                if h.antecedent != frozenset(true_cond):
                    false_verified += 1
            d = pol.diagnostics()
            row = {
                "seed": s, "condition": condition,
                "success": bool(w.finished),
                "invalid": invalid,
                "true_D1": len(true_events["D1"]),
                "true_D2": len(true_events["D2"]),
                "believed_D1": len(believed_events["D1"]),
                "believed_D2": len(believed_events["D2"]),
                "verified_total": verified_total,
                "false_verified": false_verified,
                "replay_applied": d["memory"]["replay_applied"],
                "mem_generated": d["memory"]["generated"],
                "confirmed_events": d.get("confirmed_events_total"),
                "dropped_candidates": d.get("dropped_candidates_total"),
            }
            out.append(row)
    return out


def main():
    from metargon_v11 import MetargonV11
    out_dir = root / "results" / "v11"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, make in (("v10", lambda s: MetargonV10(s)),
                       ("v11", lambda s: MetargonV11(s))):
        rows = run(seeds=4, episodes=40, make=make)
        report[name] = rows
        n = len(rows)
        t = sum(r["true_D1"] + r["true_D2"] for r in rows)
        be = sum(r["believed_D1"] + r["believed_D2"] for r in rows)
        print(f"{name}: eps={n} true_events={t} believed_events={be} "
              f"ratio={be / max(1, t):.2f} invalid/ep="
              f"{sum(r['invalid'] for r in rows) / n:.3f} "
              f"false_verified/ep={sum(r['false_verified'] for r in rows) / n:.2f} "
              f"verified/ep={sum(r['verified_total'] for r in rows) / n:.2f} "
              f"replay/ep={sum(r['replay_applied'] for r in rows) / n:.1f} "
              f"success={sum(r['success'] for r in rows) / n * 100:.1f}%")
        if name == "v11":
            print(f"  v11 confirmed/ep={sum(r['confirmed_events'] or 0 for r in rows) / n:.1f} "
                  f"dropped/ep={sum(r['dropped_candidates'] or 0 for r in rows) / n:.1f}")
    (out_dir / "diagnostics_noise.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
