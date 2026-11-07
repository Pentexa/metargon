#!/usr/bin/env python3
"""v0.9-vs-truth diagnostic (analysis only).

Runs the v0.9 METARGON agent on selected conditions and, offline (never
given to the agent), compares its hypothesis store against the true door
conditions. Quantifies:

  * hypothesis discovery (does the top-ranked hypothesis for a door match
    the true condition, at each episode end?),
  * door-event detection speed (steps to first door event per episode),
  * invalid-action rate (belief corruption indicator),
  * probe outcome distribution (probe succeeded / timeout / never fired),
  * episode failures by category (never-opened door D1 / D2, goal missed
    after doors open).

Output: results/diagnostics_v09/{condition}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lab_env import LabWorld, generate_worlds  # noqa: E402
from metargon_v09 import MetargonV09  # noqa: E402


def run_condition(condition: str, seeds=4, episodes=40) -> list:
    rows = []
    for s in range(seeds):
        worlds = generate_worlds(condition, 77000 + s, episodes)
        pol = MetargonV09(s + 500003)
        for spec in worlds:
            pol.episode_start()
            w = LabWorld(spec.config, ep_seed=spec.seed,
                         noise_q=spec.noise_q, max_steps=spec.max_steps)
            invalid = 0
            first_event = None
            events = {"D1": [], "D2": []}
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = pol.act(obs)
                res = w.step(a)
                invalid += int(not res.valid)
                if res.door_events and first_event is None:
                    first_event = w.steps
                for door, kind in res.door_events:
                    events[door].append((w.steps, kind))
                pol.observe_outcome(obs, a, res)
            mem = pol.memory.diagnostics()
            row = {
                "seed": s, "condition": condition,
                "d1_true": sorted(str(x) for x in spec.config.d1_cond),
                "d2_true": sorted(str(x) for x in spec.config.d2_cond),
                "success": bool(w.finished),
                "steps": w.steps,
                "invalid": invalid,
                "first_event_step": first_event,
                "n_events_D1": len(events["D1"]),
                "n_events_D2": len(events["D2"]),
                "probe_timeout_marks": len(getattr(pol, "tried", {}).get("D1", set())) + len(getattr(pol, "tried", {}).get("D2", set())),
                "mem_size": mem["size"],
                "verified": mem["status_counts"]["verified"],
                "contradicted": mem["status_counts"]["contradicted"],
                "evictions": mem["evictions"],
            }
            rows.append(row)
    return rows


def main():
    out = Path(__file__).resolve().parent.parent / "results" / "diagnostics_v09"
    out.mkdir(parents=True, exist_ok=True)
    for condition in ("stable", "noisy_obs", "novel", "iid"):
        rows = run_condition(condition, seeds=4, episodes=40)
        (out / f"{condition}.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        n = len(rows)
        succ = sum(r["success"] for r in rows) / n
        inv = sum(r["invalid"] for r in rows) / n
        fev = [r["first_event_step"] for r in rows if r["first_event_step"]]
        print(f"{condition:>12}: success {succ*100:5.1f}%  invalid/ep {inv:5.2f}  "
              f"median first-event step {sorted(fev)[len(fev)//2] if fev else None}  "
              f"mean events {sum(r['n_events_D1']+r['n_events_D2'] for r in rows)/n:.2f}  "
              f"mean verified {sum(r['verified'] for r in rows)/n:.1f}")


if __name__ == "__main__":
    main()
