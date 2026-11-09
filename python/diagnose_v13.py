#!/usr/bin/env python3
"""METARGON v0.13 — temporal discovery diagnostics (Mission A).

Reproduces the exact paired world stream used by `benchmark_v12.py` and
instruments the temporal memory to explain WHY exact ordered-rule
discovery succeeds in some seeds and fails in others.

For every seed × episode it records:
  * the true ordered rule per door (first, second);
  * every ordered pair of belief-transition flips observed by the temporal
    memory (the *candidate-generation floor*: if the true pair never
    appears as consecutive belief flips with the current generator, no
    correct candidate can ever be created);
  * every temporal hypothesis created, with the step of creation, and
    whether it matches the true rule (correct candidate);
  * support / contradiction / eviction / status transitions of correct
    candidates;
  * temporal probe attempts by the agent and how many were blocked by the
    atom-protection mechanism.

Usage:
  python python/diagnose_v13.py --family l --conditions stable,novel \
      --seeds 4 --episodes 40 --json results/v13/diag_l.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "generalization_audit"))

from lab_env import LabEpisode  # noqa: E402
from benchmark_v12 import family_pools, make_worlds, make_world  # noqa: E402
from metargon_v12 import MetargonV12  # noqa: E402
from metargon_v13 import MetargonV13  # noqa: E402
from metargon_v09 import atoms_of_state  # noqa: E402
from temporal_memory import (  # noqa: E402
    TemporalHypothesis, VERIFIED, PREDICTED, CONTRADICTED, HYPOTHESIS,
)

_SEED_OFFSET = 11800
_TRUE_RULE_KEY = "true_rule"


def _true_rule(spec: LabEpisode, door: str):
    first = frozenset(spec.config.first_for(door))
    second = frozenset(spec.config.second_for(door))
    return (first, second)


class InstrumentedTemporal:
    """Wraps a TemporalMemory with counters for the diagnostic analysis."""

    def __init__(self, temporal, true_rule):
        self.t = temporal
        self.true_rule = true_rule
        self.created: list[dict] = []
        self.correct_created = 0
        self.correct_supported = 0
        self.correct_contradicted = 0
        self.correct_evicted = 0
        self.correct_verified_max = 0
        self.correct_predicted_max = 0
        self.observed_flips: list[tuple] = []   # (door, atom, step)
        self.observed_pairs: list[tuple] = []   # (door, first, second, step)
        self._correct_hids: set[int] = set()
        self.evicted_hids: set[int] = set()
        self._orig_add = temporal._add
        self._orig_evict = temporal._evict_one
        self._orig_support = temporal._support
        self._orig_contradict = temporal._contradict
        temporal._add = self._add
        temporal._evict_one = self._evict_one
        temporal._support = self._support
        temporal._contradict = self._contradict

    # -- hooks -----------------------------------------------------------
    def _is_correct(self, h) -> bool:
        door = h.door
        true_first, true_second = self.true_rule.get(door)
        return (frozenset(h.first), frozenset(h.second)) == (
            true_first, true_second)

    def _add(self, first, second, consequence):
        h = self._orig_add(first, second, consequence)
        self.created.append({
            "step": self.t.step,
            "door": h.door,
            "first": sorted(tuple(first)),
            "second": sorted(tuple(second)),
            "h_id": h.h_id,
            "correct": self._is_correct(h) if h.door in self.true_rule else False,
        })
        if h.door in self.true_rule and self._is_correct(h):
            self.correct_created += 1
            self._correct_hids.add(h.h_id)
        return h

    def _evict_one(self):
        if not self.t.hypotheses:
            return
        victim = min(self.t.hypotheses,
                     key=lambda h: (h.score(), -h.last_update, -h.h_id))
        if victim.h_id in self._correct_hids:
            self.correct_evicted += 1
        self.evicted_hids.add(victim.h_id)
        self._orig_evict()

    def _support(self, h, step, weight=1.0):
        if h.h_id in self._correct_hids and h.n_sup + 1 > h.n_sup:
            self.correct_supported += 1
        self._orig_support(h, step, weight=weight)
        if h.h_id in self._correct_hids:
            self.correct_verified_max = max(
                self.correct_verified_max, int(h.status == VERIFIED))
            self.correct_predicted_max = max(
                self.correct_predicted_max, int(h.status in (VERIFIED, PREDICTED)))

    def _contradict(self, h):
        if h.h_id in self._correct_hids:
            self.correct_contradicted += 1
        self._orig_contradict(h)

    # -- flip observation (called by the agent wrapper) -------------------
    def observe_flips(self, door: str, step: int, newly_true):
        for atom in sorted(newly_true, key=str):
            self.observed_flips.append((door, atom, step))

    def summary(self) -> dict:
        # candidate-generation floor: did the *raw belief flip stream*
        # ever contain the true pair in the correct order within the
        # window? If not, no generator could ever create the candidate.
        floor = {}
        for door, (first, second) in self.true_rule.items():
            first_atom = next(iter(first))
            second_atom = next(iter(second))
            first_steps = [s for d, a, s in self.observed_flips
                           if d == door and a == first_atom]
            observed = False
            for s2 in [s for d, a, s in self.observed_flips
                       if d == door and a == second_atom]:
                if any(0 < s2 - s1 <= self.t.window for s1 in first_steps):
                    observed = True
                    break
            floor[door] = observed
        return {
            "correct_created": self.correct_created,
            "correct_supported": self.correct_supported,
            "correct_contradicted": self.correct_contradicted,
            "correct_evicted": self.correct_evicted,
            "correct_verified_max": self.correct_verified_max,
            "correct_predicted_max": self.correct_predicted_max,
            "hypotheses_created_total": len(self.created),
            "hypotheses_current": len(self.t.hypotheses),
            "evictions_total": self.t.evictions,
            "true_pair_observed": floor,
            "flips_observed": len(self.observed_flips),
            "true_rule": {d: [sorted(tuple(k)) for k in r]
                          for d, r in self.true_rule.items()},
        }

    def close(self):
        self.t._add = self._orig_add
        self.t._evict_one = self._orig_evict
        self.t._support = self._orig_support
        self.t._contradict = self._orig_contradict


class DiagAgent(MetargonV12):
    """MetargonV12 with hooks to observe flips and temporal probes."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._diag = None
        self._probe_targets = []
        self.temporal_probe_blocked = 0
        self.temporal_probe_attempts = 0


class DiagAgentV13(MetargonV13):
    """MetargonV13 with the same hooks."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._diag = None
        self._probe_targets = []
        self.temporal_probe_blocked = 0
        self.temporal_probe_attempts = 0

    def _feed_temporal(self, pre_atoms, post_atoms):
        if self.temporal is None:
            return
        step = self.memory.step
        newly_true = post_atoms - pre_atoms
        for atom in newly_true:
            if atom[0] in ("door_open",):
                continue
            comp = atom[1]
            door = "D1" if comp in ("P1", "S1") else "D2"
            self.temporal.transitions[door].append((step, atom))
        if self._diag is not None:
            for door in ("D1", "D2"):
                sub_new = frozenset(a for a in newly_true
                                    if (a[1] in ("P1", "S1") and door == "D1")
                                    or (a[1] in ("P2", "S2") and door == "D2"))
                self._diag.observe_flips(door, step, sub_new)
        super()._feed_temporal(pre_atoms, post_atoms)

    def _choose_probe(self):
        if self.temporal is not None:
            b = self.belief.belief
            protected = self._protected_atoms()
            for d in ("D1", "D2"):
                if b.door_open[d]:
                    continue
                h = self.temporal.best_for_door(d)
                if h is None or h.status not in (VERIFIED, PREDICTED):
                    continue
                now = atoms_of_state(b)
                target = h.first if not h.first <= now else h.second
                if target <= now:
                    continue
                self.temporal_probe_attempts += 1
                if any(a in protected for a in target):
                    self.temporal_probe_blocked += 1
        return super()._choose_probe()

    def set_diag(self, diag):
        self._diag = diag

    def _feed_temporal(self, pre_atoms, post_atoms):
        if self.temporal is None:
            return
        step = self.memory.step
        newly_true = post_atoms - pre_atoms
        for atom in newly_true:
            if atom[0] in ("door_open",):
                continue
            comp = atom[1]
            door = "D1" if comp in ("P1", "S1") else "D2"
            self.temporal.transitions[door].append((step, atom))
        if self._diag is not None:
            for door in ("D1", "D2"):
                sub_new = frozenset(a for a in newly_true
                                    if (a[1] in ("P1", "S1") and door == "D1")
                                    or (a[1] in ("P2", "S2") and door == "D2"))
                self._diag.observe_flips(door, step, sub_new)
        super()._feed_temporal(pre_atoms, post_atoms)

    def _choose_probe(self):
        # count protection-blocked temporal attempts before the original code
        if self.temporal is not None:
            b = self.belief.belief
            protected = self._protected_atoms()
            for d in ("D1", "D2"):
                if b.door_open[d]:
                    continue
                h = self.temporal.best_for_door(d)
                if h is None or h.status not in (VERIFIED, PREDICTED):
                    continue
                now = atoms_of_state(b)
                target = h.first if not h.first <= now else h.second
                if target <= now:
                    continue
                self.temporal_probe_attempts += 1
                if any(a in protected for a in target):
                    self.temporal_probe_blocked += 1
        return super()._choose_probe()

    def ep_diag(self):
        return {
            "probe_attempts": self.temporal_probe_attempts,
            "probe_blocked": self.temporal_probe_blocked,
        }


def play_episode_diag(spec, policy: DiagAgent, family="l"):
    world = make_world(spec, family)
    policy.episode_start()
    invalid = 0
    events = 0
    first_event_step = None
    door_event_flags = {"D1": False, "D2": False}
    while not world.finished and world.steps < world.max_steps:
        obs = world.observe()
        action = policy.act(obs)
        res = world.step(action)
        invalid += int(not res.valid)
        if res.door_events:
            events += len(res.door_events)
            for door, _ in res.door_events:
                door_event_flags[door] = True
            if first_event_step is None:
                first_event_step = world.steps
        policy.observe_outcome(obs, action, res)
    return {
        "success": world.finished,
        "steps": world.steps,
        "invalid": invalid,
        "events": events,
        "first_event_step": first_event_step,
        "door_event_flags": door_event_flags,
    }


def _comp_discovery(spec, policy, family):
    """Learning metric: exact/covered discovery per door using the same
    definitions as the benchmark (compositional and temporal)."""
    out = {}
    for door in ("D1", "D2"):
        if family == "l":
            true_rule = (frozenset(spec.config.first_for(door)),
                         frozenset(spec.config.second_for(door)))
        else:
            true_rule = frozenset(spec.config.cond_for(door))
        exact = False
        covered = False
        for h in getattr(getattr(policy, "memory", None), "hypotheses", []):
            if h.door != door or h.status != "verified":
                continue
            if family == "l":
                continue
            if h.antecedent == true_rule:
                exact = True
            if true_rule <= h.antecedent:
                covered = True
        if family == "l":
            tmem = getattr(policy, "temporal", None)
            if tmem is not None:
                for h in tmem.hypotheses:
                    if h.door != door or h.status != "verified":
                        continue
                    if (h.first, h.second) == true_rule:
                        exact = True
                    if h.first <= true_rule[0] and h.second <= true_rule[1]:
                        covered = True
        out[door] = (exact, covered)
    return out


def _comp_store(policy, family):
    out = {}
    for door in ("D1", "D2"):
        hyps = []
        for h in getattr(getattr(policy, "memory", None), "hypotheses", []):
            if h.door != door:
                continue
            hyps.append({"antecedent": sorted(tuple(h.antecedent)),
                         "consequence": h.consequence,
                         "status": h.status, "sup": h.sup, "con": h.con})
        out[door] = hyps
    return out


def diag_condition(condition, seed, episodes, family, policy_cls=DiagAgent):
    pools, _ = family_pools(family)
    worlds = make_worlds(condition, seed, episodes, family, pools)
    policy = policy_cls(seed + 1000003, use_temporal=(family == "l"))
    per_episode = []
    for spec in worlds:
        pre_clean = policy._clean_gated_events
        pre_dual = policy._dual_gated_events
        entry = {
            "episode": len(per_episode),
            "success": False,
            "steps": 0,
            "invalid": 0,
            "events": 0,
            "first_event_step": None,
            "door_event_flags": {"D1": False, "D2": False},
            "config": spec.config.key(),
            "agent": {"probe_attempts": 0, "probe_blocked": 0},
            "discovery": {"D1": [False, False], "D2": [False, False]},
        }
        if family == "l":
            true_rule = {d: _true_rule(spec, d) for d in ("D1", "D2")}
            diag = InstrumentedTemporal(policy.temporal, true_rule)
            policy.set_diag(diag)
            r = play_episode_diag(spec, policy, family)
            diag.close()
            policy.set_diag(None)
            entry["success"] = r["success"]
            entry["steps"] = r["steps"]
            entry["invalid"] = r["invalid"]
            entry["events"] = r["events"]
            entry["first_event_step"] = r["first_event_step"]
            entry["door_event_flags"] = r["door_event_flags"]
            entry["agent"] = policy.ep_diag()
            entry["discovery"] = {d: list(v)
                                  for d, v in _comp_discovery(spec, policy, family).items()}
            entry["true_rule"] = {d: [sorted(tuple(k)) for k in r_]
                                  for d, r_ in true_rule.items()}
            entry["temporal"] = diag.summary()
            entry["gate_events"] = {
                "clean": policy._clean_gated_events - pre_clean,
                "dual": policy._dual_gated_events - pre_dual,
            }
            entry["temporal_store"] = [
                {"first": sorted(tuple(h.first)), "second": sorted(tuple(h.second)),
                 "status": h.status, "sup": h.sup, "con": h.con,
                 "h_id": h.h_id}
                for h in policy.temporal.hypotheses
            ]
        else:
            r = play_episode_diag(spec, policy, family)
            entry["success"] = r["success"]
            entry["steps"] = r["steps"]
            entry["invalid"] = r["invalid"]
            entry["events"] = r["events"]
            entry["first_event_step"] = r["first_event_step"]
            entry["door_event_flags"] = r["door_event_flags"]
            entry["discovery"] = {d: list(v)
                                  for d, v in _comp_discovery(spec, policy, family).items()}
            entry["compositional_store"] = _comp_store(policy, family)
            entry["gate_events"] = {
                "clean": policy._clean_gated_events - pre_clean,
                "dual": policy._dual_gated_events - pre_dual,
            }
        per_episode.append(entry)
    return {"seed": seed, "condition": condition, "episodes": per_episode}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="l")
    ap.add_argument("--conditions", default="stable,novel")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--json", type=Path, default=Path("results/v13/diag.json"))
    ap.add_argument("--policy", type=str, default="v12",
                    choices=("v12", "v13"))
    args = ap.parse_args()

    policy_cls = DiagAgentV13 if args.policy == "v13" else DiagAgent
    out = {"metadata": {
        "tool": "diagnose_v13",
        "family": args.family,
        "conditions": args.conditions.split(","),
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "seed_offset": _SEED_OFFSET,
        "policy": f"metargon{12 if args.policy == 'v12' else 13} "
                  f"(use_temporal=True on l)",
        "note": "worlds identical to benchmark_v12; policy serial 11800+seed +1000003",
    }, "runs": []}

    for condition in args.conditions.split(","):
        for i in range(args.seeds):
            seed = i + _SEED_OFFSET
            out["runs"].append(diag_condition(
                condition, seed, args.episodes, args.family, policy_cls))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # quick console summary
    for run in out["runs"]:
        t = run["episodes"]
        success = sum(e["success"] for e in t)
        line = f"{run['condition']:>8} seed {run['seed']}: success {success}/{args.episodes}"
        if run["condition"] in ("stable", "novel") and t and "temporal" in t[0]:
            correct_created = sum(e["temporal"]["correct_created"] for e in t)
            correct_supported = sum(e["temporal"]["correct_supported"] for e in t)
            correct_verified = sum(e["temporal"]["correct_verified_max"] for e in t)
            correct_evicted = sum(e["temporal"]["correct_evicted"] for e in t)
            floor_d1 = sum(1 for e in t if e["temporal"]["true_pair_observed"].get("D1"))
            floor_d2 = sum(1 for e in t if e["temporal"]["true_pair_observed"].get("D2"))
            probes = sum(e["agent"]["probe_attempts"] for e in t)
            blocked = sum(e["agent"]["probe_blocked"] for e in t)
            line += (f" | correct_created {correct_created} "
                     f"supported {correct_supported} "
                     f"verified_ep {correct_verified} evicted {correct_evicted} "
                     f"| floor D1 {floor_d1}/{args.episodes} D2 {floor_d2}/{args.episodes} "
                     f"| probes {probes} blocked {blocked}")
        print(line)


if __name__ == "__main__":
    main()
