"""Memory-saturation diagnosis for the Academy student agent.

Separates the failure modes behind `66–73 contradicted hypotheses, ~0
verified`:

  1. GENERATION  — does the true rule clause ever enter the hypothesis
     pool? (hypothesis generation)
  2. ATTRIBUTION — when the true clause exists, does evidence give it
     3 supports before it is contradicted? (evidence attribution)
  3. EVIDENCE   — distribution of support/contradict counts, and how
     many hypotheses die by score-eviction vs contradiction.
  4. PROBING    — do probe attempts target true antecedents; does the
     event-reset of `tried` cause probe churn?

Everything runs offline and deterministically (seeded). The harness
touches only internal state of the student agent; no code paths of the
student are changed.
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
from academy.agents.memory import CONTRADICTED
from academy.envs.tasks import (
    DEFAULT_SPLITS_PATH, EpisodeSpec, difficulty_of, load_splits,
)
from academy.learning.loop import run_one_episode


def rule_clauses(rule):
    return rule if isinstance(rule, tuple) else (rule,)


def clause_match(antecedent, clause) -> bool:
    return antecedent == frozenset(clause)


def true_clauses(config):
    return {
        "D1": list(rule_clauses(config.d1_rule)),
        "D2": list(rule_clauses(config.d2_rule)),
    }


def make_stream(splits, episodes, seed):
    rng = random.Random(seed)
    out = []
    pool = splits["train"]
    for e in range(episodes):
        cfg = rng.choice(pool)
        out.append(EpisodeSpec(
            config=cfg, max_steps=80, noise_q=0.0,
            spec_seed=seed * 1000 + e, pool_tag="train",
            difficulty=difficulty_of(cfg)))
    return out


class MemoryProbe:
    def __init__(self, agent):
        self.memory = agent.memory
        self.agent = agent

    def status_counts(self):
        out = {}
        for h in self.memory.hypotheses:
            out[h.status] = out.get(h.status, 0) + 1
        return out


def run_diagnosis(splits=None, episodes=80, seed=20261107, max_steps=80,
                  results_dir=None, verbose=True):
    splits = splits or load_splits(DEFAULT_SPLITS_PATH)
    results_dir = results_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "results", "academy_v01", "memory_v03")
    os.makedirs(results_dir, exist_ok=True)

    agent = MetargonAcademyAgent(seed=seed)
    stream = make_stream(splits, episodes, seed)

    true_seen = {"D1": False, "D2": False}
    true_first_match_step = {"D1": None, "D2": None}
    true_max_support = {"D1": 0, "D2": 0}
    true_contradicted_before_verify = {"D1": 0, "D2": 0}
    true_trace = []
    evicted_statuses = []
    probes = {"D1": {"attempts": 0, "true_targets": 0},
              "D2": {"attempts": 0, "true_targets": 0}}

    orig_evict = agent.memory._evict_one
    def tracking_evict():
        victim = min(agent.memory.hypotheses,
                     key=lambda h: (h.score(), -h.last_update, -h.h_id))
        evicted_statuses.append(victim.status)
        return orig_evict()
    agent.memory._evict_one = tracking_evict

    for ep_i, spec in enumerate(stream):
        rec = run_one_episode(agent, spec)
        tc = true_clauses(spec.config)
        for door, clauses in tc.items():
            matches = [h for h in agent.memory.hypotheses
                       if h.door == door
                       and any(clause_match(h.antecedent, c) for c in clauses)]
            if matches:
                true_seen[door] = True
                if true_first_match_step[door] is None:
                    true_first_match_step[door] = ep_i
                m = max(h.sup for h in matches)
                true_max_support[door] = max(true_max_support[door], m)
                if all(h.status == CONTRADICTED for h in matches) and m < 3:
                    true_contradicted_before_verify[door] += 1
            tried = agent.tried.get(door, set())
            true_antes = {frozenset(c) for c in clauses}
            probes[door]["attempts"] += len(tried)
            probes[door]["true_targets"] += sum(
                1 for a in tried if a in true_antes)
        if ep_i % 10 == 0 or ep_i == episodes - 1:
            true_trace.append({
                "episode": ep_i,
                "status_counts": MemoryProbe(agent).status_counts(),
            })
        if verbose and ep_i % 20 == 0:
            print(f"  ep {ep_i}: memory={len(agent.memory.hypotheses)}")

    summary = {
        "run": "memory_v03_diagnosis",
        "seed": seed,
        "episodes": episodes,
        "status_counts_final": MemoryProbe(agent).status_counts(),
        "memory_size_final": len(agent.memory.hypotheses),
        "evictions": len(evicted_statuses),
        "evicted_status_count": {s: evicted_statuses.count(s)
                                 for s in set(evicted_statuses)},
        "generated_hypotheses": agent.memory.generated,
        "true_clause_ever_generated": true_seen,
        "true_first_match_episode": true_first_match_step,
        "true_max_support": true_max_support,
        "true_contradicted_before_verify_episodes": (
            true_contradicted_before_verify),
        "probes": probes,
        "true_trace": true_trace,
        "hash_randomization": bool(sys.flags.hash_randomization),
    }
    with open(os.path.join(results_dir, "diagnosis_memory.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if verbose:
        print(json.dumps({k: v for k, v in summary.items()
                          if k != "true_trace"}, indent=1, default=str))
    return summary


if __name__ == "__main__":
    run_diagnosis()
