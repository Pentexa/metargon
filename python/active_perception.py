#!/usr/bin/env python3
"""CORVID v0.2: budgeted active-perception benchmark (Python stdlib only).

Research scaffold, NOT a novel AI model or a claim of biological fidelity.
At each trial the agent receives a noisy cue, chooses whether to request an
extra observation under a strict quota, chooses an action, and gets noisy
feedback. The hidden cue and rule are NEVER exposed to the agent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Literal

Mode = Literal["no_query", "random_query", "targeted_query"]
LOW_ERR = 0.42
HIGH_ERR = 0.03
SECOND_ERR = 0.08
REWARD_ERR = 0.05
HAZARD = 0.012


@dataclass(frozen=True)
class Trial:
    context: int
    observed: int
    quality: str
    true_cue: int  # evaluator ONLY, never passed to agents
    extra_observed: int  # evaluator ONLY until query is spent
    hidden_mode: int  # evaluator ONLY, never passed to agents
    distractors: tuple[int, ...]
    reward_coin: float  # paired across strategies


@dataclass
class Result:
    mode: Mode
    seed: int
    trials: int
    query_budget: int
    queries: int
    accuracy: float
    pre_accuracy: float
    first_post_30_accuracy: float
    late_accuracy: float
    low_quality_accuracy: float
    decision_time_ms: float
    mistakes: int


def make_trials(seed: int, n: int, contexts: int = 4) -> list[Trial]:
    rng = random.Random(1000 + seed * 101)
    hidden_modes = [rng.randrange(2) for _ in range(contexts)]
    trials: list[Trial] = []
    for t in range(n):
        # Rule reversal halfway; the policy sees neither this nor hidden mode.
        if t == n // 2:
            hidden_modes = [1 - x for x in hidden_modes]
        context = rng.randrange(contexts)
        true_cue = rng.randrange(2)
        quality = "low" if rng.random() < 0.35 else "high"
        first_err = LOW_ERR if quality == "low" else HIGH_ERR
        observed = true_cue ^ int(rng.random() < first_err)
        extra = true_cue ^ int(rng.random() < SECOND_ERR)
        trials.append(Trial(context, observed, quality, true_cue, extra,
                            hidden_modes[context], tuple(rng.randrange(2)
                            for _ in range(rng.randint(2, 8))), rng.random()))
    return trials


def prob_true_one(first: int, err1: float, second: int | None) -> float:
    # Equal prior on the hidden binary cue. Independent sensor errors.
    def likelihood(true: int) -> float:
        x = (1 - err1) if true == first else err1
        if second is not None:
            x *= (1 - SECOND_ERR) if true == second else SECOND_ERR
        return x
    l0, l1 = likelihood(0), likelihood(1)
    return l1 / (l0 + l1)


def reward_likelihood(mode: int, action: int, p_cue1: float, reward: int) -> float:
    p_correct = 0.0
    for cue, p_cue in ((0, 1 - p_cue1), (1, p_cue1)):
        if action == (cue ^ mode):
            p_correct += p_cue
    p_reward = REWARD_ERR + (1 - 2 * REWARD_ERR) * p_correct
    return p_reward if reward else 1 - p_reward


class BeliefAgent:
    def __init__(self, contexts: int = 4) -> None:
        self.p_mode1 = [0.5] * contexts
        self.work_units = 0

    def act(self, context: int, p_cue1: float) -> int:
        old = self.p_mode1[context]
        self.p_mode1[context] = old * (1 - HAZARD) + (1 - old) * HAZARD
        self.work_units += 1
        p1 = self.p_mode1[context]
        score0 = ((1 - p1) * reward_likelihood(0, 0, p_cue1, 1)
                  + p1 * reward_likelihood(1, 0, p_cue1, 1))
        score1 = ((1 - p1) * reward_likelihood(0, 1, p_cue1, 1)
                  + p1 * reward_likelihood(1, 1, p_cue1, 1))
        return int(score1 > score0)

    def learn(self, context: int, action: int, p_cue1: float, reward: int) -> None:
        p1 = self.p_mode1[context]
        l0 = reward_likelihood(0, action, p_cue1, reward)
        l1 = reward_likelihood(1, action, p_cue1, reward)
        self.p_mode1[context] = (p1 * l1) / ((1 - p1) * l0 + p1 * l1)
        self.work_units += 1


class QueryBudget:
    """Online budget policy. No future observations, hidden states or rewards."""
    def __init__(self, mode: Mode, trials: int, quota: int, seed: int) -> None:
        self.mode = mode
        self.total = trials
        self.left = quota
        self.rng = random.Random(281 * seed + 23)

    def request(self, index: int, quality: str) -> bool:
        remaining = self.total - index
        if self.left == 0:
            return False
        if self.mode == "no_query":
            return False
        if self.mode == "random_query":
            take = self.rng.random() < self.left / remaining
        else:
            # Reserve remaining query allowance on the final episodes when
            # necessary to match the quota, but prioritize low-quality cues.
            take = quality == "low" or self.left == remaining
        if take:
            self.left -= 1
        return take


def run(mode: Mode, seed: int, n: int, fraction: float = 0.25) -> Result:
    if n < 60:
        raise ValueError("n must be >=60 for a meaningful post-switch window")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    budget = math.floor(n * fraction) if mode != "no_query" else 0
    trials = make_trials(seed, n)
    policy = QueryBudget(mode, n, budget, seed)
    agent = BeliefAgent()
    successful: list[int] = []
    quality_success: list[int] = []
    queries = 0
    decision_ns = 0
    for i, trial in enumerate(trials):
        start = time.perf_counter_ns()
        # Neither the true cue nor the hidden rule is visible here.
        take = policy.request(i, trial.quality)
        queries += int(take)
        p_cue = prob_true_one(trial.observed,
                              LOW_ERR if trial.quality == "low" else HIGH_ERR,
                              trial.extra_observed if take else None)
        # Distractors occur between observation and action. Retaining cue
        # belief is a bounded working-memory operation, not an oracle input.
        for _ in trial.distractors:
            pass
        action = agent.act(trial.context, p_cue)
        decision_ns += time.perf_counter_ns() - start
        correct = int(action == (trial.true_cue ^ trial.hidden_mode))
        # Outcomes are noisy and shared across agents to permit paired trials.
        reward = correct ^ int(trial.reward_coin < REWARD_ERR)
        agent.learn(trial.context, action, p_cue, reward)
        successful.append(correct)
        if trial.quality == "low":
            quality_success.append(correct)
    half = n // 2
    def rate(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")
    return Result(mode, seed, n, budget, queries, rate(successful),
                  rate(successful[:half]), rate(successful[half:half + 30]),
                  rate(successful[-max(30, n // 4):]), rate(quality_success),
                  decision_ns / 1e6, n - sum(successful))


def aggregate(results: list[Result]) -> dict:
    groups: dict[str, list[Result]] = {}
    for result in results:
        groups.setdefault(result.mode, []).append(result)
    out = {}
    for mode, group in groups.items():
        out[mode] = {
            "seeds": len(group),
            "accuracy_mean": statistics.mean(x.accuracy for x in group),
            "accuracy_sd": statistics.stdev(x.accuracy for x in group) if len(group) > 1 else 0,
            "pre_accuracy_mean": statistics.mean(x.pre_accuracy for x in group),
            "first_post_30_mean": statistics.mean(x.first_post_30_accuracy for x in group),
            "late_accuracy_mean": statistics.mean(x.late_accuracy for x in group),
            "low_quality_accuracy_mean": statistics.mean(x.low_quality_accuracy for x in group),
            "queries_mean": statistics.mean(x.queries for x in group),
            "decision_ms_mean": statistics.mean(x.decision_time_ms for x in group),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.trials < 60 or args.seeds < 1:
        parser.error("Require trials >= 60 and seeds >= 1")
    if not 0 <= args.query_fraction <= 1:
        parser.error("Require 0 <= query-fraction <= 1")
    modes: tuple[Mode, ...] = ("no_query", "random_query", "targeted_query")
    results = [run(mode, seed, args.trials, args.query_fraction)
               for mode in modes for seed in range(args.seeds)]
    summary = aggregate(results)
    print("CORVID v0.2 / active perception budget benchmark (not an energy measurement)")
    print(f"Trials/seed={args.trials}; seeds={args.seeds}; query budget={args.query_fraction:.0%}")
    print(f"{'Mode':<17} {'accuracy':>9} {'post30':>9} {'low cue':>9} {'queries':>9}")
    for mode in modes:
        r = summary[mode]
        print(f"{mode:<17} {r['accuracy_mean']*100:8.1f}% {r['first_post_30_mean']*100:8.1f}% "
              f"{r['low_quality_accuracy_mean']*100:8.1f}% {r['queries_mean']:9.1f}")
    print("No claim of novelty or edge-device energy efficiency follows from these numbers.")
    if args.json:
        payload = {
            "configuration": {"trials_per_seed": args.trials,
                              "seeds": args.seeds,
                              "query_fraction": args.query_fraction,
                              "low_cue_error": LOW_ERR,
                              "high_cue_error": HIGH_ERR,
                              "second_cue_error": SECOND_ERR,
                              "reward_error": REWARD_ERR,
                              "context_count": 4,
                              "rule_switch": "halfway"},
            "aggregate": summary,
            "raw": [asdict(r) for r in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
