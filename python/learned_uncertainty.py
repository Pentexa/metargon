#!/usr/bin/env python3
"""CORVID v0.3: *online* sensor calibration, not a learned general AI.

No model weights, GPU, external modules or privileged environment signals.
The agent sees only context, a primary binary reading, its optional secondary
reading (if purchased), and noisy reward. The evaluator keeps hidden sensor
error, correct cue and hidden decision rule to itself.

This is a falsifiable synthetic benchmark for the question: can an edge agent
allocate scarce observations from *experience*, without a quality label?
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics

from active_perception import BeliefAgent, SECOND_ERR, REWARD_ERR, HAZARD, prob_true_one

MODES = ("no_query", "random_query", "learned_query", "oracle_quality")
FAMILIES = ("stationary", "swap", "iid")
CONTEXTS = 4
RATES = (0.045, 0.10, 0.32, 0.44)
PRIOR_ERR = 0.225
HISTORY_SIZE = 16
SHRINKAGE = 2.0
BLOCK_SIZE = 50


@dataclass(frozen=True)
class AgentView:
    context: int
    reading: int


@dataclass(frozen=True)
class HiddenTrial:
    """ONLY the evaluator handles this; never pass to an agent or policy."""
    view: AgentView
    second: int
    true_cue: int
    hidden_rule: int
    first_error: float
    reward_coin: float


@dataclass(frozen=True)
class Metrics:
    family: str
    mode: str
    seed: int
    trials: int
    queries: int
    accuracy: float
    pre_accuracy: float
    post_40_accuracy: float
    late_accuracy: float
    high_error_queries: int
    high_error_exposures: int
    inferred_error_mae: float | None


class OnlineReliability:
    """Bounded, online estimator using conditional independence of sensors.

    P(disagree) = q + (1-2q)*e, where q is the *assumed* known error rate
    of the secondary sensor and e is the unknown primary-sensor error.
    We only learn from paired readings that the agent paid to obtain.
    Assumption can fail for correlated sensor errors; test this in v0.4.
    """
    def __init__(self, contexts: int = CONTEXTS, history: int = HISTORY_SIZE,
                 prior_error: float = PRIOR_ERR, second_error: float = SECOND_ERR):
        if contexts <= 0 or history <= 0 or not (0 < second_error < .5):
            raise ValueError("invalid calibration parameters")
        if not (0 < prior_error < .5):
            raise ValueError("prior_error must be between 0 and 0.5")
        self.history = [deque(maxlen=history) for _ in range(contexts)]
        self.second_error = second_error
        self.prior_error = prior_error

    def observe(self, context: int, first: int, second: int) -> None:
        self.history[context].append(int(first != second))

    def error(self, context: int) -> float:
        values = self.history[context]
        q = self.second_error
        prior_disagree = q + (1 - 2*q)*self.prior_error
        mean_disagree = (sum(values) + SHRINKAGE*prior_disagree) / (len(values) + SHRINKAGE)
        return min(.49, max(.01, (mean_disagree - q)/(1 - 2*q)))

    def count(self, context: int) -> int:
        return len(self.history[context])


class BlockBudget:
    """Exactly matched quota, evenly allocated in time blocks for every arm.

    An optional reading is charged regardless of whether it helps the result.
    No future observations, rewards, or hidden states are used by the scheduler.
    """
    def __init__(self, n: int, quota: int, block_size: int = BLOCK_SIZE):
        if n <= 0 or not 0 <= quota <= n or block_size <= 0:
            raise ValueError("invalid quota")
        self.n = n
        self.quota = quota
        self.block_size = block_size
        self.blocks = (n + block_size - 1)//block_size
        self.total_used = 0
        self._current = -1
        self._left = 0

    def request(self, t: int, weight: float, mean_weight: float, rng: random.Random) -> bool:
        if not 0 <= t < self.n or weight < 0 or mean_weight <= 0:
            raise ValueError("invalid decision")
        block = t // self.block_size
        if block != self._current:
            self._current = block
            self._left = ((block + 1)*self.quota)//self.blocks - (block*self.quota)//self.blocks
        end = min((block+1)*self.block_size, self.n)
        left_trials = end - t
        if self._left == 0:
            return False
        chance = min(1.0, (self._left / left_trials) * (weight / mean_weight))
        take = self._left >= left_trials or rng.random() < chance
        if take:
            self._left -= 1
            self.total_used += 1
        return take


def make_environment(seed: int, n: int, family: str) -> list[HiddenTrial]:
    if family not in FAMILIES or n < 80:
        raise ValueError("unknown family or too few trials")
    rng = random.Random(seed * 11939 + 910013)
    rates = list(RATES)
    rng.shuffle(rates)
    hidden_modes = [rng.randrange(2) for _ in range(CONTEXTS)]
    data = []
    for t in range(n):
        if t == n // 2:
            hidden_modes = [m ^ 1 for m in hidden_modes]
            if family == "swap":
                rates.reverse()
        context = rng.randrange(CONTEXTS)
        true_cue = rng.randrange(2)
        if family == "iid":
            first_error = RATES[rng.randrange(CONTEXTS)]
        else:
            first_error = rates[context]
        reading = true_cue ^ int(rng.random() < first_error)
        second = true_cue ^ int(rng.random() < SECOND_ERR)
        data.append(HiddenTrial(AgentView(context, reading), second, true_cue,
                                hidden_modes[context], first_error, rng.random()))
    return data


def weight_for_error(error: float, samples: int) -> float:
    # Deliberately simple fixed heuristic; only 'error' is inferred online.
    # Important: v0.3 does NOT learn this acquisition policy itself.
    return 0.10 + 4.5 * (error ** 2) + 0.25 / math.sqrt(1 + samples)


def simulate(family: str, mode: str, seed: int, n: int = 400,
             query_fraction: float = .25) -> Metrics:
    if mode not in MODES or family not in FAMILIES:
        raise ValueError("unknown mode or family")
    if n < 80 or not 0 <= query_fraction <= 1:
        raise ValueError("invalid trial count or query budget")
    episodes = make_environment(seed, n, family)
    budget = math.floor(n * query_fraction) if mode != "no_query" else 0
    controller = BlockBudget(n, budget)
    rng = random.Random(8181 + seed * 5719)
    learner = OnlineReliability()
    agent = BeliefAgent()
    hits: list[int] = []
    high_error_queries = 0
    high_error_exposures = 0
    for t, hidden in enumerate(episodes):
        # This view is the ONLY observation passed into the agent.
        view = hidden.view
        est = learner.error(view.context)
        if mode == "oracle_quality":
            w = weight_for_error(hidden.first_error, HISTORY_SIZE)
            # Oracle only has the *current* error rate, not the true cue/rule.
            # Comparable as an upper-bound informed scheduler, NOT deployable.
            other_rates = RATES
            mean_w = statistics.mean(weight_for_error(e, HISTORY_SIZE) for e in other_rates)
        elif mode == "learned_query":
            w = weight_for_error(est, learner.count(view.context))
            mean_w = statistics.mean(weight_for_error(learner.error(c), learner.count(c))
                                     for c in range(CONTEXTS))
        else:
            w, mean_w = 1., 1.
        take = mode != "no_query" and controller.request(t, w, mean_w, rng)
        if hidden.first_error > .25:
            high_error_exposures += 1
            high_error_queries += int(take)
        if take:
            learner.observe(view.context, view.reading, hidden.second)
        # The decision agent never sees true_cue, hidden_rule or first_error.
        # Its reliability knowledge comes from paired observations only.
        cue_err = hidden.first_error if mode == "oracle_quality" else learner.error(view.context)
        p_cue = prob_true_one(view.reading, cue_err, hidden.second if take else None)
        action = agent.act(view.context, p_cue)
        correct = int(action == (hidden.true_cue ^ hidden.hidden_rule))
        reward = correct ^ int(hidden.reward_coin < REWARD_ERR)
        agent.learn(view.context, action, p_cue, reward)
        hits.append(correct)
    def rate(xs: list[int]) -> float:
        return sum(xs) / len(xs)
    # Calibration evaluated on each context's final environment reliability,
    # only for stationary; swap and iid cannot use single end-state targets.
    error_mae: float | None = None
    if family == "stationary":
        final_rates = {}
        for e in episodes:
            final_rates[e.view.context] = e.first_error
        error_mae = statistics.mean(abs(learner.error(c)-final_rates[c]) for c in range(CONTEXTS))
    return Metrics(family, mode, seed, n, controller.total_used,
                   rate(hits), rate(hits[:n//2]), rate(hits[n//2:n//2+40]),
                   rate(hits[-n//4:]), high_error_queries, high_error_exposures,
                   error_mae)


def ci_paired(results: dict[str, list[Metrics]], a: str, b: str,
              replications: int = 3000) -> dict[str, float]:
    ra, rb = results[a], results[b]
    assert [r.seed for r in ra] == [r.seed for r in rb]
    differences = [x.accuracy-y.accuracy for x,y in zip(ra,rb)]
    rng = random.Random(94421)
    boot = sorted(statistics.mean(rng.choices(differences, k=len(differences)))
                  for _ in range(replications))
    return {"mean": statistics.mean(differences),
            "ci95_low": boot[int(.025*replications)],
            "ci95_high": boot[min(replications-1,int(.975*replications))]}


def benchmark(trials: int = 400, seeds: int = 30, fraction: float = .25) -> dict:
    if trials < 80 or seeds < 2 or not 0 <= fraction <= 1:
        raise ValueError("trials>=80, seeds>=2, fraction in [0,1]")
    results = {}
    for family in FAMILIES:
        group = {m: [simulate(family,m,s,trials,fraction) for s in range(seeds)]
                 for m in MODES}
        baseline = group["random_query"]
        results[family] = {
            "modes": {
                mode: {
                    "accuracy_mean": statistics.mean(x.accuracy for x in items),
                    "accuracy_sd": statistics.stdev(x.accuracy for x in items),
                    "post_40_mean": statistics.mean(x.post_40_accuracy for x in items),
                    "queries_mean": statistics.mean(x.queries for x in items),
                    "high_error_query_rate": sum(x.high_error_queries for x in items) /
                        max(1,sum(x.high_error_exposures for x in items)),
                    "calibration_mae": (statistics.mean(x.inferred_error_mae for x in items)
                                        if family == "stationary" else None),
                } for mode, items in group.items()},
            "learned_vs_random": ci_paired(group, "learned_query", "random_query"),
            "oracle_vs_random": ci_paired(group, "oracle_quality", "random_query"),
            "per_seed": {m: [asdict(x) for x in records] for m,records in group.items()},
        }
        assert all(x.queries == baseline[0].queries for m in MODES if m != "no_query"
                   for x in group[m]), "budget mismatch"
    return {"protocol": {
                "version": "0.3", "trials_per_seed": trials, "seed_count": seeds,
                "query_fraction": fraction, "families": FAMILIES,
                "matched_block_size": BLOCK_SIZE, "reward_error": REWARD_ERR,
                "assumed_secondary_error": SECOND_ERR, "hidden_quality_flags": True,
                "energy_measured": False, "novelty_claimed": False,
                "policy_is_learned_end_to_end": False,
                "baseline_policy": "random quota", "sensor_calibration": "paired disagreement"},
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--query-fraction", type=float, default=.25)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    data = benchmark(args.trials,args.seeds,args.query_fraction)
    print("CORVID v0.3 — quality labels hidden, online reliability inferred")
    print(f"{args.seeds} paired seeds × {args.trials} trials; quota={args.query_fraction:.0%}")
    for family in FAMILIES:
        r = data["results"][family]
        print(f"\n{family.upper():<12} {'accuracy':>11} {'post-40':>11} {'queries':>10} {'high-Q':>10}")
        for m in MODES:
            x = r["modes"][m]
            print(f"{m:<17} {100*x['accuracy_mean']:>9.1f}% {100*x['post_40_mean']:>9.1f}% "
                  f"{x['queries_mean']:>10.1f} {100*x['high_error_query_rate']:>9.1f}%")
        ci = r["learned_vs_random"]
        print("Learned minus random (paired accuracy): "
              f"{100*ci['mean']:+.2f} pp, bootstrap 95% CI "
              f"[{100*ci['ci95_low']:+.2f}, {100*ci['ci95_high']:+.2f}] pp")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data,indent=2),encoding="utf-8")
        print(f"Saved {args.json}")
    print("WARNING: no energy/performance or novel architecture claim supported.")


if __name__ == "__main__":
    main()
