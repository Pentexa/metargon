#!/usr/bin/env python3
"""CORVID v0.5 — sparse, delayed, budgeted outcome supervision.

Scientific prototype, NOT a new intelligence architecture. Every competing
algorithm sees the same generated episodes; it chooses optional observations
and costly labels BEFORE any hidden truth is revealed. Label requests deliver
truth only after a configurable delay. The evaluator alone sees all truths.

Only Python standard library is required. A separate Rust kernel implements
bounded feedback storage; the Python benchmark is the reference experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics

from adaptive_attention import (AgentView, BlockQuota, CONTEXTS, FAMILIES,
                                HiddenEpisode, make_environment, paired_ci)

CONDITIONS = (*FAMILIES, "novel")
POLICIES = ("no_feedback", "random", "uncertainty", "joint", "joint_no_decay")
WINDOW = 32


@dataclass(frozen=True)
class DeferredLabel:
    due: int
    context: int
    primary: int
    extra: tuple[int, int] | None
    baseline_action: int
    final_action: int
    truth: int


@dataclass(frozen=True)
class TrialMetrics:
    condition: str
    policy: str
    seed: int
    trials: int
    accuracy: float
    post_shift_accuracy: float
    brier_score: float
    observation_requests: int
    label_requests: int
    labels_delivered: int
    labels_pending: int
    observation_on_labelled: int
    confident_coverage: float
    confident_accuracy: float | None
    memory_entries: int
    query_value_updates: int
    query_helps_verified: int  # evaluator-only, includes trials without a label
    query_harms_verified: int  # evaluator-only, includes trials without a label


def make_novel_environment(seed: int, trials: int) -> list[HiddenEpisode]:
    """Non-v0.4 generating mechanisms. Unseen by policy. Not a claim of OOD rigor."""
    if trials < 40 or trials % 40:
        raise ValueError("trials must be a positive multiple of 40")
    rng = random.Random(510057 + seed * 6151)
    mapping = list(range(CONTEXTS))
    rng.shuffle(mapping)
    episodes = []
    for _ in range(trials):
        ctx = rng.randrange(CONTEXTS)
        profile = mapping[ctx]
        true = rng.randrange(2)
        if profile == 0:   # severe primary noise, good auxiliaries
            err = (rng.random() < .44, rng.random() < .06, rng.random() < .09)
        elif profile == 1:  # correlated optional errors
            shared = rng.random() < .41
            err = (rng.random() < .08, shared or rng.random() < .02,
                   shared or rng.random() < .02)
        elif profile == 2:  # high error; inversion might be learned with labels
            err = (rng.random() < .62, rng.random() < .10, rng.random() < .17)
        else:  # primary-secondary co-failure, somewhat informative tertiary
            shared = rng.random() < .24
            err = (shared or rng.random() < .09,
                   shared or rng.random() < .08,
                   rng.random() < .14)
        a, b, c = tuple(true ^ int(e) for e in err)
        pair = sum(x != true for x in (a, b, c)) >= 2
        episodes.append(HiddenEpisode(AgentView(ctx, a), b, c, true, profile, pair))
    return episodes


def environment(condition: str, seed: int, trials: int) -> list[HiddenEpisode]:
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    return (make_novel_environment(seed, trials) if condition == "novel"
            else make_environment(seed, trials, condition))


class SparseMemory:
    """Keeps only labeled history (at most WINDOW per sensor/context).

    Exponentially decaying weight is indexed by *label events* rather than
    wall-clock steps. This means shifts between sparse labels can be slow.
    A finite, fixed-size history bounds storage. No unlabeled agreement is
    treated as truth: jointly faulty sensors cannot be identified from votes.
    """
    def __init__(self, decay: float = .93, window: int = WINDOW) -> None:
        if not (0 < decay <= 1) or window < 1:
            raise ValueError("invalid decay/window")
        self.decay = decay
        self.window = window
        self.errors: list[list[list[int]]] = [[[] for _ in range(3)] for _ in range(CONTEXTS)]
        self.pairs: list[list[int]] = [[] for _ in range(CONTEXTS)]
        self.label_counts = [0] * CONTEXTS

    def _append(self, history: list[int], error: int) -> None:
        history.append(error)
        if len(history) > self.window:
            history.pop(0)

    def learn(self, ctx: int, primary: int, extra: tuple[int,int] | None,
              truth: int) -> None:
        self.label_counts[ctx] += 1
        self._append(self.errors[ctx][0], int(primary != truth))
        if extra is not None:
            b, c = extra
            self._append(self.errors[ctx][1], int(b != truth))
            self._append(self.errors[ctx][2], int(c != truth))
            self._append(self.pairs[ctx], int(b != truth and c != truth))

    def _estimate(self, values: list[int], prior_a: float, prior_b: float) -> float:
        weights = [self.decay**i for i in range(len(values)-1, -1, -1)]
        err = sum(v*w for v,w in zip(values, weights))
        return (prior_a + err) / (prior_a + prior_b + sum(weights))

    def error(self, ctx: int, sensor: int) -> float:
        return self._estimate(self.errors[ctx][sensor], 1., 3.)

    def cofailure(self, ctx: int) -> float:
        return self._estimate(self.pairs[ctx], 1., 7.)

    def uncertainty_of_error(self, ctx: int) -> float:
        """Heuristic uncertainty bonus; not a Bayesian confidence interval."""
        return 1 / math.sqrt(1 + self.label_counts[ctx])

    def probability(self, ctx: int, primary: int,
                    extra: tuple[int,int] | None = None) -> float:
        bits = (primary,) if extra is None else (primary, *extra)
        terms = []
        for i, val in enumerate(bits):
            e = max(.06, min(.94, self.error(ctx, i)))
            terms.append((1 if val else -1) * math.log((1 - e) / e))
        if extra is not None:
            excess = max(0., self.cofailure(ctx) - self.error(ctx,1)*self.error(ctx,2))
            scale = 1 - min(.65, 2*excess)
            terms[1] *= scale
            terms[2] *= scale
        z = sum(terms)
        return 1 / (1 + math.exp(-z))

    def entries(self) -> int:
        return sum(len(h) for row in self.errors for h in row) + sum(map(len, self.pairs))


class SparseAgent:
    def __init__(self, policy: str) -> None:
        if policy not in POLICIES:
            raise ValueError("unknown policy")
        self.policy = policy
        self.memory = SparseMemory(decay=1.0 if policy == "joint_no_decay" else .93)
        self.gain = [0.] * CONTEXTS
        self.gain_count = [0.] * CONTEXTS
        self.query_value_updates = 0

    def query_benefit(self, ctx: int) -> float:
        return self.gain[ctx] / (self.gain_count[ctx] + 3.)

    def observation_weight(self, ctx: int) -> float:
        if self.policy in ("random", "no_feedback"):
            return 1.0
        if self.policy == "uncertainty":
            risk = min(self.memory.error(ctx, 0), 1-self.memory.error(ctx,0))
            return .15 + 8*risk*risk
        # Learned benefit is conditioned on a *labeled* queried event.
        # Prior-driven exploration avoids allocating all budget to one context.
        risk = min(self.memory.error(ctx,0), 1-self.memory.error(ctx,0))
        return max(.1, .30 + 4.0*max(0., self.query_benefit(ctx))
                   + 2.0*risk + .55/(1+self.gain_count[ctx])**.5)

    def label_weight(self, ctx: int, p: float, primary: int,
                     extra: tuple[int,int] | None, before_action: int,
                     after_action: int) -> float:
        if self.policy == "random":
            return 1.0
        error_risk = 1. - max(p, 1.-p)
        if self.policy == "uncertainty":
            return .2 + 5*error_risk
        conflict = (extra is not None and (primary != extra[0] or primary != extra[1]))
        # Joint strategy: seek labels for uncertain, low-label, conflicting
        # decisions; no direct access to correctness or simulator profiles.
        return (.2 + 3*error_risk + 1.0*self.memory.uncertainty_of_error(ctx)
                + .9*int(before_action != after_action) + .3*int(conflict))

    def receive(self, e: DeferredLabel) -> None:
        self.memory.learn(e.context, e.primary, e.extra, e.truth)
        if e.extra is not None:
            gain = int(e.final_action == e.truth)-int(e.baseline_action == e.truth)
            ctx = e.context
            self.gain[ctx] = .94*self.gain[ctx] + gain
            self.gain_count[ctx] = .94*self.gain_count[ctx] + 1
            self.query_value_updates += 1


def simulate(condition: str, policy: str, seed: int, trials: int = 800,
             observation_fraction: float = .25,
             label_fraction: float = .10, delay: int = 7) -> TrialMetrics:
    if policy not in POLICIES or condition not in CONDITIONS or delay < 1:
        raise ValueError("invalid experiment arguments")
    episodes = environment(condition, seed, trials)
    agent = SparseAgent(policy)
    obs_budget = BlockQuota(trials, observation_fraction)
    label_budget = BlockQuota(trials, 0. if policy == "no_feedback" else label_fraction)
    obs_rng = random.Random(187777 + seed*997)
    label_rng = random.Random(93719 + seed*197)
    pending: dict[int, list[DeferredLabel]] = defaultdict(list)
    results: list[int] = []
    briers: list[float] = []
    confident_hits: list[int] = []
    observed_on_labeled = delivered = helps = harms = 0
    for i, e in enumerate(episodes):
        # Deliver only previously requested truths that are now due; no
        # access to truth from other episodes anywhere in the policy.
        for d in pending.pop(i, []):
            agent.receive(d)
            delivered += 1
        ctx, first = e.view.context, e.view.primary
        base_p = agent.memory.probability(ctx, first)
        baseline = int(base_p > .5) if base_p != .5 else first
        obs_weights = [agent.observation_weight(j) for j in range(CONTEXTS)]
        take_obs = obs_budget.decide(i, obs_weights[ctx],
                                     statistics.mean(obs_weights), obs_rng)
        extra = (e.secondary, e.tertiary) if take_obs else None
        p = agent.memory.probability(ctx, first, extra)
        action = int(p > .5) if p != .5 else first
        # Label acquisition is decided before truth; its decision may use the
        # observed bits but never the hidden target, profile or pair flag.
        if policy == "no_feedback":
            take_label = False
        else:
            label_weights = [agent.label_weight(j, p if j == ctx else .75,
                                                first if j == ctx else 1,
                                                extra if j == ctx else None,
                                                baseline if j == ctx else 1,
                                                action if j == ctx else 1)
                             for j in range(CONTEXTS)]
            take_label = label_budget.decide(i, label_weights[ctx],
                                           statistics.mean(label_weights), label_rng)
        if take_label:
            observed_on_labeled += int(take_obs)
            pending[i+delay].append(DeferredLabel(i+delay,ctx,first,extra,
                                                   baseline,action,e.truth))
        correct = int(action == e.truth)
        results.append(correct)
        briers.append((p - e.truth)**2)
        if max(p, 1-p) >= .8:
            confident_hits.append(correct)
        if take_obs:
            diff = correct-int(baseline == e.truth)
            helps += int(diff > 0)
            harms += int(diff < 0)
    if obs_budget.used != trials * obs_budget.per_block // 40:
        raise AssertionError("observation budget mismatch")
    if label_budget.used != trials * label_budget.per_block // 40:
        raise AssertionError("label budget mismatch")
    outstanding = sum(map(len, pending.values()))
    assert delivered + outstanding == label_budget.used
    return TrialMetrics(
        condition,policy,seed,trials,statistics.mean(results),
        statistics.mean(results[trials//2:trials//2+80]),
        statistics.mean(briers),obs_budget.used,label_budget.used,
        delivered,outstanding,observed_on_labeled,
        len(confident_hits)/trials,
        statistics.mean(confident_hits) if confident_hits else None,
        agent.memory.entries(),agent.query_value_updates,helps,harms)


def benchmark(seeds: int = 30, trials: int = 800,
              observation_fraction: float = .25,
              label_fraction: float = .10, delay: int = 7) -> dict:
    if seeds < 2:
        raise ValueError("require two or more seeds")
    results = {}
    for condition in CONDITIONS:
        per_policy = {p:[simulate(condition,p,s,trials,observation_fraction,
                                  label_fraction,delay) for s in range(seeds)]
                      for p in POLICIES}
        summary = {}
        for p, rs in per_policy.items():
            summary[p] = {key:statistics.mean(getattr(r,key) for r in rs)
                          for key in ('accuracy','post_shift_accuracy','brier_score',
                                      'observation_requests','label_requests',
                                      'labels_delivered','labels_pending',
                                      'confident_coverage','memory_entries',
                                      'query_value_updates','query_helps_verified',
                                      'query_harms_verified')}
            scores = [r.confident_accuracy for r in rs if r.confident_accuracy is not None]
            summary[p]['confident_accuracy'] = statistics.mean(scores) if scores else None
        result = {'summary':summary,
                  'joint_minus_random': paired_ci([a.accuracy-b.accuracy for a,b in
                      zip(per_policy['joint'],per_policy['random'])]),
                  'joint_minus_uncertainty':paired_ci([a.accuracy-b.accuracy for a,b in
                      zip(per_policy['joint'],per_policy['uncertainty'])]),
                  'per_seed':{p:[asdict(r) for r in rs] for p,rs in per_policy.items()}}
        results[condition] = result
    return {'protocol':{'version':'0.5','seeds':seeds,'trials':trials,
             'observation_fraction':observation_fraction,
             'label_fraction':label_fraction,'delay':delay,
             'trial_block':40,'condition_families':CONDITIONS,
             'negative_control':'iid context without stable profile mapping',
             'heldout_analogue':'novel synthetic generator (not real-world holdout)',
             'ground_truth_policy_access':False,
             'labels_requested_before_truth':True,
             'label_cost':'one feedback quota token per requested truth',
             'budgets_equal_across_labeled_policies':True,
             'all_feedback_delivered_before_end':False,
             'real_energy_measured':False,
             'novel_architecture_proven':False},'results':results}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,default=30)
    ap.add_argument('--trials',type=int,default=800)
    ap.add_argument('--observation-fraction',type=float,default=.25)
    ap.add_argument('--label-fraction',type=float,default=.10)
    ap.add_argument('--delay',type=int,default=7)
    ap.add_argument('--json',type=Path)
    args=ap.parse_args()
    data=benchmark(args.seeds,args.trials,args.observation_fraction,
                   args.label_fraction,args.delay)
    for c in CONDITIONS:
        print(f'\n{c.upper()} | Accuracy | Brier↓ | Obs | Labels requested/delivered | memory entries')
        for p in POLICIES:
            r=data['results'][c]['summary'][p]
            print(f'{p:15s} | {r["accuracy"]:7.2%} | {r["brier_score"]:.4f} | '
                  f'{r["observation_requests"]:3.0f} | '
                  f'{r["label_requests"]:3.0f}/{r["labels_delivered"]:3.0f} | '
                  f'{r["memory_entries"]:5.0f}')
        ci=data['results'][c]['joint_minus_random']
        print(f'Joint - random: {100*ci["mean"]:+.2f} pp '
              f'[{100*ci["ci95_low"]:+.2f},{100*ci["ci95_high"]:+.2f}]')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print('Saved',args.json)
    print('Synthetic feedback-limited experiment; no novelty or power claims.')

if __name__ == '__main__':
    main()
