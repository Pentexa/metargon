#!/usr/bin/env python3
"""CORVID v0.4: online query-value learning with correlated sensor failures.

Only `AgentView(context, primary)` is visible before query. A query reveals
secondary and tertiary readings. After acting, the simulator gives *delayed*
verified target feedback (truth). This is an explicit supervision assumption;
without feedback, correlated failure cannot in general be identified.

Pure Python standard library. All policies get the same trials and exact per-
block observation quotas. The hidden scenarios are for evaluation only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Literal

CONTEXTS = 4
BLOCK_SIZE = 40
FAMILIES = ("stable", "shift", "iid")
MODES = ("no_query", "random", "uncertainty", "ucb", "bandit")
WINDOW = 32


@dataclass(frozen=True)
class AgentView:
    context: int
    primary: int


@dataclass(frozen=True)
class HiddenEpisode:
    view: AgentView
    secondary: int
    tertiary: int
    truth: int
    profile: int  # evaluator only
    pair_failure: bool  # evaluator only: both of a pair wrong


@dataclass(frozen=True)
class Metrics:
    family: str
    mode: str
    seed: int
    trials: int
    queries: int
    accuracy: float
    post_60_accuracy: float
    paired_failure_accuracy: float | None
    paired_failure_exposures: int
    query_harms: int
    query_helps: int
    observed_pair_error_rate: float | None
    work_units: int


# Each profile defines a different, hidden sensing regime. Deliberately
# design two distinct forms of correlated failure: primary+secondary together
# and secondary+tertiary together. No error-rate label is given to the agent.
def trial_readings(rng: random.Random, truth: int, profile: int) -> tuple[int,int,int]:
    if profile == 0:  # expensive extra query is beneficial
        errors = (rng.random() < .36, rng.random() < .07, rng.random() < .08)
    elif profile == 1:  # primary usually best, extra sensors co-fail
        shared = rng.random() < .30
        errors = (rng.random() < .06,
                  shared or rng.random() < .05,
                  shared or rng.random() < .05)
    elif profile == 2:  # primary and secondary co-fail, third can rescue
        shared = rng.random() < .27
        errors = (shared or rng.random() < .04,
                  shared or rng.random() < .04,
                  rng.random() < .07)
    else:  # somewhat noisy, conditionally independent
        errors = (rng.random() < .17, rng.random() < .19,
                  rng.random() < .20)
    return tuple(truth ^ int(e) for e in errors)  # type: ignore[return-value]


def make_environment(seed: int, trials: int, family: str) -> list[HiddenEpisode]:
    if family not in FAMILIES or trials < BLOCK_SIZE or trials % BLOCK_SIZE:
        raise ValueError("unknown family or trials must be a multiple of 40")
    rng = random.Random(70021 + seed * 41177)
    profile_map = list(range(CONTEXTS))
    rng.shuffle(profile_map)
    records: list[HiddenEpisode] = []
    for i in range(trials):
        if i == trials // 2 and family == "shift":
            # Shift after the half-way point; tests online adaptation.
            profile_map = [profile_map[(j + 2) % CONTEXTS] for j in range(CONTEXTS)]
        context = rng.randrange(CONTEXTS)
        profile = rng.randrange(CONTEXTS) if family == "iid" else profile_map[context]
        truth = rng.randrange(2)
        a,b,c = trial_readings(rng, truth, profile)
        pair = ((a != truth and b != truth) or (b != truth and c != truth))
        records.append(HiddenEpisode(AgentView(context,a),b,c,truth,profile,pair))
    return records


class SensorMemory:
    """Sliding-window labeled reliability; never sees a hidden profile.

    Window size is fixed per context, per sensor. Pair co-failure is retained
    only when a query was made and the *later* verified truth is received.
    """
    def __init__(self, contexts: int = CONTEXTS, window: int = WINDOW) -> None:
        if not contexts or not window:
            raise ValueError("positive dimensions required")
        self.contexts = contexts
        self.window = window
        self.error: list[list[list[int]]] = [[[] for _ in range(3)] for _ in range(contexts)]
        self.pair: list[list[int]] = [[] for _ in range(contexts)]
        self.work_units = 0

    def _push(self, arr: list[int], value: int) -> None:
        arr.append(value)
        if len(arr) > self.window:
            arr.pop(0)

    def p_error(self, ctx: int, sensor: int) -> float:
        hist = self.error[ctx][sensor]
        return (1.0 + sum(hist)) / (4.0 + len(hist))  # Beta(1,3), 25% prior

    def disagreement_risk(self, ctx: int) -> float:
        # A measured, delayed-label co-failure rate, not ground-truth metadata.
        h = self.pair[ctx]
        return (1 + sum(h)) / (8 + len(h))

    def update(self, ctx: int, primary: int, observed: tuple[int,int] | None,
               verified_truth: int) -> None:
        self._push(self.error[ctx][0], int(primary != verified_truth))
        self.work_units += 1
        if observed is not None:
            b,c = observed
            self._push(self.error[ctx][1], int(b != verified_truth))
            self._push(self.error[ctx][2], int(c != verified_truth))
            self._push(self.pair[ctx], int(b != verified_truth and c != verified_truth))
            self.work_units += 3

    def decide(self, ctx: int, first: int,
               extra: tuple[int,int] | None = None) -> int:
        # Clipped log likelihood weights allow automatic polarity inversion if
        # sensor proves adversarial. Correlation-aware pair downweighting is
        # preliminary, not a mathematically exact dependent-sensor posterior.
        values = (first,) if extra is None else (first, *extra)
        strengths: list[float] = []
        for j,v in enumerate(values):
            e = min(.92, max(.08, self.p_error(ctx,j)))
            strengths.append((1 if v else -1) * math.log((1-e)/e))
            self.work_units += 1
        if extra is not None:
            e1, e2 = self.p_error(ctx,1), self.p_error(ctx,2)
            cofail = self.disagreement_risk(ctx)
            excess = max(0., cofail - e1*e2)
            # Mitigate double-counting votes from a correlated pair.
            suppression = min(.65, 2.0 * excess)
            strengths[1] *= (1.0 - suppression)
            strengths[2] *= (1.0 - suppression)
            self.work_units += 2
        score = sum(strengths)
        return int(score > 0) if score != 0 else first


class BlockQuota:
    def __init__(self, trials: int, fraction: float) -> None:
        if trials % BLOCK_SIZE or not 0.0 <= fraction <= 1.0:
            raise ValueError("invalid quota configuration")
        n = BLOCK_SIZE * fraction
        if n != int(n):
            raise ValueError("quota must be integer per block")
        self.per_block = int(n)
        self.remaining = self.per_block
        self.used = 0
        self.work_units = 0

    def decide(self, i: int, weight: float, average: float, rng: random.Random) -> bool:
        if i % BLOCK_SIZE == 0:
            self.remaining = self.per_block
        left = BLOCK_SIZE - (i % BLOCK_SIZE)
        if self.remaining == 0:
            return False
        base = self.remaining / left
        # Constrain policy bias and enforce exact quota at each block's end.
        ratio = max(.1,min(3.5,weight / max(.00001,average)))
        chance = min(1.,base*ratio)
        choose = self.remaining == left or rng.random() < chance
        if choose:
            self.remaining -= 1
            self.used += 1
        self.work_units += 1
        return choose


class QueryLearner:
    """Small contextual bandit; learns the query's realized *incremental*
    correctness advantage from post-action feedback on queried trials.
    Queries have zero benefits in many episodes and can be harmful. Each
    context gets its own exponentially discounted estimate; UCB comparator
    uses a stronger confidence bonus. No hidden simulator parameters.
    """
    def __init__(self, contexts: int = CONTEXTS, decay: float = .95) -> None:
        self.gain = [0.] * contexts
        self.count = [0.] * contexts
        self.decay = decay
        self.work_units = 0

    def observe(self, ctx: int, baseline_correct: bool, query_correct: bool) -> None:
        self.count[ctx] = self.decay * self.count[ctx] + 1
        self.gain[ctx] = self.decay * self.gain[ctx] + (int(query_correct)-int(baseline_correct))
        self.work_units += 1

    def estimate(self, ctx: int) -> float:
        return self.gain[ctx] / (self.count[ctx] + 3.0)

    def weight(self, ctx: int, mode: str, rng: random.Random) -> float:
        if mode == 'bandit':
            sample = self.estimate(ctx) + rng.gauss(0, .12/math.sqrt(1+self.count[ctx]))
            return max(.1, 1.0 + 5.0*sample)
        if mode == 'ucb':
            upper = self.estimate(ctx) + .18/math.sqrt(1+self.count[ctx])
            return max(.1, 1.0 + 5.0*upper)
        return 1.0


def simulate(family: str, mode: str, seed: int, trials: int = 400,
             fraction: float = .25) -> Metrics:
    if family not in FAMILIES or mode not in MODES:
        raise ValueError("unknown family or policy")
    env = make_environment(seed,trials,family)
    memory = SensorMemory()
    learner = QueryLearner()
    quota = BlockQuota(trials,fraction if mode != 'no_query' else 0)
    rng = random.Random(seed*997+9194)
    hits=[]
    pair_hits=[]
    harms=helps=0
    for i,hidden in enumerate(env):
        ctx, primary = hidden.view.context, hidden.view.primary
        base_action = memory.decide(ctx,primary)
        if mode == 'no_query':
            take=False
        elif mode == 'uncertainty':
            risk = memory.p_error(ctx,0)
            weight = .1 + risk*risk*9.
            average = statistics.mean(.1 + memory.p_error(j,0)**2*9.
                                      for j in range(CONTEXTS))
            take = quota.decide(i,weight,average,rng)
        elif mode in ('ucb','bandit'):
            weights=[learner.weight(j,mode,rng) for j in range(CONTEXTS)]
            take=quota.decide(i,weights[ctx],statistics.mean(weights),rng)
        else:
            take=quota.decide(i,1.0,1.0,rng)
        extra=(hidden.secondary,hidden.tertiary) if take else None
        action = memory.decide(ctx,primary,extra) if take else base_action
        hit = action == hidden.truth
        hits.append(int(hit))
        if hidden.pair_failure:
            pair_hits.append(int(hit))
        if take:
            delta=int(hit)-int(base_action == hidden.truth)
            helps += int(delta>0)
            harms += int(delta<0)
            learner.observe(ctx,base_action==hidden.truth,hit)
        # *Only now* is truth revealed to the learner in this experiment.
        memory.update(ctx,primary,extra,hidden.truth)
    if mode != 'no_query' and quota.used != trials*quota.per_block//BLOCK_SIZE:
        raise AssertionError("unequal quota")
    def fraction_correct(xs: list[int]) -> float:
        return sum(xs)/len(xs) if xs else float('nan')
    observed_pair_error = [memory.disagreement_risk(j) for j in range(CONTEXTS)]
    return Metrics(family,mode,seed,trials,quota.used,fraction_correct(hits),
                   fraction_correct(hits[trials//2:trials//2+60]),
                   fraction_correct(pair_hits) if pair_hits else None,
                   len(pair_hits),harms,helps,
                   statistics.mean(observed_pair_error) if mode != 'no_query' else None,
                   memory.work_units+learner.work_units+quota.work_units)


def paired_ci(xs: list[float], draws: int = 2000) -> dict[str,float]:
    if len(xs) < 2:
        raise ValueError("need at least 2 paired experiments")
    rng=random.Random(42211)
    means=sorted(statistics.mean(rng.choices(xs,k=len(xs))) for _ in range(draws))
    return {'mean':statistics.mean(xs),
            'ci95_low':means[int(.025*draws)],
            'ci95_high':means[int(.975*draws)]}


def benchmark(seeds: int = 30, trials: int = 400, fraction: float = .25) -> dict:
    if seeds < 2:
        raise ValueError("need >=2 seeds")
    result={}
    for family in FAMILIES:
        records={mode:[simulate(family,mode,s,trials,fraction)
                       for s in range(seeds)] for mode in MODES}
        assert len({x.queries for m,ls in records.items() if m!='no_query' for x in ls})==1
        summary={}
        for mode,ls in records.items():
            summary[mode]={
                'accuracy':statistics.mean(x.accuracy for x in ls),
                'post_60_accuracy':statistics.mean(x.post_60_accuracy for x in ls),
                'pair_failure_accuracy':statistics.mean(x.paired_failure_accuracy
                                  for x in ls if x.paired_failure_accuracy is not None),
                'queries':statistics.mean(x.queries for x in ls),
                'query_helps':statistics.mean(x.query_helps for x in ls),
                'query_harms':statistics.mean(x.query_harms for x in ls),
                'proxy_work_units':statistics.mean(x.work_units for x in ls),
            }
        result[family]={
            'summary':summary,
            'bandit_minus_random':paired_ci([a.accuracy-b.accuracy for a,b
                       in zip(records['bandit'],records['random'])]),
            'bandit_minus_ucb':paired_ci([a.accuracy-b.accuracy for a,b
                       in zip(records['bandit'],records['ucb'])]),
            'per_seed':{k:[asdict(r) for r in rs] for k,rs in records.items()},
        }
    return {
       'protocol':{'version':'0.4','seeds':seeds,'trials':trials,
          'query_fraction':fraction,'contexts':CONTEXTS,
          'sensor_count':3,'quota_block':BLOCK_SIZE,
          'feedback':'verified truth delivered AFTER action',
          'condition':'two sensors can co-fail',
          'primary_sensor_quality_known':False,
          'reference_sensor_error_known':False,
          'real_power_measured':False,
          'claimed_novel_architecture':False,
          'metric':'decision accuracy; work_units is an operation-count proxy',
          'control':'iid profile removes predictable context advantage'},
       'results':result}


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,default=30)
    ap.add_argument('--trials',type=int,default=400)
    ap.add_argument('--query-fraction',type=float,default=.25)
    ap.add_argument('--json',type=Path)
    args=ap.parse_args()
    data=benchmark(args.seeds,args.trials,args.query_fraction)
    for fam in FAMILIES:
        d=data['results'][fam]
        print(f'\n{fam.upper():12} | accuracy | post-mid-60 | queries | work units')
        for mode in MODES:
            x=d['summary'][mode]
            print(f'{mode:12} | {x["accuracy"]:7.2%} | {x["post_60_accuracy"]:13.2%} | '
                  f'{x["queries"]:7.0f} | {x["proxy_work_units"]:10.0f}')
        ci=d['bandit_minus_random']
        print('Bandit - random: '
              f'{100*ci["mean"]:+.2f} pp [95% bootstrap '
              f'{100*ci["ci95_low"]:+.2f}, {100*ci["ci95_high"]:+.2f}]')
    if args.json:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(f'\nSaved {args.json}')
    print('No new architecture, real-world sensing or measured energy claims.')


if __name__=='__main__':
    main()
