#!/usr/bin/env python3
"""CORVID v0.7: a bounded *predictive diagnostic outcome model*.

Research prototype, not a general world model or a brain simulation. It predicts
how two permitted diagnostic actions alter the *observed* evidence distribution,
then estimates the utility of both prospective results. Truth labels are scarce,
requested blindly, and delivered after a delay. A probe is reversible and purely
synthetic. No privileged state/profile enters the deployable agent.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import statistics

from adaptive_attention import BlockQuota, CONTEXTS, paired_ci
from causal_probe import (ACTIONS, BLOCK, CONDITIONS, BoundedMemory,
                          LearningAgent, simulate as old_simulate, world)

MODEL_POLICIES = ('predictive', 'uniform_outcomes', 'no_outcome_label')
REFERENCE_POLICIES = ('passive', 'probe', 'random', 'bayes_voi', 'corvid')
MAX_WINDOW = 64


def clamp(p: float) -> float:
    return max(0.001, min(0.999, p))


@dataclass(frozen=True)
class ScheduledLabel:
    due: int
    context: int
    first: int
    action: str | None
    reading: int | None
    truth: int


class PredictiveMemory:
    """Only observed diagnostic outcomes update response models.

    Verified labels (delayed) are stored in a separate set of bounded buffers.
    The response model knows the *observed* outcome, never the hidden truth.
    """
    def __init__(self, window: int = MAX_WINDOW):
        if window <= 0:
            raise ValueError('window must be positive')
        self.window = window
        self.sensor = BoundedMemory(window=window)
        self.observed = [[[
            deque(maxlen=window) for _ in ACTIONS] for _ in range(2)
        ] for _ in range(CONTEXTS)]
        self.global_observed = [[deque(maxlen=window) for _ in ACTIONS]
                                for _ in range(2)]
        self.verified = [[[[
            deque(maxlen=window) for _ in range(2)
        ] for _ in ACTIONS] for _ in range(2)] for _ in range(CONTEXTS)]
        self.global_verified = [[[deque(maxlen=window) for _ in range(2)]
                                  for _ in ACTIONS] for _ in range(2)]
        self.delivered = 0
        self.observations = 0

    @staticmethod
    def _check(context: int, first: int, action: str | None,
               reading: int | None = None) -> None:
        if not 0 <= context < CONTEXTS or first not in (0, 1):
            raise ValueError('invalid public observation')
        if action is not None and action not in ACTIONS:
            raise ValueError('unknown action')
        if reading is not None and reading not in (0, 1):
            raise ValueError('reading must be binary')

    def observe(self, context: int, first: int, action: str, reading: int) -> None:
        self._check(context, first, action, reading)
        idx = ACTIONS.index(action)
        self.observed[context][first][idx].append(reading)
        self.global_observed[first][idx].append(reading)
        self.observations += 1

    def deliver(self, label: ScheduledLabel) -> None:
        self._check(label.context, label.first, label.action, label.reading)
        if label.truth not in (0, 1):
            raise ValueError('truth must be binary')
        if label.action is None:
            if label.reading is not None:
                raise ValueError('phantom reading')
            self.sensor.update_primary(label.context, label.first, label.truth)
        else:
            if label.reading is None:
                raise ValueError('missing observed outcome')
            self.sensor.update(label.context, label.first, label.action,
                               label.reading, label.truth)
            ix = ACTIONS.index(label.action)
            self.verified[label.context][label.first][ix][label.reading].append(label.truth)
            self.global_verified[label.first][ix][label.reading].append(label.truth)
        self.delivered += 1

    def predict_outcome(self, context: int, first: int, action: str,
                        pooled: bool = True) -> float:
        """P(diagnostic readout=1 | public context, first reading, action).

        Beta shrinkage, not a learned physical dynamical model. Local evidence is
        pooled with a weak global prior across contexts for data efficiency.
        """
        self._check(context, first, action)
        ix = ACTIONS.index(action)
        local = self.observed[context][first][ix]
        global_ = self.global_observed[first][ix]
        global_p = (1 + sum(global_)) / (2 + len(global_)) if pooled else .5
        pseudo = 6.0
        return clamp((sum(local) + pseudo * global_p) / (len(local) + pseudo))

    def posterior_truth(self, context: int, first: int, action: str,
                        reading: int, use_labeled_joint: bool = True) -> float:
        """Estimated P(truth=1 | primary bit, observed diagnostic, action).

        Crucial separation: unlabeled outcomes NEVER update correctness labels.
        The historical v0.6 calibrated sensor posterior supplies a fallback.
        """
        self._check(context, first, action, reading)
        base = self.sensor.predict(context, first, action, reading)
        if not use_labeled_joint:
            return clamp(base)
        ix = ACTIONS.index(action)
        gl = self.global_verified[first][ix][reading]
        lc = self.verified[context][first][ix][reading]
        # Conservative soft correction; avoid trusting tiny rare-cell counts.
        gl_mean = (sum(gl) + 8 * base) / (len(gl) + 8)
        local_mean = (sum(lc) + 5 * gl_mean) / (len(lc) + 5)
        strength = len(lc) / (len(lc) + 9)
        return clamp((1 - strength) * gl_mean + strength * local_mean)

    def entries(self) -> int:
        return (self.sensor.entries()
                + sum(len(d) for c in self.observed for f in c for d in f)
                + sum(len(d) for c in self.verified for f in c for a in f for d in a))


class DiagnosticWorldModelAgent:
    """One-step imagined diagnostic branches using observed action responses."""
    def __init__(self, policy: str = 'predictive'):
        if policy not in MODEL_POLICIES:
            raise ValueError('unknown model policy')
        self.policy = policy
        self.model = PredictiveMemory()
        self.counts = [0, 0]
        self.prediction_loss = []
        self.choices = []

    def choose(self, context: int, first: int, rng: random.Random) -> str:
        # No truth, profile or generator parameters are available here.
        self.model._check(context, first, None)
        if rng.random() < .08:
            action = rng.choice(ACTIONS)
        else:
            scores = []
            for action in ACTIONS:
                p1 = (.5 if self.policy == 'uniform_outcomes'
                      else self.model.predict_outcome(context, first, action))
                exp_correct = 0.0
                for reading, pr in ((0, 1 - p1), (1, p1)):
                    post = self.model.posterior_truth(
                        context, first, action, reading,
                        use_labeled_joint=self.policy != 'no_outcome_label')
                    exp_correct += pr * max(post, 1 - post)
                # A **synthetic** extra burden for intervention (not joules).
                intervention_penalty = .010 if action == 'probe' else 0.0
                # Mild uncertainty bonus prevents premature exclusive selection.
                bonus = .010 / (1 + self.counts[ACTIONS.index(action)]) ** .5
                scores.append(exp_correct - intervention_penalty + bonus)
            action = ACTIONS[0] if scores[0] >= scores[1] else ACTIONS[1]
        self.counts[ACTIONS.index(action)] += 1
        self.choices.append(action)
        return action

    def observe(self, context: int, first: int, action: str, reading: int) -> None:
        forecast = self.model.predict_outcome(context, first, action)
        self.prediction_loss.append((forecast - reading) ** 2)
        self.model.observe(context, first, action, reading)

    def predict(self, context: int, first: int, action: str | None,
                reading: int | None) -> float:
        if action is None:
            return self.model.sensor.predict(context, first)
        assert reading is not None
        return self.model.posterior_truth(
            context, first, action, reading,
            use_labeled_joint=self.policy != 'no_outcome_label')


@dataclass(frozen=True)
class Metrics:
    condition: str
    policy: str
    seed: int
    trials: int
    accuracy: float
    brier: float
    response_brier: float
    diagnostic_count: int
    intervention_count: int
    passive_count: int
    requested_labels: int
    delivered_labels: int
    pending_labels: int
    immediate_outcome_updates: int
    memory_entries: int
    post_shift_accuracy: float
    synthetic_utility: float


def simulate(condition: str, policy: str, seed: int, trials: int = 800,
             diagnostic_fraction: float = .25, label_fraction: float = .10,
             delay: int = 7) -> Metrics:
    if condition not in CONDITIONS or policy not in MODEL_POLICIES or delay < 1:
        raise ValueError('invalid setup')
    episodes = world(condition, seed, trials)
    agent = DiagnosticWorldModelAgent(policy)
    budget = BlockQuota(trials, diagnostic_fraction)
    labels = BlockQuota(trials, label_fraction)
    slot_rng = random.Random(seed * 7177 + 571)
    label_rng = random.Random(seed * 2999 + 163)
    choice_rng = random.Random(seed * 5897 + 233)
    pending: dict[int, list[ScheduledLabel]] = defaultdict(list)
    correct: list[int] = []
    brier: list[float] = []
    pulses = 0
    passives = 0
    delivered = 0

    for t, e in enumerate(episodes):
        for label in pending.pop(t, []):
            agent.model.deliver(label)
            delivered += 1
        view = e.view
        # At this point the agent sees only view and delivered history.
        paid = budget.decide(t, 1.0, 1.0, slot_rng)
        action: str | None = None
        reading: int | None = None
        if paid:
            action = agent.choose(view.context, view.first, choice_rng)
            reading = e.reading(action)  # sole public outcome of executed action
            agent.observe(view.context, view.first, action, reading)
            pulses += int(action == 'probe')
            passives += int(action == 'passive')
        p = agent.predict(view.context, view.first, action, reading)
        guess = int(p > .5) if p != .5 else view.first
        request = labels.decide(t, 1.0, 1.0, label_rng)
        if request:
            pending[t + delay].append(ScheduledLabel(
                t + delay, view.context, view.first, action, reading, e.truth))
        correct.append(int(guess == e.truth))
        brier.append((p - e.truth) ** 2)
    outstanding = sum(len(x) for x in pending.values())
    assert delivered + outstanding == labels.used
    assert pulses + passives == budget.used
    return Metrics(condition, policy, seed, trials,
                   statistics.mean(correct), statistics.mean(brier),
                   statistics.mean(agent.prediction_loss) if agent.prediction_loss else 0.0,
                   budget.used, pulses, passives, labels.used, delivered,
                   outstanding, agent.model.observations, agent.model.entries(),
                   statistics.mean(correct[trials // 2:trials // 2 + 80]),
                   statistics.mean(correct) - .010 * pulses / trials)


def benchmark(seeds: int = 30, trials: int = 800,
              diagnostic_fraction: float = .25,
              label_fraction: float = .10, delay: int = 7) -> dict:
    if seeds < 2:
        raise ValueError('at least two seeds required')
    results = {}
    for condition in CONDITIONS:
        rows: dict[str, list[dict]] = {}
        for policy in REFERENCE_POLICIES:
            old_rows = [old_simulate(condition, policy, s, trials,
                                   diagnostic_fraction, label_fraction, delay)
                        for s in range(seeds)]
            rows[policy] = [{**asdict(r), 'synthetic_utility':
                             r.accuracy - .010 * r.intervention_count / trials}
                            for r in old_rows]
        for policy in MODEL_POLICIES:
            rows[policy] = [asdict(simulate(condition, policy, s, trials,
                                         diagnostic_fraction, label_fraction, delay))
                            for s in range(seeds)]
        cols = ('accuracy', 'brier', 'intervention_count', 'passive_count',
                'paid_diagnostics', 'requested_labels', 'delivered_labels',
                'pending_labels', 'memory_entries', 'post_shift_accuracy',
                'synthetic_utility')
        summary = {}
        for policy, replicates in rows.items():
            summary[policy] = {
                key: statistics.mean(r['diagnostic_count'] if key == 'paid_diagnostics'
                                     and 'diagnostic_count' in r else r[key]
                                     for r in replicates)
                for key in cols if key in replicates[0] or key == 'paid_diagnostics'
            }
            if 'response_brier' in replicates[0]:
                summary[policy]['response_brier'] = statistics.mean(
                    r['response_brier'] for r in replicates)
        def paired(other: str, key: str = 'accuracy') -> dict:
            return paired_ci([x[key] - y[key] for x, y in
                              zip(rows['predictive'], rows[other])])
        results[condition] = {
            'summary': summary,
            'predictive_minus_bayes_voi': paired('bayes_voi'),
            'predictive_minus_corvid_v06': paired('corvid'),
            'predictive_minus_probe': paired('probe'),
            'predictive_minus_no_forward': paired('uniform_outcomes'),
            'predictive_minus_bayes_voi_utility': paired('bayes_voi','synthetic_utility'),
            'per_seed': rows,
        }
    return {'protocol': {
        'version': '0.7', 'seeds': seeds, 'trials': trials,
        'diagnostic_fraction': diagnostic_fraction, 'label_fraction': label_fraction,
        'delay': delay, 'same_worlds': True, 'same_budget_schedules': True,
        'actions': ACTIONS, 'world_hidden_from_agent': True,
        'diagnostic_response_updates_without_label': True,
        'truth_updates_only_from_delayed_requested_labels': True,
        'sham_is_negative_control': True,
        'synthetic_probe_penalty_per_trial': .010,
        'real_energy_measured': False, 'real_hardware_tested': False,
        'world_model_is_one_step_response_distribution_not_general_physics': True,
        'new_scientific_architecture_proven': False,
        'holdout_worlds_from_independent_implementation': False,
    }, 'conditions': results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--trials', type=int, default=800)
    ap.add_argument('--diagnostic-fraction', type=float, default=.25)
    ap.add_argument('--label-fraction', type=float, default=.10)
    ap.add_argument('--delay', type=int, default=7)
    ap.add_argument('--json', type=Path,
                    default=Path('results/predictive_world_model_30seeds.json'))
    a = ap.parse_args()
    results = benchmark(a.seeds, a.trials, a.diagnostic_fraction,
                        a.label_fraction, a.delay)
    for condition in CONDITIONS:
        print('\n' + condition.upper())
        for policy, r in results['conditions'][condition]['summary'].items():
            print(f'  {policy:20s} acc={r["accuracy"]:.2%} '
                  f'pulses={r["intervention_count"]:.1f} '
                  f'utility={r["synthetic_utility"]:.4f}')
        delta = results['conditions'][condition]['predictive_minus_bayes_voi']
        print(f'  predictive minus bayes_voi {delta["mean"]:+.2%} '
              f'[{delta["ci95_low"]:+.2%}, {delta["ci95_high"]:+.2%}]')
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('\nSaved', a.json)
    print('Synthetic only. One-step response model, not general world intelligence.')


if __name__ == '__main__':
    main()
