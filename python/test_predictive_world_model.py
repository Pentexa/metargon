"""Tests for CORVID v0.7 forecasting, budget fairness and feedback isolation."""
from __future__ import annotations

import dataclasses
import inspect
import random
import unittest

from causal_probe import ACTIONS, CONDITIONS
from predictive_world_model import (
    DiagnosticWorldModelAgent, MODEL_POLICIES, PredictiveMemory,
    ScheduledLabel, benchmark, simulate,
)


class PredictiveWorldModelTests(unittest.TestCase):
    def test_forward_model_learns_response_frequency_without_labels(self):
        m = PredictiveMemory(window=16)
        self.assertAlmostEqual(m.predict_outcome(0, 1, 'probe'), .5)
        for _ in range(12):
            m.observe(0, 1, 'probe', 1)
        self.assertGreater(m.predict_outcome(0, 1, 'probe'), .75)
        self.assertEqual(m.delivered, 0)
        self.assertEqual(m.sensor.entries(), 0)
        self.assertEqual(m.sensor.delivered, 0)

    def test_outcome_observations_do_not_become_ground_truth(self):
        m = PredictiveMemory()
        base = m.posterior_truth(0, 1, 'probe', 1)
        for _ in range(25):
            m.observe(0, 1, 'probe', 1)
        self.assertAlmostEqual(base, m.posterior_truth(0, 1, 'probe', 1))
        self.assertEqual(m.delivered, 0)

    def test_delayed_label_updates_only_correct_channel(self):
        m = PredictiveMemory(window=4)
        m.deliver(ScheduledLabel(7, 0, 1, None, None, 0))
        self.assertEqual(m.sensor.count(0, 'passive'), 0)
        self.assertEqual(m.sensor.count(0, 'probe'), 0)
        self.assertEqual(m.delivered, 1)
        m.deliver(ScheduledLabel(8, 0, 1, 'probe', 0, 0))
        self.assertEqual(m.sensor.count(0, 'probe'), 1)
        self.assertEqual(m.sensor.count(0, 'passive'), 0)
        self.assertEqual(m.delivered, 2)
        with self.assertRaises(ValueError):
            m.deliver(ScheduledLabel(8, 0, 1, None, 1, 0))

    def test_memory_bounded(self):
        m = PredictiveMemory(window=4)
        for i in range(500):
            action = ACTIONS[i % 2]
            read = i % 2
            m.observe(i % 4, i % 2, action, read)
            m.deliver(ScheduledLabel(i + 7, i % 4, i % 2, action, read, (i // 2) % 2))
        self.assertLessEqual(m.entries(), 4 * 5 * 4 + 4 * 2 * 2 * 4 + 4 * 2 * 2 * 2 * 4)
        self.assertEqual(m.delivered, 500)
        self.assertEqual(m.observations, 500)

    def test_cannot_query_hidden_truth_or_profile_from_agent(self):
        for function in (DiagnosticWorldModelAgent.choose,
                         DiagnosticWorldModelAgent.observe,
                         DiagnosticWorldModelAgent.predict):
            sig = inspect.signature(function)
            self.assertNotIn('truth', sig.parameters)
            self.assertNotIn('profile', sig.parameters)
        a = DiagnosticWorldModelAgent()
        self.assertIn(a.choose(0, 1, random.Random(3)), ACTIONS)
        with self.assertRaises(ValueError):
            a.choose(9, 1, random.Random(3))

    def test_exact_budgets_and_delays_across_conditions(self):
        for condition in CONDITIONS:
            for policy in MODEL_POLICIES:
                x = simulate(condition, policy, 2, trials=160)
                self.assertEqual(x.diagnostic_count, 40)
                self.assertEqual(x.passive_count + x.intervention_count, 40)
                self.assertEqual(x.immediate_outcome_updates, 40)
                self.assertEqual(x.requested_labels, 16)
                self.assertEqual(x.delivered_labels + x.pending_labels, 16)
                self.assertGreaterEqual(x.memory_entries, 0)
                self.assertLessEqual(x.memory_entries, 4*5*64 + 4*2*2*64 + 4*2*2*2*64)
                self.assertTrue(0 <= x.accuracy <= 1)
                self.assertTrue(0 <= x.brier <= 1)
                self.assertTrue(0 <= x.response_brier <= 1)

    def test_no_feedback_does_not_leak_truth(self):
        x = simulate('stable', 'predictive', 1, 160, label_fraction=0)
        self.assertEqual(x.delivered_labels, 0)
        self.assertEqual(x.requested_labels, 0)
        self.assertEqual(x.immediate_outcome_updates, 40)
        # Unlabeled observed outcomes are OK; supervised sensor memory is empty.
        a = DiagnosticWorldModelAgent()
        a.observe(0, 1, 'probe', 0)
        self.assertEqual(a.model.sensor.entries(), 0)

    def test_unfinished_label_delays(self):
        x = simulate('stable', 'predictive', 3, 160, delay=500)
        self.assertEqual(x.delivered_labels, 0)
        self.assertEqual(x.pending_labels, 16)
        self.assertEqual(x.requested_labels, 16)

    def test_reproducibility_and_parameter_validation(self):
        self.assertEqual(simulate('shift','predictive',3,160),
                         simulate('shift','predictive',3,160))
        self.assertNotEqual(simulate('shift','predictive',3,160),
                            simulate('shift','predictive',4,160))
        with self.assertRaises(ValueError):
            simulate('invalid', 'predictive', 0, 160)
        with self.assertRaises(ValueError):
            simulate('stable', 'invalid', 0, 160)
        with self.assertRaises(ValueError):
            PredictiveMemory(window=0)

    def test_predictive_utility_is_bounded_and_finite(self):
        m = PredictiveMemory()
        for action in ACTIONS:
            for first in (0, 1):
                for read in (0, 1):
                    self.assertTrue(0 <= m.posterior_truth(0,first,action,read) <= 1)
                self.assertTrue(0 <= m.predict_outcome(0,first,action) <= 1)

    def test_benchmark_has_paired_controls(self):
        b = benchmark(seeds=2, trials=80)
        self.assertFalse(b['protocol']['new_scientific_architecture_proven'])
        for condition in CONDITIONS:
            rows = b['conditions'][condition]['per_seed']
            for policy in (*MODEL_POLICIES, 'passive','probe','bayes_voi','corvid'):
                self.assertEqual(len(rows[policy]),2)
                self.assertEqual(rows[policy][0]['requested_labels'],8)
                self.assertEqual(rows[policy][0].get('diagnostic_count',
                                 rows[policy][0].get('paid_diagnostics')), 20)


if __name__ == '__main__':
    unittest.main()
