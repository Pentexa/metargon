"""Leakage, delay, budget and bounded memory tests for CORVID v0.5."""
import math
import unittest

from sparse_feedback import (
    CONDITIONS, POLICIES, SparseMemory, SparseAgent, DeferredLabel,
    environment, make_novel_environment, simulate, benchmark,
)


class V05Tests(unittest.TestCase):
    def test_view_hides_truth_and_profile(self):
        view = environment('novel', 3, 80)[0].view
        self.assertEqual(set(vars(view)), {'context','primary'})
        for forbidden in ('truth', 'profile', 'secondary', 'tertiary', 'pair_failure'):
            self.assertFalse(hasattr(view, forbidden))

    def test_novel_generator_reproducible_and_distinct(self):
        self.assertEqual(make_novel_environment(1,80), make_novel_environment(1,80))
        self.assertNotEqual(make_novel_environment(1,80), make_novel_environment(2,80))
        self.assertNotEqual(make_novel_environment(1,80), environment('stable',1,80))
        self.assertEqual(set(CONDITIONS), {'stable','shift','iid','novel'})

    def test_delayed_labels_are_not_used_early(self):
        # If delivery lies beyond the end, all agent predictions must equal
        # its no-feedback counterpart with the SAME observation schedule.
        a = simulate('stable','random',5,80,delay=1000)
        b = simulate('stable','no_feedback',5,80,delay=1000)
        self.assertEqual(a.accuracy,b.accuracy)
        self.assertEqual(a.brier_score,b.brier_score)
        self.assertEqual(a.labels_delivered,0)
        self.assertEqual(a.labels_pending,8)
        self.assertEqual(a.label_requests,8)
        self.assertEqual(a.query_value_updates,0)
        self.assertEqual(b.label_requests,0)

    def test_budget_fairness_and_pending_accounting(self):
        for c in CONDITIONS:
            runs = {p: simulate(c,p,3,160,delay=7) for p in POLICIES}
            self.assertEqual({v.observation_requests for v in runs.values()}, {40})
            self.assertEqual({v.label_requests for p,v in runs.items() if p!='no_feedback'}, {16})
            self.assertEqual(runs['no_feedback'].label_requests,0)
            for v in runs.values():
                self.assertEqual(v.label_requests, v.labels_delivered+v.labels_pending)
                self.assertTrue(0<=v.confident_coverage<=1)
                self.assertTrue(0<=v.accuracy<=1)

    def test_memory_bounds_and_true_label_updates_only(self):
        m=SparseMemory(decay=.93,window=8)
        self.assertAlmostEqual(m.error(0,0), .25)
        for _ in range(50):
            m.learn(0,1,(0,0),0)
        self.assertEqual(m.label_counts[0],50)
        self.assertLessEqual(m.entries(),4*8)
        self.assertGreater(m.error(0,0),.6)
        self.assertLess(m.error(0,1),.15)
        self.assertGreater(m.probability(0,0),.5)
        self.assertAlmostEqual(m.error(1,0), .25)

    def test_sparse_feedback_can_reverse_bad_prior(self):
        a=SparseAgent('joint')
        before=a.memory.probability(0,1)
        for i in range(20):
            a.receive(DeferredLabel(i,0,1,(0,0),1,0,0))
        after=a.memory.probability(0,1)
        self.assertGreater(before,.5)
        self.assertLess(after,.5)
        self.assertGreater(a.query_value_updates,0)
        self.assertGreater(a.query_benefit(0),0)
        self.assertEqual(a.memory.label_counts[0],20)

    def test_empty_history_probability_not_overconfident(self):
        m=SparseMemory()
        self.assertAlmostEqual(m.probability(0,1),.75)
        self.assertAlmostEqual(m.probability(0,0),.25)
        self.assertLess(m.probability(0,1,(1,1)),1.0)

    def test_zero_label_budget_prevents_hidden_learning(self):
        # With every label request forbidden, the learned policy must not
        # spontaneously gain knowledge from unobserved simulator truths.
        a=simulate('stable','joint',3,160,label_fraction=0)
        b=simulate('stable','random',3,160,label_fraction=0)
        self.assertEqual(a.labels_delivered,0)
        self.assertEqual(a.memory_entries,0)
        self.assertEqual(a.query_value_updates,0)
        self.assertEqual(a.accuracy,b.accuracy)
        self.assertEqual(a.brier_score,b.brier_score)

    def test_no_labels_no_updates_under_normal_delay(self):
        result=simulate('shift','no_feedback',1,160)
        self.assertEqual(result.memory_entries,0)
        self.assertEqual(result.query_value_updates,0)
        self.assertEqual(result.labels_delivered,0)

    def test_delay_and_budget_validation(self):
        for d in [0,-1]:
            with self.assertRaises(ValueError): simulate('stable','random',1,80,delay=d)
        with self.assertRaises(ValueError): simulate('stable','random',1,81)
        with self.assertRaises(ValueError):simulate('stable','random',1,80,label_fraction=.13)
        with self.assertRaises(ValueError):SparseMemory(decay=0)
        with self.assertRaises(ValueError):SparseMemory(window=0)

    def test_seed_reproducibility(self):
        a=simulate('novel','joint',2,160)
        self.assertEqual(a,simulate('novel','joint',2,160))
        self.assertNotEqual(a,simulate('novel','joint',4,160))

    def test_compatible_report_includes_limitations(self):
        x=benchmark(seeds=2,trials=80)
        p=x['protocol']
        self.assertFalse(p['ground_truth_policy_access'])
        self.assertTrue(p['labels_requested_before_truth'])
        self.assertFalse(p['novel_architecture_proven'])
        self.assertFalse(p['real_energy_measured'])
        self.assertTrue(p['budgets_equal_across_labeled_policies'])
        self.assertTrue(math.isfinite(x['results']['iid']['summary']['joint']['brier_score']))


if __name__ == '__main__':
    unittest.main()
