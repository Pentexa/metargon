"""Regression tests for the v0.4 benchmark. Execute via unittest discovery."""
import math
import unittest

from adaptive_attention import (
    AgentView, BlockQuota, QueryLearner, SensorMemory, FAMILIES, MODES,
    make_environment, simulate, benchmark, trial_readings,
)
import random


class TestV04(unittest.TestCase):
    def test_only_context_and_primary_are_visible(self):
        episode=make_environment(0,80,'stable')[0]
        self.assertEqual(set(vars(episode.view)),{'context','primary'})
        for name in ('truth','secondary','tertiary','profile','pair_failure'):
            self.assertFalse(hasattr(episode.view,name))

    def test_environment_reproducible_and_shifted(self):
        self.assertEqual(make_environment(4,160,'shift'),make_environment(4,160,'shift'))
        self.assertNotEqual(make_environment(4,160,'shift'),make_environment(7,160,'shift'))
        self.assertNotEqual(make_environment(4,160,'stable'),make_environment(4,160,'shift'))

    def test_double_correlated_sensor_errors_exist(self):
        rng=random.Random(19)
        # Scenario 1 intentionally makes sensors 1 and 2 fail together.
        draws=[trial_readings(rng,0,1) for _ in range(1000)]
        double=sum(b == 1 and c == 1 for _,b,c in draws)
        self.assertGreater(double,200)
        # Scenario 2 intentionally makes primary and secondary fail together.
        draws=[trial_readings(rng,0,2) for _ in range(1000)]
        double=sum(a == 1 and b == 1 for a,b,_ in draws)
        self.assertGreater(double,200)

    def test_no_known_reference_error_rate_or_quality_flag(self):
        memory=SensorMemory()
        self.assertAlmostEqual(memory.p_error(0,0),.25)
        self.assertAlmostEqual(memory.p_error(0,1),.25)
        self.assertEqual(memory.p_error(0,1),memory.p_error(0,2))
        # Contexts are initially indistinguishable to the agent.
        self.assertEqual([memory.p_error(c,0) for c in range(4)],[.25]*4)

    def test_labeled_feedback_changes_trust_without_oracle_rate(self):
        memory=SensorMemory(contexts=2,window=8)
        for _ in range(12):
            memory.update(0,1,(0,0),0)
        self.assertGreater(memory.p_error(0,0),.65)
        self.assertLess(memory.p_error(0,1),.15)
        self.assertAlmostEqual(memory.p_error(1,0),.25)
        self.assertLessEqual(len(memory.error[0][0]),8)
        self.assertLessEqual(len(memory.pair[0]),8)
        self.assertEqual(memory.decide(0,1,None),0) # learned inversion

    def test_bounded_pair_cofailure_detection(self):
        m=SensorMemory(contexts=1,window=16)
        for _ in range(16):
            m.update(0,0,(1,1),0)
        self.assertGreater(m.disagreement_risk(0),.7)
        for _ in range(16):
            m.update(0,0,(0,0),0)
        self.assertLess(m.disagreement_risk(0),.1)

    def test_online_bandit_learns_incremental_gain_and_loss(self):
        learning=QueryLearner(contexts=2)
        for _ in range(15):
            learning.observe(0,False,True)
            learning.observe(1,True,False)
        self.assertGreater(learning.estimate(0),.5)
        self.assertLess(learning.estimate(1),-.5)
        self.assertLess(learning.weight(1,'ucb',random.Random(0)),
                        learning.weight(0,'ucb',random.Random(0)))

    def test_budget_is_exact_in_every_block(self):
        q=BlockQuota(400,.25)
        rng=random.Random(10)
        decisions=[q.decide(i,3 if i%2 else .2,1.,rng) for i in range(400)]
        self.assertEqual(sum(decisions),100)
        self.assertEqual([sum(decisions[i:i+40]) for i in range(0,400,40)], [10]*10)
        with self.assertRaises(ValueError):
            BlockQuota(401,.25)

    def test_all_strategies_share_matched_budget(self):
        for family in FAMILIES:
            modes={m:simulate(family,m,2,160,.25) for m in MODES}
            self.assertEqual(modes['no_query'].queries,0)
            self.assertEqual({x.queries for m,x in modes.items() if m!='no_query'},{40})
            self.assertTrue(all(math.isfinite(x.accuracy) for x in modes.values()))

    def test_seed_reproducibility(self):
        self.assertEqual(simulate('shift','bandit',3,160,.25),
                         simulate('shift','bandit',3,160,.25))

    def test_negative_control_and_known_limitations_documented(self):
        report=benchmark(seeds=2,trials=80,fraction=.25)
        self.assertIn('iid',report['results'])
        self.assertFalse(report['protocol']['real_power_measured'])
        self.assertFalse(report['protocol']['claimed_novel_architecture'])
        self.assertIn('AFTER action',report['protocol']['feedback'])
        self.assertFalse(report['protocol']['reference_sensor_error_known'])


if __name__=='__main__':
    unittest.main()
