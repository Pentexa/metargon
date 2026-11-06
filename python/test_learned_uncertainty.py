import math
import unittest

from learned_uncertainty import (
    OnlineReliability, BlockBudget, make_environment, simulate, benchmark,
    FAMILIES, MODES, weight_for_error,
)
import random


class TestOnlineReliability(unittest.TestCase):
    def test_prior_and_direction(self):
        r = OnlineReliability()
        self.assertAlmostEqual(r.error(0), .225)
        for _ in range(16):
            r.observe(0, 0, 1)
        self.assertGreater(r.error(0), .40)
        for _ in range(16):
            r.observe(0, 0, 0)
        self.assertLess(r.error(0), .1)
        self.assertEqual(r.count(0), 16)

    def test_isolation_and_bounded_memory(self):
        r = OnlineReliability(contexts=2, history=3)
        for _ in range(40):
            r.observe(0, 0, 1)
        self.assertEqual(r.count(0), 3)
        self.assertEqual(r.count(1), 0)
        self.assertAlmostEqual(r.error(1), .225)

    def test_calibration_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            OnlineReliability(contexts=0)
        with self.assertRaises(ValueError):
            OnlineReliability(second_error=.5)


class TestBudget(unittest.TestCase):
    def test_exact_budget_and_block_balance(self):
        for quota in (0, 1, 3, 25, 100):
            b = BlockBudget(100, quota, block_size=20)
            rng = random.Random(4)
            q = [b.request(t, 1., 1., rng) for t in range(100)]
            self.assertEqual(sum(q), quota)
            block_quotas = [sum(q[t:t+20]) for t in range(0,100,20)]
            self.assertLessEqual(max(block_quotas)-min(block_quotas), 1)

    def test_no_future_or_hidden_state_in_agent_view(self):
        env = make_environment(42, 100, 'swap')
        self.assertEqual(set(vars(env[0].view)), {'context','reading'})
        self.assertNotIn('first_error', vars(env[0].view))
        self.assertNotIn('true_cue', vars(env[0].view))
        self.assertEqual(len(env), 100)


class TestBenchmark(unittest.TestCase):
    def test_seed_reproducibility_and_hard_cap(self):
        a=simulate('swap','learned_query', 1, 200, .25)
        b=simulate('swap','learned_query', 1, 200, .25)
        self.assertEqual(a,b)
        self.assertEqual(a.queries,50)

    def test_all_modes_equal_nonzero_budget(self):
        for family in FAMILIES:
            q={m:simulate(family,m,3,200,.25).queries for m in MODES}
            self.assertEqual(q['no_query'],0)
            self.assertEqual(set(q[m] for m in MODES if m != 'no_query'),{50})

    def test_negative_control_is_present(self):
        data=benchmark(trials=100,seeds=2,fraction=.25)
        self.assertIn('iid', data['results'])
        self.assertFalse(data['protocol']['energy_measured'])
        self.assertFalse(data['protocol']['policy_is_learned_end_to_end'])
        self.assertTrue(math.isfinite(data['results']['iid']['learned_vs_random']['mean']))

    def test_sensing_weights_reflect_error_and_uncertainty(self):
        self.assertGreater(weight_for_error(.40, 16), weight_for_error(.05, 16))
        self.assertGreater(weight_for_error(.2, 0), weight_for_error(.2, 16))


if __name__ == '__main__':
    unittest.main()
