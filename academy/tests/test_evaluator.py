"""Tests for the evaluator: task isolation enforcement, paired seeds, and
bootstrap metrics."""

import unittest

from academy.agents.baselines import RandomPolicy
from academy.envs.tasks import get_splits
from academy.eval.evaluator import Evaluator
from academy.eval.metrics import bootstrap_ci, summarize_episodes


class TestEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = get_splits()

    def test_isolation_refuses_overlap(self):
        evaluator = Evaluator(self.splits, seed=1, episodes_per_seed=4)
        # inject a training config into the eval pool to force a leak
        leaked = [self.splits["train"][0]] + self.splits["test"][:3]
        with self.assertRaises(RuntimeError):
            evaluator.run_policy(lambda s: RandomPolicy(s), leaked,
                                 n_seeds=1)

    def test_run_policy_small(self):
        evaluator = Evaluator(self.splits, seed=2, episodes_per_seed=5)
        res = evaluator.run_policy(lambda s: RandomPolicy(s),
                                   self.splits["test"][:5],
                                   n_seeds=2)
        self.assertEqual(res["n_seeds"], 2)
        self.assertTrue(0.0 <= res["summary"]["success_rate"] <= 1.0)
        self.assertEqual(len(res["per_seed"]), 2)
        self.assertEqual(len(res["records"]), 2 * 5)

    def test_bootstrap_ci(self):
        lo, hi = bootstrap_ci([1.0] * 50 + [0.0] * 50, n_boot=200, seed=0)
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_summarize(self):
        s = summarize_episodes([{"success": True, "steps": 5},
                                {"success": False, "steps": 9}])
        self.assertEqual(s["success_rate"], 0.5)
        self.assertEqual(s["mean_steps"], 7.0)


if __name__ == "__main__":
    unittest.main()
