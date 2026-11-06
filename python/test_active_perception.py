"""Run: python3 -m unittest discover -s python -v"""
import math
import unittest
from active_perception import (LOW_ERR, HIGH_ERR, BeliefAgent, QueryBudget,
                               aggregate, make_trials, prob_true_one, run)


class ActivePerceptionTests(unittest.TestCase):
    def test_reproducible_environment(self):
        self.assertEqual(make_trials(10, 200), make_trials(10, 200))
        self.assertNotEqual(make_trials(10, 200), make_trials(11, 200))

    def test_cue_posterior(self):
        self.assertAlmostEqual(prob_true_one(1, HIGH_ERR, None), 1 - HIGH_ERR)
        self.assertAlmostEqual(prob_true_one(0, LOW_ERR, None), LOW_ERR)
        self.assertGreater(prob_true_one(0, LOW_ERR, 1), 0.5)
        self.assertTrue(math.isfinite(prob_true_one(1, 0.03, 0)))

    def test_query_budget_never_exceeded(self):
        for mode in ("no_query", "random_query", "targeted_query"):
            for seed in range(5):
                policy = QueryBudget(mode, 100, 25 if mode != "no_query" else 0, seed)
                used = sum(policy.request(i, "low" if i % 3 == 0 else "high")
                           for i in range(100))
                self.assertEqual(used, 0 if mode == "no_query" else 25)

    def test_learning_changes_belief(self):
        agent = BeliefAgent()
        cue_probability = prob_true_one(0, HIGH_ERR, None)
        for _ in range(12):
            action = agent.act(0, cue_probability)
            agent.learn(0, action, cue_probability, int(action == 1))
        self.assertGreater(agent.p_mode1[0], 0.9)
        self.assertAlmostEqual(agent.p_mode1[1], 0.5)

    def test_seed_reproducibility(self):
        r1 = run("targeted_query", 3, 100)
        r2 = run("targeted_query", 3, 100)
        # Timings are intentionally not compared: wall clock is nondeterministic.
        for field in ("queries", "accuracy", "first_post_30_accuracy", "mistakes"):
            self.assertEqual(getattr(r1, field), getattr(r2, field))

    def test_matched_query_counts(self):
        rs = [run(mode, 2, 100) for mode in ("no_query", "random_query", "targeted_query")]
        self.assertEqual([r.queries for r in rs], [0, 25, 25])
        self.assertIn("targeted_query", aggregate(rs))


if __name__ == "__main__":
    unittest.main()
