"""v0.9 benchmark harness verification (Phase G)."""
import unittest

from benchmark_v09 import (
    DEPLOYABLE, POLICIES, benchmark, benchmark_parallel, score_policy,
)


class BenchmarkTests(unittest.TestCase):
    def test_score_policy_reproducible(self):
        a = score_policy("stable", 3, "metargon", 12)
        b = score_policy("stable", 3, "metargon", 12)
        for k in ("success_rate", "mean_steps", "planner_ops", "plan_calls"):
            self.assertEqual(a[k], b[k], k)

    def test_benchmark_repeatable_serial(self):
        a = benchmark(2, 12)
        b = benchmark(2, 12)
        self.assertEqual(a, b)

    def test_benchmark_parallel_matches_serial(self):
        a = benchmark(2, 8)
        b = benchmark_parallel(2, 8, workers=2)
        self.assertEqual(a["summary"], b["summary"])
        # raw ordering may differ; compare per (condition, seed) rows
        for cond in a["raw"]:
            for i in range(2):
                for p in POLICIES:
                    self.assertEqual(a["raw"][cond][i][p], b["raw"][cond][i][p], (cond, i, p))

    def test_metadata_and_ranking_structure(self):
        r = benchmark(1, 8)
        m = r["metadata"]
        self.assertTrue(m["oracle_has_hidden_config"])
        self.assertFalse("oracle" in m["deployable_ranking"])
        self.assertEqual(set(m["deployable_ranking"]), set(DEPLOYABLE))
        self.assertIn("oracle", POLICIES)
        for cond in r["summary"]:
            self.assertIn("comparison:metargon_minus_bayes", r["summary"][cond])
            self.assertIn("comparison:bayes_minus_qlearning", r["summary"][cond])

    def test_summary_values_are_finite(self):
        r = benchmark(1, 8)
        for cond, row in r["summary"].items():
            for p in DEPLOYABLE + ("oracle",):
                x = row[p]
                for k in ("success_rate", "mean_steps", "cpu_s", "peak_mem_mb"):
                    self.assertTrue(0.0 <= x[k] < float("inf"), (cond, p, k))


if __name__ == "__main__":
    unittest.main()
