"""v0.10 benchmark harness verification (Phase G)."""
import unittest

from benchmark_v10 import (
    ALL_POLICIES, DEPLOYABLE_V10, benchmark, benchmark_parallel, score_policy,
)


class BenchmarkV10Tests(unittest.TestCase):
    BEHAVIORAL = ("success_rate", "mean_steps", "invalid_actions",
                  "door_events_total", "repeat_loops_total")

    def _behavioral(self, row):
        return {k: row[p][k] for p in ALL_POLICIES for k in self.BEHAVIORAL
                if k in row[p]}

    def test_score_policy_reproducible(self):
        a = score_policy("stable", 3, "metargon10", 12)
        b = score_policy("stable", 3, "metargon10", 12)
        for k in ("success_rate", "mean_steps", "repeat_loops_total"):
            self.assertEqual(a[k], b[k], k)

    def test_benchmark_repeatable_serial(self):
        a = benchmark(2, 6, conditions=["stable"])
        b = benchmark(2, 6, conditions=["stable"])
        self.assertEqual(a["metadata"], b["metadata"])
        for cond in a["summary"]:
            self.assertEqual(self._behavioral(a["summary"][cond]),
                             self._behavioral(b["summary"][cond]), cond)
            for k, v in a["summary"][cond].items():
                if k.startswith("comparison:"):
                    self.assertEqual(v, b["summary"][cond][k], (cond, k))

    def test_benchmark_parallel_matches_serial(self):
        a = benchmark(2, 6, conditions=["stable"])
        b = benchmark_parallel(2, 6, workers=2, conditions=["stable"])
        for cond in a["summary"]:
            self.assertEqual(self._behavioral(a["summary"][cond]),
                             self._behavioral(b["summary"][cond]), cond)
        for cond in a["raw"]:
            for i in range(2):
                for p in ALL_POLICIES:
                    for k in self.BEHAVIORAL:
                        self.assertEqual(a["raw"][cond][i][p][k],
                                         b["raw"][cond][i][p][k],
                                         (cond, i, p, k))

    def test_metadata_and_ranking_structure(self):
        r = benchmark(1, 8, conditions=["stable"])
        m = r["metadata"]
        self.assertFalse("oracle" in m["deployable_ranking"])
        self.assertEqual(set(m["deployable_ranking"]), set(DEPLOYABLE_V10))
        self.assertIn("metargon09", ALL_POLICIES)
        for cond in r["summary"]:
            self.assertIn("comparison:metargon10_minus_metargon09",
                          r["summary"][cond])
            self.assertIn("comparison:metargon10_minus_bayes",
                          r["summary"][cond])

    def test_summary_values_are_finite(self):
        r = benchmark(1, 8, conditions=["stable"])
        for cond, row in r["summary"].items():
            for p in DEPLOYABLE_V10 + ("bayes", "oracle"):
                x = row[p]
                for k in ("success_rate", "mean_steps", "cpu_s"):
                    self.assertTrue(0.0 <= x[k] < float("inf"), (cond, p, k))


if __name__ == "__main__":
    unittest.main()
