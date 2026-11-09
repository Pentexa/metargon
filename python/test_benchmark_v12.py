"""v0.12 benchmark harness verification."""
import unittest

from benchmark_v12 import benchmark, benchmark_parallel, score_policy


class BenchmarkV12Tests(unittest.TestCase):
    KEYS = ("success_rate", "mean_steps", "invalid_actions",
            "exact_discovery_D1")

    def test_score_policy_reproducible(self):
        from benchmark_v12 import family_pools
        pools, _ = family_pools("l")
        a = score_policy("stable", 3, "metargon12", 8, "l", pools=pools)
        b = score_policy("stable", 3, "metargon12", 8, "l", pools=pools)
        for k in self.KEYS:
            self.assertEqual(a[k], b[k], k)

    def test_benchmark_repeatable_serial(self):
        a = benchmark(1, 8, family="l", conditions=["stable"])
        b = benchmark(1, 8, family="l", conditions=["stable"])
        self.assertEqual(a["metadata"], b["metadata"])
        for cond in a["summary"]:
            for p in a["summary"][cond]:
                if p.startswith("comparison:"):
                    self.assertEqual(a["summary"][cond][p],
                                     b["summary"][cond][p], p)

    def test_benchmark_parallel_matches_serial(self):
        a = benchmark(1, 8, family="l", conditions=["stable"])
        b = benchmark_parallel(1, 8, family="l", conditions=["stable"],
                               workers=2)
        for cond in a["summary"]:
            for p in a["summary"][cond]:
                if p.startswith("comparison:"):
                    self.assertEqual(a["summary"][cond][p],
                                     b["summary"][cond][p], p)


if __name__ == "__main__":
    unittest.main()
