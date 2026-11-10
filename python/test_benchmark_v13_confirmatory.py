"""v0.13 confirmatory runner tests (seed independence, checkpoint/resume)."""
import json
import tempfile
import unittest
from pathlib import Path

from benchmark_v13 import benchmark, benchmark_parallel
from benchmark_v13_confirmatory import run, _load_checkpoint
from analysis_v13_confirmatory import delta_report, paired_bootstrap


POLS = ["metargon12", "metargon13"]


class ConfirmatorySeedIndependenceTests(unittest.TestCase):
    def test_seed_base_changes_worlds(self):
        # two seeds far apart must produce different traces for the same policy
        a = benchmark(2, 4, family="l", conditions=["stable"],
                      seed_base=21800, policies=POLS)
        b = benchmark(2, 4, family="l", conditions=["stable"],
                      seed_base=22800, policies=POLS)
        # metadata must record the base
        self.assertEqual(a["metadata"]["seed_base"], 21800)
        self.assertEqual(b["metadata"]["seed_base"], 22800)
        # worlds differ: at least one mean step count differs
        pa = a["summary"]["stable"]["metargon13"]["mean_steps"]
        pb = b["summary"]["stable"]["metargon13"]["mean_steps"]
        self.assertNotEqual(pa, pb)

    def test_parallel_matches_serial_with_seed_base(self):
        ser = benchmark(2, 4, family="l", conditions=["stable"],
                        seed_base=21800, policies=POLS)
        par = benchmark_parallel(2, 4, family="l", conditions=["stable"],
                                 workers=2, seed_base=21800, policies=POLS)
        self.assertEqual(ser["metadata"], par["metadata"])
        for cond in ser["summary"]:
            for p, s in ser["summary"][cond].items():
                if p.startswith("comparison:"):
                    self.assertEqual(s, par["summary"][cond][p], p)

    def test_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as td:
            jsonl = Path(td) / "runs.jsonl"
            # simulate two completed rows then verify loading
            rec1 = {"condition": "stable", "seed": 21800,
                    "row": {"p": {"success_rate": 0.5}}}
            rec2 = {"condition": "stable", "seed": 21801,
                    "row": {"p": {"success_rate": 0.75}}}
            with open(jsonl, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec1) + "\n")
                f.write(json.dumps(rec2) + "\n")
            rows = _load_checkpoint(jsonl)
            self.assertIn(("stable", 21800), rows)
            self.assertIn(("stable", 21801), rows)


class AnalysisTests(unittest.TestCase):
    def test_delta_report_bootstrap(self):
        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        r = delta_report(a, b, "t")
        self.assertAlmostEqual(r["delta"], 0.1)
        self.assertEqual(r["n_seeds"], 3)
        self.assertEqual(r["n_positive_delta_seeds"], 3)

    def test_paired_bootstrap_matches_harness(self):
        d = [0.1, -0.2, 0.3, 0.0]
        lo, hi = paired_bootstrap(d, seed=4301, rounds=1500)
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()
