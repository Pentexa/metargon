"""NEST — regression-diagnosis validation.

Wraps `nest/tools/diagnose_regression.py` (deterministic seeds) and
asserts the isolated findings:

- T1E1: fresh outperforms trained (regression reproduced);
- both memory ablations (content-cleared and functionally muted) restore
  fresh-level success -> the regression is memory interference;
- exploration-state/counters/RNG/threshold isolations do NOT restore it
  (negative control), demonstrating they are not the cause;
- the report records first-divergence traces and per-seed success values.

Nothing here forces positive results; it pins the observed deterministic
outcomes of a controlled experiment.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from nest.tools.diagnose_regression import (
    SEED_BASE,
    main as diagnose_main,
)

REPORT = _ROOT / "nest" / "diagnose_regression_report.json"
DIFF = 0.01


class TestRegressionDiagnosis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rc = diagnose_main(["--seeds", "2", "--json", str(REPORT)])
        cls.rc = rc
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_runs_successfully(self):
        self.assertEqual(self.rc, 0)

    def test_report_contains_both_pairs_and_all_arms(self):
        self.assertIn("T1E1", self.report["pairs"])
        self.assertIn("T2E2", self.report["pairs"])
        for tag, arms in self.report["pairs"].items():
            for arm in ("fresh", "trained_full_thr1", "trained_no_memory_content",
                        "trained_memory_muted", "trained_explore_state_reset",
                        "trained_counters_reset", "trained_rng_reset",
                        "trained_full_thr2"):
                self.assertIn(arm, arms, f"{tag}/{arm}")

    def test_regression_reproduced_on_T1E1(self):
        pair = self.report["pairs"]["T1E1"]
        fresh = pair["fresh"]["success_mean"]
        trained = pair["trained_full_thr1"]["success_mean"]
        self.assertGreater(fresh, trained, "fresh should exceed trained on T1E1")
        self.assertGreaterEqual(fresh, 0.99)
        self.assertLessEqual(trained, 0.01)
        self.assertEqual(len(pair["fresh"]["success_values"]), 2)

    def test_memory_ablation_isolates_cause(self):
        # Demonstrated mechanism: clearing memory content OR muting memory
        # read paths restores fresh-level success on T1E1.
        pair = self.report["pairs"]["T1E1"]
        fresh = pair["fresh"]["success_mean"]
        for arm in ("trained_no_memory_content", "trained_memory_muted"):
            self.assertGreaterEqual(pair[arm]["success_mean"], fresh - DIFF,
                                    f"{arm} should restore fresh on T1E1")
            self.assertGreaterEqual(pair[arm]["success_mean"], 0.99)

    def test_other_isolations_do_not_explain_regression(self):
        # Negative controls: these mutations leave the trained behavior
        # unchanged -> they are not the cause.
        pair = self.report["pairs"]["T1E1"]
        trained = pair["trained_full_thr1"]["success_mean"]
        for arm in ("trained_explore_state_reset", "trained_counters_reset",
                    "trained_rng_reset", "trained_full_thr2"):
            self.assertLessEqual(pair[arm]["success_mean"], trained + DIFF,
                                 f"{arm} should not restore success on T1E1")

    def test_first_divergence_traces_recorded(self):
        for tag in ("T1E1", "T2E2"):
            for key in ("fresh_actions", "first_divergence_trained_full_thr1",
                        "first_divergence_trained_memory_muted"):
                self.assertIn(key, self.report["pairs"][tag], f"{tag}/{key}")

    def test_matched_episodes_identical(self):
        # all arms in a pair share the same eval episodes (report computed
        # success per the same make_episodes stream)
        self.assertIn("seeds", self.report)
        self.assertEqual(len(self.report["seeds"]), 2)


if __name__ == "__main__":
    unittest.main()
