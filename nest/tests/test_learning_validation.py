"""NEST learning-validation wrapper — runs the controlled CLI experiment and
asserts the report structure, matched budgets, and honest reporting. It
never asserts a particular success ordering (results may be negative)."""
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

from nest.tools.diagnose_learning import main as diagnose_main

ARMS = ("fresh", "train_no_guidance", "train_guidance", "train_claim",
        "train_misleading", "train_wiped")


class TestLearningValidationReport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report_path = _ROOT / "nest" / "diagnose_learning_report.json"
        rc = diagnose_main(["--seeds", "2", "--json", str(cls.report_path)])
        cls.rc = rc
        cls.report = json.loads(cls.report_path.read_text(encoding="utf-8"))

    def test_runs_successfully(self):
        self.assertEqual(self.rc, 0)

    def test_all_arms_present(self):
        for arm in ARMS:
            self.assertIn(arm, self.report["arms"], arm)

    def test_matched_budgets_and_metrics(self):
        for arm in ARMS:
            r = self.report["arms"][arm]
            for metric in ("success", "mean_steps", "invalid_total"):
                self.assertIn(metric, r)
                self.assertEqual(len(r[metric]["values"]), 2, arm)
            for k in ("verified", "contradicted", "uncertain", "generated"):
                self.assertIn(k, r["memory"], f"{arm}.{k}")

    def test_wipe_control_removes_influence(self):
        wiped = self.report["arms"]["train_wiped"]
        trained = self.report["arms"]["train_no_guidance"]
        self.assertEqual(wiped["trace"]["memory_actions"], 0.0)
        self.assertLess(wiped["trace"]["hypothesis_retrieved"],
                        trained["trace"]["hypothesis_retrieved"])

    def test_training_and_eval_separated(self):
        self.assertNotEqual(self.report["T_key"], self.report["E_key"])


if __name__ == "__main__":
    unittest.main()
