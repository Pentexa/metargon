"""End-to-end smoke of the v0.2 runner (tiny sizes)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generalization_audit.cognitive_v02 import config as C
from generalization_audit.cognitive_v02.arms import all_arms
from generalization_audit.cognitive_v02.runner import run_seed, build_scorecard
from generalization_audit.cognitive_v02.report import render_scorecard

TINY = {"train": 4, "eval": 2, "ab_train": 2, "ab_eval": 2}


class TestRunnerSmoke(unittest.TestCase):

    def test_run_seed_covers_arms_and_phases(self):
        res = run_seed(0, TINY, ["standard", "ab"], with_oracle=True)
        for arm in all_arms():
            self.assertIn(arm, res, f"missing arm {arm}")
            self.assertIn("train", res[arm])
            for pool, _, _ in C.EVAL_POOLS:
                self.assertIn(pool, res[arm])
            self.assertIn("ab", res[arm])
            self.assertIn("A1_eval", res[arm]["ab"])
            self.assertIn("B_eval", res[arm]["ab"])
            self.assertIn("A2_eval", res[arm]["ab"])
            self.assertIn("retention", res[arm]["ab"])
        self.assertIn("_floors", res)
        self.assertIn("random", res["_floors"])
        self.assertIn("oracle", res["_floors"])

    def test_scorecard_and_report_render(self):
        res = run_seed(0, TINY, ["standard", "ab"], with_oracle=True)
        meta = {
            "git_head": "test", "date": "2026-11-12", "seeds": [0],
            "sizes": TINY, "phases": ["standard", "ab"],
            "splits": {}, "arms": {}, "stream_seed_base": 0,
            "questions": [], "methodology": [], "limitations": [],
        }
        pairs = [
            ("trained-fresh.test_clean.success_rate",
             "trained", "test_clean", "success_rate",
             "fresh", "test_clean", "success_rate"),
            ("ab.forgetting.trained.a2-a1", "trained",
             "ab.retention", "a2_success",
             "trained", "ab.retention", "a1_success"),
        ]
        sc = build_scorecard({0: res}, meta, pairs)
        self.assertIn("arms", sc)
        for arm in all_arms():
            self.assertIn("train", sc["arms"][arm])
            self.assertIn("test_clean", sc["arms"][arm])
            self.assertIn("ab.retention", sc["arms"][arm])
        self.assertIn("paired", sc)
        self.assertIn("floors", sc)
        # json serializable
        json.dumps(sc)
        with tempfile.TemporaryDirectory() as td:
            p = render_scorecard(sc, Path(td) / "report.md")
            text = p.read_text(encoding="utf-8")
            self.assertIn("Questions asked", text)
            self.assertIn("A/B/A", text)
            self.assertIn("Limitations", text)

    def test_raw_records_stripped_are_small(self):
        res = run_seed(0, TINY, ["standard"], with_oracle=False)
        from generalization_audit.cognitive_v02.runner import _strip_bulk
        stripped = _strip_bulk(res)
        blob = json.dumps(stripped)
        self.assertLess(len(blob), 2_000_000)
        self.assertNotIn('"antecedent"', blob)


if __name__ == "__main__":
    unittest.main()
