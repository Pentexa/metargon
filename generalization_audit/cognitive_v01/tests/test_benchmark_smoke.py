"""End-to-end smoke of the benchmark runner (tiny sizes)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generalization_audit.cognitive_v01.benchmark_cognitive import run_seed
from generalization_audit.cognitive_v01.metrics import build_scorecard, truth_rules
from generalization_audit.cognitive_v01.report import render_scorecard
from generalization_audit.cognitive_v01.protocols import (
    revisit_occurrences,
    revisit_stream,
    switch_split_point,
    switch_stream,
    train_stream,
)
from generalization_audit.cognitive_v01 import config

TINY = {"train": 3, "eval": 2, "revisit": 7, "switch_half": 2}


class TestBenchmarkSmoke(unittest.TestCase):

    def test_run_seed_produces_all_policies_and_protocols(self):
        res, extra = run_seed(0, TINY, include_restart=True)
        for pol in config.ALL_POLICIES:
            self.assertIn(pol, res)
        for pol in ("metargon_trained", "metargon_muted"):
            self.assertIn("train", res[pol])
            self.assertIn("eval_val", res[pol])
            self.assertIn("eval_test", res[pol])
            self.assertIn("eval_ood", res[pol])
            self.assertIn("revisit", res[pol])
            self.assertIn("switch", res[pol])
            self.assertIn("restart", res[pol])
            self.assertEqual(res[pol]["restart"]["recs"]["direct"][0]["success"],
                             res[pol]["eval_test"]["recs"][0]["success"])
        self.assertIsNotNone(extra["switch_split"])

    def test_scorecard_and_report_render(self):
        res, extra = run_seed(0, TINY, include_restart=True)
        train = train_stream(0, TINY["train"])
        switch = switch_stream(0, TINY["switch_half"])
        meta = {
            "date": "2026-01-01", "git_head": "test", "seeds": [0],
            "sizes": TINY,
            "splits": {},
            "train_truths": len(truth_rules(train)),
            "switch_target_door": "D1",
            "switch_split": switch_split_point(switch),
            "revisit_occurrences": revisit_occurrences(revisit_stream(0, 3)),
            "protocols": list(config.PROTOCOLS),
        }
        extra_vectors = {}
        for k, v in extra["zero_shot"].items():
            extra_vectors[k] = v
        sc = build_scorecard({0: res}, meta, extra_vectors,
                             train_stream_specs=train)
        self.assertIn("dimensions", sc)
        self.assertEqual(len(sc["dimensions"]), 6)
        # bounded proportion metrics use the Wilson method
        succ = sc["dimensions"]["generalization"]["metrics"]
        self.assertIn("metargon_trained.eval_test.success_rate", succ)
        self.assertEqual(succ["metargon_trained.eval_test.success_rate"]["method"],
                         "wilson")
        # the misleading hit-rate metric must not reappear
        plan = sc["dimensions"]["planning_efficiency"]["metrics"]
        self.assertNotIn("metargon_trained.eval_test.plan_cache_hit_rate", plan)
        self.assertIn("metargon_trained.eval_test.plan_cache_hits_per_step", plan)
        self.assertIn("metargon_trained.eval_test.plans_used_per_step", plan)
        with tempfile.TemporaryDirectory() as td:
            p = render_scorecard(sc, Path(td) / "report.md")
            self.assertTrue(p.exists())
            text = p.read_text(encoding="utf-8")
            self.assertIn("Causal discovery", text)
            self.assertIn("Planning efficiency", text)
            self.assertIn("Wilson", text)


if __name__ == "__main__":
    unittest.main()
