"""Metric unit tests on synthetic data."""
from __future__ import annotations

import unittest

from academy.envs.gridworld import EnvConfig

from generalization_audit.cognitive_v01.metrics import (
    _rate,
    discovery_metrics,
    eval_metrics,
    first_episode_with,
    learning_curve,
    match_hypotheses,
    restart_metrics,
    revisit_metrics,
    summarize,
    summarize_proportion,
    switch_metrics,
    truth_rules,
    zero_shot_coverage,
)
from generalization_audit.cognitive_v01.protocols import EpisodeSpec


def _cfg(rule1, rule2=None, tp="TP1"):
    return EnvConfig(topology=tp,
                     d1_rule=rule1,
                     d2_rule=rule2 if rule2 is not None
                     else frozenset({("empty", "P2")}),
                     d1_delay=False, d2_delay=False)


class TestMetrics(unittest.TestCase):

    def test_summarize_ci(self):
        s = summarize([0.5, 0.6, 0.7, 0.8])
        self.assertAlmostEqual(s["mean"], 0.65)
        self.assertIsNotNone(s["ci"])
        s2 = summarize([None, 0.2])
        self.assertEqual(s2["mean"], 0.2)
        self.assertEqual(summarize([])["mean"], None)

    def test_summarize_proportion_wilson_respects_bounds(self):
        # extreme pooled proportions must stay inside [0, 1]
        s = summarize_proportion([(0, 4), (1, 8)])
        self.assertAlmostEqual(s["mean"], 1 / 12)
        self.assertEqual(s["method"], "wilson")
        self.assertGreaterEqual(s["ci"][0], 0.0)
        self.assertLessEqual(s["ci"][1], 1.0)
        s1 = summarize_proportion([(4, 4), (8, 8)])
        self.assertEqual(s1["mean"], 1.0)
        self.assertEqual(s1["ci"][1], 1.0)   # upper bound clamped to 1
        self.assertGreater(s1["ci"][0], 0.7)  # Wilson lower bound < 1
        self.assertIsNone(summarize_proportion([None])["mean"])

    def test_plan_cache_metrics_renamed_no_fake_rate(self):
        recs = [
            {"success": True, "steps": 10, "invalid": 0,
             "door_open": {"D1": True}, "events": [("D1", "open")],
             "trace_delta": {"plans_used": 5, "plan_cache_hits": 40,
                             "planner_ops": 120,
                             "random_fallback_steps": 1}},
        ]
        m = eval_metrics(recs)
        self.assertNotIn("plan_cache_hit_rate", m)
        self.assertIn("plan_cache_hits_per_step", m)
        self.assertIn("plans_used_per_step", m)
        self.assertEqual(m["plan_cache_hits_per_step"], 4.0)
        self.assertEqual(m["plans_used_per_step"], 0.5)
        self.assertEqual(m["_counts"]["success_rate"], (1, 1))
        self.assertEqual(m["_counts"]["fallback_rate"], (1, 10))

    def test_truth_rules_from_stream(self):
        cfg = _cfg(frozenset({("sw", "S1", True)}))
        spec = EpisodeSpec(config=cfg, max_steps=80, noise_q=0.0,
                           spec_seed=0, pool_tag="train")
        truths = truth_rules([spec])
        self.assertIn(("D1", frozenset({("sw", "S1", True)})), truths)
        self.assertIn(("D2", frozenset({("empty", "P2")})), truths)

    def test_match_hypotheses_and_discovery(self):
        truths = {("D1", frozenset({("sw", "S1", True)}))}
        mem = [
            {"door": "D1", "antecedent": [["sw", "S1", True]],
             "status": "verified", "sup": 3, "con": 0, "id": 1},
            {"door": "D1", "antecedent": [["sw", "S2", True]],
             "status": "verified", "sup": 2, "con": 0, "id": 2},
            {"door": "D1", "antecedent": [["sw", "S1", True]],
             "status": "hypothesis", "sup": 0, "con": 0, "id": 3},
        ]
        m = match_hypotheses(mem, truths)
        self.assertEqual(len(m["verified_correct"]), 1)
        self.assertEqual(len(m["verified_wrong"]), 1)
        self.assertEqual(len(m["candidates_correct"]), 1)
        d = discovery_metrics(mem, truths)
        self.assertEqual(d["verified_precision"], 0.5)
        self.assertAlmostEqual(d["discovery_coverage"], 1.0)

    def test_learning_curve_and_first_episode(self):
        truths = {("D1", frozenset({("sw", "S1", True)}))}
        recs = [
            {"memory": [{"door": "D1", "antecedent": [["sw", "S1", True]],
                         "status": "hypothesis", "sup": 0, "con": 0,
                         "id": 1}]},
            {"memory": [{"door": "D1", "antecedent": [["sw", "S1", True]],
                         "status": "verified", "sup": 3, "con": 0,
                         "id": 1}]},
        ]
        curve = learning_curve(recs, truths)
        self.assertEqual(curve, [0, 1])
        self.assertEqual(first_episode_with(
            recs, lambda r: match_hypotheses(r["memory"], truths)["verified_correct"]), 2)
        self.assertIsNone(first_episode_with([], lambda r: True))

    def test_switch_metrics(self):
        recs = [
            {"success": True}, {"success": True},
            {"success": False}, {"success": True},
        ]
        recs[0]["memory"] = [{"door": "D1", "status": "verified",
                              "antecedent": [["sw", "S1", True]], "id": 1}]
        recs[1]["memory"] = [{"door": "D1", "status": "verified",
                              "antecedent": [["sw", "S1", True]], "id": 1}]
        recs[-1]["memory"] = [{"door": "D1", "status": "contradicted",
                               "antecedent": [["sw", "S1", True]], "id": 1}]
        m = switch_metrics(recs, split_point=2, target_door="D1")
        self.assertEqual(m["success_before"], 1.0)
        self.assertEqual(m["success_after"], 0.5)
        self.assertEqual(m["recovery_lag"], 2)
        self.assertEqual(m["wrong_belief_correction"], 1.0)

    def test_revisit_metrics(self):
        recs = [{"success": True}, {"success": False}, {"success": False},
                {"success": True}]
        occ = {"A": [0, 3], "B": [1, 2]}
        m = revisit_metrics(recs, occ)
        self.assertEqual(m["success_early"], 0.5)
        self.assertEqual(m["success_late"], 0.5)
        self.assertEqual(m["retention_preserved"], 1.0)

    def test_restart_parity(self):
        m = restart_metrics([{"success": True}, {"success": True}],
                            [{"success": True}, {"success": False}])
        self.assertEqual(m["success_direct"], 1.0)
        self.assertEqual(m["success_restored"], 0.5)
        self.assertEqual(m["parity"], 0.5)

    def test_zero_shot_coverage(self):
        cfg = _cfg(frozenset({("sw", "S1", True)}))
        stream = [EpisodeSpec(config=cfg, max_steps=80, noise_q=0.0,
                              spec_seed=0, pool_tag="test")]
        pre_mem = [{"door": "D1", "antecedent": [["sw", "S1", True]],
                    "status": "verified", "sup": 3, "con": 0, "id": 1}]
        self.assertEqual(zero_shot_coverage(pre_mem, stream), (1, 1))
        self.assertEqual(zero_shot_coverage([], stream), (0, 1))

    def test_rate_helper(self):
        self.assertEqual(_rate([{"success": True}, {"success": False}], "success"), 0.5)
        self.assertIsNone(_rate([], "success"))


if __name__ == "__main__":
    unittest.main()
