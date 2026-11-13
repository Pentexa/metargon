"""Metric unit tests (raw builders, aggregation, discovery)."""
from __future__ import annotations

import unittest

from academy.envs.gridworld import EnvConfig

from generalization_audit.cognitive_v02.metrics import (
    ab_phase_raw,
    discovery_counts,
    eval_pool_raw,
    match_hypotheses,
    paired_deltas,
    retention_raw,
    standard_train_raw,
    summarize,
    summarize_proportion,
    truth_rules,
)
from generalization_audit.cognitive_v02.streams import single_config_stream


def _cfg(d1, d2=None, tp="TP1"):
    return EnvConfig(topology=tp, d1_rule=d1,
                     d2_rule=d2 if d2 is not None
                     else frozenset({("empty", "P2")}),
                     d1_delay=False, d2_delay=False)


class TestAggregation(unittest.TestCase):

    def test_summarize_t(self):
        s = summarize([0.5, 0.6, 0.7, 0.8])
        self.assertAlmostEqual(s["mean"], 0.65)
        self.assertEqual(s["method"], "t")

    def test_wilson_bounds(self):
        s = summarize_proportion([(0, 4), (1, 8)])
        self.assertEqual(s["method"], "wilson")
        self.assertGreaterEqual(s["ci"][0], 0.0)
        self.assertLessEqual(s["ci"][1], 1.0)
        s2 = summarize_proportion([(12, 12), (8, 8)])
        self.assertEqual(s2["ci"][1], 1.0)
        self.assertIsNone(summarize_proportion([None])["mean"])

    def test_paired_deltas(self):
        d = paired_deltas([0.8, 0.6, 0.9], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(d["mean"], 0.2666667, places=4)
        self.assertEqual(d["n"], 3)
        self.assertIsNotNone(d["ci"])


class TestDiscovery(unittest.TestCase):

    def test_discovery_counts(self):
        truths = {("D1", frozenset({("sw", "S1", True)}))}
        mem = [
            {"door": "D1", "antecedent": [["sw", "S1", True]],
             "status": "verified", "sup": 3, "con": 0, "id": 1,
             "created_step": 5},
            {"door": "D1", "antecedent": [["sw", "S2", True]],
             "status": "verified", "sup": 2, "con": 0, "id": 2,
             "created_step": 6},
            {"door": "D1", "antecedent": [["sw", "S1", True]],
             "status": "contradicted", "sup": 1, "con": 1, "id": 3,
             "created_step": 7},
        ]
        m = match_hypotheses(mem, truths)
        self.assertEqual(len(m["verified_correct"]), 1)
        self.assertEqual(len(m["verified_wrong"]), 1)
        self.assertEqual(len(m["contradicted_truth"]), 1)
        d = discovery_counts(mem, truths)
        self.assertEqual(d["coverage_counts"], (1, 1))
        self.assertEqual(d["precision_counts"], (1, 2))


class TestRawBuilders(unittest.TestCase):

    def _recs(self):
        base = {
            "success": True, "steps": 10, "invalid": 0, "actions": 10,
            "door_open": True, "events": [("D1", "open")],
            "first_open_step": 4,
            "trace_delta": {"planner_ops": 50, "random_fallback_steps": 2,
                            "plan_cache_hits": 30, "plans_used": 3,
                            "memory_actions": 1, "memory_correct": 1,
                            "memory_incorrect": 0},
            "mem_before_step": 0, "mem_after_step": 20,
            "discovered_this_episode": True,
            "mem": [{"id": 1, "door": "D1",
                     "antecedent": [["sw", "S1", True]],
                     "status": "verified", "sup": 3, "con": 0,
                     "created_step": 4}],
        }
        fail = dict(base, success=False, steps=10,
                    discovered_this_episode=False,
                    trace_delta=dict(base["trace_delta"],
                                     random_fallback_steps=0))
        return [base, fail]

    def test_eval_pool_raw_shape(self):
        recs = self._recs()
        pre_mem = recs[0]["mem"]
        truths = truth_rules(single_config_stream(
            _cfg(frozenset({("sw", "S1", True)})), 0, 2))
        raw = eval_pool_raw(recs, pre_mem, truths)
        self.assertEqual(raw["success_rate"], (1, 2))
        self.assertEqual(raw["within_episode_discovery"], (1, 2))
        self.assertIsInstance(raw["planner_ops_per_step"], float)
        self.assertEqual(raw["zero_shot_coverage"][0], 1)

    def test_standard_train_raw_shape(self):
        recs = self._recs()
        truths = truth_rules(single_config_stream(
            _cfg(frozenset({("sw", "S1", True)})), 0, 2))
        raw = standard_train_raw(recs, truths)
        self.assertEqual(raw["verified_correct"], 1)
        self.assertEqual(raw["coverage"], (1, 2))
        self.assertEqual(raw["precision"], (1, 1))

    def test_ab_phase_raw(self):
        raw = ab_phase_raw(self._recs())
        self.assertEqual(raw["success_rate"], (1, 2))
        self.assertIsInstance(raw["mean_steps_to_success"], float)

    def test_retention_raw(self):
        cfg = _cfg(frozenset({("sw", "S1", True)}))
        recs = self._recs()
        mem = recs[0]["mem"]
        raw = retention_raw(recs, recs, recs, cfg, mem)
        self.assertEqual(raw["a1_success"], (1, 2))
        self.assertGreaterEqual(raw["a_rule_retained_final"][0], 1)


if __name__ == "__main__":
    unittest.main()
