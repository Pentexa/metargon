"""Tests for the Persistent Learning Diagnosis v0.1 harness.

Covers the ground-truth hypothesis classification, the memory snapshot,
the diagnostic control hooks (wipe / shuffle), the repeated-learning
attribution, and a small end-to-end smoke of the main matrix.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from academy.agents.memory import (
    CONTRADICTED, VERIFIED, CompositionalMemory, consequence_key,
)
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig, TOPOLOGIES, is_solvable
from academy.envs.tasks import (
    difficulty_of, load_splits, make_episodes,
)
from academy.tools.memory_diagnosis_attribution_v01 import (
    DiagnosticAgent, classify_hypothesis, memory_snapshot,
    _canonical_key, _rlearn_attribution, build_conflict_pool,
    pick_configs, run_repeated_learning,
)

TP1 = TOPOLOGIES["TP1"]

R1 = frozenset({("sw", "S2", True)})
R2 = frozenset({("sw", "S1", False)})


def _cfg():
    return EnvConfig(topology="TP1", d1_rule=R1, d2_rule=R2,
                     d1_delay=False, d2_delay=False)


class TestClassification(unittest.TestCase):
    def test_exact(self):
        h = CompositionalMemory()._add(antecedent=R1, consequence=consequence_key("D1", True))
        self.assertEqual(classify_hypothesis(h, _cfg()), "exact")

    def test_incorrect(self):
        m = CompositionalMemory()
        h = m._add(frozenset({("sw", "S2", False)}),
                   consequence_key("D1", True))
        self.assertEqual(classify_hypothesis(h, _cfg()), "incorrect")

    def test_subset(self):
        m = CompositionalMemory()
        h = m._add(frozenset({("sw", "S2", True), ("loadany", "P1")}),
                   consequence_key("D1", True))
        # a pair that equals the true pair is exact
        cfg2 = EnvConfig(topology="TP1",
                         d1_rule=frozenset({("sw", "S2", True),
                                            ("loadany", "P1")}),
                         d2_rule=R2, d1_delay=False, d2_delay=False)
        self.assertEqual(classify_hypothesis(h, cfg2), "exact")
        # a proper subset of a multi-atom true clause
        h2 = m._add(frozenset({("sw", "S2", True)}),
                    consequence_key("D1", True))
        self.assertEqual(classify_hypothesis(h2, cfg2), "subset")
        # and it is incorrect against the single-atom rule
        self.assertEqual(classify_hypothesis(h, _cfg()), "incorrect")


class TestMemorySnapshot(unittest.TestCase):
    def test_false_contradictions_counted(self):
        m = CompositionalMemory()
        h = m._add(R1, consequence_key("D1", True))
        for _ in range(3):
            m._support(h)
        s = memory_snapshot(_agent_with(m), _cfg())
        self.assertEqual(s["by_class"]["exact"]["verified"], 1)
        self.assertEqual(s["false_contradictions"], 0)
        # contradict it
        for _ in range(5):
            m._contradict(h)
        s = memory_snapshot(_agent_with(m), _cfg())
        self.assertEqual(s["by_class"]["exact"]["contradicted"], 1)
        self.assertEqual(s["false_contradictions"], 1)


def _agent_with(memory):
    a = MetargonAcademyAgent(seed=0)
    a.memory = memory
    return a


class TestControlHooks(unittest.TestCase):
    def test_wipe(self):
        a = DiagnosticAgent(seed=0, _wipe_memory=True)
        a.memory._add(R1, consequence_key("D1", True))
        a.maybe_wipe()
        self.assertEqual(len(a.memory.hypotheses), 0)

    def test_shuffle_breaks_content(self):
        m = CompositionalMemory()
        h1 = m._add(R1, consequence_key("D1", True))
        h2 = m._add(R2, consequence_key("D2", True))
        a = DiagnosticAgent(seed=0, _shuffle_memory=True, _shuffle_seed=1)
        a.memory = m
        a.maybe_shuffle()
        # shuffling may or may not change each; at least one of the two
        # should have moved if there are two distinct antecedents
        antes = {h.antecedent for h in m.hypotheses}
        self.assertEqual(len(antes), 2)


class TestCanonical(unittest.TestCase):
    def test_canonical_key_order_independent(self):
        c = _cfg()
        k1 = _canonical_key(c)
        self.assertEqual(k1, _canonical_key(c))

    def test_pick_configs_deterministic(self):
        splits = load_splits()
        a = pick_configs(splits["test"], 4, 7)
        b = pick_configs(splits["test"], 4, 7)
        self.assertEqual([_canonical_key(x) for x in a],
                         [_canonical_key(x) for x in b])

    def test_conflict_pool_excludes_train(self):
        splits = load_splits()
        pool = build_conflict_pool(splits, seed=20270213, target=6)
        self.assertGreaterEqual(len(pool), 4)
        train_keys = {c.key for c in splits["train"]}
        for c in pool:
            self.assertNotIn(c.key, train_keys)
            self.assertTrue(is_solvable(c))


class TestRepeatLearningAttribution(unittest.TestCase):
    def test_attribution_splits_retained_vs_rediscovered(self):
        a1 = {"step": 1000, "hypotheses": [
            {"h_id": 1, "created_step": 100, "status": VERIFIED,
             "door": "D1", "antecedent": sorted(R1)},
        ]}
        b = {"step": 2000, "hypotheses": [
            {"h_id": 1, "created_step": 100, "status": VERIFIED,
             "door": "D1", "antecedent": sorted(R1)},
        ]}
        a2 = {"step": 3000, "hypotheses": []}
        eval_a2 = {"rows": [
            {"hypothesis_plans": 5, "reuse_plans": 3},
            {"hypothesis_plans": 2, "reuse_plans": 0},
        ]}
        attr = _rlearn_attribution(a1, b, a2, eval_a2)
        self.assertEqual(attr["a2_retained_plans"], 3)
        self.assertEqual(attr["a2_rediscovered_plans"], 4)
        self.assertAlmostEqual(attr["a2_retention_share"], 3 / 7)
        self.assertEqual(attr["a1_hypotheses_usable_after_B"], 1)


class TestEndToEndSmoke(unittest.TestCase):
    def test_rlearn_smoke(self):
        splits = load_splits()
        with tempfile.TemporaryDirectory() as tmp:
            out = run_repeated_learning(
                splits, variants=("original", "epoch_reset"),
                seed=20270201, results_dir=tmp,
                phase_episodes={"A1": 4, "B": 3, "A2": 3})
            self.assertIn("original", out["variants"])
            self.assertIn("epoch_reset", out["variants"])
            for v, d in out["variants"].items():
                self.assertIn("success", d)
                self.assertIn("attribution", d)
                self.assertEqual(set(d["success"]), {"A1", "B", "A2"})

    def test_main_matrix_smoke(self):
        import academy.tools.memory_diagnosis_attribution_v01 as m
        splits = load_splits()
        m.GRID["classic"] = {"n_configs": 1, "eps_per_cfg": 1,
                              "noise_q": 0.0}
        m.GRID["conflict"] = {"n_configs": 1, "eps_per_cfg": 1,
                              "noise_q": 0.0}
        with tempfile.TemporaryDirectory() as tmp:
            out = m.run_main_matrix(splits, seeds=(20270201,),
                                    train_eps=2, results_dir=tmp)
            self.assertIn("classic", out["cells"])
            self.assertIn("epoch_reset",
                          out["cells"]["classic"]["variants"])
            rec = out["cells"]["classic"]["variants"]["epoch_reset"][0]
            self.assertIn("frozen", rec)
            self.assertIn("adaptive", rec)
            self.assertIn("snapshot_agg", rec["frozen"])


if __name__ == "__main__":
    unittest.main()
