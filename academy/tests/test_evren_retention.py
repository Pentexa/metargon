"""Tests for the retention (anti-forgetting) protocol."""

from __future__ import annotations

import unittest

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig
from academy.envs.tasks import get_splits
from academy.evren.retention import RetentionSuite, sample_substrate


def _suite_from(pools):
    return pools


class TestSampleSubstrate(unittest.TestCase):
    def test_deterministic_subset(self):
        splits = get_splits()
        a = sample_substrate(splits["train"], 5, seed=1)
        b = sample_substrate(splits["train"], 5, seed=1)
        self.assertEqual([str(c.key) for c in a], [str(c.key) for c in b])
        self.assertEqual(len(a), 5)
        keys = {str(c.key) for c in a}
        self.assertEqual(len(keys), 5)      # no duplicates

    def test_subset_of_pool(self):
        splits = get_splits()
        pool_keys = {str(c.key) for c in splits["train"]}
        for c in sample_substrate(splits["train"], 4, seed=2):
            self.assertIn(str(c.key), pool_keys)

    def test_capped_at_pool_size(self):
        self.assertEqual(len(sample_substrate([C1, C2], 5, 0)), 2)

    def test_seeds_change_choice(self):
        splits = get_splits()
        a = sample_substrate(splits["train"], 8, seed=1)
        b = sample_substrate(splits["train"], 8, seed=2)
        self.assertNotEqual([str(c.key) for c in a],
                            [str(c.key) for c in b])


C1 = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
               frozenset({("load", "P2", "B")}), False, False)
C2 = EnvConfig("TP1", frozenset({("sw", "S2", False)}),
               frozenset({("load", "P2", "B")}), False, False)


class TestRetentionSuite(unittest.TestCase):
    def test_probe_appends_and_bounds(self):
        suite = RetentionSuite([C1, C2], max_steps=20,
                               episodes_per_config=1, seeds=(0,))
        snap = suite.run_probe(None, lambda s: MetargonAcademyAgent(seed=s),
                               episode_i=0)
        self.assertEqual(len(suite.snaps), 1)
        self.assertEqual(len(snap["records"]), 2)
        s = snap["summary"]
        self.assertIn("success_rate", s)
        self.assertGreaterEqual(s["success_rate"], 0.0)

    def test_deltas_and_summary(self):
        suite = RetentionSuite([C1], max_steps=20, episodes_per_config=1,
                               seeds=(0,))
        suite.run_probe(None, lambda s: MetargonAcademyAgent(seed=s), 0)
        suite.run_probe(None, lambda s: MetargonAcademyAgent(seed=s), 5)
        self.assertGreaterEqual(len(suite.snaps), 2)
        d = suite.retention_delta()
        self.assertGreaterEqual(d, -1.0)
        self.assertLessEqual(d, 1.0)
        summ = suite.summary()
        self.assertEqual(summ["substrate_n"], 1)
        self.assertIn(str(C1.key), summ["per_config"])

    def test_loaded_state_used(self):
        agent = MetargonAcademyAgent(seed=0)
        state = agent.save_state()
        suite = RetentionSuite([C1], max_steps=30, episodes_per_config=1,
                               seeds=(0,))
        snap = suite.run_probe(
            state, lambda s: MetargonAcademyAgent(seed=s), episode_i=0)
        # restored-state probe must at least reproduce counters
        rows = snap["records"]
        self.assertTrue(all("success" in r and "steps" in r for r in rows))

    def test_restore_determinism_same_state(self):
        """Two probes from the identical restored state produce identical
        outcomes (deterministic matched probe streams)."""
        agent = MetargonAcademyAgent(seed=0)
        state = agent.save_state()
        def probe():
            s2 = RetentionSuite([C1], max_steps=40, episodes_per_config=2,
                                seeds=(0,))
            s2.run_probe(state, lambda s: MetargonAcademyAgent(seed=s), 0)
            return [(r["success"], r["steps"]) for r in s2.snaps[0]["records"]]
        self.assertEqual(probe(), probe())


if __name__ == "__main__":
    unittest.main()
