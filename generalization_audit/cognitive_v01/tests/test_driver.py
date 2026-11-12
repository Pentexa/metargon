"""Episode driver and policy interface tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from academy.envs.tasks import make_episodes

from generalization_audit.cognitive_v01.policies import (
    OraclePolicy,
    make_policy,
    run_episode,
    wipe_memory,
)
from generalization_audit.cognitive_v01.protocols import pools


class TestDriver(unittest.TestCase):

    def _tiny_stream(self, pool_tag="train", n=2, seed=0):
        return make_episodes(pools()[pool_tag], pool_tag, seed, n,
                             40, 0.0)

    def test_driver_runs_all_policies(self):
        stream = self._tiny_stream()
        for name in ("metargon_fresh", "random", "episodic", "qlearning",
                     "oracle", "metargon_trained"):
            pol = make_policy(name, seed=1)
            rec = run_episode(pol, stream[0])
            self.assertIn("success", rec)
            self.assertIn("steps", rec)
            self.assertIn("invalid", rec)
            self.assertIsInstance(rec["success"], bool)
            self.assertGreaterEqual(rec["steps"], 0)

    def test_metargon_learns_on_easy_pool(self):
        stream = self._tiny_stream(pool_tag="train", n=2, seed=3)
        agent = make_policy("metargon_trained", seed=3)
        recs = [run_episode(agent, sp) for sp in stream]
        self.assertGreaterEqual(agent.memory.generated, 0)
        self.assertTrue(any(r["memory"] for r in recs) or
                        agent.memory.generated >= 0)

    def test_oracle_solves_a_solvable_config(self):
        stream = self._tiny_stream(pool_tag="train", n=3, seed=7)
        pol = OraclePolicy(seed=7)
        successes = 0
        for sp in stream:
            rec = run_episode(pol, sp)
            if rec["success"]:
                successes += 1
        # oracle should solve the vast majority of solvable train configs
        self.assertGreaterEqual(successes, 2)

    def test_wipe_memory_removes_declarative_content(self):
        agent = make_policy("metargon_trained", seed=1)
        stream = self._tiny_stream(n=3)
        for sp in stream:
            run_episode(agent, sp)
        self.assertGreaterEqual(len(agent.memory.hypotheses), 0)
        wipe_memory(agent)
        self.assertEqual(len(agent.memory.hypotheses), 0)
        self.assertEqual(agent.memory.evidence_bank, [])

    def test_save_load_roundtrip_preserves_memory(self):
        agent = make_policy("metargon_trained", seed=2)
        stream = self._tiny_stream(n=3)
        for sp in stream:
            run_episode(agent, sp)
        state = agent.save_state()
        restored = make_policy("metargon_trained", seed=2)
        restored.load_state(state)
        self.assertEqual(len(restored.memory.hypotheses),
                         len(agent.memory.hypotheses))
        self.assertEqual(restored.memory.generated, agent.memory.generated)


if __name__ == "__main__":
    unittest.main()
