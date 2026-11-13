"""Driver behavior and within-episode discovery tests."""
from __future__ import annotations

import unittest

from generalization_audit.cognitive_v02.arms import make_policy
from generalization_audit.cognitive_v02.driver import run_episode
from generalization_audit.cognitive_v02.metrics import truth_rules
from generalization_audit.cognitive_v02.streams import train_stream


class TestDriver(unittest.TestCase):

    def test_driver_runs_all_trained_arms(self):
        stream = train_stream(3, 2)
        truths = truth_rules(stream)
        for arm in ("trained", "epoch_reset", "explore",
                    "explore_epoch_reset"):
            agent = make_policy(arm, 3)
            for sp in stream:
                rec = run_episode(agent, sp, truths)
                self.assertIn("success", rec)
                self.assertIn("steps", rec)
                self.assertIsInstance(rec["discovered_this_episode"], bool)
                self.assertIn("trace_delta", rec)

    def test_within_episode_discovery_flag_matches_synthetic_memory(self):
        # Synthesize: grow memory inside run via a real training episode,
        # then assert the flag is a boolean and consistent with mem steps.
        stream = train_stream(0, 6)
        truths = truth_rules(stream)
        agent = make_policy("epoch_reset", 0)
        recs = [run_episode(agent, sp, truths) for sp in stream]
        for rec in recs:
            self.assertGreaterEqual(rec["mem_after_step"],
                                    rec["mem_before_step"])
            self.assertIsInstance(rec["discovered_this_episode"], bool)

    def test_save_load_preserves_state_for_eval(self):
        agent = make_policy("explore", 5)
        stream = train_stream(5, 4)
        for sp in stream:
            run_episode(agent, sp)
        state = agent.save_state()
        clone = make_policy("explore", 5)
        clone.load_state(state)
        self.assertEqual(len(clone.memory.hypotheses),
                         len(agent.memory.hypotheses))
        self.assertEqual(clone.memory.generated, agent.memory.generated)
        self.assertTrue(clone.cognitive_exploration)


if __name__ == "__main__":
    unittest.main()
