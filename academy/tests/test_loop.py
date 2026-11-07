"""End-to-end learning-loop tests: the loop writes curves, changes agent
state, and the replay toggle runs without error."""

import unittest

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.tasks import get_splits
from academy.learning.loop import make_train_stream, train_agent


class TestLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = get_splits()

    def test_train_agent_small(self):
        stream = make_train_stream(self.splits["train"],
                                   ["easy"] * 4 + ["medium"] * 4,
                                   seed=123)
        agent = MetargonAcademyAgent(seed=0)
        r = train_agent(agent, stream, {"replay_capacity": 500,
                                         "run_seed": 123}, "t1",
                        replay_enabled=True, replay_frequency=3,
                        replay_batch=8)
        self.assertEqual(r.episodes, 8)
        self.assertEqual(len(r.curve), 8)
        self.assertGreaterEqual(r.success_rate, 0.0)
        self.assertLessEqual(r.success_rate, 1.0)
        self.assertGreater(agent.diagnostics()["memory"]["generated"], 0)

    def test_replay_toggle_runs(self):
        stream = make_train_stream(self.splits["train"],
                                   ["easy"] * 6, seed=7)
        for replay in (True, False):
            agent = MetargonAcademyAgent(seed=0, replay_enabled=replay)
            r = train_agent(agent, stream, {"replay_capacity": 200,
                                             "run_seed": 7}, "t2",
                            replay_enabled=replay, replay_frequency=2,
                            replay_batch=4)
            self.assertEqual(r.episodes, 6)
            if replay:
                self.assertGreater(r.replay_count, 0)

    def test_curve_has_success_and_steps(self):
        stream = make_train_stream(self.splits["train"],
                                   ["easy"] * 3, seed=9)
        agent = MetargonAcademyAgent(seed=0)
        r = train_agent(agent, stream, {"replay_capacity": 100,
                                         "run_seed": 9}, "t3",
                        replay_enabled=False)
        self.assertTrue(all("success" in c and "steps" in c
                            for c in r.curve))


if __name__ == "__main__":
    unittest.main()
