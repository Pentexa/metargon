"""Tests for the curriculum manager: measured progress, adaptive
proposals, and checkpoint roundtrip."""

import random
import unittest

from academy.learning.curriculum import BINS, CurriculumManager
from academy.envs.tasks import Difficulty, EpisodeSpec
from academy.envs.gridworld import EnvConfig


def spec_for(bin_name):
    diff = Difficulty(2, False, "TP1", False)
    diff = Difficulty(2, False, "TP1", False)
    # rebuild with the requested bin
    diff = Difficulty(1, bin_name == "easy", "TP1", False)
    return EpisodeSpec(EnvConfig("TP1", frozenset(), frozenset(), False, False),
                       80, 0.0, 0, "train", diff)


class TestCurriculum(unittest.TestCase):
    def test_progress_records(self):
        cm = CurriculumManager(window=10)
        cm.record_episode(spec_for("easy"), True, 10)
        cm.record_episode(spec_for("easy"), False, 20)
        p = cm.progress()
        self.assertEqual(p["easy"]["n"], 2)
        self.assertAlmostEqual(p["easy"]["success_rate"], 0.5)

    def test_propose_advances_when_mastered(self):
        cm = CurriculumManager(success_target=0.85, min_confidence=0.5)
        for _ in range(20):
            cm.record_episode(spec_for("easy"), True, 5)
        rng = random.Random(0)
        b = cm.propose(rng)
        self.assertIn(b, BINS)
        self.assertNotEqual(b, "easy")

    def test_propose_returns_bin(self):
        cm = CurriculumManager()
        self.assertIn(cm.propose(), BINS)

    def test_roundtrip(self):
        cm = CurriculumManager(seed=3)
        for i in range(10):
            cm.record_episode(spec_for("medium"), i % 2 == 0, i + 1)
        s = cm.save_state()
        cm2 = CurriculumManager(seed=0)
        cm2.load_state(s)
        self.assertEqual(cm.active, cm2.active)
        self.assertEqual(cm.counts, cm2.counts)
        self.assertEqual([len(v) for v in cm.windows.values()],
                         [len(v) for v in cm2.windows.values()])


if __name__ == "__main__":
    unittest.main()
