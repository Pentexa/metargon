"""Tests for the bounded experience memory."""

import unittest

from academy.core.interfaces import Experience
from academy.envs.gridworld import Observation
from academy.learning.experience_memory import BoundedExperienceMemory


def mk_exp(i):
    return Experience(Observation(0, 0, 1, 1, 2, 2, False, False, "none",
                                  "none", False, False, False),
                      "N",
                      Observation(0, 1, 1, 1, 2, 2, False, False, "none",
                                  "none", False, False, False),
                      0.0, False, (), i, 0, "train")


class TestExperienceMemory(unittest.TestCase):
    def test_bounded_fifo(self):
        mem = BoundedExperienceMemory(capacity=10, seed=0)
        for i in range(25):
            mem.push(mk_exp(i))
        self.assertEqual(len(mem), 10)

    def test_sample_batch(self):
        mem = BoundedExperienceMemory(capacity=100, seed=0)
        for i in range(50):
            mem.push(mk_exp(i))
        batch = mem.sample_batch(batch=8)
        self.assertEqual(len(batch), 8)
        self.assertEqual(len({e.step_index for e in batch}), 8)

    def test_roundtrip(self):
        mem = BoundedExperienceMemory(capacity=20, seed=1)
        for i in range(15):
            mem.push(mk_exp(i))
        s = mem.save_state()
        mem2 = BoundedExperienceMemory(capacity=0, seed=0)
        mem2.load_state(s)
        self.assertEqual(len(mem2), 15)
        self.assertEqual(mem2.buffer[3].step_index, 3)


if __name__ == "__main__":
    unittest.main()
