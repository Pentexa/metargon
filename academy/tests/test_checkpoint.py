"""Tests for checkpointing: atomic snapshots, fingerprint verification,
and exact state restoration (deterministic continuation)."""

import os
import shutil
import tempfile
import unittest

from academy.ckpt.checkpoint import CheckpointManager
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.tasks import EpisodeSpec, difficulty_of, get_splits
from academy.learning.loop import run_one_episode


class TestCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="academy_ckpt_")
        cls.splits = get_splits()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _spec(self, i=0):
        c = self.splits["train"][i]
        return EpisodeSpec(config=c, max_steps=80, noise_q=0.0,
                           spec_seed=5, pool_tag="train",
                           difficulty=difficulty_of(c))

    def test_fingerprint_stable(self):
        cfg = {"a": 1, "b": [1, 2]}
        self.assertEqual(CheckpointManager("x", "r", cfg, 2).fingerprint,
                         CheckpointManager("x", "r", cfg, 2).fingerprint)

    def test_save_load_and_verify(self):
        cm = CheckpointManager(self.tmp, "run1", {"k": 1}, 3)
        p = cm.save(10, {"state": 1}, {}, {})
        self.assertTrue(os.path.exists(p))
        snap = cm.load_latest()
        self.assertIsNotNone(snap)
        self.assertTrue(cm.verify(snap))

    def test_restore_determinism(self):
        cm = CheckpointManager(self.tmp, "run1", {"k": 1}, 3)
        a1 = MetargonAcademyAgent(seed=0)
        run_one_episode(a1, self._spec(0))
        run_one_episode(a1, self._spec(1))
        cm.save(1, a1.save_state(), {}, {})
        snap = cm.load_latest()
        a2 = MetargonAcademyAgent(seed=0)
        a2.load_state(snap["agent"])
        # repeat the next episode: identical traces required
        r1 = run_one_episode(a1, self._spec(2), collect_trace=True)
        r2 = run_one_episode(a2, self._spec(2), collect_trace=True)
        self.assertEqual(r1["success"], r2["success"])
        self.assertEqual(r1["trace"], r2["trace"])
        self.assertEqual(r1["steps"], r2["steps"])

    def test_fingerprint_mismatch_rejected(self):
        cm = CheckpointManager(self.tmp, "run2", {"k": 1}, 2)
        cm.save(0, {}, {}, {})
        cm_bad = CheckpointManager(self.tmp, "run2", {"k": 2}, 2)
        snap = cm_bad.load_latest()
        self.assertIsNotNone(snap)
        self.assertFalse(cm_bad.verify(snap))


if __name__ == "__main__":
    unittest.main()
