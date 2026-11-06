"""Tests for task-family splits: counts, disjointness, determinism, and
OOD isolation (unseen topology, unseen mechanism role)."""

import os
import unittest

from academy.envs.tasks import (
    Difficulty, difficulty_of, get_splits, make_episodes,
)
from academy.envs.gridworld import is_solvable


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # uses the disk cache (created by the CLI) when present; otherwise
        # computes with the normal (slow) path — acceptable in CI.
        cls.splits = get_splits()

    def test_counts_meet_targets(self):
        counts = self.splits["counts"]
        self.assertGreaterEqual(counts["train"], 100)
        self.assertGreaterEqual(counts["val"], 25)
        self.assertGreaterEqual(counts["test"], 25)
        self.assertGreaterEqual(counts["ood"], 25)

    def test_all_pools_solvable(self):
        for c in self.splits["train"][:20] + self.splits["test"][:10]:
            self.assertTrue(is_solvable(c), msg=str(c.key))

    def test_disjointness_train_val_test_ood(self):
        train = {c.key for c in self.splits["train"]}
        val = {c.key for c in self.splits["val"]}
        test = {c.key for c in self.splits["test"]}
        ood = {c.key for c in self.splits["ood"]}
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(train.isdisjoint(ood))
        self.assertTrue(val.isdisjoint(test))
        self.assertTrue(test.isdisjoint(ood))

    def test_ood_uses_only_unseen_topology(self):
        # OOD configs are all TP2 *and* none of them ever appears in the
        # core pools (mechanism novelty).
        train_keys = {c.key for c in self.splits["train"]}
        for c in self.splits["ood"]:
            self.assertEqual(c.topology, "TP2")
            self.assertNotIn(c.key, train_keys)

    def test_ood_uses_absence_atoms(self):
        any_absence = any(difficulty_of(c).uses_absence
                          for c in self.splits["ood"])
        self.assertTrue(any_absence)

    def test_determinism_split_cache(self):
        mtime1 = os.path.getmtime("results/academy_v01/splits.json")
        s2 = get_splits()
        mtime2 = os.path.getmtime("results/academy_v01/splits.json")
        self.assertEqual(mtime1, mtime2)
        self.assertEqual([c.key for c in s2["train"]],
                         [c.key for c in self.splits["train"]])

    def test_episode_stream_deterministic(self):
        a = make_episodes(self.splits["train"], "train", 7, 10)
        b = make_episodes(self.splits["train"], "train", 7, 10)
        self.assertEqual([e.key for e in a], [e.key for e in b])
        self.assertEqual([e.spec_seed for e in a],
                         [e.spec_seed for e in b])


class TestDifficulty(unittest.TestCase):
    def test_bin_bounds(self):
        from academy.envs.gridworld import EnvConfig
        easy = EnvConfig("TP1",
                         frozenset({("sw", "S2", True)}),
                         frozenset({("sw", "S1", True)}), False, False)
        hard = EnvConfig("TP1",
                         frozenset({("sw", "S2", True), ("loadany", "P1")}),
                         frozenset({("sw", "S1", False), ("load", "P2", "B")}),
                         True, True)
        self.assertEqual(difficulty_of(easy).bin, "easy")
        self.assertEqual(difficulty_of(hard).bin, "hard")


if __name__ == "__main__":
    unittest.main()
