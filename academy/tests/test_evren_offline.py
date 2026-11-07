"""Tests for the offline procedural generator (fallback when EVREN is
unavailable) and its approval pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from academy.evren.offline_generator import (
    OfflineTaskGenerator, run_offline_pilot,
)
from academy.evren.validator import STATUS_APPROVED, independent_solvable
from academy.tests.helpers import cached_splits, temp_registry


class TestOfflineGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = cached_splits()

    def test_deterministic_proposals(self):
        a = OfflineTaskGenerator(seed=42, splits=self.splits)
        b = OfflineTaskGenerator(seed=42, splits=self.splits)
        ra, rb = a.propose_raw(), b.propose_raw()
        self.assertEqual(ra["topology"], rb["topology"])
        self.assertEqual(ra["d1_rule"], rb["d1_rule"])
        self.assertEqual(ra["d2_rule"], rb["d2_rule"])
        self.assertEqual(ra["d1_delay"], rb["d1_delay"])
        self.assertEqual(ra["d2_delay"], rb["d2_delay"])

    def test_proposals_are_canonical_shape(self):
        gen = OfflineTaskGenerator(seed=1, splits=self.splits)
        for _ in range(20):
            raw = gen.propose_raw()
            self.assertIn("task_id", raw)
            self.assertIn("topology", raw)
            self.assertIn("d1_rule", raw)
            self.assertIn("d2_rule", raw)
            self.assertIn("d1_delay", raw)
            self.assertIn("d2_delay", raw)
            self.assertIn("difficulty_intent", raw)
            self.assertIsInstance(raw["d1_delay"], bool)

    def test_generate_approves_novel_solvable_tasks(self):
        gen = OfflineTaskGenerator(seed=3, splits=self.splits)
        results = gen.generate(10, max_attempts=400)
        approved = [r for r in results if r.approved]
        self.assertEqual(len(approved), 10)
        eval_keys = {c.key for c in (self.splits["val"] + self.splits["test"]
                                     + self.splits["ood"])}
        keys = set()
        for r in approved:
            self.assertTrue(r.solvable)
            self.assertTrue(independent_solvable(r.config))
            self.assertNotIn(r.config.key, eval_keys)
            self.assertNotIn(r.config.key, keys)
            keys.add(r.config.key)

    def test_generate_is_self_consistent_with_seed(self):
        r1 = OfflineTaskGenerator(seed=9, splits=self.splits).generate(5)
        r2 = OfflineTaskGenerator(seed=9, splits=self.splits).generate(5)
        k1 = [r.config.key for r in r1 if r.approved]
        k2 = [r.config.key for r in r2 if r.approved]
        self.assertEqual(k1, k2)

    def test_run_offline_pilot_persists_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = temp_registry(tmp)
            out = run_offline_pilot(seed=11, n=5, splits=self.splits,
                                    registry=registry)
            self.assertEqual(len(out["approved"]), 5)
            for p in ("approved_tasks.json", "rejected_candidates.json",
                      "usage_log.json", "manifest.json"):
                self.assertTrue(os.path.exists(os.path.join(tmp, p)))
            approved = registry.approved_tasks()
            self.assertEqual(len(approved), 5)
            for t in approved:
                self.assertEqual(t.model, "offline")
                self.assertIn("proposal", t.as_dict())
            self.assertEqual(registry.task_ids(), {t.task_id for t in approved})
            summ = registry.token_summary()
            self.assertEqual(summ["totals"]["requests"], 0)  # no API calls


if __name__ == "__main__":
    unittest.main()
