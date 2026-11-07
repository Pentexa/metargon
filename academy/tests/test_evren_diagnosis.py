"""Tests for the v0.1 null-result diagnosis (offline, tiny budgets)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from academy.envs.tasks import get_splits
from academy.evren.diagnosis import run_diagnosis


class TestDiagnosis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="diagnosis-smoke-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_runs_with_fallback_training(self):
        splits = get_splits()
        s = run_diagnosis(results_dir=self.tmp, splits=splits,
                          n_configs=3, episodes=3, seed=11,
                          train_fallback=2, max_steps=60)
        self.os_checks(s)

    def os_checks(self, s):
        self.assertTrue(s["offline"])
        self.assertEqual(s["api_calls"], 0)
        self.assertEqual(s["n_configs"], 3)
        agg = s["aggregate"]
        self.assertIn("fresh", agg)
        fresh = agg["fresh"]
        for key in ("success_rate", "delta_vs_fresh"):
            # fresh has no trained-state to differ from
            self.assertNotIn(key, fresh) if key == "delta_vs_fresh" else \
                self.assertIn(key, fresh)
        # fallback trained state measured
        self.assertIn("fallback_trained", s["trained_agents"])
        tr = agg["fallback_trained"]
        for key in ("success_rate", "mean_contradicted", "mean_verified",
                    "mean_tried_entries", "delta_vs_fresh"):
            self.assertIn(key, tr)
        self.assertAlmostEqual(tr["delta_vs_fresh"],
                               tr["success_rate"] - fresh["success_rate"],
                               places=9)
        # summary persisted
        path = os.path.join(self.tmp, "diagnosis_summary.json")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
        self.assertEqual(reloaded["run_id"], "diagnosis")

    def test_no_states_is_reported_cleanly(self):
        # without fallback training and without saved policies the
        # diagnosis still measures the fresh agent only
        splits = get_splits()
        s = run_diagnosis(results_dir=self.tmp, splits=splits,
                          n_configs=2, episodes=2, seed=13,
                          train_fallback=None, max_steps=60)
        self.assertEqual(s["states_measured"], [])
        self.assertEqual(len(s["fresh_agent"]), 2)
        self.assertTrue(s["offline"])


if __name__ == "__main__":
    unittest.main()
