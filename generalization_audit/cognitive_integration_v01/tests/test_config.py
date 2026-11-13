"""Preregistration invariants and protocol configuration tests."""
from __future__ import annotations

import unittest
from pathlib import Path

from generalization_audit.cognitive_integration_v01 import config as C


class TestPreregistration(unittest.TestCase):

    def test_confirmatory_seeds_fixed_and_untouched(self):
        self.assertEqual(C.CONFIRMATORY_SEEDS, (101, 103, 107, 109))
        # smoke must never use confirmatory seeds
        self.assertNotIn(C.SMOKE_SEED, C.CONFIRMATORY_SEEDS)

    def test_extra_controls_cover_integration_attribution(self):
        self.assertIn("muted_epoch_reset", C.EXTRA_CONTROLS)
        self.assertIn("muted_explore_epoch_reset", C.EXTRA_CONTROLS)
        self.assertIn("shuffled_explore_epoch_reset", C.EXTRA_CONTROLS)
        self.assertEqual(
            C.EXTRA_CONTROLS["muted_explore_epoch_reset"]["base"],
            "explore_epoch_reset")
        self.assertEqual(
            C.EXTRA_CONTROLS["muted_explore_epoch_reset"]["transform"],
            "wipe")
        self.assertEqual(
            C.EXTRA_CONTROLS["shuffled_explore_epoch_reset"]["transform"],
            "shuffle")

    def test_primary_pool_in_eval_pools(self):
        names = [p[0] for p in C.EVAL_POOLS]
        self.assertIn(C.PRIMARY_POOL, names)
        for p in C.SECONDARY_POOLS:
            self.assertIn(p, names)

    def test_preregistration_document_exists(self):
        doc = (Path(__file__).resolve().parent.parent / "docs"
               / "PREREGISTRATION.md")
        self.assertTrue(doc.exists())
        text = doc.read_text(encoding="utf-8")
        self.assertIn("101, 103, 107, 109", text)
        self.assertIn("preregistered", text.lower())
        self.assertIn("H1", text)
        self.assertIn("H4", text)

    def test_arm_order_includes_all_controls(self):
        for arm in C.EXTRA_CONTROLS:
            self.assertIn(arm, C.ARM_ORDER)
        for arm in ("fresh", "trained", "muted", "shuffled",
                    "epoch_reset", "explore", "explore_epoch_reset"):
            self.assertIn(arm, C.ARM_ORDER)


if __name__ == "__main__":
    unittest.main()
