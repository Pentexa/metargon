"""Interaction and attribution analysis unit tests."""
from __future__ import annotations

import unittest

from generalization_audit.cognitive_integration_v01.interaction import (
    build_analysis,
    delta_summary,
    interaction_summary,
    interaction_term,
    per_seed_value,
)


def _seed_results():
    """Synthetic per-seed results with 4 seeds."""
    out = {}
    for seed in range(4):
        out[seed] = {}
        for arm in ("trained", "epoch_reset", "explore",
                    "explore_epoch_reset", "muted_epoch_reset",
                    "muted_explore_epoch_reset",
                    "shuffled_explore_epoch_reset"):
            out[seed][arm] = {
                "val_clean": {"metrics": {"success_rate": (seed % 2, 6),
                                          "zero_shot_coverage": (1, 6)}},
            }
        out[seed]["trained"]["val_clean"]["metrics"]["success_rate"] = (1, 6)
        out[seed]["epoch_reset"]["val_clean"]["metrics"]["success_rate"] = (3, 6)
        out[seed]["explore"]["val_clean"]["metrics"]["success_rate"] = (2, 6)
        out[seed]["explore_epoch_reset"]["val_clean"]["metrics"]["success_rate"] = (5, 6)
        out[seed]["muted_epoch_reset"]["val_clean"]["metrics"]["success_rate"] = (3, 6)
        out[seed]["muted_explore_epoch_reset"]["val_clean"]["metrics"]["success_rate"] = (4, 6)
        out[seed]["shuffled_explore_epoch_reset"]["val_clean"]["metrics"]["success_rate"] = (4, 6)
    return out


class TestInteraction(unittest.TestCase):

    def test_per_seed_value_rates(self):
        sr = _seed_results()
        v = per_seed_value(sr, "trained", "val_clean", "success_rate")
        self.assertEqual(v, [1 / 6] * 4)

    def test_interaction_term_math(self):
        sr = _seed_results()
        terms = interaction_term(sr, "val_clean")
        # trained=.1667, epoch=.5, explore=.3333, integ=.8333
        # (integ - epoch) - (explore - trained) = .3333 - .1667 = .1667
        for t in terms:
            self.assertAlmostEqual(t, 1 / 6, places=6)

    def test_delta_summary(self):
        sr = _seed_results()
        d = delta_summary(sr, "explore_epoch_reset", "epoch_reset",
                          "val_clean")
        self.assertEqual(d["n"], 4)
        self.assertAlmostEqual(d["mean"], 2 / 6, places=6)
        self.assertIsNotNone(d["ci"])

    def test_interaction_summary(self):
        sr = _seed_results()
        s = interaction_summary(sr, "val_clean")
        self.assertAlmostEqual(s["mean"], 1 / 6, places=6)
        self.assertEqual(s["n"], 4)
        self.assertIsNotNone(s["ci"])

    def test_build_analysis_shape(self):
        sr = _seed_results()
        a = build_analysis(sr, pools=("val_clean",))
        self.assertIn("deltas", a)
        self.assertIn("interactions", a)
        key = "explore_epoch_reset-epoch_reset.val_clean.success_rate"
        self.assertIn(key, a["deltas"])
        self.assertIn("val_clean.success_rate", a["interactions"])
        self.assertIn("per_seed", a)


if __name__ == "__main__":
    unittest.main()
