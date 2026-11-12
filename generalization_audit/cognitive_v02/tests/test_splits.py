"""Frozen-split integrity tests (v0.2)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generalization_audit.cognitive_v02.splits import (
    canonical_config_dict,
    config_from_canonical,
    freeze_splits,
    load_frozen_splits,
    pools_dict,
    verify_disjoint,
    verify_frozen_splits,
    verify_solvable_sample,
)


class TestFrozenSplits(unittest.TestCase):

    def test_frozen_matches_canonical_splits(self):
        self.assertTrue(verify_frozen_splits(raise_on_mismatch=True))

    def test_pools_stable_and_disjoint(self):
        pools, counts = load_frozen_splits()
        self.assertEqual(counts["train"], 147)
        self.assertEqual(counts["test"], 49)
        self.assertEqual(counts["ood"], 25)
        verify_solvable_sample(pools["train"])
        verify_disjoint(pools["train"], pools["val"], pools["test"],
                        pools["ood"])

    def test_canonical_roundtrip_stable(self):
        pools, _ = load_frozen_splits()
        cfg = config_from_canonical(pools["train"][0])
        d1 = canonical_config_dict(cfg)
        self.assertEqual(d1, canonical_config_dict(cfg))
        self.assertEqual(canonical_config_dict(config_from_canonical(d1)), d1)

    def test_freeze_writes_tmp(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "frozen.json"
            freeze_splits(p)
            self.assertTrue(p.exists())
            pools, counts = load_frozen_splits(p)
            self.assertEqual(len(pools["train"]), counts["train"])


if __name__ == "__main__":
    unittest.main()
