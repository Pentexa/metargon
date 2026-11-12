"""Frozen-split integrity tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generalization_audit.cognitive_v01.splits import (
    canonical_config_dict,
    freeze_splits,
    load_frozen_splits,
    rule_from_canonical,
    verify_disjoint,
    verify_frozen_splits,
    verify_solvable,
)


class TestFrozenSplits(unittest.TestCase):

    def test_frozen_matches_canonical_splits(self):
        verify_frozen_splits(raise_on_mismatch=True, light=True)

    def test_frozen_pools_are_solvable_and_disjoint(self):
        pools, _ = load_frozen_splits()
        # full solvability scan is expensive; verify a deterministic sample
        sample = pools["train"][:10] + pools["val"][:5] + pools["test"][:5]
        verify_solvable(sample)
        verify_disjoint(pools["train"], pools["val"], pools["test"],
                        pools["ood"])

    def test_counts_are_stable(self):
        pools, counts = load_frozen_splits()
        self.assertEqual(counts["train"], 147)
        self.assertEqual(counts["val"], 49)
        self.assertEqual(counts["test"], 49)
        self.assertEqual(counts["ood"], 25)
        self.assertEqual(len(pools["train"]), counts["train"])

    def test_canonical_serialization_roundtrip_and_stability(self):
        pools, _ = load_frozen_splits()
        cfg = rule_from_canonical(pools["train"][0])
        d1 = canonical_config_dict(cfg)
        d2 = canonical_config_dict(cfg)
        self.assertEqual(d1, d2)              # stable / hash-order free
        cfg2 = rule_from_canonical(d1)
        self.assertEqual(canonical_config_dict(cfg2), d1)

    def test_freeze_writes_event_file_in_tmp(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "frozen.json"
            freeze_splits(p)
            self.assertTrue(p.exists())
            pools, counts = load_frozen_splits(p)
            self.assertEqual(len(pools["train"]), counts["train"])


if __name__ == "__main__":
    unittest.main()
