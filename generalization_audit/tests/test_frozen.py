"""Frozen-configuration tests: freeze determinism and solvability."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from generalization_audit.envs.pools import (
    build_pools, freeze_pools, load_frozen_pools, config_from_json,
    SOLVABLE,
)
from generalization_audit.metrics.shortcuts import assert_all_solvable


class TestFrozenConfigs(unittest.TestCase):
    def test_build_pools_deterministic(self):
        for family in ("r", "l"):
            pools1, n1 = build_pools(family)
            pools2, n2 = build_pools(family)
            self.assertEqual(n1, n2)
            for a, b in zip(pools1, pools2):
                self.assertEqual([c.key() for c in a],
                                 [c.key() for c in b])

    def test_freeze_roundtrip(self):
        for family in ("r", "l"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"frozen_{family}.json"
                doc = freeze_pools(family, path)
                loaded = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(loaded["family"], family)
                train, val, test = load_frozen_pools(loaded)
                self.assertEqual(len(train), doc["n_train"])
                self.assertEqual(len(val), doc["n_val"])
                self.assertEqual(len(test), doc["n_test"])
                # keys preserved
                orig = build_pools(family)[0]
                for got, exp in zip(train, orig[0]):
                    self.assertEqual(got.key(), exp.key())

    def test_all_frozen_configs_solvable(self):
        for family in ("r", "l"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"frozen_{family}.json"
                doc = freeze_pools(family, path)
                assert_all_solvable(doc, SOLVABLE[family])

    def test_disjoint_splits(self):
        for family in ("r", "l"):
            (train, val, test), _ = build_pools(family)
            keys = {c.key() for c in train}
            self.assertFalse(any(c.key() in keys for c in val))
            self.assertFalse(any(c.key() in keys for c in test))
            vkeys = {c.key() for c in val}
            self.assertFalse(any(c.key() in vkeys for c in test))


if __name__ == "__main__":
    unittest.main()
