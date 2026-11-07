"""Tests for diversity / generalization metrics."""

from __future__ import annotations

import unittest

from academy.envs.gridworld import EnvConfig
from academy.evren.diversity import (
    _normalized_entropy, atoms_of, coverage_dims, diversity_metrics,
    family_distribution, jaccard, signature,
)


C1 = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
               frozenset({("load", "P2", "B")}), False, False)
C2 = EnvConfig("TP1", frozenset({("empty", "P1")}),
               frozenset({("sw", "S1", False)}), True, True)
C3 = EnvConfig("TP2", frozenset({("sw", "S2", True)}),
               frozenset({("load", "P2", "B")}), False, False)


class TestAtoms(unittest.TestCase):
    def test_atoms_of_union(self):
        a = atoms_of(C1)
        self.assertEqual(a, frozenset({("sw", "S2", True), ("load", "P2", "B")}))

    def test_jaccard_bounds(self):
        self.assertEqual(jaccard(frozenset(), frozenset()), 0.0)
        self.assertEqual(jaccard(frozenset({1}), frozenset({1})), 0.0)
        self.assertEqual(jaccard(frozenset({1}), frozenset({2})), 1.0)


class TestSignature(unittest.TestCase):
    def test_canonical_and_jsonable(self):
        import json
        s = signature(C1)
        json.dumps(s)                      # must be serializable
        self.assertEqual(s["topology"], "TP1")
        self.assertFalse(s["d1_delay"])


class TestMetrics(unittest.TestCase):
    def test_normalized_entropy_bounds(self):
        self.assertEqual(_normalized_entropy([]), 0.0)
        self.assertEqual(_normalized_entropy([5]), 0.0)
        self.assertAlmostEqual(_normalized_entropy([1, 1]), 1.0)

    def test_family_distribution_counts(self):
        fams = family_distribution([C1, C1, C3])
        # C1 and C3 differ only by topology => distinct families
        self.assertEqual(sum(fams.values()), 3)
        self.assertEqual(len(fams), 2)

    def test_unique_ratio_and_jaccard_stats(self):
        dm = diversity_metrics([C1, C2, C3])
        self.assertEqual(dm["n"], 3)
        self.assertEqual(dm["unique_configs"], 3)
        self.assertEqual(dm["unique_ratio"], 1.0)
        self.assertGreaterEqual(dm["pairwise_jaccard"]["mean"], 0.0)
        self.assertLessEqual(dm["pairwise_jaccard"]["max"], 1.0)

    def test_novelty_vs_reference(self):
        ref = [C2]
        same = diversity_metrics([C2], reference=ref)
        self.assertAlmostEqual(same["novelty_vs_reference"]["mean"], 0.0)
        disjoint = diversity_metrics(
            [EnvConfig("TP1", frozenset({("load", "P2", "B")}),
                       frozenset({("loadany", "P2")}), False, False)],
            reference=ref)
        self.assertEqual(disjoint["novelty_vs_reference"]["min"], 1.0)

    def test_coverage_dims_content(self):
        cov = coverage_dims([C1, C2])
        self.assertEqual(cov["topology"], {"TP1": 2})
        self.assertEqual(cov["delay_flags"], {"(False, False)": 1,
                                              "(True, True)": 1})
        self.assertIn("d1:conjunctive", cov["clause_shape"])

    def test_duplicate_ratio(self):
        dm = diversity_metrics([C1, C1, C1])
        self.assertEqual(dm["unique_ratio"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
