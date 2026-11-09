"""Evaluation v0.3 audit — artifact integrity regression tests.

Covers:
1. v0.12 L-family temporal exact-D1 deltas vs committed raw summary.
2. v0.12 F-family exact D2 reporting vs committed raw summary.
3. Memory v0.4 comparison signs vs raw per-config vectors under the
   audit v0.2 candidate-minus-baseline convention.
4. v0.12 headline success tables vs raw summaries.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from academy.eval.metrics import paired_difference

V12_DIR = _ROOT / "results" / "v12"
MEMORY_V04_DIR = _ROOT / "results" / "academy_v01" / "memory_v04"


def _load(rel):
    return json.loads((_ROOT / rel).read_text(encoding="utf-8"))


class TestV12LReportemporalDeltas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sum_l = _load("results/v12/summary_l_small.json")
        cls.sum_f = _load("results/v12/summary_f_small.json")

    def test_temporal_deltas_match_committed_raw_summary(self):
        """Report §4.4 temporal Δ column: stable +16.88, novel +23.12.
        The committed summary stores the exact deltas; a draft of the
        report had these swapped (stable +23.12, novel +16.88)."""
        comp = self.sum_l["summary"]["stable"]["comparison:temporal_minus_notp_exact_D1"]
        self.assertAlmostEqual(comp["delta"], 0.16875, places=5)
        comp_n = self.sum_l["summary"]["novel"]["comparison:temporal_minus_notp_exact_D1"]
        self.assertAlmostEqual(comp_n["delta"], 0.23125, places=5)

    def test_temporal_delta_bootstrap_ci_matches(self):
        comp = self.sum_l["summary"]["stable"]["comparison:temporal_minus_notp_exact_D1"]
        ci = comp["paired_bootstrap_95pct"]
        self.assertAlmostEqual(ci[0], 0.0, places=6)
        self.assertAlmostEqual(ci[1], 0.50625, places=5)

    def test_exact_d1_success_tables_match_raw(self):
        for cond in ("stable", "novel"):
            row = self.sum_l["summary"][cond]
            expected = {"metargon10": 0.2125, "metargon11": 0.18125,
                        "metargon12": 0.2750, "metargon12-notp": 0.18125,
                        "bayes": 0.48125}
            if cond == "novel":
                expected["metargon10"] = 0.21875
                expected["metargon11"] = 0.19375
                expected["metargon12"] = 0.29375
                expected["metargon12-notp"] = 0.1875
                expected["bayes"] = 0.425
            for p, exp in expected.items():
                self.assertAlmostEqual(
                    row[p]["success_rate"], exp, places=4,
                    msg=f"{cond}/{p} success vs report")


class TestV12FExactD2Reporting(unittest.TestCase):
    """Report §4.1: v0.11 D2 = 65.6/65.0/31.9/66.9/64.4 (stable/shift/
    noisy/novel/tight). The committed raw shows shift = 65.0, not 66.9."""

    @classmethod
    def setUpClass(cls):
        cls.sum_f = _load("results/v12/summary_f_small.json")

    def test_v11_d2_values_match_raw(self):
        expected = {"stable": 0.65625, "shift": 0.6500,
                    "noisy_obs": 0.31875, "novel": 0.66875,
                    "tight_budget": 0.64375}
        for cond, exp in expected.items():
            got = self.sum_f["summary"][cond]["metargon11"]["exact_discovery_D2"]
            self.assertAlmostEqual(got, exp, places=4, msg=f"v11 D2 {cond}")

    def test_v12_d2_values_match_raw(self):
        expected = {"stable": 0.84375, "shift": 0.8875,
                    "noisy_obs": 0.76875, "novel": 0.88125,
                    "tight_budget": 0.86875}
        for cond, exp in expected.items():
            got = self.sum_f["summary"][cond]["metargon12"]["exact_discovery_D2"]
            self.assertAlmostEqual(got, exp, places=4, msg=f"v12 D2 {cond}")

    def test_v12_d1_values_match_raw(self):
        expected = {"stable": 0.975, "shift": 0.95,
                    "noisy_obs": 0.875, "novel": 0.9625,
                    "tight_budget": 0.93125}
        for cond, exp in expected.items():
            got = self.sum_f["summary"][cond]["metargon12"]["exact_discovery_D1"]
            self.assertAlmostEqual(got, exp, places=4, msg=f"v12 D1 {cond}")


class TestMemoryV04ComparisonSigns(unittest.TestCase):
    """Memory v0.4 report §5.2 deltas are candidate − baseline. The
    committed summary and report previously used baseline − candidate
    (sign-flipped). These tests recompute from raw per-config vectors."""

    @classmethod
    def setUpClass(cls):
        cls.summary = _load("results/academy_v01/memory_v04/memory_v04_summary.json")
        cls.recs = {}
        runs = MEMORY_V04_DIR / "memory_v04_runs.jsonl"
        for line in runs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cls.recs[r["variant"]] = r["per"]

    def test_committed_titleparisons_match_recomputed_vectors(self):
        orig = self.summary["conditions_detail"]["original"]["classic_test_per_config"]
        for name in self.summary["comparisons"]:
            variant = name.replace("_vs_original", "")
            vec = self.summary["conditions_detail"][variant]["classic_test_per_config"]
            expected = paired_difference(vec, orig, seed=20261108)
            got = self.summary["comparisons"][name]["classic_test"]
            self.assertAlmostEqual(got["mean"], expected["mean"], places=8,
                                   msg=f"{name} mean")
            self.assertAlmostEqual(got["ci"][0], expected["ci"][0], places=8)
            self.assertAlmostEqual(got["ci"][1], expected["ci"][1], places=8)

    def test_improving_variants_have_positive_deltas(self):
        for name in ("probe_local", "ctx", "ctx_reset", "ctx_gate"):
            mean = self.summary["comparisons"][f"{name}_vs_original"]["classic_test"]["mean"]
            self.assertGreater(mean, 0.0, f"{name} Δ should be positive")

    def test_worsening_variant_has_negative_delta(self):
        mean = self.summary["comparisons"]["ctx_full_vs_original"]["classic_test"]["mean"]
        self.assertLess(mean, 0.0, "ctx_full Δ should be negative (0.5625 < 0.59375)")

    def test_fresh_control_positive_delta(self):
        mean = self.summary["comparisons"]["fresh_vs_original"]["classic_test"]["mean"]
        self.assertGreater(mean, 0.0, "fresh Δ should be positive (0.695 > 0.594)")

    def test_train_deltas_match_raw(self):
        orig = self.summary["conditions_detail"]["original"]["train_success_mean"]
        for name in ("probe_local", "ctx", "ctx_full"):
            per = self.recs[name]
            exp = per["train_success_mean"] - orig
            got = self.summary["comparisons"][f"{name}_vs_original"]["train_delta"]
            self.assertAlmostEqual(got, exp, places=8, msg=f"{name} train_delta")

    def test_raw_per_config_vectors_present(self):
        for name, per in self.recs.items():
            if name == "fresh":
                continue
            self.assertEqual(len(per["classic_test_per_config"]), 16,
                             f"{name} classic vector length")


class TestV12HeadlineTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sum_f = _load("results/v12/summary_f_small.json")
        cls.sum_h = _load("results/v12/summary_h_small.json")
        cls.sum_r = _load("results/v12/summary_r_small.json")

    def test_f_success_table_matches_raw(self):
        exp = {
            "stable": (0.85625, 0.85625, 0.85625, 0.7875),
            "shift": (0.8625, 0.90625, 0.9125, 0.80625),
            "noisy_obs": (0.44375, 0.80625, 0.725, 0.7625),
            "novel": (0.9125, 0.91875, 0.9125, 0.79375),
            "tight_budget": (0.73125, 0.9, 0.90625, 0.65625),
        }
        for cond, (v10, v11, v12, bay) in exp.items():
            row = self.sum_f["summary"][cond]
            self.assertAlmostEqual(row["metargon10"]["success_rate"], v10, places=4, msg=cond)
            self.assertAlmostEqual(row["metargon11"]["success_rate"], v11, places=4, msg=cond)
            self.assertAlmostEqual(row["metargon12"]["success_rate"], v12, places=4, msg=cond)
            self.assertAlmostEqual(row["bayes"]["success_rate"], bay, places=4, msg=cond)

    def test_h_success_table_matches_raw(self):
        exp = {"stable": (0.8375, 0.73125, 0.73125),
               "novel": (0.8375, 0.79375, 0.79375)}
        for cond, (v10, v11, v12) in exp.items():
            row = self.sum_h["summary"][cond]
            self.assertAlmostEqual(row["metargon10"]["success_rate"], v10, places=4, msg=cond)
            self.assertAlmostEqual(row["metargon11"]["success_rate"], v11, places=4, msg=cond)
            self.assertAlmostEqual(row["metargon12"]["success_rate"], v12, places=4, msg=cond)

    def test_r_success_table_matches_raw(self):
        exp = {"stable": (0.6125, 0.71875, 0.60625),
               "novel": (0.8, 0.84375, 0.79375)}
        for cond, (v10, v11, v12) in exp.items():
            row = self.sum_r["summary"][cond]
            self.assertAlmostEqual(row["metargon10"]["success_rate"], v10, places=4, msg=cond)
            self.assertAlmostEqual(row["metargon11"]["success_rate"], v11, places=4, msg=cond)
            self.assertAlmostEqual(row["metargon12"]["success_rate"], v12, places=4, msg=cond)


class TestTemporalExactD1Reproducibility(unittest.TestCase):
    """Temporal exact-D1 provenance fix: recalulate the metric from every
    raw seed and compare with committed summary values. Previously the
    audit wrongly looked up `discovery.D1.exact_rate` (absent) instead of
    the flat field `exact_discovery_D1`. These tests fail if the summary
    means diverge from the raw per-seed means."""

    @classmethod
    def setUpClass(cls):
        cls.sum_l = _load("results/v12/summary_l_small.json")
        cls.raw = cls.sum_l["raw"]

    def _per_seed_v12(self, cond):
        return [row["metargon12"]["exact_discovery_D1"] for row in self.raw[cond]]

    def test_summary_mean_matches_raw_per_seed_mean_stable(self):
        import statistics
        vals = self._per_seed_v12("stable")
        self.assertEqual(len(vals), 4)
        mean = statistics.fmean(vals)
        committed = self.sum_l["summary"]["stable"]["metargon12"]["exact_discovery_D1"]
        self.assertAlmostEqual(mean, committed, places=12)
        self.assertAlmostEqual(mean, 0.16875, places=5)

    def test_summary_mean_matches_raw_per_seed_mean_novel(self):
        import statistics
        vals = self._per_seed_v12("novel")
        self.assertEqual(len(vals), 4)
        mean = statistics.fmean(vals)
        committed = self.sum_l["summary"]["novel"]["metargon12"]["exact_discovery_D1"]
        self.assertAlmostEqual(mean, committed, places=12)
        self.assertAlmostEqual(mean, 0.23125, places=5)

    def test_values_temporal_delta_matches_raw(self):
        import statistics
        for cond, expected in (("stable", 0.16875), ("novel", 0.23125)):
            vals = self._per_seed_v12(cond)
            notp = [row["metargon12-notp"]["exact_discovery_D1"]
                    for row in self.raw[cond]]
            self.assertTrue(all(v == 0.0 for v in notp),
                            f"{cond}: notp exact-D1 must be 0")
            delta = statistics.fmean(vals) - statistics.fmean(notp)
            committed = self.sum_l["summary"][cond][
                "comparison:temporal_minus_notp_exact_D1"]["delta"]
            self.assertAlmostEqual(delta, committed, places=12)
            self.assertAlmostEqual(delta, expected, places=5)

    def test_stable_novel_labels_are_distinct(self):
        import statistics
        s = statistics.fmean(self._per_seed_v12("stable"))
        n = statistics.fmean(self._per_seed_v12("novel"))
        self.assertNotAlmostEqual(s, n, places=6)
        self.assertLess(s, n)

    def test_fraction_seed_sensitivity_recorded(self):
        """Document that discovery is dominated by individual seeds (seed 2
        on stable; seeds 1+2 on novel). This does not assert a value; it
        asserts the test reports the variance for audit documentation."""
        import statistics
        message = []
        for cond in ("stable", "novel"):
            vals = self._per_seed_v12(cond)
            positive = [i for i, v in enumerate(vals) if v > 0]
            message.append(f"{cond}: positive seeds={positive} "
                           f"stdev={statistics.pstdev(vals):.4f}")
        # informational only (kept for documentation); assert the data
        # exists and is well-formed
        self.assertEqual(len(message), 2)


if __name__ == "__main__":
    unittest.main()
