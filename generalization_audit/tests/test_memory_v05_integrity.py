"""Evaluation v0.4 audit — Memory v0.5 comparison-sign regression tests.

Confirms (independently of the report) that:

1. The committed `memory_v05_summary.json` comparison deltas were computed
   with the REVERSED convention (baseline - candidate): the merge tool
   calls `paired_difference(orig, candidate)`. Under the project-wide
   document - baseline convention (audit v0.2/v0.3) every delta must be
   sign-flipped.
2. The same summary's `train_delta` column already uses candidate - baseline
   (it is computed separately and is correct).
3. The raw per-config vectors and per-seed detail in the summary are
   bit-identical to the committed JSONL records (historical observations
   are preserved).
4. The corrected values reproduce exactly from the raw vectors with the
   canonical bootstrap (seed 20261108), including CI sign-flip symmetry.

These tests DO NOT modify any Academy file. They provide the exact
reproducible correction specification for the EVREN agent (see
docs/AUDIT_V04.md).
"""
from __future__ import annotations

import json
import statistics
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from academy.eval.metrics import paired_difference  # noqa: E402

_MEM_DIR = _ROOT / "results" / "academy_v01" / "memory_v05"
_SUMMARY = _MEM_DIR / "memory_v05_summary.json"
_RUNS = _MEM_DIR / "memory_v05_runs.jsonl"
_BOOT_SEED = 20261108

# Expected corrected deltas (candidate - baseline) under the canonical
# paired bootstrap (seed 20261108). These equal the negation of the
# committed (reversed-convention) values, because reversing the two
# vectors negates every bootstrap difference, so mean -> -mean and
# CI (lo, hi) -> (-hi, -lo) exactly, with the same random seed.
_EXPECTED_CORRECTED = {
    "fresh_vs_original": {
        "classic_frozen": (0.0303125, (0.0, 0.09375)),
        "classic_adapt": (0.0996640625, (0.0, 0.2109375)),
    },
    "ctx_full_vs_original": {
        "classic_frozen": (-0.0303125, (-0.09375, 0.0)),
        "classic_adapt": (-0.0303125, (-0.09375, 0.0)),
    },
    "bank_ctx_full_vs_original": {
        "classic_frozen": (-0.0071796875, (-0.09375, 0.0703125)),
        "classic_adapt": (0.037203125, (-0.0390625, 0.1328125)),
    },
}


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestMemoryV05SummaryPreserved(unittest.TestCase):
    """Raw JSONL records must equal the committed conditions_detail exactly."""

    @classmethod
    def setUpClass(cls):
        cls.summary = _load(_SUMMARY)
        cls.recs = {}
        for line in _RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cls.recs[r["variant"]] = r["per"]

    def test_jsonl_variants_match_summary_variants(self):
        self.assertEqual(sorted(self.recs), sorted(self.summary["variants"]))

    def test_conditions_detail_matches_jsonl_records(self):
        for v in self.summary["variants"]:
            self.assertEqual(
                self.summary["conditions_detail"][v], self.recs[v],
                f"conditions_detail[{v}] drifted from committed JSONL")

    def test_conditions_detail_matches_per_seed_detail(self):
        for v in self.summary["variants"]:
            per = self.summary["conditions_detail"][v]
            seeds = per["per_seed"]
            for mode in ("classic_frozen", "classic_adapt"):
                arr = per[f"{mode}_per_config"]
                keys = list(seeds[0][mode]["per_config"])
                recon = [statistics.fmean(s[mode]["per_config"][k]
                                          for s in seeds)
                         for k in keys]
                # same config order (dict order is preserved from JSON)
                self.assertEqual(len(recon), len(arr), f"{v}/{mode} len")
                for a, b in zip(arr, recon):
                    self.assertAlmostEqual(a, b, places=12,
                                           msg=f"{v}/{mode} per-config")


class TestMemoryV05CommittedComparisonsAreReversed(unittest.TestCase):
    """Committed deltas must equal baseline - candidate (reversed).

    This is the regression test that would have caught the v0.5 sign bug
    before merge: every committed comparison must equal
    `paired_difference(original, candidate)` exactly.
    """

    @classmethod
    def setUpClass(cls):
        cls.summary = _load(_SUMMARY)

    def test_every_committed_comparison_equals_reversed_recomputation(self):
        orig = self.summary["conditions_detail"]["original"]
        for name, cmp_ in self.summary["comparisons"].items():
            variant = name.replace("_vs_original", "")
            per = self.summary["conditions_detail"][variant]
            for mode in ("classic_frozen", "classic_adapt"):
                a = orig[f"{mode}_per_config"]
                b = per[f"{mode}_per_config"]
                expected = paired_difference(a, b, seed=_BOOT_SEED)
                got = cmp_[mode]
                self.assertAlmostEqual(got["mean"], expected["mean"],
                                       places=8, msg=f"{name}/{mode} mean")
                self.assertAlmostEqual(got["ci"][0], expected["ci"][0],
                                       places=8, msg=f"{name}/{mode} ci0")
                self.assertAlmostEqual(got["ci"][1], expected["ci"][1],
                                       places=8, msg=f"{name}/{mode} ci1")

    def test_train_deltas_already_candidate_minus_baseline(self):
        orig = self.summary["conditions_detail"]["original"]
        for name, cmp_ in self.summary["comparisons"].items():
            variant = name.replace("_vs_original", "")
            per = self.summary["conditions_detail"][variant]
            expected = per["train_success_mean"] - orig["train_success_mean"]
            self.assertAlmostEqual(cmp_["train_delta"], expected, places=8,
                                   msg=f"{name} train_delta")

    def test_committed_fresh_delta_is_negative_under_old_convention(self):
        # Documents the bug: the committed summary reads negative for fresh
        # even though fresh > original in BOTH regimes.
        fr = self.summary["comparisons"]["fresh_vs_original"]
        self.assertLess(fr["classic_frozen"]["mean"], 0.0)
        self.assertLess(fr["classic_adapt"]["mean"], 0.0)
        self.assertGreater(
            self.summary["conditions_detail"]["fresh"]["classic_frozen"]["config_mean_success"],
            self.summary["conditions_detail"]["original"]["classic_frozen"]["config_mean_success"])


class TestMemoryV05CorrectedDeltas(unittest.TestCase):
    """Corrected values under candidate - baseline must match raw vectors."""

    @classmethod
    def setUpClass(cls):
        cls.summary = _load(_SUMMARY)

    def test_corrected_fresh_deltas_positive(self):
        fr = self.summary["comparisons"]["fresh_vs_original"]
        for mode in ("classic_frozen", "classic_adapt"):
            mean = -fr[mode]["mean"]
            ci = (-fr[mode]["ci"][1], -fr[mode]["ci"][0])
            self.assertGreater(mean, 0.0, mode)
            exp_mean, exp_ci = _EXPECTED_CORRECTED["fresh_vs_original"][mode]
            self.assertAlmostEqual(mean, exp_mean, places=6, msg=mode)
            self.assertAlmostEqual(ci[0], exp_ci[0], places=6, msg=mode)
            self.assertAlmostEqual(ci[1], exp_ci[1], places=6, msg=mode)

    def test_corrected_ctx_full_deltas_negative(self):
        for mode in ("classic_frozen", "classic_adapt"):
            cmp_ = self.summary["comparisons"]["ctx_full_vs_original"][mode]
            mean = -cmp_["mean"]
            ci = (-cmp_["ci"][1], -cmp_["ci"][0])
            self.assertLess(mean, 0.0, mode)
            exp_mean, exp_ci = _EXPECTED_CORRECTED["ctx_full_vs_original"][mode]
            self.assertAlmostEqual(mean, exp_mean, places=6, msg=mode)
            self.assertAlmostEqual(ci[0], exp_ci[0], places=6, msg=mode)
            self.assertAlmostEqual(ci[1], exp_ci[1], places=6, msg=mode)

    def test_corrected_bank_ctx_full_adapt_positive_ci_crosses_zero(self):
        cmp_ = self.summary["comparisons"]["bank_ctx_full_vs_original"]["classic_adapt"]
        mean = -cmp_["mean"]
        ci = (-cmp_["ci"][1], -cmp_["ci"][0])
        self.assertGreater(mean, 0.0)
        exp_mean, exp_ci = _EXPECTED_CORRECTED["bank_ctx_full_vs_original"]["classic_adapt"]
        self.assertAlmostEqual(mean, exp_mean, places=6)
        self.assertAlmostEqual(ci[0], exp_ci[0], places=6)
        self.assertAlmostEqual(ci[1], exp_ci[1], places=6)

    def test_report_table_deltas_match_committed_summary(self):
        # §4.2 table (as committed) equals the committed comparison block.
        fr = self.summary["comparisons"]["fresh_vs_original"]
        self.assertAlmostEqual(fr["classic_frozen"]["mean"], -0.0303125, places=6)
        self.assertAlmostEqual(fr["classic_adapt"]["mean"], -0.0996640625, places=6)
        self.assertAlmostEqual(fr["classic_frozen"]["ci"][0], -0.09375, places=6)
        self.assertAlmostEqual(fr["classic_frozen"]["ci"][1], 0.0, places=6)
        cf = self.summary["comparisons"]["ctx_full_vs_original"]
        self.assertAlmostEqual(cf["classic_frozen"]["mean"], 0.0303125, places=6)
        self.assertAlmostEqual(cf["classic_adapt"]["mean"], 0.0303125, places=6)
        bcf = self.summary["comparisons"]["bank_ctx_full_vs_original"]
        self.assertAlmostEqual(bcf["classic_adapt"]["mean"], -0.037203125, places=6)

    def test_report_means_match_conditions_detail(self):
        # §4.1 held-out table equals conditions_detail config_mean_success.
        exp = {
            "fresh": (None, 0.3125, 0.6953125),
            "original": (0.340625, 0.28125, 0.59375),
            "memory_wiped": (0.340625, 0.28125, 0.59375),
            "ctx": (0.36875, 0.28125, 0.625),
            "ctx_full": (0.496875, 0.25, 0.5625),
            "bank": (0.340625, 0.28125, 0.59375),
            "bank_ctx": (0.36875, 0.28125, 0.625),
            "bank_ctx_full": (0.23125, 0.2734375, 0.6328125),
        }
        for v, (tr, fr, ad) in exp.items():
            per = self.summary["conditions_detail"][v]
            if tr is not None:
                self.assertAlmostEqual(per["train_success_mean"], tr, places=6, msg=f"{v} train")
            self.assertAlmostEqual(per["classic_frozen"]["config_mean_success"], fr, places=6, msg=f"{v} frozen")
            self.assertAlmostEqual(per["classic_adapt"]["config_mean_success"], ad, places=6, msg=f"{v} adapt")


class TestPairedDifferenceConvention(unittest.TestCase):
    """The canonical bootstrap must have exact CI sign-flip symmetry."""

    def test_reversing_arguments_negates_mean_exactly(self):
        a = [0.0, 1.0, 0.5, 0.25, 1.0, 0.75]
        b = [0.5, 0.0, 0.25, 0.5, 0.0, 0.25]
        fwd = paired_difference(a, b, seed=20261108)
        rev = paired_difference(b, a, seed=20261108)
        self.assertAlmostEqual(fwd["mean"], -rev["mean"], places=12)
        self.assertAlmostEqual(fwd["ci"][0], -rev["ci"][1], places=12)
        self.assertAlmostEqual(fwd["ci"][1], -rev["ci"][0], places=12)

    def test_positive_delta_means_candidate_improves(self):
        a = [0.8] * 8
        b = [0.5] * 8
        d = paired_difference(a, b, seed=7)
        self.assertGreater(d["mean"], 0.0)
        self.assertGreater(d["ci"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
