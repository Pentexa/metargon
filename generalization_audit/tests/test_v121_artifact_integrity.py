"""Evaluation v0.4 audit — v0.12.1 artifact-integrity tests.

Checks the committed v0.12.1 artifacts (`results/v121/*`) for:

1. Raw-to-summary reproducibility (every committed summary mean must
   equal the mean of the committed raw per-seed rows).
2. Per-seed variability reporting (seed-sensitivity visibility).
3. Matched seed / episode alignment across artifacts.
4. The legacy-ablation semantics documented issues
   - `metargon12-legacy` lacks temporal memory (so it cannot measure
     "recent-credit vs temporal" interactions on family L);
   - the diagnosis-phase `-norc` rows faithfully reproduce the v0.12
     configuration, and the final `metargon12-legacy` rows reproduce the
     diagnosis `metargon12` rows exactly on F/H;
   - the final `metargon12` (recent-credit off) D2 values disagree with
     the diagnosis `-norc` D2 values on F/H (documented discrepancy).
5. The L temporal preservation chain: `eval2_l.json` metargon12 rows
   match the v0.12 study `summary_l_small.json` exactly; final_l keeps
   exact-D1 per-seed discovery; final_l legacy has zero temporal probes.
6. Source-level conventions (harness ablation semantics, defaults,
   public-observation boundary, equal budgets).

These tests only read committed artifacts and committed source; they do
not modify research or academy files.
"""
from __future__ import annotations

import ast
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

_V121 = _ROOT / "results" / "v121"
_V12 = _ROOT / "results" / "v12"

FINAL_F = _V121 / "final_f.json"
FINAL_H = _V121 / "final_h.json"
FINAL_L = _V121 / "final_l.json"
DIAG_F = _V121 / "diag_f_noisy.json"
DIAG_H = _V121 / "diag_h_stable.json"
EVAL2_L = _V121 / "eval2_l.json"
V12_L = _V12 / "summary_l_small.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_rows(artifact, cond, policy):
    d = _load(artifact)
    rows = d["raw"].get(cond, [])
    return [r[policy] for r in rows]


def _mean(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


class TestV121MetadataAndRawSummaryReproducibility(unittest.TestCase):

    ARTIFACTS = [FINAL_F, FINAL_H, FINAL_L, DIAG_F, DIAG_H, EVAL2_L]
    CONDITIONS = {
        FINAL_F: ["noisy_obs"],
        FINAL_H: ["stable"],
        FINAL_L: ["stable", "novel"],
        DIAG_F: ["noisy_obs"],
        DIAG_H: ["stable"],
        EVAL2_L: ["stable", "novel"],
    }
    METRICS = ("success_rate", "mean_steps", "invalid_actions",
               "exact_discovery_D1", "exact_discovery_D2",
               "covered_discovery_D1", "covered_discovery_D2")

    def test_metadata_present_and_matched(self):
        for art in self.ARTIFACTS:
            m = _load(art)["metadata"]
            self.assertEqual(m.get("seeds"), 4, f"{art.name}: seeds")
            self.assertEqual(m.get("episodes_per_seed"), 40, f"{art.name}: episodes")
            self.assertTrue(m.get("conditions"), f"{art.name}: conditions")
            self.assertTrue(m.get("policies"), f"{art.name}: policies")
            self.assertEqual(m.get("version"), "0.12", f"{art.name}: version")

    def test_summary_means_match_raw_per_seed_means(self):
        for art, conds in self.CONDITIONS.items():
            d = _load(art)
            for cond in conds:
                summary = d["summary"][cond]
                rows = d["raw"].get(cond, [])
                self.assertEqual(len(rows), 4, f"{art.name}/{cond}: raw seeds")
                for pol, entry in summary.items():
                    if pol.startswith("comparison:"):
                        continue
                    pol_rows = [r[pol] for r in rows if pol in r]
                    for k in self.METRICS:
                        if k in entry and entry[k] is not None:
                            got = _mean(pol_rows, k)
                            self.assertIsNotNone(got, f"{art.name}/{cond}/{pol}/{k}")
                            self.assertAlmostEqual(
                                entry[k], got, places=12,
                                msg=f"{art.name}/{cond}/{pol}/{k} summary vs raw")

    def test_per_seed_rows_all_present(self):
        for art, conds in self.CONDITIONS.items():
            d = _load(art)
            for cond in conds:
                for i, row in enumerate(d["raw"][cond]):
                    for pol in d["metadata"]["policies"]:
                        if pol in row:
                            for k in ("success_rate", "mean_steps",
                                      "exact_discovery_D2"):
                                self.assertIn(k, row[pol],
                                              f"{art.name}/{cond}/seed{i}/{pol}")

    def test_success_rates_are_fractions_of_40_episodes(self):
        for art, conds in self.CONDITIONS.items():
            d = _load(art)
            for cond in conds:
                for row in d["raw"][cond]:
                    for pol, p in row.items():
                        sr = p.get("success_rate")
                        if sr is not None:
                            self.assertAlmostEqual(sr * 40, round(sr * 40),
                                                   places=6,
                                                   msg=f"{art.name}/{cond}/{pol}")

    def test_per_seed_variability_is_reportable(self):
        # frozen criterion: per-seed values must be present and vary; a
        # single-seed mechanism must be visible in the artifact.
        for art, conds in self.CONDITIONS.items():
            d = _load(art)
            for cond in conds:
                for pol, entry in d["summary"][cond].items():
                    if pol.startswith("comparison:"):
                        continue
                    if entry.get("exact_discovery_D1") is not None:
                        vals = [r[pol]["exact_discovery_D1"]
                                for r in d["raw"][cond] if pol in r]
                        self.assertEqual(len(vals), 4, f"{art.name}/{cond}/{pol}")
                        self.assertAlmostEqual(statistics.fmean(vals),
                                               entry["exact_discovery_D1"],
                                               places=12)


class TestV121LegacyAblationSemantics(unittest.TestCase):
    """Priority B.1: `metargon12-legacy` cannot test recent-credit on L."""

    def test_final_l_legacy_has_zero_temporal_activity(self):
        d = _load(FINAL_L)
        for cond in ("stable", "novel"):
            legacy = d["summary"][cond]["metargon12-legacy"]
            self.assertEqual(legacy["temporal_probes"], 0.0,
                             f"{cond}: legacy probes must be 0")
            self.assertEqual(legacy["exact_discovery_D1"], 0.0,
                             f"{cond}: legacy exact D1 must be 0")

    def test_final_l_metargon12_has_temporal_activity(self):
        d = _load(FINAL_L)
        for cond, exp_d1 in (("stable", 0.16875), ("novel", 0.23125)):
            m12 = d["summary"][cond]["metargon12"]
            self.assertGreater(m12["temporal_probes"], 0, cond)
            self.assertAlmostEqual(m12["exact_discovery_D1"], exp_d1, places=5, msg=cond)

    def test_final_l_metargon12_per_seed_exact_d1(self):
        d = _load(FINAL_L)
        for cond, exp in (("stable", [0.0, 0.0, 0.675, 0.0]),
                          ("novel", [0.0, 0.25, 0.675, 0.0])):
            vals = _raw_rows(FINAL_L, cond, "metargon12")
            got = [r["exact_discovery_D1"] for r in vals]
            self.assertEqual(got, exp, f"{cond}: per-seed exact D1")

    def test_eval2_l_reproduces_v12_study_exactly(self):
        # eval2_l.json metargon12 (diagnosis-phase, pre-default-flip) must
        # equal the v0.12 study's metargon12 rows bit-for-bit on L.
        study = _load(V12_L)
        for cond in ("stable", "novel"):
            study_rows = _raw_rows(V12_L, cond, "metargon12")
            eval2_rows = _raw_rows(EVAL2_L, cond, "metargon12")
            self.assertEqual(len(study_rows), 4)
            self.assertEqual(len(eval2_rows), 4)
            for s, e in zip(study_rows, eval2_rows):
                for k in ("success_rate", "exact_discovery_D1",
                          "exact_discovery_D2", "temporal_probes"):
                    self.assertEqual(s.get(k), e.get(k),
                                     f"{cond}: {k} study vs eval2_l")

    def test_final_l_metargon12_preserves_exact_d1(self):
        d = _load(FINAL_L)
        for cond in ("stable", "novel"):
            vals = _raw_rows(FINAL_L, cond, "metargon12")
            got = [r["exact_discovery_D1"] for r in vals]
            self.assertEqual(got, [0.0, 0.0, 0.675, 0.0] if cond == "stable"
                             else [0.0, 0.25, 0.675, 0.0], cond)


class TestV121DiagnosisVsFinalD2(unittest.TestCase):
    """Priority B.2: diagnosis-phase vs final exact-D2 disagreement."""

    def test_diagnosis_norc_vs_final_m12_d2_discrepancy(self):
        # Documented >25pp discrepancy in both families. If a corrective
        # rerun replaces these artifacts, update the expected values here.
        for diag, final, cond, pol_diag, pol_final in (
                (DIAG_F, FINAL_F, "noisy_obs", "metargon12-norc", "metargon12"),
                (DIAG_H, FINAL_H, "stable", "metargon12-norc", "metargon12")):
            d2_diag = statistics.fmean(
                r[pol_diag]["exact_discovery_D2"] for r in _load(diag)["raw"][cond])
            d2_final = statistics.fmean(
                r[pol_final]["exact_discovery_D2"] for r in _load(final)["raw"][cond])
            self.assertGreater(abs(d2_diag - d2_final), 0.25,
                               f"{cond}: discrepancy must be documented")
            self.assertGreater(d2_diag, d2_final, f"{cond}: diag > final")

    def test_final_legacy_reproduces_diagnosis_metargon12_exactly(self):
        # Same configuration (v0.12 defaults, recent-credit on) in both
        # artifact generations => identical per-seed rows on F/H.
        for diag, final, cond in (
                (DIAG_F, FINAL_F, "noisy_obs"),
                (DIAG_H, FINAL_H, "stable")):
            diag_rows = _raw_rows(diag, cond, "metargon12")
            final_rows = _raw_rows(final, cond, "metargon12-legacy")
            self.assertEqual(len(diag_rows), 4)
            self.assertEqual(len(final_rows), 4)
            for i, (a, b) in enumerate(zip(diag_rows, final_rows)):
                for k in ("success_rate", "exact_discovery_D1",
                          "exact_discovery_D2", "mean_steps"):
                    self.assertEqual(a.get(k), b.get(k),
                                     f"{cond}/seed{i}/{k}: diag m12 vs final legacy")

    def test_final_metargon12_h_matches_diagnosis_nodc(self):
        # The committed final `metargon12` (recent-credit off) rows on H
        # equal the diagnosis `-nodc` (dual-context off) rows exactly.
        # This is a striking fingerprint of the unresolved diagnosis-vs-
        # final disagreement and is recorded as evidence, not asserted as
        # a desired mechanism.
        diag_rows = _raw_rows(DIAG_H, "stable", "metargon12-nodc")
        final_rows = _raw_rows(FINAL_H, "stable", "metargon12")
        for i, (a, b) in enumerate(zip(diag_rows, final_rows)):
            for k in ("success_rate", "exact_discovery_D1",
                      "exact_discovery_D2", "mean_steps"):
                self.assertEqual(a.get(k), b.get(k),
                                 f"seed{i}/{k}: diag nodc vs final m12")

    def test_final_metargon12_f_d2_close_to_diagnosis_nodc(self):
        d2_final = statistics.fmean(
            r["metargon12"]["exact_discovery_D2"] for r in _load(FINAL_F)["raw"]["noisy_obs"])
        d2_nodc = statistics.fmean(
            r["metargon12-nodc"]["exact_discovery_D2"] for r in _load(DIAG_F)["raw"]["noisy_obs"])
        self.assertAlmostEqual(d2_final, d2_nodc, delta=0.06)

    def test_report_d2_header_cells_trace_to_diagnosis(self):
        # RESEARCH_V121 §4 reports F D2 75.6 / H D2 70.0 for v0.12.1;
        # those trace to the diagnosis -norc runs, not to final artifacts.
        f_diag = statistics.fmean(
            r["metargon12-norc"]["exact_discovery_D2"] for r in _load(DIAG_F)["raw"]["noisy_obs"])
        h_diag = statistics.fmean(
            r["metargon12-norc"]["exact_discovery_D2"] for r in _load(DIAG_H)["raw"]["stable"])
        self.assertAlmostEqual(f_diag, 0.75625, places=4)
        self.assertAlmostEqual(h_diag, 0.70, places=4)
        f_final = statistics.fmean(
            r["metargon12"]["exact_discovery_D2"] for r in _load(FINAL_F)["raw"]["noisy_obs"])
        h_final = statistics.fmean(
            r["metargon12"]["exact_discovery_D2"] for r in _load(FINAL_H)["raw"]["stable"])
        self.assertAlmostEqual(f_final, 0.45, places=4)
        self.assertAlmostEqual(h_final, 0.4125, places=4)


class TestV121RosterConsistency(unittest.TestCase):
    """Priority B.3: roster / metadata / ablation consistency."""

    def test_final_l_metadata_lists_notp_twice_documented(self):
        m = _load(FINAL_L)["metadata"]["policies"]
        self.assertGreater(len(m), len(set(m)), "final_l metadata duplicates")
        self.assertEqual(m.count("metargon12-notp"), 2)

    def test_summary_policy_keys_unique_and_within_metadata(self):
        d = _load(FINAL_L)
        meta = set(d["metadata"]["policies"])
        for cond in ("stable", "novel"):
            keys = [k for k in d["summary"][cond] if not k.startswith("comparison:")]
            self.assertEqual(len(keys), len(set(keys)), cond)
            self.assertTrue(set(keys) <= meta, cond)

    def test_metargon12_notp_rows_close_to_legacy_rows_on_l(self):
        # Both are temporal-off on L (legacy also keeps recent-credit on).
        # Success may differ by a few episodes; exact D1 is 0 for both and
        # the success difference must be small (documented closeness).
        d = _load(FINAL_L)
        for cond in ("stable", "novel"):
            notp = d["summary"][cond]["metargon12-notp"]
            legacy = d["summary"][cond]["metargon12-legacy"]
            self.assertEqual(notp["exact_discovery_D1"], 0.0, cond)
            self.assertEqual(legacy["exact_discovery_D1"], 0.0, cond)
            self.assertAlmostEqual(notp["success_rate"], legacy["success_rate"],
                                   delta=0.02, msg=f"{cond} success")


class TestV121SourceConventions(unittest.TestCase):
    """Source-level checks that guard the documented interpretations."""

    @classmethod
    def setUpClass(cls):
        cls.harness = (_ROOT / "python" / "benchmark_v12.py").read_text(encoding="utf-8")
        cls.agent = (_ROOT / "python" / "metargon_v12.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.harness)

    def _assign_dict(self, name):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
        return None

    def test_legacy_ablation_lacks_temporal(self):
        abl = self._assign_dict("ABLATIONS")
        self.assertIn("metargon12-legacy", abl)
        legacy = abl["metargon12-legacy"]
        self.assertEqual(legacy.get("dual_context_mode"), "dual")
        self.assertIs(legacy.get("use_recent_credit"), True)
        self.assertNotIn("use_temporal", legacy,
                         "legacy ablation must not set temporal")

    def test_temporal_only_metargon12_on_family_l(self):
        # make_policy: only (family=="l" and name=="metargon12") gets
        # use_temporal=True.
        src = self.harness
        self.assertIn('if family == "l" and name == "metargon12":', src)
        self.assertIn("use_temporal=True", src)

    def test_agent_defaults_recent_credit_off_temporal_off(self):
        src = self.agent
        self.assertIn("use_recent_credit: bool = False", src)
        self.assertIn("use_temporal: bool = False", src)
        self.assertIn("adaptive_confirmed_events: bool = False", src)

    def test_student_policies_get_no_hidden_config(self):
        # play_episode passes no spec to students; bayes gets only noise_q;
        # oracle gets config.
        src = self.harness
        self.assertIn("policy.episode_start()", src)
        self.assertIn("policy.episode_start(noise_q=spec.noise_q)", src)
        self.assertIn("policy.episode_start(config=spec.config)", src)

    def test_equal_episode_budgets_in_harness(self):
        # r/l worlds: 80 steps normally; 44 only for tight_budget; noise_q
        # 0.08 only for noisy_obs. Same budget for every policy in a run.
        src = self.harness
        self.assertIn("max_steps = 44 if condition == \"tight_budget\" else 80", src)
        self.assertIn("noise_q = 0.08 if condition == \"noisy_obs\" else 0.0", src)

    def test_frozen_pools_loaded_read_only_for_r_l(self):
        src = self.harness
        self.assertIn("FROZEN", src)
        self.assertIn("frozen_r.json", src)
        self.assertIn("frozen_l.json", src)
        self.assertIn("load_frozen_pools", src)


if __name__ == "__main__":
    unittest.main()
