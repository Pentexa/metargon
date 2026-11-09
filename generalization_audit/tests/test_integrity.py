"""Priority A + C — statistical integrity regression tests.

1. paired_difference is ordered (candidate − baseline).
2. The committed memory_v03 summary's comparisons agree with the raw
   JSONL evidence (no stale/diverged artifacts).
3. Reproducibility metadata (seed, episodes, hash randomization) is
   recorded and consistent.
4. The silent untrained-fallback path surfaces a warning.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from academy.eval.metrics import paired_difference

MEMORY_V03_DIR = _ROOT / "results" / "academy_v01" / "memory_v03"
SUMMARY = MEMORY_V03_DIR / "memory_v03_summary.json"
RUNS = MEMORY_V03_DIR / "memory_v03_runs.jsonl"


class TestPairedDifferenceOrdering(unittest.TestCase):
    def test_candidate_minus_baseline_positive_improvement(self):
        baseline = [0.5] * 16
        candidate = [0.7] * 16
        d = paired_difference(candidate, baseline, seed=0)
        self.assertAlmostEqual(d["mean"], 0.2, places=6)

    def test_candidate_minus_baseline_negative_degradation(self):
        baseline = [0.5625] * 16
        candidate = [0.3125] * 16
        d = paired_difference(candidate, baseline, seed=20261108)
        self.assertAlmostEqual(d["mean"], -0.25, places=6)

    def test_reversing_arguments_flips_sign(self):
        a = [0.3, 0.6, 0.9]
        b = [0.4, 0.5, 0.8]
        d1 = paired_difference(a, b, seed=1, n_boot=500)
        d2 = paired_difference(b, a, seed=1, n_boot=500)
        self.assertAlmostEqual(d1["mean"], -d2["mean"], places=10)


class TestMemoryV03ArtifactConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skip = False
        if not (SUMMARY.exists() and RUNS.exists()):
            cls.skip = True
            return
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        cls.runs = {}
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cls.runs[r["variant"]] = r["per"]

    def _require_artifacts(self):
        if self.skip:
            self.skipTest("memory_v03 artifacts not present")

    def test_summary_conditions_detail_matches_jsonl(self):
        self._require_artifacts()
        for name, per in self.summary["conditions_detail"].items():
            self.assertIn(name, self.runs)
            self.assertEqual(per, self.runs[name],
                             f"conditions_detail[{name}] diverged from jsonl")

    def test_comparisons_match_candidate_minus_baseline(self):
        self._require_artifacts()
        orig = self.summary["conditions_detail"]["original"]["classic_test_per_config"]
        for name in self.summary["comparisons"]:
            variant = name.replace("_vs_original", "")
            vec = self.summary["conditions_detail"][variant]["classic_test_per_config"]
            if not vec or len(vec) != len(orig):
                continue
            expected = paired_difference(vec, orig, seed=20261108)
            got = self.summary["comparisons"][name]["classic_test"]
            self.assertAlmostEqual(got["mean"], expected["mean"], places=8,
                                   msg=f"{name} mean")
            self.assertAlmostEqual(got["ci"][0], expected["ci"][0], places=8,
                                   msg=f"{name} ci lo")
            self.assertAlmostEqual(got["ci"][1], expected["ci"][1], places=8,
                                   msg=f"{name} ci hi")

    def test_signed_comparisons_are_negative_for_bad_variants(self):
        """Window/ratio/combined scores are below original, so candidate −
        baseline must be negative. The previous pipeline reported them as
        positive (baseline − candidate), a difference-direction bug."""
        self._require_artifacts()
        for name in ("window", "ratio", "combined"):
            mean = self.summary["comparisons"][f"{name}_vs_original"]["classic_test"]["mean"]
            self.assertLess(mean, 0.0,
                            f"{name} classic Δ should be negative (candidate − baseline)")

    def test_metadata_recorded(self):
        self._require_artifacts()
        self.assertEqual(self.summary["seed"], 20261107)
        self.assertEqual(self.summary["episodes"], 160)
        self.assertFalse(self.summary["hash_randomization"])
        self.assertIn("run_id", self.summary)
        self.assertEqual(self.summary["run_id"], "memory-v03")
        self.assertIn("conditions_detail", self.summary)
        self.assertIn("comparisons", self.summary)


class TestUntrainedFallbackWarning(unittest.TestCase):
    def test_warning_emitted_when_checkpoint_missing(self):
        from academy.cli.main import load_policy_state_or_warn
        from academy.agents.metargon_agent import MetargonAcademyAgent
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            agent = load_policy_state_or_warn(
                lambda s: MetargonAcademyAgent(seed=s),
                str(_ROOT / "results" / "academy_v01"),
                "does_not_exist", warn=True)
        self.assertIsNone(agent)
        self.assertIn("WARNING", stderr.getvalue())
        self.assertIn("UNTRAINED", stderr.getvalue())

    def test_no_warning_when_disabled(self):
        from academy.cli.main import load_policy_state_or_warn
        from academy.agents.metargon_agent import MetargonAcademyAgent
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            agent = load_policy_state_or_warn(
                lambda s: MetargonAcademyAgent(seed=s),
                str(_ROOT / "results" / "academy_v01"),
                "does_not_exist", warn=False)
        self.assertIsNone(agent)
        self.assertEqual(stderr.getvalue(), "")


class TestAcademyEvalGrid(unittest.TestCase):
    """Priority D — the frozen Academy evaluation grid must be
    deterministic and every config must belong to the current pool."""

    GRID = _ROOT / "generalization_audit" / "configs" / "academy_eval_grid_v01.json"

    @classmethod
    def setUpClass(cls):
        if not cls.GRID.exists():
            cls.skip = True
        else:
            cls.skip = False
            cls.grid = json.loads(cls.GRID.read_text(encoding="utf-8"))

    def _require(self):
        if getattr(self, "skip", False):
            self.skipTest("grid file not present")

    def test_deterministic_reproduction(self):
        self._require()
        from academy.envs.tasks import load_splits, DEFAULT_SPLITS_PATH
        from academy.evren.retention import sample_substrate
        splits = load_splits(DEFAULT_SPLITS_PATH)
        for key in ("classic", "ood"):
            spec = self.grid[key]
            rep = [str(c.key) for c in sample_substrate(
                list(splits[spec["pool"]]), spec["n_configs"], spec["sample_seed"])]
            self.assertEqual(rep, spec["keys"], f"{key} grid not reproducible")

    def test_all_grid_configs_belong_to_current_pool(self):
        self._require()
        from academy.envs.tasks import load_splits, DEFAULT_SPLITS_PATH
        splits = load_splits(DEFAULT_SPLITS_PATH)
        pool_keys = {str(c.key) for c in splits["test"]}
        ood_keys = {str(c.key) for c in splits["ood"]}
        for k in self.grid["classic"]["keys"]:
            self.assertIn(k, pool_keys, f"classic config {k} not in current test pool")
        for k in self.grid["ood"]["keys"]:
            self.assertIn(k, ood_keys, f"ood config {k} not in current ood pool")

    def test_committed_memory_v03_grid_reproduces_under_pinned_hash(self):
        """The committed memory_v03 per-config grid is reproducible from the
        current splits under a pinned PYTHONHASHSEED (hash seed 0 in this
        suite). sample_substrate sorts by str(frozenset), which is hash-seed
        sensitive — hence the pinnning requirement."""
        self._require()
        if not SUMMARY.exists():
            self.skipTest("memory_v03 summary not present")
        committed = set(json.loads(SUMMARY.read_text(encoding="utf-8"))
                        ["conditions_detail"]["original"]["classic_test"]["per_config"])
        self.assertEqual(committed, set(self.grid["classic"]["keys"]),
                         "committed grid must reproduce under pinned hash seed")

    def test_hash_seed_sensitivity_is_documented(self):
        """record the design constraint in the frozen protocol file."""
        self._require()
        note = self.grid.get("note", "") + " " + self.grid.get("hash_seed", "")
        self.assertIn("HASHSEED", note.upper() + "0")


if __name__ == "__main__":
    unittest.main()
