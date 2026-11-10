"""v0.13 confirmatory diagnostics/analysis tool tests."""
import unittest
from pathlib import Path

from classify_v13_failures import classify_episode
from analyze_hr_regressions import report as hr_report
from analysis_v13_confirmatory import delta_report
from analyze_hr_diag import analyze as hr_diag_analyze


def _episode(floor=True, created=1, verified=0, contradicted=0,
             evicted=0, probes=1):
    return {
        "temporal": {
            "true_pair_observed": {"D1": floor, "D2": False},
            "correct_created": created,
            "correct_verified_max": verified,
            "correct_contradicted": contradicted,
            "correct_evicted": evicted,
        },
        "agent": {"probe_attempts": probes, "probe_blocked": 0},
    }


class FailureClassificationTests(unittest.TestCase):
    def test_c1_sequence_never_observed(self):
        e = _episode(floor=False, created=0)
        self.assertEqual(classify_episode(e),
                         "C1_sequence_never_observed")

    def test_c2_candidate_not_generated(self):
        e = _episode(floor=True, created=0)
        self.assertEqual(classify_episode(e),
                         "C2_candidate_not_generated")

    def test_c3_generated_not_verified(self):
        e = _episode(floor=True, created=1, verified=0)
        self.assertEqual(classify_episode(e),
                         "C3_generated_not_verified")

    def test_c4_contradicted(self):
        e = _episode(floor=True, created=1, verified=0, contradicted=1)
        self.assertEqual(classify_episode(e),
                         "C4_contradicted_or_evicted")

    def test_c4_evicted(self):
        e = _episode(floor=True, created=1, verified=0, evicted=1)
        self.assertEqual(classify_episode(e),
                         "C4_contradicted_or_evicted")

    def test_c5_discovered_but_unused(self):
        e = _episode(floor=True, created=1, verified=1, probes=0)
        self.assertEqual(classify_episode(e),
                         "C5_discovered_but_unused")

    def test_success_path_none(self):
        e = _episode(floor=True, created=1, verified=1, probes=2)
        self.assertIsNone(classify_episode(e))


class HrDiagTests(unittest.TestCase):
    def test_hr_diag_analyze_classifies(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "diag.json"
            p.write_text(json.dumps({
                "runs": [{
                    "seed": 21800,
                    "episodes": [
                        {"discovery": {"D1": [True, True],
                                       "D2": [False, False]},
                         "success": True},
                        {"discovery": {"D1": [False, False],
                                       "D2": [False, False]},
                         "success": False},
                    ],
                }],
            }), encoding="utf-8")
            rows = hr_diag_analyze([p])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["cls"], "learned+success")
            self.assertEqual(rows[1]["cls"], "failed_no_exact_learn")


class AnalysisDeltaTests(unittest.TestCase):
    def test_delta_report(self):
        r = delta_report([0.8, 0.9], [0.7, 0.6], "t")
        self.assertAlmostEqual(r["delta"], 0.2)
        self.assertEqual(r["n_seeds"], 2)
        self.assertEqual(r["n_positive_delta_seeds"], 2)

    def test_hr_report(self):
        class FakeArt:
            def __init__(self):
                self.__dict__["raw"] = {"stable": [
                    {"metargon13": {"success_rate": 0.7},
                     "metargon12": {"success_rate": 0.6}},
                    {"metargon13": {"success_rate": 0.8},
                     "metargon12": {"success_rate": 0.7}},
                ]}
            def __getitem__(self, k):
                return self.__dict__[k]
        # avoid exposing a dict; reuse per_seed via duck typing
        art = {"metadata": {"name": "x", "conditions": ["stable"],
                            "policies": ["metargon13", "metargon12"]},
               "raw": {"stable": [
                   {"metargon13": {"success_rate": 0.7},
                    "metargon12": {"success_rate": 0.6}},
                   {"metargon13": {"success_rate": 0.8},
                    "metargon12": {"success_rate": 0.7}},
               ]}}
        r = hr_report(art, "stable", "metargon13", "metargon12",
                      "success_rate", "t")
        self.assertAlmostEqual(r["delta"], 0.1)
        self.assertEqual(r["n_seeds"], 2)


if __name__ == "__main__":
    unittest.main()
