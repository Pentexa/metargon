"""Evaluation v0.4 audit — frozen future-evaluation criteria (v02).

Validates `generalization_audit/configs/frozen_eval_criteria_v02.json`
(the criteria defined by audit v0.4 for v0.13 / Memory v0.6 before any
such results exist). Also asserts the criteria do not accidentally
reference artifacts that do not exist yet, keeping the protocol
independent of the implementation.
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

_CRITERIA = _ROOT / "generalization_audit" / "configs" / "frozen_eval_criteria_v02.json"
_REQUIRED_SECTIONS = (
    "pilot_requirements", "statistical_uncertainty",
    "zero_shot_vs_within_episode", "resource_fairness",
    "regression_preservation", "evidence_for_improvement_claim",
    "independence",
)


class TestFrozenCriteriaV02(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.criteria = json.loads(_CRITERIA.read_text(encoding="utf-8"))

    def test_protocol_and_audit_meta(self):
        self.assertEqual(self.criteria["protocol"], "frozen-eval-criteria-v02")
        self.assertEqual(self.criteria["audit"], "evaluation-v04")
        self.assertTrue(self.criteria["created"].startswith("2026-11-"))

    def test_all_required_sections_present(self):
        for sec in _REQUIRED_SECTIONS:
            self.assertIn(sec, self.criteria, f"missing section {sec}")

    def test_every_section_has_content(self):
        for sec in _REQUIRED_SECTIONS:
            body = self.criteria[sec]
            if isinstance(body, dict):
                # requirement arrays must be non-empty
                for k, v in body.items():
                    if k == "requirement":
                        self.assertTrue(isinstance(v, list) and v, f"{sec}.requirement")
                    elif k in ("minimum", "confirmatory", "registration", "note"):
                        self.assertTrue(str(v).strip(), f"{sec}.{k}")
            else:
                self.assertTrue(str(body).strip(), sec)

    def test_improvement_rule_covers_evidence_levels(self):
        rule = self.criteria["evidence_for_improvement_claim"]["requirement"]
        text = "\n".join(rule).lower()
        for token in ("confirmatory", "bootstrap", "per-seed", "ablation",
                      "reproducible", "regression"):
            self.assertIn(token, text, f"improvement rule must mention {token}")

    def test_future_targets_have_no_committed_artifacts_yet(self):
        # Criteria must be defined before results exist: no results/v13 or
        # memory_v06 directory should be present in the workspace yet.
        self.assertFalse((_ROOT / "results" / "v13").exists(),
                         "v0.13 artifacts must not exist while criteria are frozen")
        self.assertFalse((_ROOT / "results" / "academy_v01" / "memory_v06").exists(),
                         "Memory v0.6 artifacts must not exist while criteria are frozen")

    def test_criteria_do_not_require_specific_implementation(self):
        text = json.dumps(self.criteria)
        for banned in ("metargon_v12.py", "benchmark_v12.py"):
            self.assertNotIn(banned, text,
                             "frozen criteria must not reference implementation files")


if __name__ == "__main__":
    unittest.main()
