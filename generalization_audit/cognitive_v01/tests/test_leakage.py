"""Leakage prevention checks.

The benchmark never hands hidden wiring to the student. These tests
verify (a) that the student's source does not reference hidden internals
(the same guarantee the Academy's own leakage tests enforce), and (b)
that the shared driver only feeds public observations to policies.
"""
from __future__ import annotations

import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AGENT_SOURCES = (
    _ROOT / "academy" / "agents" / "metargon_agent.py",
    _ROOT / "academy" / "agents" / "memory.py",
    _ROOT / "academy" / "agents" / "planner.py",
    _ROOT / "academy" / "agents" / "common.py",
)
FORBIDDEN = (
    "transition_with_hidden",
    "_config",
    "d1_rule",
    "d2_rule",
    "rule_for",
)


class TestNoHiddenStateLeakage(unittest.TestCase):

    def test_student_sources_never_reference_hidden_internals(self):
        for src in AGENT_SOURCES:
            text = src.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                self.assertNotIn(token, text,
                                 f"{src.name} references {token!r}")

    def test_driver_uses_only_public_observation_stream(self):
        from generalization_audit.cognitive_v01.policies import run_episode
        import inspect
        src = inspect.getsource(run_episode)
        # hidden-dynamics access is never allowed in the driver
        self.assertNotIn("transition_with_hidden", src)
        # the ONLY legitimate use of spec.config is constructing the public
        # environment (which needs the wiring to simulate hidden dynamics)
        self.assertEqual(src.count("spec.config"), 1)
        # policies receive only public observations / results
        self.assertNotIn("policy.act(spec", src)
        self.assertNotIn("policy.learn(spec", src)


if __name__ == "__main__":
    unittest.main()
