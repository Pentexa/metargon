"""Priority B — enforce the public-observation boundary (Academy).

These tests verify that the runner boundary (evaluator, learning loop,
evaluation grids) hands student policies a *public* episode spec that
cannot expose the hidden EnvConfig, while privileged diagnostic
comparators (the oracle) still receive the full spec.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from academy.agents.baselines import OraclePolicy
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.tasks import EpisodeSpec, difficulty_of, get_splits
from academy.eval.evaluator import Evaluator
from academy.eval.protocol import (
    is_privileged, public_episode_spec, spec_for,
)
from academy.learning.loop import run_one_episode

SPLITS = get_splits()
SPEC = EpisodeSpec(
    config=SPLITS["train"][0], max_steps=80, noise_q=0.0,
    spec_seed=5, pool_tag="train", difficulty=difficulty_of(SPLITS["train"][0]))


class _EvilStudent:
    """Adversarial policy: attempts to read the hidden config through the
    episode spec it receives from the runner."""

    name = "evil_student"

    def __init__(self, seed=0):
        self.seed = seed
        self.saw_config = False
        self.error = None

    def episode_start(self, spec, geom=None):
        try:
            _ = spec.config
        except AttributeError:
            self.error = "AttributeError"
        else:
            self.saw_config = True

    def act(self, obs):
        return "N"

    def learn(self, *args, **kwargs):
        pass

    def finish_episode(self, success):
        pass

    def diagnostics(self):
        return {}


class TestProtocolBoundary(unittest.TestCase):
    def test_public_spec_hides_config(self):
        pub = public_episode_spec(SPEC)
        self.assertFalse(hasattr(pub, "config"),
                         "public spec must not expose .config")
        self.assertEqual(pub.max_steps, 80)
        self.assertEqual(pub.noise_q, 0.0)
        self.assertEqual(pub.spec_seed, 5)
        self.assertEqual(pub.pool_tag, "train")

    def test_oracle_is_privileged(self):
        self.assertTrue(is_privileged(OraclePolicy()))
        self.assertFalse(is_privileged(MetargonAcademyAgent()))
        self.assertFalse(is_privileged(_EvilStudent()))

    def test_spec_for_returns_public_for_student_full_for_oracle(self):
        self.assertFalse(hasattr(spec_for(MetargonAcademyAgent(), SPEC), "config"))
        self.assertIs(spec_for(OraclePolicy(), SPEC), SPEC)

    def test_adversarial_student_cannot_read_config_via_runner(self):
        evil = _EvilStudent()
        r = run_one_episode(evil, SPEC, max_steps=20)
        self.assertEqual(evil.error, "AttributeError",
                         "student-facing spec must hide the hidden config")
        self.assertFalse(evil.saw_config)
        self.assertIn("success", r)

    def test_evaluator_passes_public_spec_to_student(self):
        captured = {}

        class SpyAgent(MetargonAcademyAgent):
            def episode_start(self, spec, geom=None):  # noqa: D102
                captured["spec"] = spec
                return super().episode_start(spec, geom)

        ev = Evaluator(SPLITS, seed=1, episodes_per_seed=2, max_steps=40)
        result = ev.run_policy(lambda s: SpyAgent(seed=s), SPLITS["test"], n_seeds=1)
        self.assertFalse(hasattr(captured["spec"], "config"),
                         "evaluator must pass a public spec to students")
        self.assertEqual(result["policy"], "metargon_full")

    def test_evaluator_passes_full_spec_to_oracle(self):
        captured = {}

        class SpyOracle(OraclePolicy):
            def episode_start(self, spec, geom=None):  # noqa: D102
                captured["spec"] = spec
                return super().episode_start(spec, geom)

        ev = Evaluator(SPLITS, seed=2, episodes_per_seed=2, max_steps=40)
        ev.run_policy(lambda s: SpyOracle(seed=s), SPLITS["test"], n_seeds=1)
        self.assertTrue(hasattr(captured["spec"], "config"),
                        "oracle must still receive the full spec")


if __name__ == "__main__":
    unittest.main()
