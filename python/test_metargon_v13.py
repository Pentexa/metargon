"""v0.13 agent tests."""
import unittest

from lab_env import generate_worlds, LabWorld, LabConfig, F1_TEMPLATES, F2_TEMPLATES
from metargon_v13 import MetargonV13


class MetargonV13Tests(unittest.TestCase):
    def test_no_hidden_access(self):
        a = MetargonV13(1)
        for attr in ("config", "_config", "hidden", "family"):
            self.assertFalse(hasattr(a, attr), attr)

    def test_reproducible(self):
        def run(seed):
            pol = MetargonV13(seed, use_temporal=True)
            trace = []
            for spec in generate_worlds("stable", 7, 6):
                pol.episode_start()
                w = LabWorld(spec.config, ep_seed=spec.seed,
                             noise_q=spec.noise_q, max_steps=spec.max_steps)
                while not w.finished and w.steps < w.max_steps:
                    obs = w.observe()
                    a = pol.act(obs)
                    res = w.step(a)
                    pol.observe_outcome(obs, a, res)
                    trace.append(a)
            return trace
        self.assertEqual(run(42), run(42))

    def test_budgets_bounded(self):
        pol = MetargonV13(9, use_temporal=True)
        spec = generate_worlds("stable", 11, 4)[0]
        pol.episode_start()
        w = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                     max_steps=spec.max_steps)
        before = pol.plan_calls
        for _ in range(12):
            obs = w.observe()
            a = pol.act(obs)
            res = w.step(a)
            pol.observe_outcome(obs, a, res)
        self.assertLessEqual(pol.plan_calls - before, 12 * 5 + 8)
        if pol.temporal is not None:
            self.assertLessEqual(len(pol.temporal), 64)

    def test_variants_constructible(self):
        for kw in ({"use_dual_context": False},
                   {"use_recent_credit": False},
                   {"use_adaptive_confirm": False},
                   {"use_temporal": True},
                   {"temporal_completion_mode": "active_first"},
                   {"temporal_support_weighting": "flat"},
                   {"temporal_target_exempt": False}):
            a = MetargonV13(1, **kw)
            self.assertTrue(hasattr(a, "memory"))

    def test_temporal_modes_wired(self):
        a = MetargonV13(3, use_temporal=True)
        self.assertEqual(a.temporal.completion_mode, "rolling")
        self.assertEqual(a.temporal.support_weighting, "flat")
        self.assertFalse(a.temporal_target_exempt)
        self.assertEqual(a.dual_persist_gate, 0)
        b = MetargonV13(3, use_temporal=True,
                        temporal_completion_mode="active_first",
                        temporal_support_weighting="proximity",
                        temporal_target_exempt=True,
                        dual_persist_gate=2)
        self.assertEqual(b.temporal.completion_mode, "active_first")
        self.assertEqual(b.temporal.support_weighting, "proximity")
        self.assertTrue(b.temporal_target_exempt)
        self.assertEqual(b.dual_persist_gate, 2)

    def test_protected_split(self):
        pol = MetargonV13(4)
        hold, hyp = pol._protected_atoms_split()
        self.assertIsInstance(hold, frozenset)
        self.assertIsInstance(hyp, frozenset)

    def test_learns_easy_config(self):
        easy = LabConfig(d1_cond=F1_TEMPLATES[0], d2_cond=F2_TEMPLATES[0],
                         d1_delay=False, d2_delay=False)
        successes = 0
        for seed in (2, 5, 11):
            pol = MetargonV13(seed)
            for _ in range(8):
                pol.episode_start()
                w = LabWorld(easy, ep_seed=1, noise_q=0.0, max_steps=80)
                while not w.finished and w.steps < w.max_steps:
                    obs = w.observe()
                    a = pol.act(obs)
                    res = w.step(a)
                    pol.observe_outcome(obs, a, res)
                successes += int(w.finished)
        self.assertGreaterEqual(successes, 8)


if __name__ == "__main__":
    unittest.main()
