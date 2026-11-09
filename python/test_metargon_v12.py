"""v0.12 agent tests."""
import unittest

from lab_env import generate_worlds, LabWorld
from metargon_v12 import MetargonV12


class MetargonV12Tests(unittest.TestCase):
    def test_no_hidden_access(self):
        a = MetargonV12(1)
        for attr in ("config", "_config", "hidden", "family"):
            self.assertFalse(hasattr(a, attr), attr)

    def test_reproducible(self):
        def run(seed):
            pol = MetargonV12(seed)
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
        pol = MetargonV12(9, use_temporal=True)
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
        self.assertLessEqual(len(pol.temporal), 64) if pol.temporal else None

    def test_learns_easy_config(self):
        from lab_env import LabConfig, F1_TEMPLATES, F2_TEMPLATES
        easy = LabConfig(d1_cond=F1_TEMPLATES[0], d2_cond=F2_TEMPLATES[0],
                         d1_delay=False, d2_delay=False)
        successes = 0
        for seed in (2, 5, 11):
            pol = MetargonV12(seed)
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

    def test_variants_constructible(self):
        for kw in ({"use_dual_context": False},
                   {"use_recent_credit": False},
                   {"use_adaptive_confirm": False},
                   {"use_temporal": True}):
            a = MetargonV12(1, **kw)
            self.assertTrue(hasattr(a, "memory"))


if __name__ == "__main__":
    unittest.main()
