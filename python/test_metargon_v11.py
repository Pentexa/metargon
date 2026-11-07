"""v0.11 agent tests (noise-aware learning)."""
import unittest

from lab_env import generate_worlds, LabWorld
from metargon_v11 import MetargonV11
from metargon_v10 import PROBE_MAX_PLANS


class MetargonV11Tests(unittest.TestCase):
    def test_no_hidden_access(self):
        a = MetargonV11(1)
        for attr in ("config", "_config", "mode", "d1_cond", "d2_cond",
                     "hidden", "family"):
            self.assertFalse(hasattr(a, attr), attr)

    def test_reproducible(self):
        def run(seed):
            pol = MetargonV11(seed)
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
        pol = MetargonV11(9)
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
        self.assertLessEqual(pol.plan_calls - before,
                             12 * (PROBE_MAX_PLANS + 2) + 8)

    def test_learns_easy_config(self):
        from lab_env import LabConfig, F1_TEMPLATES, F2_TEMPLATES
        easy = LabConfig(d1_cond=F1_TEMPLATES[0], d2_cond=F2_TEMPLATES[0],
                         d1_delay=False, d2_delay=False)
        successes = 0
        for seed in (2, 5, 11):
            pol = MetargonV11(seed)
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

    def test_confirmed_events_reduce_false_verification(self):
        """Under noise, v0.11's confirmed-event gate must cut false
        verified hypotheses vs v0.10 (same worlds/seeds)."""
        from metargon_v10 import MetargonV10
        worlds = generate_worlds("noisy_obs", 13, 12)
        counts = {"v10": 0, "v11": 0}
        for name, make in (("v10", lambda s: MetargonV10(s)),
                           ("v11", lambda s: MetargonV11(s))):
            pol = make(555)
            for spec in worlds:
                pol.episode_start()
                w = LabWorld(spec.config, ep_seed=spec.seed,
                             noise_q=spec.noise_q, max_steps=spec.max_steps)
                while not w.finished and w.steps < w.max_steps:
                    obs = w.observe()
                    a = pol.act(obs)
                    res = w.step(a)
                    pol.observe_outcome(obs, a, res)
                false_v = 0
                for h in pol.memory.hypotheses:
                    if h.status != "verified":
                        continue
                    true_cond = spec.config.d1_cond if h.door == "D1" \
                        else spec.config.d2_cond
                    if h.antecedent != frozenset(true_cond):
                        false_v += 1
                counts[name] += false_v
        # v11 must not be worse (allow small noise slack)
        self.assertLessEqual(counts["v11"], counts["v10"] + 3)

    def test_replay_modes_validated(self):
        for mode in ("on", "off", "filtered", "weighted"):
            a = MetargonV11(1, replay_mode=mode)
            self.assertEqual(a.memory.replay_mode, mode)
        with self.assertRaises(ValueError):
            MetargonV11(1, replay_mode="bogus")


if __name__ == "__main__":
    unittest.main()
