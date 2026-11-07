"""v0.10 agent tests (Phase G)."""
import unittest

from lab_env import (
    LabConfig, LabWorld, F1_TEMPLATES, F2_TEMPLATES, generate_worlds,
)
from metargon_v10 import MetargonV10
from metargon_v09 import PROBE_MAX_PLANS

EASY = LabConfig(
    d1_cond=F1_TEMPLATES[0],
    d2_cond=F2_TEMPLATES[0],
    d1_delay=False, d2_delay=False,
)


class MetargonV10Tests(unittest.TestCase):
    def test_agent_has_no_hidden_config_access(self):
        a = MetargonV10(1)
        for attr in ("config", "_config", "mode", "d1_cond", "d2_cond",
                     "hidden", "family"):
            self.assertFalse(hasattr(a, attr), attr)

    def test_seed_reproducible_actions(self):
        specs = generate_worlds("stable", 7, 6)

        def run(seed):
            pol = MetargonV10(seed)
            trace = []
            for spec in specs:
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

    def test_planner_budget_per_decision_bounded(self):
        pol = MetargonV10(9)
        spec = generate_worlds("stable", 11, 4)[0]
        pol.episode_start()
        w = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                     max_steps=spec.max_steps)
        calls_before = pol.plan_calls
        for _ in range(12):
            obs = w.observe()
            a = pol.act(obs)
            res = w.step(a)
            pol.observe_outcome(obs, a, res)
        calls = pol.plan_calls - calls_before
        self.assertLessEqual(calls, 12 * (PROBE_MAX_PLANS + 2) + 8)

    def test_learns_and_opens_doors_on_easy_config(self):
        successes = 0
        for seed in (2, 5, 11, 17):
            pol = MetargonV10(seed)
            for _ in range(12):
                pol.episode_start()
                w = LabWorld(EASY, ep_seed=1, noise_q=0.0, max_steps=80)
                while not w.finished and w.steps < w.max_steps:
                    obs = w.observe()
                    a = pol.act(obs)
                    res = w.step(a)
                    pol.observe_outcome(obs, a, res)
                successes += int(w.finished)
        self.assertGreaterEqual(successes, 8)

    def test_anti_freeze_probes_continue_without_events(self):
        """After a run with no door events, the agent must still try new
        antecedents instead of permanently exhausting its probe menu (the
        v0.9 discovery-freeze failure mode)."""
        pol = MetargonV10(3)
        specs = generate_worlds("stable", 5, 8)
        for spec in specs:
            pol.episode_start()
            w = LabWorld(spec.config, ep_seed=spec.seed,
                         noise_q=spec.noise_q, max_steps=spec.max_steps)
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = pol.act(obs)
                res = w.step(a)
                pol.observe_outcome(obs, a, res)
        d = pol.diagnostics()
        # hypothesis-backed probes must have been attempted
        self.assertGreaterEqual(d["hyp_probe_actions"], 5)
        self.assertGreaterEqual(d["plan_calls"], 10)

    def test_memory_bounded(self):
        pol = MetargonV10(3)
        specs = generate_worlds("rapid_shift", 5, 20)
        for spec in specs:
            pol.episode_start()
            w = LabWorld(spec.config, ep_seed=spec.seed,
                         noise_q=spec.noise_q, max_steps=spec.max_steps)
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = pol.act(obs)
                res = w.step(a)
                pol.observe_outcome(obs, a, res)
        self.assertLessEqual(len(pol.memory), pol.memory.max_hypotheses)

    def test_noisy_condition_bounded_invalid_actions(self):
        pol = MetargonV10(5)
        spec = generate_worlds("noisy_obs", 13, 4)[0]
        pol.episode_start()
        w = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                     max_steps=spec.max_steps)
        invalid = 0
        while not w.finished and w.steps < w.max_steps:
            obs = w.observe()
            a = pol.act(obs)
            res = w.step(a)
            invalid += int(not res.valid)
            pol.observe_outcome(obs, a, res)
        # robust belief should keep invalid actions far below raw noise rate
        self.assertLess(invalid, 10)

    def test_ablation_flags_constructible(self):
        for kw in ({"use_robust_belief": False},
                   {"use_cooldowns": False},
                   {"use_explore_fallback": False},
                   {"use_replay": False}):
            a = MetargonV10(1, **kw)
            self.assertEqual(a.use_robust_belief, kw.get("use_robust_belief", True))
            self.assertTrue(hasattr(a, "memory"))


if __name__ == "__main__":
    unittest.main()
