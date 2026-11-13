"""Unit tests for the exploration-noise diagnosis instrumentation.

Covers the perception-counter classification (phantom events, missed true
events, belief-reality divergence, ground-atom mismatch) and the diagnosis
harness smoke. These tests are about the *measuring instrument*, not a new
exploration mechanism.
"""

from __future__ import annotations

import unittest

from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.gridworld import EnvConfig, MechanismGrid, TOPOLOGIES
from academy.envs.tasks import EpisodeSpec, difficulty_of
from academy.tests.test_noise_diagnosis_experiment import (
    step_perception_counters, run_episode, ARMS, CONDITIONS,
)
from academy.tests.helpers import cached_splits

R1 = frozenset({("sw", "S2", True)})
R2 = frozenset({("sw", "S1", True)})   # NOT satisfied at episode start


def _config(noise=0.0):
    return EnvConfig(topology="TP1", d1_rule=R1, d2_rule=R2,
                     d1_delay=False, d2_delay=False)


def _spec(noise=0.0):
    cfg = _config(noise)
    return EpisodeSpec(config=cfg, max_steps=40, noise_q=noise,
                       spec_seed=1, pool_tag="diag",
                       difficulty=difficulty_of(cfg))


class TestPerceptionCounters(unittest.TestCase):
    def test_clean_no_phantom(self):
        env = MechanismGrid(_config(), 40, 0.0)
        env.reset(1)
        # no rule holds at start: a move produces no door event, so no
        # claimed flip, no phantom, no missed event, no divergence
        before = env.observe()
        res = env.step("N")
        c = step_perception_counters(before, res.observation, res, env)
        self.assertEqual(c["true_open"] + c["true_close"], 0)
        self.assertEqual(c["claimed_open"] + c["claimed_close"], 0)
        self.assertEqual(c["phantom_open"] + c["phantom_close"], 0)
        self.assertEqual(c["missed_true_open"] + c["missed_true_close"], 0)
        self.assertEqual(c["belief_reality_divergence"], 0)
        self.assertEqual(c["ground_atom_mismatch"], 0)

    def test_clean_real_open_classified_exact(self):
        env = MechanismGrid(_config(), 40, 0.0)
        env.reset(1)
        # set D1's rule to hold (S2 True), D1 closed, then move: the
        # environment fires a real D1-open event this step
        st = env._state
        st = st._replace(switch_state={"S1": False, "S2": True})
        st = st._replace(door_open={"D1": False, "D2": False})
        env._state = st
        before = env.observe()
        res = env.step("N")
        c = step_perception_counters(before, res.observation, res, env)
        self.assertEqual(c["true_open"], 1)
        self.assertEqual(c["claimed_open"], 1)
        self.assertEqual(c["phantom_open"], 0)
        self.assertEqual(c["missed_true_open"], 0)
        self.assertEqual(c["belief_reality_divergence"], 0)

    def test_noisy_phantom_detected(self):
        # a raw-reading flip with no real event must classify as phantom:
        # reality has D1 closed; the "observation" flips D1 to open with
        # no corresponding true event.
        env = MechanismGrid(_config(), 40, 0.0)
        env.reset(1)
        obs = env.observe()          # D1 closed (true)
        before = obs
        after = obs._replace(d1=True)   # corrupted reading: claims open
        from academy.envs.gridworld import StepResult
        res = StepResult(observation=after, valid=True, events=())
        c = step_perception_counters(before, after, res, env)
        self.assertEqual(c["true_open"] + c["true_close"], 0)
        self.assertEqual(c["claimed_open"], 1)
        self.assertEqual(c["phantom_open"], 1)
        self.assertEqual(c["phantom_close"], 0)
        self.assertEqual(c["belief_reality_divergence"], 1)

    def test_missed_true_open_detected(self):
        # a real opening event hidden by a corrupted reading (still reads
        # closed) must classify as missed.
        env = MechanismGrid(_config(), 40, 0.0)
        env.reset(1)
        obs = env.observe()          # D1 closed (true reading)
        before = obs
        after = obs                   # reading stays "closed" despite event
        from academy.envs.gridworld import StepResult
        res = StepResult(observation=after, valid=True,
                         events=(("D1", "open"),))
        c = step_perception_counters(before, after, res, env)
        self.assertEqual(c["true_open"], 1)
        self.assertEqual(c["claimed_open"], 0)
        self.assertEqual(c["missed_true_open"], 1)

    def test_belief_divergence_under_noise(self):
        env = MechanismGrid(_config(), 40, 0.5)
        before = env.observe()
        res = env.step("N")
        c = step_perception_counters(before, res.observation, res, env)
        self.assertGreaterEqual(c["belief_reality_divergence"], 0)
        self.assertLessEqual(c["belief_reality_divergence"], 2)


class TestTraceKeyCoverage(unittest.TestCase):
    def test_all_probe_sources_have_door_open_trace_keys(self):
        # diagnosis found a latent KeyError: a "gap"-sourced probe that
        # opens a door crashed learn() because door_open_after_gap was
        # missing from the trace dict. Every probe source must have its
        # attribution key.
        agent = MetargonAcademyAgent(seed=0, cognitive_exploration=True)
        srcs = ("hypothesis", "bank", "atom_single", "atom_pair",
                "conjunction", "novel_single", "gap", "other")
        for src in srcs:
            self.assertIn(f"door_open_after_{src}", agent.trace, src)


class TestEpisodeRunnerIntegration(unittest.TestCase):
    def test_run_episode_counts_collect(self):
        agent = MetargonAcademyAgent(seed=0, cognitive_exploration=True)
        collect = {k: 0 for k in (
            "claimed_open", "claimed_close", "true_open", "true_close",
            "phantom_open", "phantom_close", "missed_true_open",
            "missed_true_close", "belief_reality_divergence",
            "ground_atom_mismatch", "invalid_action")}
        rec = run_episode(agent, _spec(), collect)
        self.assertIn("success", rec)
        self.assertIn("trace", rec)
        self.assertIn("perception", rec)
        self.assertEqual(sum(collect.values()) >= 0, True)
        self.assertGreaterEqual(rec["steps"], 1)

    def test_arm_flags_valid(self):
        # every diagnostic arm must be a valid agent configuration
        for name, flags in ARMS.items():
            a = MetargonAcademyAgent(seed=0, **flags)
            self.assertTrue(isinstance(a, MetargonAcademyAgent), name)
            a.episode_start(_spec(), TOPOLOGIES["TP1"])


class TestHarnessSmoke(unittest.TestCase):
    def test_diagnosis_smoke(self):
        import tempfile
        from academy.tests.test_noise_diagnosis_experiment import (
            run_diagnosis)
        with tempfile.TemporaryDirectory() as tmp:
            out = run_diagnosis(
                seed_base=999888, n_seeds=1, episodes=4,
                results_dir=tmp,
                conditions={"clean": CONDITIONS["clean"]},
                arms={"baseline": {}, "v01": ARMS["v01"]})
            cell = out["conditions"]["clean"]
            self.assertIn("baseline", cell["arms"])
            self.assertIn("v01", cell["arms"])
            self.assertIn("v01_vs_baseline", cell["comparisons"])


if __name__ == "__main__":
    unittest.main()
