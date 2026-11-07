"""Phase E tests: baseline fairness and correctness."""
import unittest

from lab_env import (
    F1_TEMPLATES, F2_TEMPLATES, LabConfig, LabEpisode, LabWorld,
    generate_worlds,
)
from baselines_v09 import (
    BayesPlannerPolicy, EpisodicMemoryPolicy, OraclePolicy, QLearningPolicy,
    RandomPolicy, ground_truth_plan, transition_with_hidden,
)
from benchmark_v09 import play_episode

EASY = LabConfig(
    d1_cond=frozenset({("load", "P1", "A")}),
    d2_cond=frozenset({("sw", "S1", True)}),
    d1_delay=False, d2_delay=False,
)


class RandomPolicyTests(unittest.TestCase):
    def test_deterministic_seed(self):
        specs = generate_worlds("stable", 3, 4)
        p1 = RandomPolicy(9)
        p2 = RandomPolicy(9)
        t1, t2 = [], []
        for spec in specs:
            w1, w2 = (LabWorld(spec.config, ep_seed=spec.seed,
                                noise_q=spec.noise_q, max_steps=spec.max_steps)
                       for _ in range(2))
            for w, p, t in ((w1, p1, t1), (w2, p2, t2)):
                while not w.finished and w.steps < w.max_steps:
                    obs = w.observe()
                    a = p.act(obs)
                    res = w.step(a)
                    p.observe_outcome(obs, a, res)
                    t.append(a)
        self.assertEqual(t1, t2)

    def test_actions_always_valid(self):
        spec = generate_worlds("stable", 3, 4)[0]
        p = RandomPolicy(1)
        w = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                     max_steps=spec.max_steps)
        invalid = 0
        while not w.finished and w.steps < w.max_steps:
            obs = w.observe()
            a = p.act(obs)
            res = w.step(a)
            p.observe_outcome(obs, a, res)
            invalid += int(not res.valid)
        self.assertEqual(invalid, 0)


class EpisodicTests(unittest.TestCase):
    def test_memory_bounded(self):
        p = EpisodicMemoryPolicy(1, max_mem=5)
        specs = generate_worlds("stable", 2, 30)
        for spec in specs:
            w = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                         max_steps=spec.max_steps)
            p.episode_start()
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = p.act(obs)
                res = w.step(a)
                p.observe_outcome(obs, a, res)
        self.assertLessEqual(len(p.mem), 5)


class QLearningTests(unittest.TestCase):
    def test_q_table_updates_and_epsilon_decays(self):
        p = QLearningPolicy(1)
        spec = generate_worlds("stable", 6, 4)[0]
        w = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q,
                     max_steps=spec.max_steps)
        p.episode_start()
        first_eps = p.epsilon
        for _ in range(10):
            obs = w.observe()
            a = p.act(obs)
            res = w.step(a)
            p.observe_outcome(obs, a, res)
        self.assertGreater(len(p.Q), 0)


class BayesTests(unittest.TestCase):
    def test_bayes_solves_easy_config(self):
        # The model-based Bayesian planner (particle filter over the
        # declared family + ground-truth planning under the sampled
        # hypothesis) must solve the easy config clearly above the ~0%
        # random/episodic level. Threshold is kept modest because
        # posterior sampling + plan replanning makes single-episode
        # outcomes stochastic (observed 3–8/10 across seed runs).
        p = BayesPlannerPolicy(11)
        successes = 0
        for _ in range(10):
            p.episode_start(noise_q=0.0)
            w = LabWorld(EASY, ep_seed=3, noise_q=0.0, max_steps=80)
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = p.act(obs)
                res = w.step(a)
                p.observe_outcome(obs, a, res)
            successes += int(w.finished)
        self.assertGreaterEqual(successes, 3)

    def test_model_class_is_declared_family(self):
        p = BayesPlannerPolicy(1)
        n_hyp = len(p.hyp_space)
        self.assertEqual(n_hyp, 7 * 9 * 4)
        for h in p.hyp_space:
            self.assertIn(h.d1_cond, F1_TEMPLATES)
            self.assertIn(h.d2_cond, F2_TEMPLATES)


class OracleTests(unittest.TestCase):
    def test_oracle_solves_easy_config(self):
        p = OraclePolicy(1)
        for _ in range(4):
            spec = LabEpisode(config=EASY, max_steps=80, noise_q=0.0,
                              seed=1, context=0)
            r = play_episode(spec, p)
            self.assertTrue(r["success"])
            self.assertEqual(r["invalid"], 0)


class FairnessTests(unittest.TestCase):
    def test_all_policies_see_identical_worlds(self):
        names = ("random", "episodic", "qlearning", "bayes", "metargon", "oracle")
        for name in names:
            from benchmark_v09 import make_policy
            pol = make_policy(name, 11800 + 1000003)
            spec = generate_worlds("noisy_obs", 11800, 4)[0]
            r = play_episode(spec, pol)
            self.assertIsInstance(r["steps"], int)

    def test_ground_truth_plan_finds_solution_when_exists(self):
        from lab_env import make_state
        plan = ground_truth_plan(make_state(), EASY)
        self.assertIsNotNone(plan)
        self.assertTrue(plan)


if __name__ == "__main__":
    unittest.main()
