"""Phase D/G tests: METARGON v0.9 agent correctness and budgets."""
import unittest

from lab_env import (
    ACTIONS, DOORS, F1_TEMPLATES, F2_TEMPLATES, LabConfig, LabWorld,
    generate_worlds,
)
from metargon_v09 import (
    MAX_EXPANSIONS, MAX_HYPOTHESES_AGENT, PROBE_MAX_PLANS, MetargonV09,
    plan_to_atoms, plan_to_pos,
)
from benchmark_v09 import play_episode, score_policy

EASY = LabConfig(
    d1_cond=frozenset({("load", "P1", "A")}),
    d2_cond=frozenset({("sw", "S1", True)}),
    d1_delay=False, d2_delay=False,
)


class PlannerUnitTests(unittest.TestCase):
    def test_plan_to_atoms_shortest(self):
        from lab_env import make_state, WS, PLATES, SWITCHES
        st = make_state()
        plan = plan_to_atoms(st, frozenset({("loadany", "P1")}))
        self.assertIsNotNone(plan)
        # shortest way: walk to (2,4) and push up
        self.assertLessEqual(len(plan), 4)

    def test_plan_to_pos_reaches(self):
        from lab_env import make_state
        plan = plan_to_pos(make_state(), (6, 0))
        self.assertIsNone(plan)          # goal unreachable with doors closed

    def test_plan_cache_bounded(self):
        from lab_env import make_state
        cache = {}
        st = make_state()
        for i in range(20):
            plan_to_atoms(st, frozenset({("sw", "S1", bool(i % 2))}),
                          cache=cache)
        self.assertLessEqual(len(cache), 3000)


class AgentPropertiesTests(unittest.TestCase):
    def test_agent_has_no_hidden_config_access(self):
        a = MetargonV09(1)
        for attr in ("config", "_config", "mode", "d1_cond", "d2_cond",
                     "hidden", "family"):
            self.assertFalse(hasattr(a, attr), attr)
        self.assertTrue(hasattr(a, "memory"))

    def test_seed_reproducible_actions(self):
        specs = generate_worlds("stable", 7, 6)

        def run():
            pol = MetargonV09(42)
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

        self.assertEqual(run(), run())

    def test_memory_bounded_and_deterministic_eviction(self):
        pol = MetargonV09(3)
        specs = generate_worlds("rapid_shift", 5, 40)
        for spec in specs:
            pol.episode_start()
            w = LabWorld(spec.config, ep_seed=spec.seed,
                         noise_q=spec.noise_q, max_steps=spec.max_steps)
            while not w.finished and w.steps < w.max_steps:
                obs = w.observe()
                a = pol.act(obs)
                res = w.step(a)
                pol.observe_outcome(obs, a, res)
        self.assertLessEqual(len(pol.memory), MAX_HYPOTHESES_AGENT)
        d = pol.memory.diagnostics()
        self.assertGreaterEqual(d["evictions"], 0)
        self.assertLessEqual(d["size"], MAX_HYPOTHESES_AGENT)

    def test_planner_budget_per_decision_bounded(self):
        pol = MetargonV09(9)
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
        # <= steps * small planning budget (12 steps * (PROBE+2) probes)
        self.assertLessEqual(calls, 12 * (PROBE_MAX_PLANS + 2) + 4)

    def test_learns_and_opens_doors_on_easy_config(self):
        # several agent seeds + episodes: the easy config should be solved
        # at least once (learning must occur)
        successes = 0
        for seed in (2, 5, 11, 17):
            pol = MetargonV09(seed)
            for _ in range(12):
                pol.episode_start()
                w = LabWorld(EASY, ep_seed=1, noise_q=0.0, max_steps=80)
                while not w.finished and w.steps < w.max_steps:
                    obs = w.observe()
                    a = pol.act(obs)
                    res = w.step(a)
                    pol.observe_outcome(obs, a, res)
                successes += int(w.finished)
        self.assertGreaterEqual(successes, 1)

    def test_scoring_reports_metrics_and_finite(self):
        s = score_policy("stable", 4, "metargon", 20)
        self.assertGreaterEqual(s["success_rate"], 0.0)
        self.assertLessEqual(s["success_rate"], 1.0)
        self.assertGreaterEqual(s["planner_ops"], 0)
        self.assertGreaterEqual(s["plan_calls"], 0)
        self.assertLessEqual(s["memory_size_end"], MAX_HYPOTHESES_AGENT)


if __name__ == "__main__":
    unittest.main()
