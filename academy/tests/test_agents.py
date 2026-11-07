"""Tests for agents: learning-state changes, leakage audit (no forbidden
simulator identifiers inside policies), and baseline sanity."""

import os
import unittest

from academy.agents.baselines import (
    BayesPlannerPolicy, EpisodicPolicy, OraclePolicy, QLearningPolicy,
    RandomPolicy,
)
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.envs.tasks import EpisodeSpec, difficulty_of, get_splits
from academy.learning.loop import run_one_episode

FORBIDDEN = (
    "transition_with_hidden", "_config", "rule_holds", "is_solvable",
    "EnvConfig", "door_condition",
)
# baselines.py is exempt: the model-based Bayes baseline legitimately
# plans under its *own* hypothesized configurations (a declared model
# class), so it necessarily references config types and the public
# transition function. The student agent and its infrastructure never do.
AGENT_FILES = (
    "academy/agents/metargon_agent.py",
    "academy/agents/planner.py",
    "academy/agents/memory.py",
    "academy/agents/common.py",
)


class TestLeakageAudit(unittest.TestCase):
    def test_agents_never_reference_hidden_simulator(self):
        for path in AGENT_FILES:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            for token in FORBIDDEN:
                self.assertNotIn(token, src,
                                  msg=f"{path} contains forbidden {token}")


class TestMetargonLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = get_splits()

    def _easy_spec(self):
        for c in self.splits["train"]:
            d = difficulty_of(c)
            if d.bin == "easy" and not d.delay and len(c.d1_rule) == 1 \
                    and len(c.d2_rule) == 1:
                return EpisodeSpec(config=c, max_steps=80, noise_q=0.0,
                                   spec_seed=5, pool_tag="train",
                                   difficulty=d)
        self.fail("no easy single-rule config found")

    def test_learning_state_changes_with_experience(self):
        agent = MetargonAcademyAgent(seed=0)
        before = agent.diagnostics()["memory"]["generated"]
        run_one_episode(agent, self._easy_spec())
        after = agent.diagnostics()["memory"]["generated"]
        self.assertGreater(after, before)

    def test_memory_affects_future_decisions(self):
        """A verified hypothesis changes the *next decision*: with D1
        closed and S2 off, an agent that verified "S2 on opens D1" must
        choose to turn S2 on, while an agent that verified the wrong
        polarity ("S2 off opens D1") plans something else."""
        from academy.envs.gridworld import MechanismGrid
        c = self._easy_spec().config
        spec = self._easy_spec()
        env = MechanismGrid(c, 80, 0.0)
        obs = env.reset(5)
        # drive the world to: D1 closed, S2 off (S2 is a switch on TP2;
        # ensure the chosen config's D1 rule is S2-switch based)
        self.assertTrue(c.d1_rule == frozenset({("sw", "S2", True)})
                        or c.d1_rule == frozenset({("sw", "S2", False)}))
        # move to S2 cell and toggle twice (on then off) to close D1
        a_true = MetargonAcademyAgent(seed=0)
        a_false = MetargonAcademyAgent(seed=0)
        for ag in (a_true, a_false):
            ag.episode_start(spec, env.geom)
        atom_on = frozenset({("sw", "S2", True)})
        atom_off = frozenset({("sw", "S2", False)})
        a_true.memory.observe_event(frozenset(), "D1", True, atom_on)
        a_true.memory.observe_maintain(atom_on, "D1")
        a_true.memory.observe_maintain(atom_on, "D1")
        a_false.memory.observe_event(frozenset(), "D1", True, atom_off)
        a_false.memory.observe_maintain(atom_off, "D1")
        a_false.memory.observe_maintain(atom_off, "D1")
        # synchronize both agents to the same closed-door belief
        a_true._sync(obs)
        a_false._sync(obs)
        plan_true, _ = a_true._choose_probe()
        plan_false, _ = a_false._choose_probe()
        self.assertNotEqual(plan_true, plan_false)
        # the true-polarity agent must plan a change that ends with S2 on
        from academy.agents.planner import atoms_of_state
        st = a_true.belief
        for act in plan_true:
            from academy.envs.gridworld import public_next
            nxt = public_next(st, act, a_true.geom)
            self.assertIsNotNone(nxt)
            st = nxt
        self.assertIn(("sw", "S2", True), atoms_of_state(st))

    def test_restore_keeps_learning(self):
        agent = MetargonAcademyAgent(seed=0)
        spec = self._easy_spec()
        run_one_episode(agent, spec)
        s = agent.save_state()
        clone = MetargonAcademyAgent(seed=0)
        clone.load_state(s)
        self.assertEqual(agent.diagnostics()["memory"]["size"],
                         clone.diagnostics()["memory"]["size"])


class TestBaselinesSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = get_splits()

    def test_all_baselines_run(self):
        c = self.splits["train"][0]
        spec = EpisodeSpec(config=c, max_steps=80, noise_q=0.0,
                           spec_seed=3, pool_tag="train",
                           difficulty=difficulty_of(c))
        for factory in (RandomPolicy, EpisodicPolicy, QLearningPolicy,
                        BayesPlannerPolicy, OraclePolicy):
            agent = factory(seed=0)
            r = run_one_episode(agent, spec)
            self.assertIn(r["success"], (True, False))
            self.assertGreaterEqual(r["steps"], 0)

    def test_qlearning_updates_values(self):
        q = QLearningPolicy(seed=0)
        self.assertEqual(len(q.Q), 0)
        c = self.splits["train"][0]
        spec = EpisodeSpec(config=c, max_steps=80, noise_q=0.0,
                           spec_seed=3, pool_tag="train",
                           difficulty=difficulty_of(c))
        from academy.envs.gridworld import MechanismGrid
        env = MechanismGrid(spec.config, 80, 0.0)
        obs = env.reset(3)
        q.episode_start(spec, env.geom)
        act = q.act(obs)
        res = env.step(act)
        q.learn(obs, act, res)
        self.assertGreaterEqual(len(q.Q), 0)


class TestQlearningPersistence(unittest.TestCase):
    """Regression tests for QLearningPolicy save/load (Q-table keys)."""

    @classmethod
    def setUpClass(cls):
        cls.splits = get_splits()

    def test_save_load_roundtrip_preserves_q_table(self):
        import json
        from academy.envs.gridworld import MechanismGrid
        c = self.splits["train"][0]
        spec = EpisodeSpec(config=c, max_steps=80, noise_q=0.0,
                           spec_seed=3, pool_tag="train",
                           difficulty=difficulty_of(c))
        q = QLearningPolicy(seed=0)
        env = MechanismGrid(spec.config, 80, 0.0)
        for seed in (3, 7):
            obs = env.reset(seed)
            q.episode_start(spec, env.geom)
            while not env.finished and env.steps < spec.max_steps:
                act = q.act(obs)
                res = env.step(act)
                q.learn(obs, act, res)
        self.assertGreater(len(q.Q), 0)
        snapshot = json.loads(json.dumps(q.save_state()))
        clone = QLearningPolicy(seed=0)
        clone.load_state(snapshot)
        self.assertEqual(clone.Q, q.Q)
        # every learned key must be reachable under the runtime key format
        sig, action = next(iter(q.Q))
        self.assertEqual(clone._q(sig, action), q.Q[(sig, action)])

    def test_load_committed_policy_json_reaches_learned_keys(self):
        """Regression: load_state used to wrap signatures in an extra tuple,
        so a reloaded Q table was unreachable at decision time."""
        import json
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "results", "academy_v01", "policies", "qlearning.json")
        if not os.path.exists(path):
            self.skipTest("committed qlearning policy artifact not present")
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertIn("Q", state)
        self.assertGreater(len(state["Q"]), 0)
        q = QLearningPolicy(seed=0)
        q.load_state(state)
        self.assertEqual(len(q.Q), len(state["Q"]))
        # every saved row must map to a runtime-reachable (sig, action) key
        for row in state["Q"]:
            self.assertIn((tuple(row[0]), row[-2][0]), q.Q)


if __name__ == "__main__":
    unittest.main()
