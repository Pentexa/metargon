"""v0.8 regression and validity checks, using only the standard library."""
import unittest

from metargon_world import (
    CONDITIONS, MAX_STEPS, Observation, Policy, State, Workshop,
    benchmark, deterministic_transition, generate_worlds, plan_to_gate,
    play_episode, score_policy,
)


def uniform_world(mode='key', key_random=0.5, switch_random=0.5, sham=False):
    return Workshop(mode, {'key': (key_random,) * 4,
                           'switch': (switch_random,) * 4}, sham=sham)


class WorldTests(unittest.TestCase):
    def test_missing_key_and_switch_dont_open_gate(self):
        world = uniform_world()
        for _ in range(3):
            world.step('right')
        result = world.step('open')
        self.assertFalse(result.valid)
        self.assertIsNone(result.gate_result)
        self.assertFalse(result.observation.gate_open)

    def test_door_blocks_right_move_before_success(self):
        world = uniform_world()
        for _ in range(3):
            world.step('right')
        result = world.step('right')
        self.assertFalse(result.valid)
        self.assertEqual(result.observation.position, 3)

    def test_key_recipe_opens_key_gate(self):
        world = uniform_world(mode='key')
        trace = []
        for a in plan_to_gate(world.observe(), 'key') + ['open', 'right']:
            trace.append(world.step(a))
        self.assertTrue(trace[-1].observation.finished)
        self.assertEqual(trace[-2].gate_result, True)
        self.assertEqual(world.steps, 6)

    def test_wrong_recipe_does_not_open_when_random_is_high(self):
        world = uniform_world(mode='key')
        for a in plan_to_gate(world.observe(), 'switch'):
            world.step(a)
        self.assertFalse(world.step('open').gate_result)
        self.assertFalse(world.finished)

    def test_wrong_recipe_can_be_recovered_from_state(self):
        world = uniform_world(mode='switch')
        for a in plan_to_gate(world.observe(), 'key'):
            self.assertTrue(world.step(a).valid)
        self.assertFalse(world.step('open').gate_result)
        for a in plan_to_gate(world.observe(), 'switch'):
            self.assertTrue(world.step(a).valid)
        self.assertTrue(world.step('open').gate_result)
        self.assertTrue(world.step('right').observation.finished)
        self.assertLessEqual(world.steps, MAX_STEPS)

    def test_transition_model_only_known_public_actions(self):
        self.assertEqual(deterministic_transition(State(0, False, False), 'right'),
                         State(1, False, False))
        self.assertIsNone(deterministic_transition(State(0, False, False), 'open'))

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            uniform_world().step('jump')

    def test_bounded_steps(self):
        world = uniform_world()
        for _ in range(MAX_STEPS):
            world.step('left')
        with self.assertRaises(RuntimeError):
            world.step('left')

    def test_observation_is_read_only_and_has_no_mode(self):
        obs = uniform_world().observe()
        self.assertIsInstance(obs, Observation)
        self.assertFalse(hasattr(obs, 'mode'))
        with self.assertRaises(AttributeError):
            obs.position = 10

    def test_no_impossible_route(self):
        with self.assertRaises(ValueError):
            plan_to_gate(uniform_world().observe(), 'neither')


class LearningTests(unittest.TestCase):
    def test_belief_updates_only_after_gate_feedback(self):
        agent = Policy('metargon', 5)
        agent.start(0)
        before = agent.posteriors[0]
        self.assertEqual(agent.learning_events, 0)
        a = agent.choose(0)
        self.assertEqual(agent.posteriors[0], before)
        prediction = agent.predict(0, a)
        agent.feedback(0, a, False, prediction)
        self.assertEqual(agent.learning_events, 1)
        self.assertNotEqual(agent.posteriors[0], before)

    def test_predict_returns_probabilities(self):
        for policy in ('bayes', 'metargon'):
            a = Policy(policy, 1)
            a.start(2)
            for recipe in ('key', 'switch'):
                self.assertTrue(0.0 <= a.predict(2, recipe) <= 1.0)

    def test_context_posteriors_separate(self):
        a = Policy('metargon', 1)
        a.start(1)
        pred = a.predict(1, 'key')
        a.feedback(1, 'key', True, pred)
        self.assertNotEqual(a.posteriors[1], 0.5)
        self.assertEqual(a.posteriors[0], 0.5)
        self.assertEqual(a.posteriors[2], 0.5)

    def test_surprise_can_change_belief(self):
        a = Policy('metargon', 1)
        a.posteriors[0] = 0.0001
        a.start(0)
        p = a.predict(0, 'key')
        a.feedback(0, 'key', False, p)
        self.assertGreater(a.surprises[0], 0)

    def test_paired_world_generation_is_deterministic(self):
        for condition in CONDITIONS:
            self.assertEqual(generate_worlds(condition, 123, 80),
                             generate_worlds(condition, 123, 80))

    def test_changing_rule_is_hidden_not_missing(self):
        seq = generate_worlds('shift', 5, 40)
        early = {i: set() for i in range(3)}
        late = {i: set() for i in range(3)}
        for idx, spec in enumerate(seq):
            (early if idx < 20 else late)[spec.context].add(spec.mode)
        for idx in range(3):
            if early[idx] and late[idx]:
                self.assertNotEqual(early[idx], late[idx])

    def test_novel_context_appears_only_after_training(self):
        worlds = generate_worlds('novel', 10, 80)
        self.assertTrue(all(w.context != 2 for w in worlds[:40]))
        self.assertTrue(all(w.context == 2 for w in worlds[40:]))

    def test_privileged_oracle_explicitly_separated(self):
        specs = generate_worlds('stable', 2, 80)
        self.assertTrue(all(s.mode in ('key','switch') for s in specs))
        non_privileged = Policy('metargon', 3)
        self.assertFalse(hasattr(non_privileged, '_mode'))
        result = play_episode(specs[0], non_privileged)
        self.assertEqual(result['invalid'], 0)
        self.assertLessEqual(result['steps'], MAX_STEPS)

    def test_scoring_has_no_invalid_actions(self):
        for name in ('metargon', 'bayes', 'win_stay', 'always_key'):
            score = score_policy('shift', 7, name, 40)
            self.assertEqual(score['invalid_actions'], 0)
            self.assertGreaterEqual(score['success_rate'], 0)

    def test_tight_budget_changes_success_metric(self):
        specs = generate_worlds('tight_budget', 15, 40)
        self.assertTrue(all(spec.max_steps == 7 for spec in specs))
        score = score_policy('tight_budget', 15, 'metargon', 40)
        self.assertLess(score['mean_steps'], 7.01)

    def test_noise_gate_misses_model_assumptions(self):
        specs = generate_worlds('noisy_gate', 15, 40)
        self.assertTrue(all(abs(spec.match_success - 0.72) < 1e-12 for spec in specs))
        self.assertTrue(all(abs(spec.mismatch_success - 0.28) < 1e-12 for spec in specs))

    def test_benchmark_repeatable(self):
        a = benchmark(2, 12)
        b = benchmark(2, 12)
        self.assertEqual(a, b)

    def test_score_rates_are_finite(self):
        b = benchmark(1, 12)
        for row in b['summary'].values():
            for key in ('metargon', 'bayes'):
                self.assertGreaterEqual(row[key]['success_rate'], 0)
                self.assertLessEqual(row[key]['success_rate'], 1)
                self.assertLessEqual(row[key]['mean_steps'], MAX_STEPS)


if __name__ == '__main__':
    unittest.main()
