"""v0.9 Mechanism Lab environment validity tests (Phase B gates)."""
import unittest
import random

from lab_env import (
    ACTIONS, CONDITIONS, DOORS, DOOR_CELL, F1_TEMPLATES, F2_TEMPLATES, GOAL,
    H, MAX_STEPS, OBJECT_HOME, PLATE_CELL, PLATES, START, SWITCH_CELL,
    SWITCHES, TIGHT_STEPS, TEST_POOL, TRAIN_POOL, VAL_POOL, W, WALLS,
    LabConfig, LabWorld, Observation, StepResult, WS,
    door_condition_satisfied, eval_atoms, generate_worlds, make_state,
    public_next, sample_config, split_pools,
)

KEY_CFG = LabConfig(
    d1_cond=frozenset({("sw", "S1", True), ("load", "P1", "A")}),
    d2_cond=frozenset({("sw", "S2", False), ("load", "P2", "B")}),
    d1_delay=False, d2_delay=False,
)

# A configuration whose doors are both open-able from the left/start side
# (D1 by P1+A; D2 by S1 on) so the goal is reachable in a short plan.
EASY_CFG = LabConfig(
    d1_cond=frozenset({("load", "P1", "A")}),
    d2_cond=frozenset({("sw", "S1", True)}),
    d1_delay=False, d2_delay=False,
)


def plan_moves(world: LabWorld, path):
    """Execute a list of actions, returning (valid_all, final_obs)."""
    valid = True
    obs = None
    for a in path:
        r = world.step(a)
        valid = valid and r.valid
        obs = r.observation
    return valid, obs


class TopologyTests(unittest.TestCase):
    def test_world_dimensions_and_key_cells(self):
        self.assertEqual((W, H), (7, 5))
        self.assertIn(START, [(x, y) for x in range(W) for y in range(H)])
        self.assertIn(GOAL, [(x, y) for x in range(W) for y in range(H)])
        self.assertTrue(all(c not in WALLS for c in PLATE_CELL.values()))
        self.assertTrue(all(c not in WALLS for c in SWITCH_CELL.values()))
        self.assertTrue(all(c not in WALLS for c in DOOR_CELL.values()))
        self.assertTrue(all(c not in WALLS for c in OBJECT_HOME.values()))

    def test_both_doors_are_on_every_start_goal_route(self):
        """D1 and D2 must be required: removing either makes the goal
        unreachable under public physics alone."""
        from collections import deque

        def reachable(open_doors):
            start = WS(pos=START, objs=dict(OBJECT_HOME),
                       plate_load={p: "none" for p in PLATES},
                       switch_state={s: False for s in SWITCHES},
                       door_open={d: (d in open_doors) for d in DOORS},
                       armed={d: False for d in DOORS})
            seen = {start.pos}
            q = deque([start])
            while q:
                st = q.popleft()
                if st.pos == GOAL:
                    return True
                for a in ("N", "E", "S", "W"):
                    nxt = public_next(st, a)
                    if nxt is not None and nxt.pos not in seen:
                        seen.add(nxt.pos)
                        q.append(nxt)
            return False

        self.assertTrue(reachable({"D1", "D2"}))
        self.assertFalse(reachable(set()))
        self.assertFalse(reachable({"D1"}))
        self.assertFalse(reachable({"D2"}))

    def test_all_components_reachable_in_principle(self):
        """With both doors open, every plate / switch / door cell is
        reachable from start on foot. (Object home cells and P2 are only
        reachable by pushing, so they are covered by separate push tests.)"""
        from collections import deque
        open_all = {"D1", "D2"}
        start = WS(pos=START, objs=dict(OBJECT_HOME),
                   plate_load={p: "none" for p in PLATES},
                   switch_state={s: False for s in SWITCHES},
                   door_open={d: (d in open_all) for d in DOORS},
                   armed={d: False for d in DOORS})
        seen = {start.pos}
        q = deque([start])
        reachable_cells = {start.pos}
        while q:
            st = q.popleft()
            for a in ("N", "E", "S", "W"):
                nxt = public_next(st, a)
                if nxt is not None and nxt.pos not in seen:
                    seen.add(nxt.pos)
                    reachable_cells.add(nxt.pos)
                    q.append(nxt)
        required = [PLATE_CELL["P1"], SWITCH_CELL["S1"], SWITCH_CELL["S2"],
                    DOOR_CELL["D1"], DOOR_CELL["D2"]]
        for c in required:
            self.assertIn(c, reachable_cells)
        # P2 is loaded by pushing B; it is not walkable-into before that.
        self.assertNotIn(PLATE_CELL["P2"], reachable_cells)


class WorldMechanicsTests(unittest.TestCase):
    def test_move_and_blocked_walls(self):
        w = LabWorld(EASY_CFG)
        r = w.step("N")
        self.assertTrue(r.valid)
        self.assertEqual((r.observation.x, r.observation.y), (0, 3))
        # wall at (0,2)
        r = w.step("N")
        self.assertFalse(r.valid)
        self.assertEqual((r.observation.x, r.observation.y), (0, 3))

    def test_closed_door_blocks_movement(self):
        w = LabWorld(EASY_CFG)
        # go from (0,4) -> (1,4) -> (1,3) -> (1,2) -> (1,1) -> (2,1)
        for a in ("E", "N", "N", "N", "E"):
            r = w.step(a)
            self.assertTrue(r.valid, a)
        # at (2,1); try to move E onto D1 (3,1) while closed
        r = w.step("E")
        self.assertFalse(r.valid)
        self.assertEqual((r.observation.x, r.observation.y), (2, 1))

    def test_push_object_onto_plate_updates_load(self):
        w = LabWorld(EASY_CFG)
        # A at (2,3); stand at (2,4) and push N so A lands on P1 (2,2)
        for a in ("E", "E"):
            w.step(a)
        self.assertEqual((w.observe().x, w.observe().y), (2, 4))
        r = w.step("PushN")                # push A -> (2,2)=P1
        self.assertTrue(r.valid)
        self.assertEqual(r.observation.p1, "A")
        self.assertEqual((r.observation.ax, r.observation.ay), (2, 2))
        self.assertEqual((r.observation.x, r.observation.y), (2, 3))

    def test_push_object_off_plate_clears_load(self):
        w = LabWorld(EASY_CFG)
        for a in ("E", "E", "PushN"):
            w.step(a)
        self.assertEqual(w.observe().p1, "A")
        w.step("PushN")                     # A -> (2,1), off P1
        self.assertEqual(w.observe().p1, "none")
        self.assertEqual((w.observe().ax, w.observe().ay), (2, 1))

    def test_toggle_switch_requires_standing_on_it(self):
        w = LabWorld(EASY_CFG)
        r = w.step("Toggle")
        self.assertFalse(r.valid)
        # S1 at (0,1); from (0,4): N->(0,3), E->(1,3), N->(1,2), N->(1,1), W->(0,1)
        for a in ("N", "E", "N", "N", "W"):
            w.step(a)
        self.assertEqual((w.observe().x, w.observe().y), (0, 1))
        r = w.step("Toggle")
        self.assertTrue(r.valid)
        self.assertTrue(w.observe().s1)
        r = w.step("Toggle")
        self.assertFalse(w.observe().s1)

    def test_cannot_push_into_wall_or_out_of_bounds(self):
        # A at (2,3); from (1,3) pushing E would push A into wall (3,3)
        w = LabWorld(EASY_CFG)
        for a in ("N", "E"):
            w.step(a)
        self.assertEqual((w.observe().x, w.observe().y), (1, 3))
        r = w.step("PushE")
        self.assertFalse(r.valid)
        self.assertEqual((w.observe().ax, w.observe().ay), (2, 3))


class HiddenRuleTests(unittest.TestCase):
    def test_door_opens_when_atoms_satisfied(self):
        w = LabWorld(EASY_CFG)
        # satisfy D1: P1 loaded with A (stand at (2,4), push N)
        for a in ("E", "E", "PushN"):
            w.step(a)
        self.assertTrue(w.observe().d1)
        # D1's cell is (3,1); walk there and confirm it is traversable
        for a in ("W", "N", "N", "E", "E"):
            w.step(a)
        r = w.step("E")                     # through D1 into (4,1)
        self.assertTrue(r.valid)
        self.assertEqual((r.observation.x, r.observation.y), (4, 1))

    def test_door_closes_when_condition_lost(self):
        w = LabWorld(EASY_CFG)
        for a in ("E", "E", "PushN"):
            w.step(a)
        self.assertTrue(w.observe().d1)
        w.step("PushN")                     # A off plate -> (2,1)
        self.assertFalse(w.observe().d1)

    def test_delayed_effect_opens_one_step_later(self):
        cfg = LabConfig(d1_cond=frozenset({("load", "P1", "A")}),
                        d2_cond=frozenset({("sw", "S1", True)}),
                        d1_delay=True, d2_delay=False)
        w = LabWorld(cfg)
        for a in ("E", "E", "PushN"):
            w.step(a)
        # at end of the push step the condition holds; with delay the door
        # is still closed this step
        self.assertFalse(w.observe().d1)
        # any subsequent step while condition persists opens it
        w.step("W")
        self.assertTrue(w.observe().d1)

    def test_door_never_opens_on_wrong_atoms(self):
        cfg = LabConfig(d1_cond=frozenset({("sw", "S1", True), ("load", "P1", "A")}),
                        d2_cond=frozenset({("sw", "S2", False), ("load", "P2", "B")}),
                        d1_delay=False, d2_delay=False)
        w = LabWorld(cfg)
        # load P1 with A but never touch S1
        for a in ("E", "E", "PushN"):
            w.step(a)
        for _ in range(6):
            w.step("W" if w.observe().x > 0 else "E")
        self.assertFalse(w.observe().d1)

    def test_eval_atoms_semantics(self):
        st = make_state()
        st = st._replace(plate_load={"P1": "A", "P2": "none"},
                         switch_state={"S1": True, "S2": False})
        self.assertTrue(eval_atoms({("sw", "S1", True), ("load", "P1", "A")}, st))
        self.assertFalse(eval_atoms({("load", "P2", "A")}, st))
        self.assertTrue(eval_atoms({("loadany", "P1")}, st))
        self.assertFalse(eval_atoms({("loadany", "P2")}, st))


class IntegrityTests(unittest.TestCase):
    def test_observation_has_no_hidden_fields(self):
        w = LabWorld(EASY_CFG)
        obs = w.observe()
        self.assertIsInstance(obs, Observation)
        for field in ("config", "mode", "rule", "d1_cond", "d2_cond",
                      "seed", "noise_q"):
            self.assertFalse(hasattr(obs, field))

    def test_world_object_does_not_expose_config_to_policies(self):
        w = LabWorld(EASY_CFG)
        # the *environment* holds a config, but the public observation and
        # step result never carry it
        self.assertTrue(hasattr(w, "_config"))
        obs = w.observe()
        self.assertNotIn("_config", obs.__class__.__dict__ if hasattr(obs.__class__, "__dict__") else {})
        for attr in ("_config", "_noise_q", "_ep_seed"):
            self.assertFalse(hasattr(obs, attr))

    def test_deterministic_replay_same_seed(self):
        def run(seed):
            w = LabWorld(EASY_CFG, ep_seed=seed)
            trace = []
            for a in ("E", "E", "S", "S", "N", "PushS", "E", "E", "Toggle"):
                r = w.step(a)
                trace.append((r.valid, r.observation))
            return trace
        self.assertEqual(run(7), run(7))

    def test_worlds_fixed_different_seeds(self):
        a = generate_worlds("stable", 22, 40)
        b = generate_worlds("stable", 23, 40)
        # same length; configurations themselves will differ
        self.assertEqual(len(a), len(b))
        self.assertEqual(a, generate_worlds("stable", 22, 40))

    def test_noisy_readings_matched_across_policies(self):
        """Two worlds with the same episode seed must produce *identical*
        corrupted observations at the same step index, even under different
        action histories."""
        spec = generate_worlds("noisy_obs", 5, 6)[0]
        w1 = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q)
        w2 = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q)
        # w1 takes two steps; w2 takes two different steps; both end at
        # step index 2, so readings at step 2 must be identical
        w1.step("N")
        w1.step("E")
        w2.step("S")
        w2.step("W")
        self.assertEqual(w1.steps, 2)
        self.assertEqual(w2.steps, 2)
        # readings keyed by step index only => identical. Positions differ
        # (they are true state, never corrupted), so compare readings only.
        o1, o2 = w1.observe(), w2.observe()
        for field in ("s1", "s2", "p1", "p2", "d1", "d2"):
            self.assertEqual(getattr(o1, field), getattr(o2, field), field)

    def test_observation_noise_respects_episode_seed(self):
        spec = generate_worlds("noisy_obs", 5, 6)[0]
        w0 = LabWorld(spec.config, ep_seed=spec.seed, noise_q=0.0)
        w1 = LabWorld(spec.config, ep_seed=spec.seed, noise_q=spec.noise_q)
        r0 = w0.step("N")
        r1 = w1.step("N")
        self.assertEqual(r0.observation.x, r1.observation.x)  # pos not corrupted
        # switch/plate/door readings are corrupted w.p. q, so they may
        # differ; but both are valid booleans/strings
        for field in ("s1", "s2", "d1", "d2", "p1", "p2"):
            self.assertIn(getattr(r1.observation, field),
                          (True, False) if field in ("s1", "s2", "d1", "d2")
                          else ("none", "A", "B"))

    def test_train_val_test_pools_disjoint(self):
        train, val, test = TRAIN_POOL, VAL_POOL, TEST_POOL
        self.assertGreaterEqual(len(train), 100)
        self.assertGreaterEqual(len(val), 30)
        self.assertGreaterEqual(len(test), 30)
        st, sv, ss = set(train), set(val), set(test)
        self.assertFalse(st & sv)
        self.assertFalse(st & ss)
        self.assertFalse(sv & ss)

    def test_conditions_generate_sane_worlds(self):
        for cond in CONDITIONS:
            worlds = generate_worlds(cond, 11, 80)
            self.assertEqual(len(worlds), 80)
            for spec in worlds:
                self.assertIsInstance(spec.config, LabConfig)
                self.assertIn(spec.config.d1_cond, F1_TEMPLATES)
                self.assertIn(spec.config.d2_cond, F2_TEMPLATES)
                self.assertGreaterEqual(spec.max_steps, TIGHT_STEPS)

    def test_shift_changes_wiring_at_midpoint(self):
        worlds = generate_worlds("shift", 5, 80)
        early = worlds[10].config
        late = worlds[70].config
        self.assertNotEqual(early, late)

    def test_novel_uses_held_out_pool_in_second_half(self):
        worlds = generate_worlds("novel", 5, 80)
        first, second = worlds[10].config, worlds[70].config
        self.assertIn(first, TRAIN_POOL)
        self.assertIn(second, TEST_POOL)

    def test_goal_reachable_with_both_doors_open(self):
        w = LabWorld(EASY_CFG)
        # open D1: stand at (2,4), push A onto P1
        for a in ("E", "E", "PushN"):
            w.step(a)
        self.assertTrue(w.observe().d1)
        # walk to D1 (3,1) and cross it: (2,3)->(1,3)->(1,2)->(1,1)->(2,1)->(3,1)->(4,1)
        for a in ("W", "N", "N", "E", "E", "E"):
            w.step(a)
        self.assertEqual((w.observe().x, w.observe().y), (4, 1))
        # go west to S1 and toggle it -> D2 opens: (4,1)->(3,1)->(2,1)->(1,1)->(0,1)
        for a in ("W", "W", "W", "W", "Toggle"):
            w.step(a)
        self.assertTrue(w.observe().s1)
        self.assertTrue(w.observe().d2)
        # cross D2 and reach goal: (0,1)->(1,1)->(2,1)->(3,1)->(4,1)->(4,2)->(5,2)->(6,2)->(6,1)->(6,0)
        for a in ("E", "E", "E", "E", "S", "E", "E", "N", "N"):
            w.step(a)
        self.assertTrue(w.observe().finished)


if __name__ == "__main__":
    unittest.main()
