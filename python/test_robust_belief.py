"""v0.10 robust belief tests (Phase G)."""
import unittest

from lab_env import LabConfig, LabWorld, F1_TEMPLATES, F2_TEMPLATES
from robust_belief import RobustBelief


class RobustBeliefTests(unittest.TestCase):
    def _world(self, noise_q=0.0):
        cfg = LabConfig(d1_cond=F1_TEMPLATES[0], d2_cond=F2_TEMPLATES[0],
                        d1_delay=False, d2_delay=False)
        return LabWorld(cfg, ep_seed=7, noise_q=noise_q)

    def test_positions_are_never_corrupted(self):
        rb = RobustBelief(1)
        w = self._world(noise_q=0.5)
        obs = w.observe()
        rb.observe(obs)
        self.assertEqual(rb.belief.pos, (obs.x, obs.y))
        self.assertEqual(rb.belief.objs["A"], (obs.ax, obs.ay))
        self.assertEqual(rb.belief.objs["B"], (obs.bx, obs.by))

    def test_toggle_explains_switch_immediately(self):
        rb = RobustBelief(1)
        w = self._world()
        obs0 = w.observe()
        rb.observe(obs0)
        # go stand on S1 and toggle
        for a in ("N", "E", "N", "N", "W"):
            res = w.step(a)
            rb.observe(res.observation)
        assert rb.belief.pos == (0, 1)
        res = w.step("Toggle")
        rb.observe(res.observation)
        rb.apply_transition("Toggle", True, res.observation)
        self.assertTrue(rb.belief.switch_state["S1"])

    def test_failed_move_proves_door_closed(self):
        rb = RobustBelief(1)
        w = self._world()
        obs = w.observe()
        rb.observe(obs)
        # D1 sits at (3,1); from (0,4) walk east a few steps then attempt
        # moving into the closed door cell from (2,1)
        seq = []
        for a in ("N", "E", "N", "N", "E", "E"):
            res = w.step(a)
            rb.observe(res.observation)
            rb.apply_transition(a, res.valid, res.observation)
            seq.append(a)
        # agent should now be at (2,1) (D1 at (3,1) is closed)
        self.assertEqual(rb.belief.pos, (2, 1))
        res = w.step("E")   # into closed D1 -> invalid
        rb.observe(res.observation)
        rb.apply_transition("E", res.valid, res.observation)
        self.assertFalse(res.valid)
        self.assertFalse(rb.belief.door_open["D1"])

    def test_confirm_needs_persistence(self):
        rb = RobustBelief(1)
        w = self._world()
        obs = w.observe()
        rb.observe(obs)
        # simulate a single flip then revert (like one corrupted reading)
        class FakeObs:
            def __init__(self, d1):
                self.d1 = d1
                self.d2 = False
                self.s1 = False
                self.s2 = False
                self.p1 = "none"
                self.p2 = "none"
                self.x = 0
                self.y = 4
                self.ax = 2
                self.ay = 3
                self.bx = 4
                self.by = 3
        rb.observe(FakeObs(True))     # flip
        rb.observe(FakeObs(False))    # revert before confirmation
        self.assertEqual(rb.poll_confirmed_events(), [])
        self.assertGreaterEqual(rb.dropped_candidates, 1)

    def test_confirm_on_persistence(self):
        rb = RobustBelief(1)
        class FakeObs:
            def __init__(self, d1):
                self.d1 = d1
                self.d2 = False
                self.s1 = False
                self.s2 = False
                self.p1 = "none"
                self.p2 = "none"
                self.x = 0
                self.y = 4
                self.ax = 2
                self.ay = 3
                self.bx = 4
                self.by = 3
        rb.observe(FakeObs(False))
        rb.observe(FakeObs(True))     # flip  -> candidate
        rb.observe(FakeObs(True))     # persist 1
        rb.observe(FakeObs(True))     # persist 2 -> confirm
        ev = rb.poll_confirmed_events()
        self.assertIn(("D1", True), ev)
        self.assertEqual(len(ev), 1)

    def test_movement_proof_confirms_and_dedupes(self):
        rb = RobustBelief(1)
        # synthetic: stand on (2,1), move E into door cell (3,1), valid
        # (door truly open)
        class FakeObs:
            def __init__(self, x, y, d1=True):
                self.x = x
                self.y = y
                self.d1 = d1
                self.d2 = False
                self.s1 = True
                self.s2 = False
                self.p1 = "A"
                self.p2 = "none"
                self.ax = 2
                self.ay = 2
                self.bx = 4
                self.by = 3
        rb.observe(FakeObs(2, 1, d1=False))
        rb.apply_transition("E", True, FakeObs(3, 1, d1=True))
        self.assertTrue(rb.belief.door_open["D1"])
        ev = rb.poll_confirmed_events()
        self.assertIn(("D1", True), ev)
        # repeated transit through the same open door: no new event
        rb.observe(FakeObs(3, 1, d1=True))
        rb.apply_transition("E", True, FakeObs(4, 1, d1=True))
        rb.observe(FakeObs(4, 1, d1=True))
        rb.apply_transition("W", True, FakeObs(3, 1, d1=True))
        self.assertEqual(rb.poll_confirmed_events(), [])

    def test_debounce_filters_single_flip(self):
        rb = RobustBelief(1, debounce=2)
        vb = rb.belief
        # simulate: 1 corrupted reading then recovery (platform, switch)
        class FakeObs:
            def __init__(self, s1):
                self.s1 = s1
                self.s2 = False
                self.p1 = "none"
                self.p2 = "none"
                self.d1 = False
                self.d2 = False
                self.x = 0
                self.y = 4
                self.ax = 2
                self.ay = 3
                self.bx = 4
                self.by = 3
        rb.observe(FakeObs(True))      # corrupted
        self.assertFalse(rb.belief.switch_state["S1"])
        rb.observe(FakeObs(False))     # recovers
        self.assertFalse(rb.belief.switch_state["S1"])
        rb.observe(FakeObs(False))
        self.assertFalse(rb.belief.switch_state["S1"])
        rb.observe(FakeObs(True))      # persistent? no, consecutive needed
        rb.observe(FakeObs(True))
        self.assertTrue(rb.belief.switch_state["S1"])

    def test_push_onto_plate_adopted_immediately(self):
        rb = RobustBelief(1)
        w = self._world()
        obs0 = w.observe()
        rb.observe(obs0)
        # path pushing A (at (2,3)) north onto P1 (2,2): stand at (2,4)
        for a in ("E", "E", "S", "S", "S", "W", "W"):
            res = w.step(a)
            rb.observe(res.observation)
            rb.apply_transition(a, res.valid, res.observation)
        self.assertEqual(rb.belief.pos, (0, 4))
        # walk to (2,4)
        for a in ("E", "E"):
            res = w.step(a)
            rb.observe(res.observation)
            rb.apply_transition(a, res.valid, res.observation)
        self.assertEqual(rb.belief.pos, (2, 4))
        res = w.step("PushN")
        rb.observe(res.observation)
        rb.apply_transition("PushN", res.valid, res.observation)
        self.assertTrue(res.valid)
        self.assertEqual(rb.belief.plate_load["P1"], "A")
        self.assertEqual(rb.belief.objs["A"], (2, 2))


if __name__ == "__main__":
    unittest.main()
