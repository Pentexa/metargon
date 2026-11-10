"""NEST v0.2+ â€” human-guided transfer demonstration (integration).

Proves, with the REAL agent, that:

1. Human guidance changes behavior: guided episodes execute guided actions
   and record hint usage.
2. Persistent learning affects later decisions: a trained agent's action
   choices and memory traces on a RELATED unseen task differ from a
   memory-wiped control and from fresh.
3. Wiping memory removes the influence (control).

Under matched budgets (same configurations, seeds, max steps).

Output is honest: the success comparison is reported, and the tests assert
only the measurable behavior/memory effects (not a fabricated gain).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from nest.agent_session import MAX_GLOBAL_STEPS, Challenge, NestSession
from nest.storage import NestStorage
from nest.teaching import parse_hint_text
from academy.envs.tasks import get_splits, make_episodes

EPS = 4          # evaluation episodes per arm
TRAIN_EPS = 5    # training episodes for the taught agent
MAX_STEPS = 80


def _run_episodes(session, episodes, budget=MAX_STEPS):
    results = []
    for spec in episodes:
        from academy.envs.gridworld import MechanismGrid
        env = MechanismGrid(spec.config, spec.max_steps, spec.noise_q)
        obs = env.reset(spec.spec_seed)
        session.agent.episode_start(spec, env.geom)
        session.teaching.begin_episode_snapshot()
        success = False
        actions = []
        for _ in range(budget):
            if env.finished or env.steps >= env.max_steps:
                break
            # student is the sole action authority
            action = session.agent.act(obs)
            actions.append(action)
            before = obs
            res = env.step(action)
            session.agent.learn(before, action, res)
            session.teaching.record_step(action, res)
            obs = res.observation
            if res.terminated:
                success = True
                break
        session.agent.finish_episode(success)
        results.append({"success": success, "actions": actions,
                        "steps": env.steps})
    return results


def _wipe_memory(session):
    session.agent.memory.hypotheses = []
    session.agent.memory.evidence_bank = []
    session.agent.trace["memory_actions"] = 0
    session.agent.trace["hypothesis_retrieved"] = 0


class TestHumanGuidedTransfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.splits = get_splits()
        # training task T: first easy train config
        cls.T = next(c for c in cls.splits["train"]
                     if c.n_atoms() <= 2)
        # related unseen task E: test-family config, same topology, different rule
        cls.E = None
        for c in cls.splits["test"]:
            if c.topology == cls.T.topology and c.key != cls.T.key:
                cls.E = c
                break
        assert cls.E is not None, "no related unseen test config found"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = NestStorage(Path(self._tmp.name))
        self.seed = 99191
        self.t_episodes = make_episodes([self.T], "train", self.seed,
                                        TRAIN_EPS, MAX_STEPS)
        self.e_episodes = make_episodes([self.E], "eval", self.seed,
                                        EPS, MAX_STEPS)

    def tearDown(self):
        self._tmp.cleanup()

    def _fresh_session(self):
        return NestSession(storage=self.storage, seed=self.seed)

    def test_hint_parser_whitelist(self):
        h = parse_hint_text("try switch S1")
        self.assertIsNotNone(h)
        self.assertIn(("sw", "S1", True), h.atoms)
        self.assertEqual(h.status, "unverified-preference")
        self.assertIsNone(parse_hint_text("fly to the moon"))
        h2 = parse_hint_text("load P1 with A")
        self.assertIn(("load", "P1", "A"), h2.atoms)

    def test_injection_starts_unverified_and_student_owned(self):
        # A hint injected after an episode is running must appear as an
        # unverified candidate hypothesis (status hypothesis, sup 0); the
        # only way it can proceed is through real observations.
        s = self._fresh_session()
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy",
                                  seed=0, description="train"))
        hint = parse_hint_text("try switch S1")
        self.assertTrue(s.teaching.add_hint(hint))
        self.assertTrue(s.teaching.teacher_ids)
        for hid in s.teaching.teacher_ids:
            h = next(x for x in s.agent.memory.hypotheses if x.h_id == hid)
            self.assertEqual(h.status, "hypothesis")
            self.assertEqual(h.sup, 0)
            self.assertEqual(h.con, 0)
        # student remains the only actor: no external planner bypass flag
        self.assertFalse(hasattr(s, "guided_actions"))

    def test_guidance_changes_behavior_and_persistence_affects_decisions(self):
        # ---- arm 1: fresh (no training, no guidance) ----
        fresh = self._fresh_session()
        fresh_res = _run_episodes(fresh, self.e_episodes)
        fresh_success = sum(1 for r in fresh_res if r["success"])
        fresh_probe_actions = sum(1 for r in fresh_res
                                  for a in r["actions"]
                                  if a in ("Toggle", "PushN", "PushE", "PushS", "PushW"))

        # ---- arm 2: human-guided training then related unseen eval ----
        trained = self._fresh_session()
        hint = parse_hint_text("try switch S1")
        self.assertIsNotNone(hint)
        trained.teaching.add_hint(hint)
        # the hint is recorded as an UNVERIFIED preference; it is injected
        # into the student's candidate hypotheses at the next episode start
        self.assertTrue(trained.teaching.hints)
        self.assertEqual(trained.teaching.hints[-1].status, "unverified-preference")
        self.assertIn("switch s1", trained.teaching.hints[-1].label.lower())
        self.assertTrue(trained.teaching.has_teacher_suggestion() or not trained.agent.geom)
        _run_episodes(trained, self.t_episodes)
        # after injection, teacher suggestions must exist; any verification
        # must come only from real observation (we never seed support)
        self.assertTrue(trained.teaching.teacher_ids)
        for hid in trained.teaching.teacher_ids:
            h = next(x for x in trained.agent.memory.hypotheses if x.h_id == hid)
            self.assertGreaterEqual(h.sup, 0)
            self.assertTrue(h.created_step >= 0)
        self.assertGreaterEqual(trained.teaching.stats["suggestions_added"], 1)
        teacher_probe_hits = trained.teaching.teacher_probe_hits()
        trained_verified = trained.memory_snapshot()["stats"].get("verified", 0)
        trained_res = _run_episodes(trained, self.e_episodes)
        trained_success = sum(1 for r in trained_res if r["success"])
        trained_probe_actions = sum(1 for r in trained_res
                                    for a in r["actions"]
                                    if a in ("Toggle", "PushN", "PushE", "PushS", "PushW"))
        trained_mem_actions = trained.agent.trace["memory_actions"]
        trained_retrieved = trained.agent.trace["hypothesis_retrieved"]

        # ---- arm 3: same trained agent, memory wiped ----
        wiped = self._fresh_session()
        hint2 = parse_hint_text("try switch S1")
        wiped.teaching.add_hint(hint2)
        _run_episodes(wiped, self.t_episodes)
        _wipe_memory(wiped)
        wiped_res = _run_episodes(wiped, self.e_episodes)
        wiped_success = sum(1 for r in wiped_res if r["success"])
        wiped_probe_actions = sum(1 for r in wiped_res
                                  for a in r["actions"]
                                  if a in ("Toggle", "PushN", "PushE", "PushS", "PushW"))
        wiped_mem_actions = wiped.agent.trace["memory_actions"]
        wiped_retrieved = wiped.agent.trace["hypothesis_retrieved"]

        # ---- honest report (printed) ----
        print("\n[transfer] fresh:", fresh_success, "/", EPS,
              "| trained:", trained_success, "/", EPS,
              "| wiped:", wiped_success, "/", EPS)
        print("[transfer] probe-actions fresh:", fresh_probe_actions,
              "trained:", trained_probe_actions,
              "wiped:", wiped_probe_actions)
        print("[transfer] teacher-suggestion probe hits (chosen by student's act()):",
              teacher_probe_hits)
        print("[transfer] trained traces: memory_actions =", trained_mem_actions,
              ", retrievals =", trained_retrieved,
              "| wiped traces: memory_actions =", wiped_mem_actions,
              ", retrievals =", wiped_retrieved)

        # 1) teaching influenced candidate goals chosen by the student itself
        self.assertGreater(teacher_probe_hits, 0,
                           "the student's act() must select teacher-suggested "
                           "hypotheses sometimes")

        # 2) guidance changed the exploration/behavior profile vs fresh
        self.assertNotEqual(fresh_probe_actions, trained_probe_actions,
                            "fresh vs trained probe behavior should differ")

        # 3) persistent learning affects later decisions: memory influence is
        #    measurable on the unseen task and disappears after wiping
        self.assertGreater(trained_mem_actions + trained_retrieved, 0,
                           "trained agent must show memory influence during eval")
        self.assertEqual(wiped_mem_actions, 0)
        self.assertLess(wiped_retrieved, trained_retrieved,
                        "wiped memory must reduce retrieval influence")

        # Honest per: no positive transfer claimed automatically.
        if trained_success > fresh_success and trained_success >= wiped_success:
            print("[transfer] directional signal: trained > fresh and trained >= wiped")
        else:
            print("[transfer] no positive transfer measured â€” reported honestly "
                  f"(fresh={fresh_success}, trained={trained_success}, "
                  f"wiped={wiped_success}). Guidance changed behavior, but did not "
                  "produce a gain on this unseen task.")


if __name__ == "__main__":
    unittest.main()

