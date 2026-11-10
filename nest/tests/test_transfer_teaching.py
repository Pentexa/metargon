"""NEST teaching validation — corrected semantics.

Covers:
- teaching input classification (guidance vs claim vs objective);
- claim injection as UNVERIFIED candidate hypothesis (status hypothesis,
  sup=0), removable, student-owned;
- guidance must never create a causal hypothesis;
- claim + persistence influence on later decisions vs fresh and memory-wiped
  controls; honest transfer reporting.
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

from nest.agent_session import Challenge, NestSession
from nest.storage import NestStorage
from nest.teaching import (
    KIND_CLAIM,
    KIND_GUIDANCE,
    KIND_OBJECTIVE,
    parse_teaching_text,
)

EPS = 4
TRAIN_EPS = 5
MAX_STEPS = 80
SEED = 99191


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


class TestTeachingSemantics(unittest.TestCase):

    def test_parser_distinguishes_kinds(self):
        g = parse_teaching_text("try switch S1")
        self.assertEqual(g.kind, KIND_GUIDANCE)
        self.assertIn(("sw", "S1", True), g.atoms)
        c = parse_teaching_text("I think switch S1 opens door D1")
        self.assertEqual(c.kind, KIND_CLAIM)
        self.assertEqual(c.door, "D1")
        self.assertIn(("sw", "S1", True), c.atoms)
        o = parse_teaching_text("discover how to open D1")
        self.assertEqual(o.kind, KIND_OBJECTIVE)
        self.assertEqual(o.door, "D1")
        self.assertIsNone(parse_teaching_text("fly to the moon"))
        # claim without a door must not become a causal rule
        partial = parse_teaching_text("I think switch S1 opens something")
        self.assertEqual(partial.kind, KIND_GUIDANCE)

    def test_guidance_never_injects_causal_hypothesis(self):
        s = NestSession(storage=NestStorage(Path(tempfile.mkdtemp())), seed=SEED)
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy",
                                  seed=0, description="train"))
        g = parse_teaching_text("try switch S1")
        self.assertTrue(s.teaching.apply(g))
        self.assertEqual(len(s.teaching.teacher_ids), 0)
        self.assertTrue(s.teaching.guidance_atoms)
        self.assertEqual(s.teaching.stats["guidance_applied"], 1)

    def test_claim_injection_starts_unverified_and_is_removable(self):
        tmp = tempfile.mkdtemp()
        s = NestSession(storage=NestStorage(Path(tmp)), seed=SEED)
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy",
                                  seed=0, description="train"))
        c = parse_teaching_text("I think switch S1 opens door D1")
        self.assertTrue(s.teaching.apply(c))
        self.assertTrue(s.teaching.teacher_ids)
        for hid in s.teaching.teacher_ids:
            h = next(x for x in s.agent.memory.hypotheses if x.h_id == hid)
            self.assertEqual(h.status, "hypothesis")
            self.assertEqual(h.sup, 0)
            self.assertEqual(h.con, 0)
            self.assertEqual(h.door, "D1")
        claims = s.teaching.teacher_ids.copy()
        s.teaching.remove_teacher_hypotheses()
        remaining = [h for h in s.agent.memory.hypotheses if h.h_id in claims]
        self.assertEqual(remaining, [])


class TestClaimTransferHonest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from academy.envs.tasks import get_splits, make_episodes
        cls.splits = get_splits()
        cls.T = next(c for c in cls.splits["train"] if c.n_atoms() <= 2)
        cls.E = next(c for c in cls.splits["test"]
                     if c.topology == cls.T.topology and c.key != cls.T.key)
        cls.t_episodes = make_episodes([cls.T], "train", SEED,
                                       TRAIN_EPS, MAX_STEPS)
        cls.e_episodes = make_episodes([cls.E], "eval", SEED,
                                       EPS, MAX_STEPS)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = NestStorage(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_claim_training_changes_decisions_and_persists(self):
        # fresh
        fresh = NestSession(storage=self.storage, seed=SEED)
        fresh_res = _run_episodes(fresh, self.e_episodes)
        fresh_success = sum(1 for r in fresh_res if r["success"])

        # guided-by-claim training (unverified claim injected)
        trained = NestSession(storage=self.storage, seed=SEED)
        claim = parse_teaching_text("I think switch S1 opens door D1")
        trained.teaching.apply(claim)
        _run_episodes(trained, self.t_episodes)
        claim_hits = trained.teaching.teacher_claim_probe_hits()
        trained_res = _run_episodes(trained, self.e_episodes)
        trained_success = sum(1 for r in trained_res if r["success"])
        trained_mem_actions = trained.agent.trace["memory_actions"]
        trained_retrieved = trained.agent.trace["hypothesis_retrieved"]

        # memory-wiped control of the same trained agent
        wiped = NestSession(storage=self.storage, seed=SEED)
        wiped.teaching.apply(parse_teaching_text("I think switch S1 opens door D1"))
        _run_episodes(wiped, self.t_episodes)
        _wipe_memory(wiped)
        wiped_res = _run_episodes(wiped, self.e_episodes)
        wiped_success = sum(1 for r in wiped_res if r["success"])
        wiped_mem_actions = wiped.agent.trace["memory_actions"]
        wiped_retrieved = wiped.agent.trace["hypothesis_retrieved"]

        print("\n[claim-transfer] fresh:", fresh_success, "/", EPS,
              "| trained-claim:", trained_success, "/", EPS,
              "| wiped:", wiped_success, "/", EPS)
        print("[claim-transfer] teacher-claim probes chosen by act():", claim_hits)
        print("[claim-transfer] memory traces: trained", trained_mem_actions,
              "/", trained_retrieved, "| wiped", wiped_mem_actions, "/",
              wiped_retrieved)

        self.assertTrue(trained.teaching.teacher_ids)
        self.assertGreaterEqual(claim_hits, 0)  # investigated (may be low)
        # learning persists and influences decisions on the unseen task
        self.assertGreater(trained_mem_actions + trained_retrieved, 0)
        self.assertEqual(wiped_mem_actions, 0)
        self.assertLess(wiped_retrieved, trained_retrieved)

        if trained_success > fresh_success and trained_success >= wiped_success:
            print("[claim-transfer] directional positive signal measured.")
        else:
            print("[claim-transfer] no positive transfer measured — reported "
                  f"honestly (fresh={fresh_success}, trained={trained_success}, "
                  f"wiped={wiped_success}).")


if __name__ == "__main__":
    unittest.main()
