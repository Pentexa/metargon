"""Arm factory, transforms, and candidate-registration tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from academy.envs.tasks import make_episodes

from generalization_audit.cognitive_v02 import config as C
from generalization_audit.cognitive_v02.arms import (
    all_arms,
    apply_transform,
    load_candidate_specs,
    make_fresh,
    make_policy,
    register_candidate,
    trained_arms,
)
from generalization_audit.cognitive_v02.driver import run_episode
from generalization_audit.cognitive_v02.splits import pools_dict
from generalization_audit.cognitive_v02.streams import train_stream


class TestArms(unittest.TestCase):

    def _train(self, agent, n=3):
        stream = train_stream(7, n)
        for sp in stream:
            run_episode(agent, sp)

    def test_factory_makes_configured_agents(self):
        a = make_policy("trained", 0)
        self.assertFalse(a.cognitive_exploration)
        self.assertFalse(a.memory_episode_scoped)
        e = make_policy("epoch_reset", 0)
        self.assertTrue(e.memory_episode_scoped)
        self.assertTrue(e.probe_reset_per_episode)
        x = make_policy("explore", 0)
        self.assertTrue(x.cognitive_exploration)
        xe = make_policy("explore_epoch_reset", 0)
        self.assertTrue(xe.cognitive_exploration)
        self.assertTrue(xe.memory_episode_scoped)
        self.assertTrue(xe.probe_reset_per_episode)

    def test_wipe_removes_declarative_knowledge(self):
        a = make_policy("trained", 1)
        self._train(a)
        self.assertGreaterEqual(len(a.memory.hypotheses), 0)
        apply_transform(a, "wipe", seed=1)
        self.assertEqual(len(a.memory.hypotheses), 0)
        self.assertEqual(a.memory.evidence_bank, [])

    def test_shuffle_changes_associations_and_preserves_size(self):
        a = make_policy("trained", 2)
        self._train(a, n=5)
        before = [(h.door, h.antecedent, h.verified_epochs)
                  for h in a.memory.hypotheses]
        a2 = make_policy("trained", 2)
        self._train(a2, n=5)
        apply_transform(a2, "shuffle", seed=2, assert_changed=True)
        after = [(h.door, h.antecedent, h.verified_epochs)
                 for h in a2.memory.hypotheses]
        self.assertEqual(len(after), len(before))
        # at least one antecedent changed (or trivial all-1 hyp case)
        changed = sum(1 for (d0, a0, v0), (d1, a1, v1)
                      in zip(before, after) if a0 != a1)
        self.assertGreaterEqual(changed, 1)

    def test_candidate_registration_interface(self):
        register_candidate("dummy_candidate", kwargs={"probe_adaptive": True})
        self.assertIn("dummy_candidate", trained_arms())
        ag = make_policy("dummy_candidate", 0)
        self.assertTrue(ag.probe_adaptive)
        # cleanup for isolation
        C.ARM_SPECS.pop("dummy_candidate", None)
        self.assertNotIn("dummy_candidate", trained_arms())

    def test_candidate_loader_reads_json_specs(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mem_v02_candidate.json"
            p.write_text(json.dumps({"name": "mem_v02_candidate",
                                     "kwargs": {"memory_atom_transfer": True}}),
                         encoding="utf-8")
            reg = load_candidate_specs(Path(td))
            self.assertIn("mem_v02_candidate", reg)
            self.assertIn("mem_v02_candidate", trained_arms())
            ag = make_policy("mem_v02_candidate", 0)
            self.assertTrue(ag.memory_atom_transfer)
            C.ARM_SPECS.pop("mem_v02_candidate", None)
            self.assertNotIn("mem_v02_candidate", trained_arms())


if __name__ == "__main__":
    unittest.main()
