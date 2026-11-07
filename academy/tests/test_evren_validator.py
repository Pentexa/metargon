"""Tests for strict validation of teacher-generated task proposals."""

from __future__ import annotations

import unittest

from academy.envs.gridworld import EnvConfig, is_solvable
from academy.evren.validator import (
    STATUS_APPROVED, STATUS_DUPLICATE, STATUS_EVAL_CONTAMINATION,
    STATUS_IMPOSSIBLE, STATUS_INVALID_SCHEMA, STATUS_LEAKED,
    TaskValidator, independent_solvable, proposal_to_config,
)
from academy.tests.helpers import cached_splits

GOOD_RAW = {
    "task_id": "evren-test-0001",
    "topology": "TP2",
    "d1_rule": [[["sw", "S2", True]]],
    "d2_rule": [[["sw", "S1", False]], [["empty", "P1"]]],
    "d1_delay": False,
    "d2_delay": True,
    "difficulty_intent": "medium",
    "rationale": "switch plus emptiness conjunct on TP2",
}


def cfg_to_raw(cfg, task_id, rationale="", intent="medium"):
    """Serialize an EnvConfig into raw proposal JSON (canonical form)."""
    def ser_clause(c):
        return [list(a) for a in sorted(c, key=str)]

    def ser_rule(r):
        if isinstance(r, tuple):
            return [ser_clause(c) for c in sorted(r, key=str)]
        return [ser_clause(r)]          # conjunctive: exactly one clause

    return {
        "task_id": task_id,
        "topology": cfg.topology,
        "d1_rule": ser_rule(cfg.d1_rule),
        "d2_rule": ser_rule(cfg.d2_rule),
        "d1_delay": cfg.d1_delay,
        "d2_delay": cfg.d2_delay,
        "difficulty_intent": intent,
        "rationale": rationale,
    }


class TestSchemaValidation(unittest.TestCase):
    def setUp(self):
        self.v = TaskValidator(splits=cached_splits())

    def _raw(self, **mut):
        raw = dict(GOOD_RAW)
        raw.update(mut)
        return raw

    def test_unknown_field_rejected(self):
        # an innocuous unknown field (not a forbidden name) fails schema
        res = self.v.validate(self._raw(temperature=0.5))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)
        self.assertTrue(any("unknown field" in r for r in res.reasons))

    def test_missing_required_field_rejected(self):
        raw = self._raw()
        del raw["d2_rule"]
        res = self.v.validate(raw)
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_bad_topology_rejected(self):
        res = self.v.validate(self._raw(topology="TP9"))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_bad_atom_kind_rejected(self):
        res = self.v.validate(self._raw(d1_rule=[[["teleport", "P1", None]]]))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_sw_value_must_be_boolean(self):
        res = self.v.validate(self._raw(d1_rule=[[["sw", "S1", "true"]]]))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_load_subject_must_be_plate(self):
        res = self.v.validate(self._raw(d1_rule=[[["load", "S1", "A"]]]))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_load_object_must_be_a_or_b(self):
        res = self.v.validate(self._raw(d1_rule=[[["load", "P1", "C"]]]))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_empty_clause_rejected(self):
        res = self.v.validate(self._raw(d1_rule=[[]]))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_duplicate_atom_in_clause_rejected(self):
        res = self.v.validate(self._raw(
            d1_rule=[[["sw", "S1", True], ["sw", "S1", True]]]))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)

    def test_atom_arity_rejected(self):
        res = self.v.validate(self._raw(d1_rule=[[["load", "P1"]]]))
        self.assertEqual(res.status, STATUS_INVALID_SCHEMA)


class TestLeakageDetection(unittest.TestCase):
    def setUp(self):
        self.v = TaskValidator(splits=cached_splits())

    def test_forbidden_key_name_rejected(self):
        raw = dict(GOOD_RAW)
        raw["hint"] = "load A onto P1"
        res = self.v.validate(raw)
        self.assertEqual(res.status, STATUS_LEAKED)

    def test_hint_text_in_rationale_rejected(self):
        raw = dict(GOOD_RAW)
        raw["rationale"] = "door opens when you load a plate — the answer is P1"
        res = self.v.validate(raw)
        self.assertEqual(res.status, STATUS_LEAKED)

    def test_nested_forbidden_key_rejected(self):
        raw = dict(GOOD_RAW)
        raw["d1_rule"].append(["empty", "P2", None])
        raw["meta"] = {"solution_policy": ["N", "E"]}
        res = self.v.validate(raw)
        self.assertEqual(res.status, STATUS_LEAKED)


class TestSolvability(unittest.TestCase):
    def test_independent_solver_agrees_with_gridworld_solver(self):
        import random
        rng = random.Random(7)
        # sample a spread of configs from cached pools
        splits = cached_splits()
        pool = (splits["train"] + splits["val"] + splits["test"]
                + splits["ood"])
        rng.shuffle(pool)
        for cfg in pool[:24]:
            self.assertEqual(independent_solvable(cfg), is_solvable(cfg),
                             f"disagreement for {cfg.key}")

    def test_impossible_task_rejected(self):
        # find a known-unsolvable config inside the enumerated space
        from academy.envs.tasks import _enumerate_configs, CORE_RULESETS
        for cfg in _enumerate_configs(CORE_RULESETS, "TP1"):
            if not is_solvable(cfg):
                break
        else:
            self.skipTest("no unsolvable config found")
        res = TaskValidator(splits=cached_splits()).validate(
            cfg_to_raw(cfg, "impossible-1", intent="hard",
                       rationale="impossible by construction"))
        self.assertEqual(res.status, STATUS_IMPOSSIBLE)
        self.assertFalse(res.solvable)


class TestApproval(unittest.TestCase):
    def setUp(self):
        self.splits = cached_splits()
        self.v = TaskValidator(splits=self.splits)

    def test_novel_solvable_task_approved(self):
        res = self.v.validate(dict(GOOD_RAW))
        self.assertEqual(res.status, STATUS_APPROVED)
        self.assertTrue(res.solvable)
        self.assertIsInstance(res.config, EnvConfig)

    def test_duplicate_rejected_after_register(self):
        res1 = self.v.validate(dict(GOOD_RAW))
        self.assertEqual(res1.status, STATUS_APPROVED)
        self.v.register(res1.config)
        res2 = self.v.validate(dict(GOOD_RAW))
        self.assertEqual(res2.status, STATUS_DUPLICATE)

    def test_eval_contamination_rejected(self):
        cfg = self.splits["test"][0]
        res = self.v.validate(cfg_to_raw(cfg, "eval-leak-1",
                                         rationale="eval contamination probe"))
        self.assertEqual(res.status, STATUS_EVAL_CONTAMINATION)

    def test_train_pool_members_rejected_as_duplicates(self):
        cfg = self.splits["train"][0]
        res = self.v.validate(cfg_to_raw(cfg, "train-dup-1",
                                         rationale="train-pool duplicate probe"))
        self.assertIn(res.status, (STATUS_DUPLICATE, STATUS_EVAL_CONTAMINATION))

    def test_proposal_to_config_roundtrip(self):
        res = self.v.validate(dict(GOOD_RAW))
        self.assertEqual(res.status, STATUS_APPROVED)
        cfg = res.config
        from academy.evren.validator import config_to_proposal
        prop = config_to_proposal(cfg, task_id="roundtrip", rationale="")
        self.v.register(cfg)
        again = self.v.validate(prop)
        self.assertEqual(again.status, STATUS_DUPLICATE)  # same key seen


if __name__ == "__main__":
    unittest.main()
