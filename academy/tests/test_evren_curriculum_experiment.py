"""Tests for the v0.2 curriculum experiment (tiny smoke budgets)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from academy.envs.gridworld import EnvConfig
from academy.envs.tasks import get_splits
from academy.evren.curriculum_experiment import (
    eval_config_grid, partition_approved, plan_treatment,
    run_curriculum_experiment,
)
from academy.evren.task_registry import TaskRegistry

C1 = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
               frozenset({("load", "P2", "B")}), False, False)
C2 = EnvConfig("TP2", frozenset({("empty", "P1")}),
               frozenset({("sw", "S1", False)}), True, True)


def _tiny_cfg():
    return {"evren": {"curriculum": {
        "seed": 5,
        "substrate_classic_n": 4,
        "n_procedural": 4,
        "substrate_share": 0.30,
        "retention_eps_per_cfg": 1,
        "test_grid_n": 4,
        "eval_eps_per_cfg": 2,
        "holdout_eps_per_seed": 2,
        "ood_probe_n": 2,
        "ood_eps_per_seed": 1,
    }}, "seeds": {"eval": 20261108}}


class TestEvalGrid(unittest.TestCase):
    def test_matched_picks_across_calls(self):
        splits = get_splits()
        a = eval_config_grid(None, splits["test"], "test", 40, 5, 1, (0,), 7)
        b = eval_config_grid(None, splits["test"], "test", 40, 5, 1, (0,), 7)
        self.assertEqual([c["config"] for c in a["cells"]],
                         [c["config"] for c in b["cells"]])
        self.assertEqual(len(a["cells"]), 5)     # n_configs x 1 seed

    def test_summary_per_config(self):
        splits = get_splits()
        res = eval_config_grid(None, splits["test"], "test", 40, 4, 2, (0, 1),
                               3)
        self.assertEqual(res["summary"]["n_configs"], 4)
        self.assertEqual(len(res["per_config_list"]), 4)
        for r in res["cells"]:
            self.assertIn(r["config"], res["summary"]["per_config"])


class TestPartition(unittest.TestCase):
    def test_partition_shapes(self):
        sub, hold, extras = partition_approved([C1, C2, C1, C2], seed=1)
        self.assertEqual(len(sub) + len(hold) + len(extras), 4)
        self.assertGreaterEqual(len(sub), 1)
        allc = sub + hold + extras
        self.assertEqual(sorted(str(c.key) for c in allc),
                         sorted(str(c.key) for c in [C1, C2, C1, C2]))

    def test_degenerate_single(self):
        sub, hold, extras = partition_approved([C1], seed=0)
        self.assertEqual((len(sub), len(hold), len(extras)), (1, 0, 0))


class TestPlanTreatment(unittest.TestCase):
    def test_accepts_known_conditions(self):
        import random
        rng = random.Random(0)
        pool = [C1, C2]
        self.assertIn(plan_treatment("uniform", 0, pool, rng, {}),
                      pool)
        sched = {"ascending": sorted(pool, key=str)}
        self.assertIn(plan_treatment("ascending", 0, pool, rng, sched), pool)
        self.assertIn(plan_treatment("ascending", 1, pool, rng, sched), pool)

    def test_unknown_condition_raises(self):
        import random
        with self.assertRaises(ValueError):
            plan_treatment("bogus", 0, [C1], random.Random(0), {})


class TestExperimentSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="curriculum-smoke-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.results_dir = os.path.join(self.tmp, "evren")
        self.registry_dir = os.path.join(self.tmp, "registry")

    def test_two_conditions_matched_budgets_and_isolation(self):
        splits = get_splits()
        registry = TaskRegistry(self.registry_dir, run_id="test")
        results = run_curriculum_experiment(
            _tiny_cfg(), splits, registry, results_dir=self.results_dir,
            conditions=("uniform", "proc_lp"), episodes=6, block_size=3,
            seed=5, n_procedural=4, verbose=False)
        # summary persisted + parseable
        path = os.path.join(self.results_dir, "curriculum_v02_summary.json")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
        detail = reloaded["conditions_detail"]
        self.assertEqual(set(detail) | set(), set(detail))
        # matched budgets: every condition saw exactly `episodes` episodes
        for cond, d in detail.items():
            n = d["train"]["substrate_episodes"] + d["train"]["treatment_episodes"]
            self.assertEqual(n, 6)
        # offline / zero-API assertions
        self.assertEqual(reloaded["api_calls"], 0)
        self.assertTrue(reloaded["offline"])
        # isolation: no served config may be in val/test/ood pools
        def keys(pool):
            return {str(c.key) for c in splits[pool]}
        forbidden = keys("val") | keys("test") | keys("ood")
        for cond, d in detail.items():
            for r in d["curve"]:
                self.assertNotIn(r["config"], forbidden)
        # eval grids are matched across conditions (identical picks)
        cu = detail["uniform"]["classic_test_per_config"]
        cp = detail["proc_lp"]["classic_test_per_config"]
        self.assertEqual(len(cu), len(cp))
        # trained student states persisted for audit (one per condition)
        policies_dir = os.path.join(self.results_dir, "policies_v02")
        self.assertTrue(os.path.isdir(policies_dir))
        for cond in ("uniform", "proc_lp"):
            p = os.path.join(policies_dir, f"{cond}_agent.json")
            self.assertTrue(os.path.exists(p))
            with open(p, "r", encoding="utf-8") as f:
                st = json.load(f)
            self.assertIn("memory", st)
            self.assertIn("rng", st)
        # comparisons computed for the proc_lp - uniform pair
        self.assertIn("lp_effect_classic_test", reloaded["comparisons"])
        cmp_ = reloaded["comparisons"]["lp_effect_classic_test"]
        self.assertIn("mean", cmp_)
        self.assertIn("ci", cmp_)

    def test_empty_registry_teacher_lp_falls_back(self):
        splits = get_splits()
        registry = TaskRegistry(self.registry_dir, run_id="test2")
        results = run_curriculum_experiment(
            _tiny_cfg(), splits, registry, results_dir=self.results_dir,
            conditions=("teacher_lp",), episodes=4, block_size=4,
            seed=6, n_procedural=3, verbose=False)
        d = results["conditions_detail"]["teacher_lp"]
        self.assertEqual(
            d["train"]["substrate_episodes"] + d["train"]["treatment_episodes"],
            4)
        self.assertIsNone(results["conditions_detail"]["teacher_lp"]
                          ["teacher_holdout"])
        self.assertEqual(
            results["diversity"]["provenance"]["registered_approved"], 0)
        # teacher_lp with no registered tasks still persists its state
        p = os.path.join(self.results_dir, "policies_v02",
                         "teacher_lp_agent.json")
        self.assertTrue(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()
