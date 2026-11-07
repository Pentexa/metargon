"""End-to-end verification of the EVREN teacher integration:
  * persistent learning state changes through the student's own experience,
  * checkpoints restore learned behavior deterministically,
  * training never leaks evaluation answers,
  * provider outages do not destroy training progress,
  * token/request limits are enforced,
  * a baseline without EVREN-generated tasks is included.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from academy.tests.helpers import cached_splits, temp_registry
from academy.evren.offline_generator import run_offline_pilot
from academy.evren.pilot import run_experiment, run_pilot
from academy.evren.task_registry import TaskRegistry
from academy.evren.provider import QuotaGuard, ProviderConfig, EvrenProvider
from academy.ckpt.checkpoint import CheckpointManager
from academy.learning.loop import run_one_episode
from academy.agents.metargon_agent import MetargonAcademyAgent
from academy.eval.metrics import forgetting_metrics


def minimal_cfg(tmp, max_calls=10):
    return {
        "evren": {
            "api_key_env": "EVREN_API_KEY",
            "max_calls_per_run": max_calls,
            "allow_unlimited": False,
            "registry_dir": tmp,
            "seed": 5,
            "temperature": 0.7,
            "max_tokens": 256,
        },
        "seeds": {"pool_split": 20260901},
        "replay_capacity": 2000,
        "run_seed": 5,
    }


class TestLearningIntegration(unittest.TestCase):
    """Shared small experiment (one heavy compute per class)."""

    @classmethod
    def setUpClass(cls):
        cls.splits = cached_splits()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = cls._tmp.name
        run_offline_pilot(seed=21, n=6, splits=cls.splits,
                          registry=temp_registry(cls.tmp, "evren-exp"))
        cls.registry = TaskRegistry(cls.tmp, "evren-exp")
        cls.exp = run_experiment(
            minimal_cfg(cls.tmp), cls.splits, cls.registry,
            episodes=8, p_teacher=0.5, seed=5, max_steps=50,
            eval_seeds=1, eval_episodes_per_seed=10,
            results_dir=cls.tmp, run_id="evren-exp")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- 1. persistent learning state changes through experience ---------
    def test_learning_state_changes_through_experience(self):
        approved = self.registry.approved_configs()
        self.assertGreaterEqual(len(approved), 4)
        stream = []
        from academy.envs.tasks import EpisodeSpec, difficulty_of
        for i, c in enumerate(approved[:4]):
            stream.append(EpisodeSpec(config=c, max_steps=40, noise_q=0.0,
                                      spec_seed=1000 + i, pool_tag="train",
                                      difficulty=difficulty_of(c)))
        agent = MetargonAcademyAgent(seed=7)
        before = agent.save_state()
        for spec in stream:
            run_one_episode(agent, spec)
        after = agent.save_state()
        self.assertGreater(agent.diagnostics()["episodes"], 0)
        self.assertGreater(agent.diagnostics()["memory"]["generated"], 0)
        self.assertNotEqual(before["steps"], after["steps"])
        self.assertNotEqual(json.dumps(before, sort_keys=True, default=str),
                            json.dumps(after, sort_keys=True, default=str))

    # -- 2. checkpoint restore retains learned behavior ------------------
    def test_checkpoint_restore_retains_behavior(self):
        approved = self.registry.approved_configs()
        stream = []
        from academy.envs.tasks import EpisodeSpec, difficulty_of
        for i, c in enumerate(approved[:5]):
            stream.append(EpisodeSpec(config=c, max_steps=40, noise_q=0.0,
                                      spec_seed=2000 + i, pool_tag="train",
                                      difficulty=difficulty_of(c)))
        agent = MetargonAcademyAgent(seed=11)
        for spec in stream[:3]:
            run_one_episode(agent, spec)
        ckpt_dir = os.path.join(self.tmp, "checkpoints")
        ckpt = CheckpointManager(ckpt_dir, "restore-test",
                                 minimal_cfg(self.tmp), keep_last=2)
        ckpt.save(episode=2, agent_state=agent.save_state(),
                  curriculum_state={}, mem_state={}, extra={})
        snap = ckpt.load_latest()
        self.assertIsNotNone(snap)
        self.assertTrue(ckpt.verify(snap))

        restored = MetargonAcademyAgent(seed=11)
        restored.load_state(snap["agent"])
        self.assertEqual(json.dumps(agent.diagnostics(), sort_keys=True),
                         json.dumps(restored.diagnostics(), sort_keys=True))
        self.assertEqual(json.dumps(agent.save_state(), sort_keys=True, default=str),
                         json.dumps(restored.save_state(), sort_keys=True, default=str))

        r1 = run_one_episode(restored, stream[3], collect_trace=True)
        restored2 = MetargonAcademyAgent(seed=11)
        restored2.load_state(snap["agent"])
        r2 = run_one_episode(restored2, stream[3], collect_trace=True)
        self.assertEqual(r1["success"], r2["success"])
        self.assertEqual(r1["steps"], r2["steps"])
        self.assertEqual(r1["trace"], r2["trace"])

    # -- 3. training does not leak evaluation answers --------------------
    def test_approved_tasks_disjoint_from_evaluation(self):
        eval_keys = {c.key for c in (self.splits["val"] + self.splits["test"]
                                     + self.splits["ood"])}
        train_keys = {c.key for c in self.splits["train"]}
        for t in self.registry.approved_tasks():
            self.assertNotIn(t.config.key, eval_keys,
                             "teacher task leaked into evaluation pool")
            self.assertNotIn(t.config.key, train_keys,
                             "teacher task duplicates a training config")

    # -- 4. provider outage does not destroy training progress -----------
    def test_provider_outage_does_not_block_training(self):
        class DownProvider(EvrenProvider):
            def discover_models(self, force=False):
                raise RuntimeError("total outage")

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_pilot(minimal_cfg(tmp), self.splits, n=4,
                                results_dir=tmp, run_id="outage-pilot",
                                provider=DownProvider())
            self.assertEqual(summary["source"], "offline_generator")
            self.assertEqual(summary["approved"], 4)
        # the shared experiment (with a registry seeded via the offline
        # path) completed despite no EVREN availability
        self.assertIn("baseline", self.exp)
        self.assertIn("teacher", self.exp)
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp, "experiment_summary.json")))

    # -- 5. limits enforced (default 10, no unlimited by default) -------
    def test_default_limits_and_config(self):
        g = QuotaGuard()
        self.assertEqual(g.max_calls, 10)
        self.assertFalse(g.allow_unlimited)
        with self.assertRaises(ValueError):
            QuotaGuard(max_calls=-1)

    def test_quota_exceeded_is_raised_during_chat(self):
        p = EvrenProvider(ProviderConfig(), quota=QuotaGuard(max_calls=1))
        p._request = lambda *a, **k: (
            200, '{"choices":[{"message":{"content":"x"}}],"usage":{}}', None)
        p.chat([{"role": "user", "content": "x"}], model="m")
        with self.assertRaises(Exception) as ctx:
            p.chat([{"role": "user", "content": "x"}], model="m")
        self.assertIn("quota", str(ctx.exception).lower())

    # -- 6. baseline without EVREN tasks is included ---------------------
    def test_experiment_includes_baseline_and_teacher(self):
        exp = self.exp
        for cond in ("baseline", "teacher"):
            self.assertIn("train_success", exp[cond])
            self.assertIn("classic_test", exp[cond])
            self.assertIn("teacher_holdout", exp[cond])
        self.assertGreaterEqual(exp["teacher_holdout_tasks"], 1)
        fg = exp["forgetting"]
        self.assertIn("baseline_probe", fg)
        self.assertIn("teacher_probe", fg)
        self.assertIn("forgetting_delta_teacher_vs_baseline", fg)
        early = [{"success": True, "steps": 3}] * 6
        late = [{"success": False, "steps": 9}] * 4
        fm = forgetting_metrics(early, late)
        self.assertAlmostEqual(fm["forgetting_delta"], 1.0)


class TestCliWiring(unittest.TestCase):
    def test_evren_offline_cli_runs_end_to_end(self):
        from academy.cli.main import main
        with tempfile.TemporaryDirectory() as tmp:
            # parent-level options must precede the subcommand
            rc = main(["--override", f"evren.registry_dir={tmp}",
                       "evren-offline", "--n", "2"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "approved_tasks.json")))
            rc2 = main(["--override", f"evren.registry_dir={tmp}",
                        "evren-status"])
            self.assertEqual(rc2, 0)

    def test_evren_status_reports_missing_registry(self):
        from academy.cli.main import main
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--override", f"evren.registry_dir={tmp}",
                        "evren-status"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
