"""End-to-end smoke of the integration runner (tiny, seed 0)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generalization_audit.cognitive_integration_v01 import config as C
from generalization_audit.cognitive_integration_v01.runner import (
    main as runner_main,
    register_protocol_arms,
)
from generalization_audit.cognitive_v02 import config as V2
from generalization_audit.cognitive_v02.runner import run_seed


class TestRunnerSmoke(unittest.TestCase):

    def test_registration_adds_integration_controls(self):
        register_protocol_arms()
        for name in C.EXTRA_CONTROLS:
            self.assertIn(name, V2.CONTROL_SPECS)

    def test_quick_run_produces_artifacts(self):
        register_protocol_arms()
        V2.TRAIN_EPISODES = 4
        V2.EVAL_EPISODES = 2
        V2.AB_TRAIN_PER_PHASE = 2
        V2.AB_EVAL_EPISODES = 2
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "results"
            seed_results = {}
            oracle_cache = {}
            seed_results[0] = run_seed(0, {"train": 4, "eval": 2,
                                           "ab_train": 2, "ab_eval": 2},
                                       ["standard", "ab"],
                                       oracle_cache=oracle_cache)
            for arm in C.ARM_ORDER:
                self.assertIn(arm, seed_results[0],
                              f"missing arm {arm}")
            self.assertIn("_floors", seed_results[0])
            # artifact writing via runner.main --quick path
            runner_main(["--quick", "--out", str(out), "--no-oracle"])
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "analysis.json").exists())
            self.assertTrue((out / "per_seed.json").exists())
            sc = json.loads((out / "summary.json").read_text(
                encoding="utf-8"))
            self.assertIn("paired", sc)
            self.assertIn("floors", sc)
            an = json.loads((out / "analysis.json").read_text(
                encoding="utf-8"))
            self.assertIn("interactions", an)


if __name__ == "__main__":
    unittest.main()
