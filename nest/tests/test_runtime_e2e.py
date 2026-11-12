"""METARGON NEST — end-to-end runtime regression tests (real HTTP server).

Exercise the founder-visible workflow against an actual local server, not
mocked DOM or fake agents:

  1. Initial world initialization and valid JSON response.
  2. Start and Single-step.
  3. A student-generated hypothesis.
  4. GET /api/state after hypotheses exist (serialization audit).
  5. Teaching guidance: "try switch S1".
  6. Teaching objective: "discover how to open D1".
  7. Teaching claim: "I think S1 opens door D1".
  8. Rejected teaching input with a clear error.
  9. Save, restart and Restore with populated memory.
 10. Frontend rendering after learning.

Every assertion inspects actual JSON responses and the real agent state.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import urllib.error
import urllib.request

from nest.agent_session import Challenge, NestSession
from nest.server import make_server
from nest.storage import NestStorage
from nest.teaching import parse_teaching_text


def is_json_native(obj) -> bool:
    """True if obj contains only JSON-native types, recursively."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return True
    if isinstance(obj, list):
        return all(is_json_native(v) for v in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and is_json_native(v)
                   for k, v in obj.items())
    return False


class TestSnapshotSerialization(unittest.TestCase):
    """Regression: a real hypothesis in memory must not break JSON output."""

    def test_snapshot_json_serializable_with_real_hypothesis(self):
        s = NestSession(storage=NestStorage(Path(tempfile.mkdtemp())), seed=7)
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy",
                                  description="easy training"))
        s.teaching.apply(parse_teaching_text("I think S1 opens door D1"))
        s.run_steps(8)  # easy config produces door events / hypotheses fast
        snap = s.snapshot()
        self.assertGreater(len(snap["memory"]["hypotheses"]), 0)
        encoded = json.dumps(snap)  # must NOT raise TypeError on frozenset
        self.assertNotIn("frozenset", encoded)
        # scientific representation is untouched under the hood
        h = next(x for x in s.agent.memory.hypotheses
                 if x.h_id in s.teaching.teacher_ids)
        self.assertIsInstance(h.antecedent, frozenset)
        # every snapshot field is JSON-native
        self.assertTrue(is_json_native(snap))

    def test_memory_snapshot_antecedent_is_list(self):
        s = NestSession(storage=NestStorage(Path(tempfile.mkdtemp())), seed=7)
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy",
                                  description="easy training"))
        s.teaching.apply(parse_teaching_text("I think S1 opens door D1"))
        s.run_steps(2)
        for h in s.memory_snapshot()["hypotheses"]:
            self.assertIsInstance(h["antecedent"], list)
            self.assertIsInstance(h["antecedent_text"], str)


class TestNestRuntimeE2E(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._storage = Path(self._tmp.name)
        self._start_server()

    def tearDown(self):
        self._stop_server()
        self._tmp.cleanup()

    # -- server lifecycle -------------------------------------------------
    def _start_server(self):
        self.server, self.app = make_server(port=0, storage_root=self._storage,
                                            api_adapter=None)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever,
                                        daemon=True)
        self._thread.start()

    def _stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)

    # -- HTTP helpers ------------------------------------------------------
    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return r.status, json.loads(r.read().decode())

    def _post(self, path, body):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"error": raw}
            return e.code, payload

    def _wait(self, cond, timeout=60.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, st = self._get("/api/state")
            if cond(st):
                return st
            time.sleep(0.03)
        self.fail("timeout waiting for server state condition")

    def _step_until(self, cond, timeout=90.0):
        """Advance via HTTP single-steps until cond(state) is true."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, st = self._get("/api/state")
            if cond(st):
                return st
            code, _ = self._post("/api/control", {"op": "step"})
            if code != 200:
                self.fail(f"single-step control failed with HTTP {code}")
            time.sleep(0.02)
        self.fail("timeout waiting for step condition")

    # -- the founder workflow ----------------------------------------------
    def test_full_runtime_workflow(self):
        # 1. initial world initialization and JSON response
        code, st = self._get("/api/state")
        self.assertEqual(code, 200)
        self.assertTrue(st["world"]["present"],
                        "Academy world visible without starting training")
        self.assertIn(st["status"], ("idle", "paused"))
        self.assertTrue(is_json_native(st))
        self.assertTrue(st["challenge"])

        # 2. Start and Single-step
        code, _ = self._post("/api/control", {"op": "start"})
        self.assertEqual(code, 200)
        st = self._wait(lambda s: s["world"]["steps"] > 0)
        self.assertGreater(st["world"]["steps"], 0)
        code, _ = self._post("/api/control", {"op": "stop"})
        self.assertEqual(code, 200)
        before = self._get("/api/state")[1]["world"]["steps"]
        code, _ = self._post("/api/control", {"op": "step"})
        self.assertEqual(code, 200)
        st = self._wait(lambda s: s["world"]["steps"] > before)
        self.assertGreaterEqual(st["world"]["steps"], before + 1)

        # use an easy training pool so the real student generates hypotheses
        code, _ = self._post("/api/control",
                              {"op": "new_challenge",
                               "params": {"pool_tag": "train",
                                          "difficulty": "easy",
                                          "description": "easy training"}})
        self.assertEqual(code, 200)

        # 3. a student-generated hypothesis
        st = self._step_until(lambda s: len(s["memory"]["hypotheses"]) > 0)
        self.assertGreater(len(st["memory"]["hypotheses"]), 0)

        # 4. GET /api/state after hypotheses exist — serialization audit
        code, st = self._get("/api/state")
        self.assertEqual(code, 200)
        hyps = st["memory"]["hypotheses"]
        self.assertGreater(len(hyps), 0)
        for h in hyps:
            self.assertIsInstance(h["antecedent"], list)
            self.assertTrue(all(isinstance(a, list) for a in h["antecedent"]))
            self.assertIsInstance(h["antecedent_text"], str)
        self.assertTrue(is_json_native(st))

        # 5. teaching guidance: "try switch S1"
        base_stats = st["teaching"]["stats"]
        code, st = self._post("/api/teaching", {"text": "try switch S1"})
        self.assertEqual(code, 200)
        self.assertEqual(st["teaching"]["stats"]["guidance_applied"],
                         base_stats["guidance_applied"] + 1)
        self.assertEqual(len(st["teaching"]["teacher_claims"]), 0,
                         "guidance must never become a causal claim")

        # 6. teaching objective: "discover how to open D1"
        code, st = self._post("/api/teaching",
                              {"text": "discover how to open D1"})
        self.assertEqual(code, 200)
        self.assertTrue(any("discover how to open D1" == o["text"]
                            for o in st["teaching"]["objectives"]))
        self.assertEqual(st["teaching"]["stats"]["objectives_added"],
                         base_stats["objectives_added"] + 1)

        # 7. teaching claim: "I think S1 opens door D1"
        hyp_before = len(st["memory"]["hypotheses"])
        code, st = self._post("/api/teaching",
                              {"text": "I think S1 opens door D1"})
        self.assertEqual(code, 200)
        self.assertTrue(st["teaching"]["teacher_claims"])
        claims = [h for h in st["memory"]["hypotheses"]
                  if h["door"] == "D1" and h["status"] == "hypothesis"
                  and h["sup"] == 0]
        self.assertTrue(claims, "claim injected as unverified candidate")
        self.assertGreaterEqual(len(st["memory"]["hypotheses"]), hyp_before)
        self.assertTrue(is_json_native(st))

        # 8. rejected teaching input with a clear error
        inputs_before = len(st["teaching"]["inputs"])
        code, resp = self._post("/api/teaching", {"text": "door D2"})
        self.assertEqual(code, 400)
        self.assertIn("error", resp)
        self.assertIn("Supported examples", resp["error"])
        _, st = self._get("/api/state")
        self.assertEqual(len(st["teaching"]["inputs"]), inputs_before,
                         "rejected instruction must not be recorded")

        # 8b. Reset episode over HTTP: world resets, memory kept
        code, st = self._post("/api/control", {"op": "reset"})
        self.assertEqual(code, 200)
        self.assertTrue(st["world"]["present"])
        self.assertEqual(st["world"]["steps"], 0)
        self.assertGreaterEqual(len(st["memory"]["hypotheses"]), hyp_before)

        # 9. save -> restart -> restore with populated memory
        code, st = self._post("/api/control", {"op": "save"})
        self.assertEqual(code, 200)
        self.assertTrue(st["memory"]["hypotheses"])
        saved_hyps = st["memory"]["hypotheses"]
        saved_generated = st["memory"]["generated"]

        self._stop_server()
        self._start_server()

        code, st = self._get("/api/state")
        self.assertEqual(code, 200)
        self.assertTrue(st["world"]["present"])
        self.assertEqual(len(st["memory"]["hypotheses"]), 0,
                         "restarted app starts with empty memory")

        code, st = self._post("/api/control", {"op": "restore"})
        self.assertEqual(code, 200)
        self.assertEqual(len(st["memory"]["hypotheses"]),
                         len(saved_hyps))
        self.assertEqual(st["memory"]["generated"], saved_generated)
        self.assertTrue(st["world"]["present"], "world visible after restore")
        self.assertTrue(is_json_native(st))

        # 10. frontend rendering after learning
        with urllib.request.urlopen(self.base + "/", timeout=30) as r:
            html = r.read().decode()
        self.assertIn("METARGON NEST", html)
        self.assertIn("teach-hints", html)
        self.assertIn('"discover how to open D1"', html)
        self.assertNotIn('"door D2"', html)  # unsupported suggestion removed
        audit = _ROOT / "nest" / "tools" / "frontend_audit.py"
        r = subprocess.run([sys.executable, str(audit), "--smoke"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("balance checks: OK", r.stdout)
        if "node --check: OK" in r.stdout or "DOM smoke test: OK" in r.stdout:
            self.assertIn("node --check: OK", r.stdout)
        # real agent state carried the learning (not a mock)
        self.assertGreater(self.app.session.agent.memory.generated, 0)


if __name__ == "__main__":
    unittest.main()
