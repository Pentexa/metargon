"""Tests for the pilot flow: teacher calls, validation gating, quota
enforcement, persistence, and the offline fallback."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from academy.evren.curriculum_teacher import CurriculumTeacher, extract_json_object
from academy.evren.provider import (
    ProviderConfig, QuotaGuard, EvrenProvider, EvrenResponseError, ChatResponse,
)
from academy.evren.pilot import run_pilot
from academy.evren.task_registry import TaskRegistry
from academy.tests.helpers import cached_splits, temp_registry

DISCOVERY_BODY = json.dumps({
    "data": [
        {"id": "glm-5.3-2026-01"},
        {"id": "deepseek-v4.1-flash"},
    ]})


class RecordingProvider(EvrenProvider):
    """Serves canned bodies; counts requests; never touches the network."""

    def __init__(self, chat_bodies, quota=None):
        super().__init__(ProviderConfig(), quota=quota or QuotaGuard(max_calls=50))
        self.chat_bodies = list(chat_bodies)
        self._models_cache = json.loads(DISCOVERY_BODY)["data"]
        self.request_count = 0

    def _request(self, method, path, payload=None, headers_extra=None, timeout=None):
        self.request_count += 1
        if path == "/models":
            return 200, DISCOVERY_BODY, None
        if not self.chat_bodies:
            raise AssertionError("no canned chat body left")
        item = self.chat_bodies.pop(0)
        if isinstance(item, tuple):      # (status, body) error simulation
            return item[0], item[1], None
        return 200, item, None


def _chat_body(content, tokens=40):
    return json.dumps({
        "choices": [{"message": {"content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens // 2,
                  "completion_tokens": tokens - tokens // 2,
                  "total_tokens": tokens},
        "model": "glm-5.3-2026-01",
    })


def _cfg(tmp, max_calls=50, allow_unlimited=False):
    return {
        "evren": {
            "api_key_env": "EVREN_API_KEY",
            "max_calls_per_run": max_calls,
            "allow_unlimited": allow_unlimited,
            "registry_dir": tmp,
            "seed": 1,
            "temperature": 0.7,
            "max_tokens": 256,
        },
        "seeds": {"pool_split": 20260901},
    }


GOOD_RAW = {
    "task_id": "evren-glm-0001",
    "topology": "TP2",
    "d1_rule": [[["sw", "S2", True]]],
    "d2_rule": [[["empty", "P2"], ["sw", "S1", True]]],
    "d1_delay": False,
    "d2_delay": False,
    "difficulty_intent": "medium",
    "rationale": "probe",
}

GOOD_RAW_2 = {
    "task_id": "evren-glm-0002",
    "topology": "TP2",
    "d1_rule": [[["sw", "S2", True]]],
    "d2_rule": [[["empty", "P1"], ["sw", "S1", False]]],
    "d1_delay": False,
    "d2_delay": False,
    "difficulty_intent": "medium",
    "rationale": "probe",
}


class TestExtractJson(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced_object(self):
        text = "```json\n{\"a\": 1}\n```"
        self.assertEqual(extract_json_object(text), {"a": 1})

    def test_prose_around_object(self):
        text = "Here you go: {\"a\": 1, \"b\": [true, null]} thanks!"
        self.assertEqual(extract_json_object(text), {"a": 1, "b": [True, None]})

    def test_no_object_returns_none(self):
        self.assertIsNone(extract_json_object("no json here"))
        self.assertIsNone(extract_json_object(""))
        self.assertIsNone(extract_json_object(None))

    def test_broken_json_returns_none(self):
        self.assertIsNone(extract_json_object('{"a": '))


class TestCurriculumTeacher(unittest.TestCase):
    def test_propose_task_ok(self):
        p = RecordingProvider([_chat_body(json.dumps(GOOD_RAW))])
        t = CurriculumTeacher(p, model="primary", seed=7)
        attempt = t.propose_task(slot=1)
        self.assertEqual(attempt.status, "ok")
        self.assertEqual(attempt.raw["task_id"], "evren-glm-0001")
        self.assertEqual(attempt.meta["model_id"], "glm-5.3-2026-01")
        self.assertEqual(attempt.meta["seed"], 7)
        self.assertGreater(attempt.meta["prompt_tokens"], 0)
        self.assertEqual(p.request_count, 1)

    def test_parse_error_status(self):
        p = RecordingProvider([_chat_body("I cannot produce JSON today")])
        t = CurriculumTeacher(p, model="primary", seed=1)
        attempt = t.propose_task()
        self.assertEqual(attempt.status, "parse_error")

    def test_quota_exhausted_stops_calls(self):
        quota = QuotaGuard(max_calls=2)
        p = RecordingProvider(
            [_chat_body(json.dumps(GOOD_RAW)), _chat_body(json.dumps(GOOD_RAW)),
             _chat_body(json.dumps(GOOD_RAW))],
            quota=quota)
        t = CurriculumTeacher(p, model="primary", seed=1)
        attempts = t.propose_batch(5)
        statuses = [a.status for a in attempts]
        self.assertEqual(statuses.count("ok"), 2)
        self.assertEqual(statuses[2], "quota")
        self.assertEqual(p.request_count, 2)  # no third network call


class TestPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = cached_splits()

    def test_pilot_online_with_recording_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 2 good, 1 invalid-schema, 1 leaked -> 2 approved, 2 rejected
            bad = dict(GOOD_RAW, task_id="bad-schema",
                       topology="TP9")
            leaked = dict(GOOD_RAW, task_id="leaked",
                          rationale="the answer is to load P1")
            provider = RecordingProvider([
                _chat_body(json.dumps(GOOD_RAW)),
                _chat_body(json.dumps(GOOD_RAW_2)),
                _chat_body(json.dumps(bad)),
                _chat_body(json.dumps(leaked)),
            ])
            summary = run_pilot(_cfg(tmp), self.splits, n=4,
                                results_dir=tmp, run_id="pilot-test",
                                provider=provider)
            self.assertEqual(summary["source"], "evren")
            self.assertEqual(summary["approved"], 2)
            self.assertEqual(summary["rejected"], 2)
            self.assertEqual(provider.request_count, 4)
            registry = TaskRegistry(tmp, run_id="pilot-test")
            self.assertEqual(len(registry.approved_tasks()), 2)
            self.assertEqual(len(registry.rejected()), 2)
            tok = registry.token_summary()
            self.assertEqual(tok["totals"]["requests"], 4)
            self.assertEqual(tok["totals"]["total_tokens"], 4 * 40)
            self.assertEqual(tok["error_count"], 0)
            # provenance is persisted
            task = registry.approved_tasks()[0]
            self.assertEqual(task.model_id, "glm-5.3-2026-01")
            self.assertEqual(task.provenance_note, "evren teacher")

    def test_pilot_quota_stops_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = RecordingProvider(
                [_chat_body(json.dumps(GOOD_RAW)) for _ in range(10)],
                quota=QuotaGuard(max_calls=3))
            summary = run_pilot(_cfg(tmp), self.splits, n=10,
                                results_dir=tmp, run_id="pilot-quota",
                                provider=provider)
            # 3 calls are allowed; the 4th slot is stopped by the quota
            # before any network request is made.
            self.assertEqual(provider.request_count, 3)
            self.assertEqual(summary["quota_calls"]["calls"], 3)
            self.assertEqual(summary["approved"] + summary["rejected"], 3)
            self.assertEqual(len(summary["attempts"]), 4)

    def test_pilot_offline_fallback_when_provider_fails(self):
        class BrokenProvider(EvrenProvider):
            def discover_models(self, force=False):
                raise EvrenResponseError("network unreachable")

            def _request(self, *a, **k):
                raise AssertionError("must not be used")

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_pilot(_cfg(tmp), self.splits, n=3,
                                results_dir=tmp, run_id="pilot-fail",
                                provider=BrokenProvider())
            self.assertEqual(summary["source"], "offline_generator")
            self.assertIsNotNone(summary["fallback_reason"])
            self.assertEqual(summary["approved"], 3)

    def test_pilot_offline_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_pilot(_cfg(tmp), self.splits, n=3,
                                results_dir=tmp, run_id="pilot-offline",
                                force_offline=True)
            self.assertEqual(summary["source"], "offline_generator")
            self.assertEqual(summary["approved"], 3)
            registry = TaskRegistry(tmp, run_id="pilot-offline")
            self.assertEqual(len(registry.approved_tasks()), 3)

    def test_pilot_persists_usage_errors_on_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = RecordingProvider([(500, "provider exploded"),
                                          (500, "provider exploded again")])
            summary = run_pilot(_cfg(tmp), self.splits, n=2,
                                results_dir=tmp, run_id="pilot-errors",
                                provider=provider)
            self.assertEqual(provider.request_count, 2)
            registry = TaskRegistry(tmp, run_id="pilot-errors")
            tok = registry.token_summary()
            self.assertEqual(tok["error_count"], 2)
            self.assertEqual(summary["rejected"], 2)


if __name__ == "__main__":
    unittest.main()
