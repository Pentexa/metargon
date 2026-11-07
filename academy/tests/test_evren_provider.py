"""Tests for the EVREN provider: quota enforcement, retries, model
resolution, usage accounting, and API-key hygiene. No network access is
ever made (all HTTP is stubbed)."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from academy.evren.provider import (
    DEFAULT_MAX_CALLS_PER_RUN, ChatResponse, EvrenAuthError, EvrenError,
    EvrenModelNotFound, EvrenProvider, EvrenQuotaExceeded,
    EvrenResponseError, ProviderConfig, QuotaGuard, UsageRecord,
    api_key_from_env, _mask_headers,
)


class TestQuotaGuard(unittest.TestCase):
    def test_default_budget_is_10(self):
        g = QuotaGuard()
        self.assertEqual(g.max_calls, DEFAULT_MAX_CALLS_PER_RUN)
        self.assertEqual(g.remaining(), 10)

    def test_zero_treated_as_default(self):
        g = QuotaGuard(max_calls=0)
        self.assertEqual(g.max_calls, DEFAULT_MAX_CALLS_PER_RUN)

    def test_negative_requires_explicit_unlimited(self):
        with self.assertRaises(ValueError):
            QuotaGuard(max_calls=-1, allow_unlimited=False)
        g = QuotaGuard(max_calls=-1, allow_unlimited=True)
        self.assertIsNone(g.max_calls)
        self.assertEqual(g.remaining(), -1)

    def test_quota_raised_before_any_network(self):
        g = QuotaGuard(max_calls=2)
        rec = UsageRecord(ts=0, endpoint="/chat/completions", model="m",
                          prompt_tokens=1, completion_tokens=1,
                          latency_ms=1.0, status="ok")
        g.record(rec)
        g.record(rec)
        with self.assertRaises(EvrenQuotaExceeded):
            g.check()

    def test_token_cap(self):
        g = QuotaGuard(max_calls=100, max_total_tokens=10)
        g.record(UsageRecord(ts=0, endpoint="e", model="m",
                             prompt_tokens=6, completion_tokens=4,
                             latency_ms=1.0, status="ok", total_tokens=10))
        with self.assertRaises(EvrenQuotaExceeded):
            g.check()

    def test_state_serializable(self):
        g = QuotaGuard(run_id="r1")
        st = g.state()
        self.assertEqual(st["max_calls"], 10)
        self.assertEqual(st["run_id"], "r1")
        self.assertEqual(st["remaining"], 10)


class TestKeyHygiene(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("EVREN_API_KEY")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("EVREN_API_KEY", None)
        else:
            os.environ["EVREN_API_KEY"] = self._old

    def test_env_key_never_surfaces_in_diagnostics(self):
        os.environ["EVREN_API_KEY"] = "sk-super-secret-value"
        p = EvrenProvider(ProviderConfig())
        d = json.dumps(p.diagnostics(), sort_keys=True)
        self.assertNotIn("sk-super-secret-value", d)
        self.assertNotIn("super-secret", d)
        self.assertIn("auth_key_present", d)

    def test_mask_headers(self):
        headers = {"Authorization": "Bearer sk-abc", "Content-Type": "application/json"}
        masked = _mask_headers(headers)
        self.assertEqual(masked["Authorization"], "***")
        self.assertNotIn("sk-abc", json.dumps(masked))

    def test_env_precedence_over_dotenv(self):
        os.environ["EVREN_API_KEY"] = "from-env"
        # _load_dotenv must not override an existing environment variable.
        with mock.patch("academy.evren.provider._load_dotenv") as m:
            self.assertEqual(api_key_from_env(), "from-env")
            m.assert_not_called()


class FakeTransport:
    """Replaces EvrenProvider._request with canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, path, payload=None, headers_extra=None, timeout=None):
        self.calls.append((method, path))
        if not self.responses:
            raise AssertionError("no canned response left")
        return self.responses.pop(0)


class FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    def read(self):
        return self._body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code):
    import urllib.error

    class _E(urllib.error.HTTPError):
        def read(self):
            return b"server error"

        def close(self):
            pass

    return _E("https://x", code, "err", {}, None)


class TestModelResolution(unittest.TestCase):
    def setUp(self):
        self.models = [
            {"id": "glm-5.3-2026-01"},
            {"id": "deepseek-v4.1-flash"},
            {"id": "glm-4.6"},
        ]

    def test_exact_id_match(self):
        p = EvrenProvider(ProviderConfig())
        p._models_cache = self.models
        self.assertEqual(p.resolve_model("glm-5.3-2026-01"), "glm-5.3-2026-01")

    def test_display_name_matched(self):
        p = EvrenProvider(ProviderConfig())
        p._models_cache = self.models
        self.assertEqual(p.resolve_model("GLM-5.3"), "glm-5.3-2026-01")
        self.assertEqual(p.resolve_model("DeepSeek V4.1 Flash"),
                         "deepseek-v4.1-flash")

    def test_unresolvable_raises(self):
        p = EvrenProvider(ProviderConfig())
        p._models_cache = self.models
        with self.assertRaises(EvrenModelNotFound):
            p.resolve_model("GPT-99")

    def test_discovery_parses_data_list(self):
        body = json.dumps({"data": [{"id": "glm-5.3-2026-01"}]})
        p = EvrenProvider(ProviderConfig())
        p._request = FakeTransport([(200, body, None)])
        models = p.discover_models()
        self.assertEqual(models[0]["id"], "glm-5.3-2026-01")
        self.assertTrue(p.connected)


class TestChatAndRetries(unittest.TestCase):
    def _provider(self, responses, quota=None):
        p = EvrenProvider(ProviderConfig(), quota=quota or QuotaGuard(max_calls=20))
        p._request = FakeTransport(responses)
        return p

    def test_chat_parses_and_records_usage(self):
        body = json.dumps({
            "choices": [{"message": {"content": "{\"a\": 1}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7,
                      "total_tokens": 19},
            "model": "glm-5.3-2026-01",
        })
        p = self._provider([(200, body, None)])
        resp = p.chat([{"role": "user", "content": "hi"}], model="glm-5.3-2026-01")
        self.assertIsInstance(resp, ChatResponse)
        self.assertIn("a", json.loads(resp.content))
        self.assertEqual(resp.total_tokens, 19)
        self.assertEqual(p.quota.calls, 1)
        self.assertEqual(p.quota.total_tokens, 19)
        self.assertEqual(len(p.usage_records()), 1)

    def test_retry_429_then_success(self):
        body = json.dumps({"choices": [{"message": {"content": "ok"},
                                        "finish_reason": "stop"}],
                           "usage": {}, "model": "m"})
        p = EvrenProvider(ProviderConfig(max_retries=3))
        with mock.patch.dict(os.environ, {"EVREN_API_KEY": "sk-test"}), \
                mock.patch("urllib.request.urlopen", side_effect=[
                    _http_error(429), FakeResponse(200, body)]):
            resp = p.chat([{"role": "user", "content": "x"}], model="m")
        self.assertEqual(resp.content, "ok")

    def test_transient_network_error_then_success(self):
        body = json.dumps({"choices": [{"message": {"content": "ok"},
                                        "finish_reason": "stop"}],
                           "usage": {}, "model": "m"})
        p = EvrenProvider(ProviderConfig(max_retries=3))
        with mock.patch.dict(os.environ, {"EVREN_API_KEY": "sk-test"}), \
                mock.patch("urllib.request.urlopen", side_effect=[
                    TimeoutError("timed out"), FakeResponse(200, body)]):
            resp = p.chat([{"role": "user", "content": "x"}], model="m")
        self.assertEqual(resp.content, "ok")

    def test_retries_exhausted_raises(self):
        p = EvrenProvider(ProviderConfig(max_retries=2))
        with mock.patch.dict(os.environ, {"EVREN_API_KEY": "sk-test"}), \
                mock.patch("urllib.request.urlopen", side_effect=[
                    _http_error(503), _http_error(503), _http_error(503)]):
            with self.assertRaises(EvrenResponseError):
                p.chat([{"role": "user", "content": "x"}], model="m")

    def test_nonretryable_4xx_raises(self):
        p = self._provider([(400, "bad request", None)])
        with self.assertRaises(EvrenResponseError):
            p.chat([{"role": "user", "content": "x"}], model="m")

    def test_auth_error_raises_evren_auth(self):
        p = self._provider([(401, "unauthorized", None)])
        with self.assertRaises(EvrenAuthError):
            p.chat([{"role": "user", "content": "x"}], model="m")

    def test_quota_blocks_network_call(self):
        body = json.dumps({"choices": [{"message": {"content": "ok"},
                                        "finish_reason": "stop"}],
                           "usage": {}, "model": "m"})
        p = self._provider([(200, body, None)], quota=QuotaGuard(max_calls=1))
        p.chat([{"role": "user", "content": "x"}], model="m")
        with self.assertRaises(EvrenQuotaExceeded):
            p.chat([{"role": "user", "content": "x"}], model="m")
        self.assertEqual(len(p._request.calls), 1)  # no second network call

    def test_malformed_completion_raises(self):
        p = self._provider([(200, json.dumps({"choices": []}), None)])
        with self.assertRaises(EvrenResponseError):
            p.chat([{"role": "user", "content": "x"}], model="m")

    def test_usage_serialization_roundtrip(self):
        rows = EvrenProvider.serialize_usage([
            UsageRecord(ts=1.0, endpoint="/chat/completions", model="m",
                        prompt_tokens=2, completion_tokens=3, latency_ms=4.0,
                        status="ok", total_tokens=5)])
        loaded = EvrenProvider.deserialize_usage(rows)
        self.assertEqual(loaded[0].total_tokens, 5)
        self.assertEqual(loaded[0].status, "ok")


if __name__ == "__main__":
    unittest.main()
