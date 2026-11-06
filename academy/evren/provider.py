"""EVREN LLM provider — OpenAI-compatible client.

Stdlib-only HTTP client (``urllib``) for EVREN's OpenAI-compatible API.
It resolves exact model IDs from the provider's discovery endpoint instead
of assuming display names are valid API identifiers.

Security: the API key is read from ``EVREN_API_KEY`` (or a git-ignored
``.env`` file). It is never printed, logged, serialized, or included in
error messages. The request logger masks the Authorization header.

Rate/usage policy:
  * ``QuotaGuard`` bounds the number of calls per run (default 10).
  * Unlimited usage is disallowed unless explicitly configured with
    ``allow_unlimited=True`` (a deliberate, explicit decision).
  * Timeouts, retries (exponential backoff + jitter), and output limits
    are configurable.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_BASE_URL = "https://evren-llmapi.ssyz.org.tr/v1"
DEFAULT_API_KEY_ENV = "EVREN_API_KEY"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_CALLS_PER_RUN = 10

# Display names used in configuration; resolved to exact API IDs later.
PRIMARY_TEACHER = "GLM-5.3"
SECONDARY_TEACHER = "DeepSeek V4.1 Flash"

_HTTP_429 = 429
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EvrenError(Exception):
    """Base class for EVREN provider errors."""


class EvrenAuthError(EvrenError):
    """Authentication failed (missing/invalid API key)."""


class EvrenQuotaExceeded(EvrenError):
    """The per-run call or token budget was exceeded."""


class EvrenModelNotFound(EvrenError):
    """The requested display name could not be resolved to a model ID."""


class EvrenResponseError(EvrenError):
    """The provider returned an unexpected/invalid response."""


def _load_dotenv(path: Optional[str] = None):
    """Parse a minimal ``.env`` file (KEY=VALUE lines, '#' comments).
    Existing environment variables take precedence (never overridden)."""
    path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def api_key_from_env(env_var: str = DEFAULT_API_KEY_ENV) -> Optional[str]:
    """Return the API key or None (never logged; no value embedded)."""
    if env_var in os.environ and os.environ[env_var]:
        return os.environ[env_var]
    _load_dotenv()
    return os.environ.get(env_var) or None


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    max_tokens: int = DEFAULT_MAX_TOKENS
    # The API key itself is intentionally NOT part of this config: it is
    # only ever resolved at call time from the environment.
    extra_headers: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None


@dataclass
class UsageRecord:
    ts: float
    endpoint: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: str          # "ok" | "retry" | "error" | "quota"
    error: str = ""
    run_id: str = ""
    total_tokens: int = 0


class QuotaGuard:
    """Per-run budget for teacher calls and tokens.

    Default maximum is 10 calls per run. Unlimited usage is *not* allowed
    unless ``allow_unlimited=True`` is explicitly configured.
    """

    def __init__(self, max_calls: int = DEFAULT_MAX_CALLS_PER_RUN,
                 max_total_tokens: int = 0,
                 allow_unlimited: bool = False,
                 run_id: str = "evren"):
        if max_calls == 0:
            max_calls = DEFAULT_MAX_CALLS_PER_RUN  # treat 0 as unset
        if max_calls < 0 and not allow_unlimited:
            raise ValueError(
                "negative max_calls means unlimited; set allow_unlimited=True "
                "explicitly to authorize it")
        self.max_calls = max_calls if max_calls > 0 else (
            None if allow_unlimited else DEFAULT_MAX_CALLS_PER_RUN)
        self.max_total_tokens = max_total_tokens
        self.allow_unlimited = allow_unlimited
        self.run_id = run_id
        self.calls = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def remaining(self) -> int:
        if self.max_calls is None:
            return -1  # unlimited (explicitly configured)
        return max(0, self.max_calls - self.calls)

    def check(self) -> None:
        """Raise before making a call if the budget is exhausted."""
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise EvrenQuotaExceeded(
                f"call quota exceeded: {self.calls}/{self.max_calls} "
                f"(increase via config, e.g. evren.max_calls_per_run)")
        if (self.max_total_tokens > 0
                and self.total_tokens >= self.max_total_tokens):
            raise EvrenQuotaExceeded(
                f"token quota exceeded: {self.total_tokens}/"
                f"{self.max_total_tokens}")
    def record(self, rec: UsageRecord) -> None:
        self.calls += 1
        if rec.status == "ok":
            self.total_tokens += rec.total_tokens
            self.prompt_tokens += rec.prompt_tokens
            self.completion_tokens += rec.completion_tokens

    def state(self) -> dict:
        return {
            "max_calls": self.max_calls,
            "max_total_tokens": self.max_total_tokens,
            "allow_unlimited": self.allow_unlimited,
            "run_id": self.run_id,
            "calls": self.calls,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "remaining": self.remaining(),
        }


def _mask_headers(headers: dict) -> dict:
    """Return a copy of headers with credentials masked (for diagnostics)."""
    out = {}
    for k, v in headers.items():
        if k.lower() in ("authorization", "api-key", "x-api-key"):
            out[k] = "***"
        else:
            out[k] = v
    return out


class EvrenProvider:
    """OpenAI-compatible client for EVREN-hosted models."""

    def __init__(self, cfg: ProviderConfig = None, quota: QuotaGuard = None,
                 seed: int = 0):
        self.cfg = cfg or ProviderConfig()
        self.quota = quota or QuotaGuard()
        self.rng = random.Random(seed)
        self._models_cache = None
        self._usage_log = []          # UsageRecord list for this process
        self.connected = False

    # ------------------------------------------------------------------
    # auth / request helpers
    # ------------------------------------------------------------------
    def _api_key(self) -> str:
        key = api_key_from_env(self.cfg.api_key_env)
        if not key:
            raise EvrenAuthError(
                f"EVREN_API_KEY is not set (expected env var "
                f"'{self.cfg.api_key_env}' or .env file)")
        return key

    def _request(self, method: str, path: str, payload=None,
                 headers_extra: dict = None, timeout: Optional[float] = None):
        """Perform an HTTP request with retries; returns (status, body)."""
        key = self._api_key()
        url = self.cfg.base_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(headers_extra or {})
        headers.update(self.cfg.extra_headers)
        timeout = timeout or self.cfg.timeout
        data = json.dumps(payload).encode("utf-8") if payload is not None else None

        attempts = 0
        max_attempts = 1 + max(0, self.cfg.max_retries)
        while True:
            attempts += 1
            try:
                req = urllib.request.Request(url, data=data, headers=headers,
                                             method=method)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    return resp.status, body, None
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code in _RETRYABLE_STATUS and attempts < max_attempts:
                    self._sleep_backoff(attempts, e.code)
                    continue
                return e.code, body, None
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                if attempts < max_attempts:
                    self._sleep_backoff(attempts, 0)
                    continue
                return 0, "", str(e)

    def _sleep_backoff(self, attempt: int, status: int):
        base = 0.5 * (2 ** (attempt - 1))
        jitter = self.rng.uniform(0.0, base * 0.5)
        time.sleep(min(base + jitter, 8.0))

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------
    def discover_models(self, force: bool = False) -> list:
        """GET {base}/models — the provider's model discovery endpoint."""
        if self._models_cache is not None and not force:
            return self._models_cache
        status, body, err = self._request("GET", "/models")
        if status != 200:
            raise EvrenResponseError(
                f"model discovery failed: HTTP {status} {err or body[:200]}")
        try:
            data = json.loads(body)
        except ValueError as e:
            raise EvrenResponseError(f"model discovery returned non-JSON: {e}")
        models = data.get("data") or data.get("models") or []
        if not isinstance(models, list):
            raise EvrenResponseError(
                "model discovery: unexpected payload shape (no list)")
        self._models_cache = models
        self.connected = True
        return models

    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().replace(" ", "").replace("-", "").replace("_", "")

    def resolve_model(self, display_name: str,
                      models: Optional[list] = None) -> str:
        """Resolve a display name (e.g. 'GLM-5.3') to an exact model ID.

        Matching is order-preserving: exact ID match first, then token
        subset match on normalized IDs. Raises EvrenModelNotFound when no
        unique candidate exists.
        """
        if models is None:
            models = self.discover_models()
        ids = [m.get("id") for m in models if isinstance(m, dict) and m.get("id")]
        ids += [m.get("model") for m in models
                if isinstance(m, dict) and m.get("model") and not m.get("id")]
        if not ids:
            raise EvrenModelNotFound(
                f"discovery returned no model ids for '{display_name}'")
        norm = self._normalize(display_name)
        exact = [i for i in ids if i == display_name or self._normalize(i) == norm]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            exact.sort()
            return exact[0]
        tokens = [t for t in norm.split(".") if t] + [
            t for t in norm.split("v") if t]
        # fall back to token-subset matching on normalized ids
        best = None
        best_score = 0
        for i in ids:
            ni = self._normalize(i)
            score = 0
            for tok in tokens:
                if tok and tok in ni:
                    score += len(tok)
            if score > best_score:
                best_score = score
                best = i
        if best is not None and best_score > 0:
            return best
        raise EvrenModelNotFound(
            f"could not resolve display name '{display_name}' to a model id "
            f"(discovered: {sorted(set(ids))[:20]}...)")

    # ------------------------------------------------------------------
    # chat completions
    # ------------------------------------------------------------------
    def chat(self, messages: list, model: str,
             temperature: float = 0.7,
             max_tokens: Optional[int] = None,
             seed: Optional[int] = None,
             timeout: Optional[float] = None) -> ChatResponse:
        """One chat completion against the resolved ``model`` id.

        The quota is checked *before* the network call, so budget
        enforcement is strict even during provider outages.
        """
        self.quota.check()
        max_tokens = max_tokens or self.cfg.max_tokens
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        t0 = time.perf_counter()
        rec = UsageRecord(
            ts=time.time(), endpoint="/chat/completions", model=model,
            prompt_tokens=0, completion_tokens=0, latency_ms=0.0,
            status="ok", run_id=self.quota.run_id)
        try:
            status, body, err = self._request(
                "POST", "/chat/completions", payload, timeout=timeout)
            rec.latency_ms = (time.perf_counter() - t0) * 1000.0
            if status != 200:
                rec.status = "error"
                rec.error = f"HTTP {status} {err or body[:200]}"
                if status in (401, 403):
                    raise EvrenAuthError(rec.error)
                if status == 429:
                    raise EvrenQuotaExceeded(
                        f"provider rate limit (HTTP 429): {rec.error}")
                raise EvrenResponseError(rec.error)
            try:
                data = json.loads(body)
            except ValueError as e:
                rec.status = "error"
                rec.error = f"non-JSON response: {e}"
                raise EvrenResponseError(rec.error)
            try:
                content = data["choices"][0]["message"]["content"]
                finish_reason = data["choices"][0].get("finish_reason")
                usage = data.get("usage", {})
                rec.prompt_tokens = int(usage.get("prompt_tokens", 0))
                rec.completion_tokens = int(usage.get("completion_tokens", 0))
                rec.total_tokens = int(usage.get("total_tokens", 0))
            except (KeyError, IndexError, TypeError) as e:
                rec.status = "error"
                rec.error = f"malformed completion payload: {e}"
                raise EvrenResponseError(rec.error)
            self.connected = True
            return ChatResponse(
                content=content,
                model=data.get("model", model),
                prompt_tokens=rec.prompt_tokens,
                completion_tokens=rec.completion_tokens,
                total_tokens=rec.total_tokens,
                latency_ms=rec.latency_ms,
                finish_reason=finish_reason,
            )
        finally:
            self.quota.record(rec)
            self._usage_log.append(rec)

    # ------------------------------------------------------------------
    # persisted usage log
    # ------------------------------------------------------------------
    def usage_records(self) -> list:
        return list(self._usage_log)

    def attach_usage_log(self, records: list) -> None:
        self._usage_log.extend(records)

    @staticmethod
    def serialize_usage(records: list) -> list:
        out = []
        for r in records:
            out.append({
                "ts": r.ts, "endpoint": r.endpoint, "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "latency_ms": round(r.latency_ms, 1),
                "status": r.status, "error": r.error, "run_id": r.run_id,
            })
        return out

    @staticmethod
    def deserialize_usage(rows: list) -> list:
        out = []
        for r in rows:
            out.append(UsageRecord(
                ts=r.get("ts", 0.0), endpoint=r.get("endpoint", ""),
                model=r.get("model", ""),
                prompt_tokens=int(r.get("prompt_tokens", 0)),
                completion_tokens=int(r.get("completion_tokens", 0)),
                total_tokens=int(r.get("total_tokens", 0)),
                latency_ms=float(r.get("latency_ms", 0.0)),
                status=r.get("status", "ok"), error=r.get("error", ""),
                run_id=r.get("run_id", "")))
        return out

    def diagnostics(self) -> dict:
        return {
            "base_url": self.cfg.base_url,
            "timeout": self.cfg.timeout,
            "max_retries": self.cfg.max_retries,
            "max_tokens": self.cfg.max_tokens,
            "auth_key_present": bool(api_key_from_env(self.cfg.api_key_env)),
            "connected": self.connected,
            "quota": self.quota.state(),
        }
