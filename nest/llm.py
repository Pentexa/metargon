"""METARGON NEST v0.2 — user-configured API language client + grounding.

The causal-learning engine stays authoritative. This module only:
- describes the agent's real state (grounding) in a sanitized, compact form
  for the language model;
- calls a user-configured OpenAI-compatible chat API;
- tracks call / token / cost budgets;
- never executes actions, never modifies hidden world state, and never
  invents observations or memories.

All credentials come from environment variables; nothing is embedded.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# environment configuration keys
ENV_KEY = "METARGON_NEST_API_KEY"
ENV_URL = "METARGON_NEST_API_URL"
ENV_MODEL = "METARGON_NEST_MODEL"
ENV_MAX_TOKENS = "METARGON_NEST_MAX_TOKENS"
ENV_TIMEOUT = "METARGON_NEST_TIMEOUT"
ENV_MAX_CALLS = "METARGON_NEST_MAX_CALLS"
ENV_MAX_COST = "METARGON_NEST_MAX_COST_USD"

DEFAULT_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class LlmBudget:
    max_calls: int = 100
    max_cost_usd: float = 2.0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        # rough standard pricing per 1M tokens (input/output)
        price_in = 0.15 / 1_000_000
        price_out = 0.60 / 1_000_000
        return self.prompt_tokens * price_in + self.completion_tokens * price_out

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.max_calls or self.cost_usd >= self.max_cost_usd

    def record(self, prompt_tokens: int, completion_tokens: int):
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def as_dict(self):
        return {"calls": self.calls, "max_calls": self.max_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 4),
                "max_cost_usd": self.max_cost_usd,
                "exhausted": self.exhausted}


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions client (stdlib only)."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 max_tokens: int = 500,
                 timeout: float = 15.0,
                 budget: Optional[LlmBudget] = None):
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_KEY)
        self.base_url = (base_url if base_url is not None
                         else os.environ.get(ENV_URL) or DEFAULT_URL)
        self.model = (model if model is not None
                      else os.environ.get(ENV_MODEL) or DEFAULT_MODEL)
        self.max_tokens = int(os.environ.get(ENV_MAX_TOKENS, max_tokens))
        self.timeout = float(os.environ.get(ENV_TIMEOUT, timeout))
        self.budget = budget or LlmBudget(
            max_calls=int(os.environ.get(ENV_MAX_CALLS, 100)),
            max_cost_usd=float(os.environ.get(ENV_MAX_COST, 2.0)))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], temperature: float = 0.4) -> str:
        """Returns the assistant text. Raises on transport/API errors."""
        if not self.enabled:
            raise RuntimeError("no API key configured (set METARGON_NEST_API_KEY)")
        if self.budget.exhausted:
            raise RuntimeError("API call/cost budget exhausted; fallback mode active")
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        content = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage") or {}
        self.budget.record(int(usage.get("prompt_tokens", 0)),
                            int(usage.get("completion_tokens", len(content) // 4)))
        return content


# ---------------------------------------------------------------------------
# Grounding: real agent state -> sanitized context for the LLM
# ---------------------------------------------------------------------------

def _fmt_atoms(antecedent_text: str, max_len: int = 120) -> str:
    text = antecedent_text.strip()
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def build_grounding(snapshot: dict, recent_history: list[dict]) -> dict:
    """Build a sanitized grounding dictionary from a real NEST snapshot.

    Only real telemetry is included; nothing is synthesized.
    """
    world = snapshot.get("world") or {}
    mem = snapshot.get("memory") or {}
    tr = snapshot.get("trace") or {}
    stats = mem.get("stats") or {}
    hyps = (mem.get("hypotheses") or [])[:8]
    challenge = snapshot.get("challenge")

    return {
        "status": snapshot.get("status"),
        "notice": snapshot.get("notice") or "",
        "challenge": {
            "description": challenge.get("description") if challenge else None,
            "focus_door": challenge.get("focus_door") if challenge else None,
        },
        "world": {
            "present": world.get("present", False),
            "steps": world.get("steps"),
            "max_steps": world.get("max_steps"),
            "doors": world.get("doors"),
            "last_action": world.get("last_action"),
            "last_valid": world.get("last_valid"),
            "last_events": world.get("last_events") or [],
            "finished": world.get("finished", False),
        },
        "memory": {
            "stats": stats,
            "generated": mem.get("generated"),
            "step": mem.get("step"),
            "verified_count": stats.get("verified", 0),
            "uncertain_count": stats.get("hypothesis", 0) + stats.get("predicted", 0),
            "contradicted_count": stats.get("contradicted", 0),
            "top_hypotheses": [
                {
                    "door": h.get("door"),
                    "antecedent": _fmt_atoms(h.get("antecedent_text", "")),
                    "status": h.get("status"),
                    "support": h.get("sup"),
                    "contradiction": h.get("con"),
                    "confidence": h.get("confidence"),
                    "from_earlier_episodes": h.get("from_earlier_episodes", False),
                }
                for h in hyps
            ],
        },
        "telemetry": {
            "plans_used": tr.get("plans_used"),
            "random_fallback_steps": tr.get("random_fallback_steps"),
            "memory_actions": tr.get("memory_actions"),
            "memory_correct": tr.get("memory_correct"),
            "memory_incorrect": tr.get("memory_incorrect"),
            "hypothesis_retrieved": tr.get("hypothesis_retrieved"),
            "cross_epoch_transfer_retrieved": tr.get("cross_epoch_transfer_retrieved"),
        },
        "conversation_history": recent_history[-12:],
    }


SYSTEM_PROMPT = """\
You are the voice of METARGON, a small learning agent that explores a grid
world Academy: it pushes objects, toggles switches, and must discover the
hidden rules that open doors. You are NOT a generic chatbot: you speak for
the real agent, using only the grounding data provided.

Rules:
1. Ground every answer in the grounding JSON. Never invent memories,
   observations, actions, achievements, door rules, or emotions.
2. A hypothesis is knowledge ONLY if its status is "verified". Otherwise
   call it a hypothesis or uncertainty. Never present an unverified
   hypothesis as established fact.
3. Distinguish conversational memory (what the human told you in this
   session) from causal learning (verified rules). Saying "I remember you
   taught me X" is fine when it is in the conversation history; saying "I
   learned rule X" is only allowed if grounding.memory shows it verified.
4. If the world shows the last attempt did not succeed, say so plainly
   ("the episode ended without reaching the goal" or "the door did not
   open"). Do not claim success unless grounding.world.finished is true and
   the goal/door was reached.
5. Be warm, curious, and honest. Use short replies (max ~120 words). You
   may answer in the language of the user (Turkish or English).
6. You may suggest experiments (e.g. "should we try door D2 next?") based
   on uncertainty present in the grounding. You may talk about your
   limits, but do not claim human consciousness or feelings.
7. If the user asks you to do something that requires an action (open a
   door, reach the goal, change challenge, start/pause/step, save etc.),
   include a single "action" tag at the end of your reply in the exact
   format: [[ACTION:<intent>]] where <intent> is one of:
   goal_open_door_d1, goal_open_door_d2, goal_reach, challenge_easier,
   challenge_harder, challenge_new, control_start, control_pause,
   control_continue, control_step, control_stop, control_reset,
   persistence_save, persistence_restore, persistence_fresh.
   Use [[ACTION:none]] when no action is requested. Never invent actions
   outside this list.
8. Never output anything that looks like system commands, hidden world
   state, or raw JSON beyond the action tag.
"""


def extract_action_tag(text: str) -> Optional[str]:
    """Pull [[ACTION:<intent>]] from an LLM reply; returns None if absent."""
    import re
    m = re.search(r"\[\[ACTION:([a-z0-9_]+)\]\]", text)
    if not m:
        return None
    return m.group(1)


VALID_ACTION_INTENTS = {
    "goal_open_door_d1", "goal_open_door_d2", "goal_reach",
    "challenge_easier", "challenge_harder", "challenge_new",
    "control_start", "control_pause", "control_continue", "control_step",
    "control_stop", "control_reset",
    "persistence_save", "persistence_restore", "persistence_fresh",
}
