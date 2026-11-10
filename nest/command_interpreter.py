"""METARGON NEST — conversation -> structured command interpreter.

A deterministic, offline interpreter covers the supported intent set so
the whole essential workflow works with NO API key. An optional
OpenAI-compatible adapter may translate free-form text into the SAME
structured intent set; it never issues decisions, observations, or
memories — it only helps understand what the user asked.

Intent rules:
- Unsolicited "success" is impossible: unsupported requests receive an
  honest explanation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

INTENTS = (
    "hello",
    "help",
    "memory_what_remember",
    "memory_current_hypothesis",
    "memory_learned_today",
    "goal_open_door_d1",
    "goal_open_door_d2",
    "goal_reach",
    "challenge_easier",
    "challenge_harder",
    "challenge_new",
    "control_start",
    "control_pause",
    "control_continue",
    "control_step",
    "control_stop",
    "control_reset",
    "persistence_save",
    "persistence_restore",
    "persistence_fresh",
    "why_last_action",
    "unsupported",
)


@dataclass
class Command:
    intent: str
    text: str = ""
    params: dict = field(default_factory=dict)
    direct_answer: Optional[str] = None

    def as_dict(self):
        return {"intent": self.intent, "text": self.text,
                "params": self.params, "direct_answer": self.direct_answer}


_HELLO_RE = re.compile(r"^\s*(hello|hi|hey|greetings|merhaba|selam)\b", re.I)
_REMEMBER_RE = re.compile(r"what do you remember|what do you know|what have you learned|ne hatırlıyorsun|ne biliyorsun", re.I)
_HYP_RE = re.compile(r"current hypothes|show me your hypothes|what are you testing|hang? hipotez|şu anki hipotez", re.I)
_LEARNED_RE = re.compile(r"what did you learn today|what did you learn|bugün ne öğrendin|ne öğrendin", re.I)
_DOOR_D1_RE = re.compile(r"open (the )?door d1|door d1|d1 kapısını aç|kapıyı aç", re.I)
_DOOR_D2_RE = re.compile(r"open (the )?door d2|door d2|d2 kapısını aç", re.I)
_GOAL_RE = re.compile(r"reach (the )?goal|go to (the )?goal|get to (the )?goal|hedefe ulaş|hedefe git", re.I)
_ANY_DOOR_RE = re.compile(r"open (the )?door|kapıyı aç|door", re.I)
_EASIER_RE = re.compile(r"easier|kolay|daha kolay", re.I)
_HARDER_RE = re.compile(r"harder|zor|daha zor|challenge", re.I)
_NEW_RE = re.compile(r"new challenge|new episode|yeni görev|yeni meydan okuma", re.I)
_START_RE = re.compile(r"^(start|begin|run|başlat|başla)\b", re.I)
_PAUSE_RE = re.compile(r"^(pause|dur|durdur)\b", re.I)
_CONTINUE_RE = re.compile(r"^(continue|resume|devam)\b", re.I)
_STEP_RE = re.compile(r"single[- ]step|one step|tek adım|adım adım", re.I)
_STOP_RE = re.compile(r"^(stop|cancel|dur|durdur)\b", re.I)
_RESET_RE = re.compile(r"reset (the )?episode|episode reset|bölümü sıfırla", re.I)
_SAVE_RE = re.compile(r"save|kaydet|save progress", re.I)
_RESTORE_RE = re.compile(r"restore|yükle|geri yükle", re.I)
_FRESH_RE = re.compile(r"fresh start|wipe|sıfırla|baştan", re.I)
_WHY_RE = re.compile(r"why did you do that|why|neden|niçin", re.I)
_TRY_DIFFERENT_RE = re.compile(r"try a different approach|farklı bir yol dene", re.I)


def interpret(text: str, api_adapter=None) -> Command:
    """Map user text to a Command. `api_adapter` (optional) is a callable
    text -> intent from the allowed set; returns None on error/unset."""
    t = text.strip()
    if not t:
        return Command("unsupported", t,
                       direct_answer="I didn't catch that. Try saying 'hello', asking what I remember, or asking me to open a door.")

    # safety: never pretend to execute shell / system commands
    if re.search(r"\b(shell|command|exec(ute)?|rm\b|delete|teleport|fly|sudo|powershell|cmd\.exe)\b", t, re.I):
        return Command("unsupported", t,
                       direct_answer=("I can't do that — and I won't pretend I did. "
                                      "I only act inside the Academy world: open doors, reach the goal, "
                                      "and learn rules. Say 'help' for what I can do."))

    if api_adapter is not None and _HELLO_RE.match(t) is None:
        try:
            intent = api_adapter(t)
            if intent in INTENTS:
                return Command(intent, t)
        except Exception:
            pass  # fall through to deterministic interpreter

    # greetings first
    if _HELLO_RE.match(t):
        return Command("hello", t,
                       direct_answer="Hello! I'm Metargon, a little learning agent. I explore the Academy world, "
                                    "learn door rules from evidence, and remember what I discover. You can ask me "
                                    "what I remember, tell me to open a door, or ask for help.")
    if _REMEMBER_RE.search(t):
        return Command("memory_what_remember", t)
    if _HYP_RE.search(t):
        return Command("memory_current_hypothesis", t)
    if _LEARNED_RE.search(t):
        return Command("memory_learned_today", t)
    if _DOOR_D1_RE.search(t):
        return Command("goal_open_door_d1", t)
    if _DOOR_D2_RE.search(t):
        return Command("goal_open_door_d2", t)
    if _GOAL_RE.search(t):
        return Command("goal_reach", t)
    if _ANY_DOOR_RE.search(t):
        return Command("goal_open_door_d1", t)
    if _HARDER_RE.search(t):
        return Command("challenge_harder", t)
    if _EASIER_RE.search(t):
        return Command("challenge_easier", t)
    if _NEW_RE.search(t):
        return Command("challenge_new", t)
    if _TRY_DIFFERENT_RE.search(t):
        return Command("challenge_new", t,
                       direct_answer="I'll try a fresh episode with what I have learned so far — a different attempt, "
                                    "not a claim that my strategy changed.")
    if _SAVE_RE.search(t):
        return Command("persistence_save", t)
    if _RESTORE_RE.search(t):
        return Command("persistence_restore", t)
    if _FRESH_RE.search(t):
        return Command("persistence_fresh", t)
    if _WHY_RE.search(t):
        return Command("why_last_action", t)
    if _STEP_RE.search(t):
        return Command("control_step", t)
    if _START_RE.match(t):
        return Command("control_start", t)
    if _CONTINUE_RE.match(t):
        return Command("control_continue", t)
    if _PAUSE_RE.match(t) or _STOP_RE.match(t):
        return Command("control_pause", t)
    if _RESET_RE.search(t):
        return Command("control_reset", t)
    if re.search(r"help|yardım", t, re.I):
        return Command("help", t, direct_answer=HELP_TEXT)
    return Command("unsupported", t,
                   direct_answer=("I don't understand that yet — and I won't pretend I did it. "
                                  "I can: greet you, answer what I remember, show my current hypothesis, "
                                  "open door D1/D2, reach the goal, start/pause/continue/step/stop/reset, "
                                  "save or restore progress, or start fresh. You can also ask 'why' about my last action."))


HELP_TEXT = (
    "I'm Metargon — a real learning agent, not a chatbot pretending. I live in the Academy world, "
    "where doors are opened by hidden rules I discover from evidence.\n\n"
    "You can say:\n"
    "  • 'Hello' — greet me\n"
    "  • 'What do you remember?' — my learned memory\n"
    "  • 'Show me your current hypothesis' — what I'm testing\n"
    "  • 'Open door D1' / 'Open door D2' — a real challenge\n"
    "  • 'Reach the goal' — the classic Academy objective\n"
    "  • 'Try a harder/easier challenge' — switch difficulty\n"
    "  • 'Start / Pause / Continue / Single-step / Stop / Reset episode'\n"
    "  • 'Why did you do that?' — explanation of my last action\n"
    "  • 'Save / Restore / Fresh start' — persistence\n"
    "  • 'Help' — this text\n\n"
    "Everything you see comes from my actual observations, actions, and memory."
)


class OpenAICompatibleAdapter:
    """Optional OpenAI-compatible NL->intent adapter (API only, user key).

    It never decides actions; it only suggests which structured intent the
    user meant. Errors degrade to the deterministic interpreter.
    """

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 5.0):
        self.api_key = api_key or os.environ.get("METARGON_NEST_API_KEY")
        self.base_url = (base_url or os.environ.get("METARGON_NEST_API_URL")
                         or "https://api.openai.com/v1")
        self.model = model or os.environ.get("METARGON_NEST_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self._session = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def __call__(self, text: str) -> Optional[str]:
        if not self.enabled:
            return None
        import urllib.request
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system",
                 "content": (f"Map the user's utterance to exactly one of these intents: "
                             f"{', '.join(INTENTS)}. Reply with only the intent token.")},
                {"role": "user", "content": text},
            ],
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
        for intent in INTENTS:
            if intent in content:
                return intent
        return None
