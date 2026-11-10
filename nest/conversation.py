"""METARGON NEST v0.2 — conversation engine.

Pipeline:

    user text -> (LLM | offline interpreter) -> validated action or reply
              -> real agent execution -> grounded reply

The LLM path is primary when a user API key is configured. The offline
interpreter remains as an explicitly labeled fallback ("command mode"),
never as a fake conversational intelligence.

Both paths ground replies in the real snapshot; actions are validated
against the supported intent set and executed only through the session.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from .llm import (
    SYSTEM_PROMPT,
    VALID_ACTION_INTENTS,
    build_grounding,
    extract_action_tag,
)

FALLBACK_LABEL = "command-mode (offline interpreter)"

INTENT_LABELS = {
    "goal_open_door_d1": ("challenge", "open door D1"),
    "goal_open_door_d2": ("challenge", "open door D2"),
    "goal_reach": ("challenge", "reach the goal"),
    "challenge_easier": ("challenge", "easier challenge"),
    "challenge_harder": ("challenge", "harder challenge"),
    "challenge_new": ("challenge", "new challenge"),
    "control_start": ("control", "start"),
    "control_pause": ("control", "pause"),
    "control_continue": ("control", "continue"),
    "control_step": ("control", "single step"),
    "control_stop": ("control", "stop"),
    "control_reset": ("control", "reset episode"),
    "persistence_save": ("persistence", "save"),
    "persistence_restore": ("persistence", "restore"),
    "persistence_fresh": ("persistence", "fresh start"),
}


@dataclass
class ConversationEngine:
    session: object                       # NestSession
    client: object = None                 # OpenAICompatibleClient (None = offline)
    interpreter: object = None            # deterministic interpreter (fallback)
    history: list = field(default_factory=list)
    max_history: int = 20

    @property
    def mode(self) -> str:
        return "llm" if (self.client is not None and self.client.enabled) else "command"

    @property
    def budget(self):
        return self.client.budget if (self.client is not None and self.client.enabled) else None

    def _append_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            del self.history[: len(self.history) - self.max_history]

    def respond(self, text: str) -> dict:
        """Return {reply, action, intent, mode, grounded, action_note, snapshot}."""
        snapshot = self.session.snapshot()
        if self.mode == "llm":
            try:
                reply, action = self._respond_llm(text)
                note = self._execute_action(action) if action else None
                snapshot = self.session.snapshot()
                return {"reply": reply, "action": action, "intent": action,
                        "mode": "llm", "grounded": True, "action_note": note,
                        "snapshot": snapshot}
            except Exception as exc:
                # provider failure: honest degradation, never fake success
                fallback = self._respond_fallback(text)
                return {"reply": (f"(language model unavailable: {exc})\n"
                                  + fallback["reply"]),
                        "action": fallback["action"],
                        "intent": fallback["intent"],
                        "mode": "command", "grounded": True,
                        "action_note": None,
                        "snapshot": self.session.snapshot()}
        return self._respond_fallback(text)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------
    def _respond_llm(self, text: str) -> tuple[str, Optional[str]]:
        grounding = build_grounding(self.session.snapshot(), self.history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system",
             "content": "GROUNDING (real agent state, JSON):\n"
                         + json.dumps(grounding, ensure_ascii=False)},
        ]
        for turn in self.history[-10:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": text})
        raw = self.client.chat(messages)
        self._append_turn("user", text)
        action = extract_action_tag(raw)
        if action not in VALID_ACTION_INTENTS:
            action = None
        clean = raw
        if action is not None:
            clean = raw.split("[[ACTION:")[0].strip()
        self._append_turn("assistant", clean)
        return clean, action

    # ------------------------------------------------------------------
    # Offline fallback (explicitly labeled, grounded in telemetry)
    # ------------------------------------------------------------------
    def _respond_fallback(self, text: str) -> dict:
        cmd = self.interpreter(text)
        reply, action = self._fallback_reply(cmd)
        note = self._execute_action(action) if action else None
        return {"reply": reply, "action": action, "intent": cmd.intent,
                "mode": "command", "grounded": True, "action_note": note,
                "snapshot": self.session.snapshot()}

    def _fallback_reply(self, cmd) -> tuple[str, Optional[str]]:
        intent = cmd.intent
        if cmd.direct_answer:
            return cmd.direct_answer, None
        if intent == "hello":
            return ("Hello! I'm Metargon. I explore the Academy, learn door "
                    "rules from evidence, and remember what I find. Ask me "
                    "what I remember, or give me a challenge."), None
        if intent == "help":
            return cmd.direct_answer or "I can open doors, reach the goal, and explain my learning.", None
        if intent == "memory_what_remember":
            return self._reply_what_remember(), None
        if intent == "memory_current_hypothesis":
            return self._reply_hypotheses(), None
        if intent == "memory_learned_today":
            return self._reply_learned_today(), None
        if intent == "goal_open_door_d1" or intent == "goal_open_door_d2":
            door = "D1" if intent.endswith("d1") else "D2"
            return None, f"goal_open_door_{door.lower()}"
        if intent == "goal_reach":
            return None, "goal_reach"
        if intent == "challenge_easier":
            return None, "challenge_easier"
        if intent == "challenge_harder":
            return None, "challenge_harder"
        if intent == "challenge_new":
            return None, "challenge_new"
        if intent == "control_start":
            return None, "control_start"
        if intent == "control_pause":
            return None, "control_pause"
        if intent == "control_continue":
            return None, "control_continue"
        if intent == "control_step":
            return None, "control_step"
        if intent == "control_stop":
            return None, "control_stop"
        if intent == "control_reset":
            return None, "control_reset"
        if intent == "persistence_save":
            return None, "persistence_save"
        if intent == "persistence_restore":
            return None, "persistence_restore"
        if intent == "persistence_fresh":
            return None, "persistence_fresh"
        if intent == "why_last_action":
            return self._reply_why_last_action(), None
        return ("I don't understand that yet — and I won't pretend I did it. "
                "Say 'help' to see what I can do."), None

    def _execute_action(self, action: str) -> Optional[str]:
        """Execute a validated intent through the session; returns a short
        status note or None. Never invents outcomes."""
        kind, label = INTENT_LABELS.get(action, (None, action))
        s = self.session
        if kind == "challenge":
            from .agent_session import Challenge
            if action == "goal_open_door_d1":
                ch = Challenge(focus_door="D1", seed=s.episode_index,
                               description="open door D1")
            elif action == "goal_open_door_d2":
                ch = Challenge(focus_door="D2", seed=s.episode_index,
                               description="open door D2")
            elif action == "goal_reach":
                ch = Challenge(seed=s.episode_index, description="reach the goal")
            elif action == "challenge_easier":
                ch = Challenge(difficulty="easy", seed=s.episode_index,
                               description="easy challenge")
            elif action == "challenge_harder":
                ch = Challenge(difficulty="hard", seed=s.episode_index,
                               description="hard challenge")
            else:
                ch = Challenge(seed=s.episode_index, description="new challenge")
            s.new_challenge(ch)
            return f"Goal accepted: {label}. Use Start/Step to execute."
        if kind == "control":
            if action == "control_start":
                s.start(); return "Started."
            if action == "control_pause":
                s.pause(); return "Paused."
            if action == "control_continue":
                s.continue_running(); return "Continuing."
            if action == "control_step":
                s.single_step(); return "One step executed."
            if action == "control_stop":
                s.stop(); return "Stopped."
            if action == "control_reset":
                s.reset_episode(); return "Episode reset (memory kept)."
            return None
        if kind == "persistence":
            if action == "persistence_save":
                info = s.save()
                return f"Saved. Checkpoint at {info['agent_state']}."
            if action == "persistence_restore":
                meta = s.restore()
                if meta is None:
                    return "No saved checkpoint found."
                return "Restored. Learned memory is intact."
            if action == "persistence_fresh":
                s.fresh()
                return "Fresh start. All learned memory cleared."
        return None

    # ------------------------------------------------------------------
    # Telemetry-grounded fallback replies
    # ------------------------------------------------------------------
    def _mem(self):
        return self.session.memory_snapshot()

    def _reply_what_remember(self):
        mem = self._mem()
        stats = mem["stats"]
        verified = stats.get("verified", 0)
        contradicted = stats.get("contradicted", 0)
        pending = stats.get("hypothesis", 0) + stats.get("predicted", 0)
        hv = [h for h in mem["hypotheses"] if h["status"] == "verified"]
        if hv:
            h = max(hv, key=lambda x: x["confidence"])
            return (f"I remember {verified} verified rule{'s' if verified != 1 else ''} "
                    f"(and {contradicted} contradicted, {pending} uncertain). "
                    f"My strongest verified belief: door {h['door']} opens when "
                    f"{self._atoms(h['antecedent_text'])}. I only say this because "
                    f"the evidence supports it.")
        if contradicted:
            return (f"I remember {contradicted} contradicted hypotheses and {pending} uncertain. "
                    f"None are verified yet — so I can't claim I know a door rule.")
        return "I don't remember any verified rules yet. I'm still exploring."

    def _reply_hypotheses(self):
        mem = self._mem()
        hs = mem["hypotheses"]
        if not hs:
            return "I have no hypotheses yet — I haven't observed enough."
        lines = []
        for h in hs[:8]:
            lines.append(f"- door {h['door']} opens when {self._atoms(h['antecedent_text'])} "
                         f"[{h['status']}, support {h['sup']}, contradiction {h['con']}]")
        tail = f" … and {len(hs)-8} more" if len(hs) > 8 else ""
        return "Current hypotheses:\n" + "\n".join(lines) + tail

    def _reply_learned_today(self):
        mem = self._mem()
        stats = mem["stats"]
        s = self.session
        return (f"This session so far: {s.episode_index} episodes, "
                f"{s.episode_successes} successful. Memory: "
                f"{stats.get('verified', 0)} verified, {stats.get('contradicted', 0)} "
                f"contradicted, {mem['generated']} generated total.")

    def _reply_why_last_action(self):
        s = self.session
        if s.last_action is None:
            return "I haven't acted yet."
        tr = s.trace_snapshot()
        src = None
        for key, count in tr.items():
            if key.startswith("source_") and count:
                src = key.replace("source_", "")
                break
        reason = s.last_action_reason or (f"source: {src}" if src else
                                          "I picked an action from my planner/probe logic.")
        return (f"My last action was {s.last_action} ({reason}). "
                f"It came from my actual planning/learning loop, not from a script.")

    @staticmethod
    def _atoms(text: str) -> str:
        return text.replace(" ∧ ", ", ") if text else "(empty)"
