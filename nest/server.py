"""METARGON NEST — local HTTP backend (stdlib only, localhost by default).

Serves the browser UI and exposes a minimal JSON API:

- GET  /api/state       -> full session snapshot (world, memory, trace)
- POST /api/command      -> {"text": "..."} conversational command
- POST /api/control      -> {"op": "start|pause|continue|step|stop|reset|new_challenge|save|restore|fresh", "params": {...}}
- GET  /api/config       -> configuration info (api mode, storage, intents)
- GET  /api/checkpoint   -> storage info

Responds with JSON; all mutations are confined to the application-owned
storage directory. The server never spawns arbitrary processes.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .agent_session import (
    Challenge, NestSession, STATUS_IDLE, STATUS_PAUSED, STATUS_RUNNING,
    STATUS_STOPPED, STATUS_FINISHED,
)
from .command_interpreter import HELP_TEXT, INTENTS, OpenAICompatibleAdapter, interpret
from .storage import NestStorage

STATIC_DIR = Path(__file__).resolve().parent / "static"


class NestApp:
    """Ties session + interpreter together and answers from real telemetry."""

    def __init__(self, storage_root=None, api_adapter=None):
        self.storage = NestStorage(storage_root)
        self.session = NestSession(storage=self.storage)
        self.api_adapter = api_adapter
        self._lock = threading.RLock()
        self.reply_cache = {}

    # -- conversational replies (deterministic, from real state) --------
    def handle_command(self, text: str) -> dict:
        cmd = interpret(text, api_adapter=self.api_adapter)
        reply = self._execute(cmd)
        return {"command": cmd.as_dict(), "reply": reply,
                "snapshot": self.session.snapshot()}

    def _execute(self, cmd) -> str:
        intent = cmd.intent
        if cmd.direct_answer:
            return cmd.direct_answer
        if intent == "hello":
            return ("Hello! I'm Metargon. I explore the Academy, learn door rules from evidence, "
                    "and remember what I find. Ask me what I remember, or give me a challenge.")
        if intent == "help":
            return HELP_TEXT
        if intent == "memory_what_remember":
            return self._reply_what_remember()
        if intent == "memory_current_hypothesis":
            return self._reply_hypotheses()
        if intent == "memory_learned_today":
            return self._reply_learned_today()
        if intent == "goal_open_door_d1" or intent == "goal_open_door_d2":
            door = "D1" if intent.endswith("d1") else "D2"
            return self._start_challenge_for_door(door)
        if intent == "goal_reach":
            return self._start_goal_challenge()
        if intent == "challenge_easier":
            return self._start_difficulty("easy")
        if intent == "challenge_harder":
            return self._start_difficulty("hard")
        if intent == "challenge_new":
            return self._start_difficulty("medium")
        if intent == "control_start":
            self.session.start()
            return "Starting. I'm running my observe → plan → act → learn cycle."
        if intent == "control_pause":
            self.session.pause()
            return "Paused."
        if intent == "control_continue":
            self.session.continue_running()
            return "Continuing."
        if intent == "control_step":
            self.session.single_step()
            return "One step executed."
        if intent == "control_stop":
            self.session.stop()
            return "Stopped."
        if intent == "control_reset":
            self.session.reset_episode()
            return "Episode reset (memory kept)."
        if intent == "persistence_save":
            info = self.session.save()
            return (f"Progress saved. Learned memory checkpoint written to "
                    f"{info['agent_state']}.")
        if intent == "persistence_restore":
            meta = self.session.restore()
            if meta is None:
                return "No saved checkpoint found — nothing to restore."
            return ("Restored saved progress. Learned memory is intact: "
                    f"{meta.get('episode_index', 0)} episodes, "
                    f"{meta.get('episode_successes', 0)} successes.")
        if intent == "persistence_fresh":
            self.session.fresh()
            return "Fresh start. All learned memory cleared."
        if intent == "why_last_action":
            return self._reply_why_last_action()
        return ("I don't understand that yet. I won't pretend I did it. "
                "Say 'help' to see what I can do.")

    # -- telemetry-based replies ----------------------------------------
    def _reply_what_remember(self):
        mem = self.session.memory_snapshot()
        stats = mem["stats"]
        verified = stats.get("verified", 0)
        contradicted = stats.get("contradicted", 0)
        pending = stats.get("hypothesis", 0) + stats.get("predicted", 0)
        hv = [h for h in mem["hypotheses"] if h["status"] == "verified"]
        # choose the most confident verified hypothesis to speak about
        if hv:
            h = max(hv, key=lambda x: x["confidence"])
            return (f"I remember {verified} verified rule{'s' if verified != 1 else ''} "
                    f"(and {contradicted} contradicted, {pending} uncertain). "
                    f"My strongest verified belief: door {h['door']} opens when "
                    f"{self._atoms_text(h['antecedent_text'])}. I only say this "
                    f"because the evidence supports it.")
        if contradicted:
            return (f"I remember {contradicted} contradicted hypotheses and {pending} uncertain. "
                    f"None are verified yet — so I can't claim I know a door rule.")
        return "I don't remember any verified rules yet. I'm still exploring."

    def _reply_hypotheses(self):
        mem = self.session.memory_snapshot()
        hs = mem["hypotheses"]
        if not hs:
            return "I have no hypotheses yet — I haven't observed enough."
        lines = []
        actionable = [h for h in hs if h["status"] in ("hypothesis", "predicted")]
        for h in hs[:8]:
            status = h["status"]
            lines.append(f"- door {h['door']} opens when {self._atoms_text(h['antecedent_text'])} "
                         f"[{status}, support {h['sup']}, contradiction {h['con']}]")
        tail = f" … and {len(hs)-8} more" if len(hs) > 8 else ""
        return "Current hypotheses:\n" + "\n".join(lines) + tail

    def _reply_learned_today(self):
        mem = self.session.memory_snapshot()
        stats = mem["stats"]
        session = self.session
        return (f"This session so far: {session.episode_index} episodes, "
                f"{session.episode_successes} successful. Memory: "
                f"{stats.get('verified', 0)} verified, {stats.get('contradicted', 0)} "
                f"contradicted, {mem['generated']} generated total.")

    def _start_challenge_for_door(self, door: str) -> str:
        ch = Challenge(pool_tag="train", difficulty="medium",
                       focus_door=door, seed=self.session.episode_index,
                       description=f"open door {door}")
        self.session.new_challenge(ch)
        return f"Challenge set: try to open door {door}. I'm using my real planner and learning. (Use Start/Step to run.)"

    def _start_goal_challenge(self) -> str:
        ch = Challenge(pool_tag="train", difficulty="medium",
                       seed=self.session.episode_index,
                       description="reach the goal")
        self.session.new_challenge(ch)
        return "Challenge set: reach the goal cell. I'll plan and learn along the way."

    def _start_difficulty(self, difficulty: str) -> str:
        ch = Challenge(pool_tag="train", difficulty=difficulty,
                       seed=self.session.episode_index,
                       description=f"{difficulty} challenge")
        self.session.new_challenge(ch)
        return f"New {difficulty} challenge set. (My memory is retained.)"

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
        if s.last_action_reason:
            reason = s.last_action_reason
        elif src:
            reason = f"source: {src}"
        else:
            reason = "I picked an action from my planner/probe logic."
        return (f"My last action was {s.last_action} ({reason}). "
                f"It came from my actual planning/learning loop, not from a script.")

    @staticmethod
    def _atoms_text(antecedent_text: str) -> str:
        return antecedent_text.replace(" ∧ ", ", ") if antecedent_text else "(empty)"

    # -- lifecycle -------------------------------------------------------
    def config(self):
        return {
            "api_mode": ("provider" if (self.api_adapter and self.api_adapter.enabled) else "none"),
            "api_provider": self.api_adapter.base_url if (self.api_adapter and self.api_adapter.enabled) else None,
            "intents": list(INTENTS),
            "storage_root": str(self.storage.root),
            "checkpoint": self.storage.checkpoint_info(),
            "max_global_steps": 4000,
        }

    def control(self, op: str, params: dict | None = None):
        params = params or {}
        if op == "start":
            self.session.start()
        elif op == "pause":
            self.session.pause()
        elif op == "continue":
            self.session.continue_running()
        elif op == "step":
            self.session.single_step()
        elif op == "stop":
            self.session.stop()
        elif op == "reset":
            self.session.reset_episode()
        elif op == "new_challenge":
            ch = Challenge(**params) if params else Challenge(description="reach the goal")
            self.session.new_challenge(ch)
        elif op == "save":
            self.session.save()
        elif op == "restore":
            self.session.restore()
        elif op == "fresh":
            self.session.fresh()
        else:
            raise ValueError(f"unknown control op: {op}")
        return self.session.snapshot()


class NestHandler(BaseHTTPRequestHandler):
    app: NestApp = None

    def log_message(self, fmt, *args):
        pass  # keep console quiet

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            return self._send(200, self.app.session.snapshot())
        if parsed.path == "/api/config":
            return self._send(200, self.app.config())
        if parsed.path == "/api/checkpoint":
            return self._send(200, self.app.storage.checkpoint_info())
        # static files
        rel = parsed.path.lstrip("/") or "index.html"
        candidate = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
            return self._send(403, {"error": "forbidden"})
        if not candidate.exists():
            return self._send(404, {"error": "not found"})
        ctype = "text/html" if candidate.suffix == ".html" else (
            "text/css" if candidate.suffix == ".css" else (
                "application/javascript" if candidate.suffix == ".js"
                else "application/octet-stream"))
        self._send(200, candidate.read_bytes(), ctype)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        if parsed.path == "/api/command":
            text = body.get("text", "")
            return self._send(200, self.app.handle_command(text))
        if parsed.path == "/api/control":
            try:
                return self._send(200, self.app.control(body.get("op", ""), body.get("params")))
            except Exception as exc:
                return self._send(400, {"error": str(exc)})
        return self._send(404, {"error": "unknown endpoint"})


def make_server(host="127.0.0.1", port=8765, storage_root=None,
                api_adapter=None) -> ThreadingHTTPServer:
    app = NestApp(storage_root=storage_root, api_adapter=api_adapter)
    NestHandler.app = app
    server = ThreadingHTTPServer((host, port), NestHandler)
    return server, app
