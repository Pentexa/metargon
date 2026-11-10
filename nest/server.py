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
from .conversation import ConversationEngine
from .llm import OpenAICompatibleClient
from .storage import NestStorage

STATIC_DIR = Path(__file__).resolve().parent / "static"


class NestApp:
    """Ties session + conversation engine together."""

    def __init__(self, storage_root=None, api_adapter=None, llm_client=None):
        self.storage = NestStorage(storage_root)
        self.session = NestSession(storage=self.storage)
        self.api_adapter = api_adapter
        self.llm_client = llm_client if llm_client is not None else OpenAICompatibleClient()
        self.engine = ConversationEngine(
            session=self.session,
            client=self.llm_client,
            interpreter=lambda text: interpret(text, api_adapter=self.api_adapter),
        )
        self._lock = threading.RLock()

    # -- conversational replies (from real state) ------------------------
    def handle_command(self, text: str) -> dict:
        result = self.engine.respond(text)
        return {"command": {"intent": result.get("intent") or result.get("action")
                            or "conversation",
                            "text": text},
                "reply": result["reply"],
                "mode": result["mode"],
                "action": result.get("action"),
                "action_note": result.get("action_note"),
                "grounded": result.get("grounded", True),
                "snapshot": result["snapshot"]}

    # -- lifecycle -------------------------------------------------------
    def config(self):
        return {
            "api_mode": self.engine.mode,
            "api_provider": (self.llm_client.base_url
                             if self.llm_client.enabled else None),
            "api_model": (self.llm_client.model
                          if self.llm_client.enabled else None),
            "llm_budget": (self.llm_client.budget.as_dict()
                           if self.llm_client.enabled else None),
            "intents": list(INTENTS),
            "conversation_mode": self.engine.mode,
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
