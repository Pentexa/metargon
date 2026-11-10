"""METARGON NEST — application-owned persistence.

Storage lives in a dedicated directory (default: ``<repo>/.nest/``, or
``METARGON_NEST_HOME``). Only NEST writes there. Learned memory is stored
via the agent's own ``save_state()/load_state()`` (checkpoint
serialization), never reconstructed or invented.

Two files are kept:
- ``agent_state.json``  — the agent checkpoint (memory, RNG, counters).
- ``session_meta.json`` — NEST session metadata (challenge, episode stats),
  which is *conversational/administrative*, not learned memory.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from academy.agents.metargon_agent import MetargonAcademyAgent

AGENT_STATE = "agent_state.json"
SESSION_META = "session_meta.json"
MIN_AGENT_KEYS = {"seed", "rng", "memory", "steps", "episodes"}


class NestStorage:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else self._default_root()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_root() -> Path:
        env = os.environ.get("METARGON_NEST_HOME")
        if env:
            return Path(env)
        # application-owned directory inside the repository, gitignored
        return Path(__file__).resolve().parent.parent / ".nest"

    # -- compatibility checks --------------------------------------------
    @staticmethod
    def agent_state_compatible(state: dict) -> tuple[bool, str]:
        missing = [k for k in MIN_AGENT_KEYS if k not in state]
        if missing:
            return False, f"checkpoint missing required keys: {missing}"
        mem = state.get("memory")
        if not isinstance(mem, dict):
            return False, "checkpoint memory section missing"
        return True, "ok"

    # -- save / restore / clear ------------------------------------------
    def save(self, agent: MetargonAcademyAgent, meta: dict) -> dict:
        state = agent.save_state()
        ok, why = self.agent_state_compatible(state)
        if not ok:
            raise ValueError(f"refusing to save incompatible checkpoint: {why}")
        agent_path = self.root / AGENT_STATE
        tmp = agent_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, agent_path)
        meta_path = self.root / SESSION_META
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return {"saved": True, "agent_state": str(agent_path),
                "meta": str(meta_path)}

    def restore(self, agent: MetargonAcademyAgent) -> Optional[dict]:
        agent_path = self.root / AGENT_STATE
        if not agent_path.exists():
            return None
        try:
            state = json.loads(agent_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        ok, why = self.agent_state_compatible(state)
        if not ok:
            raise ValueError(f"checkpoint incompatible: {why}")
        agent.load_state(state)
        meta_path = self.root / SESSION_META
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
        return meta

    def clear(self):
        for name in (AGENT_STATE, SESSION_META):
            p = self.root / name
            if p.exists():
                p.unlink()

    def checkpoint_info(self) -> dict:
        agent_path = self.root / AGENT_STATE
        if not agent_path.exists():
            return {"exists": False}
        size = agent_path.stat().st_size
        try:
            state = json.loads(agent_path.read_text(encoding="utf-8"))
        except Exception:
            return {"exists": True, "size": size, "compatible": False}
        ok, why = self.agent_state_compatible(state)
        return {
            "exists": True,
            "size": size,
            "compatible": ok,
            "reason": why,
            "episodes": state.get("episodes"),
            "steps": state.get("steps"),
            "generated": state.get("memory", {}).get("generated"),
        }
