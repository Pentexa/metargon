"""METARGON Academy — checkpoint manager (Phase 7).

Crash-safe, deterministic snapshots: atomic write (temp + os.replace),
config fingerprint, agent state (including RNG), curriculum state, replay
buffer, and run metadata. Restoring a snapshot reconstructs the exact
agent state (same seed -> identical continuation).
"""

from __future__ import annotations

import hashlib
import json
import os
import time


def config_fingerprint(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, directory: str, run_id: str, cfg: dict,
                 keep_last: int = 5):
        self.directory = directory
        self.run_id = run_id
        self.cfg = cfg
        self.keep_last = keep_last
        self.fingerprint = config_fingerprint(cfg)
        self.episodes_since_checkpoint = 0
        os.makedirs(directory, exist_ok=True)

    def _path(self, episode: int) -> str:
        return os.path.join(self.directory,
                            f"{self.run_id}_ep{episode:06d}.json")

    def _latest_path(self) -> str:
        return os.path.join(self.directory, f"{self.run_id}_latest.json")

    def save(self, episode: int, agent_state: dict, curriculum_state: dict,
             mem_state: dict, extra: dict = None) -> str:
        snap = {
            "run_id": self.run_id,
            "config_fingerprint": self.fingerprint,
            "episode": episode,
            "saved_at": time.time(),
            "agent": agent_state,
            "curriculum": curriculum_state,
            "memory": mem_state,
            "extra": extra or {},
        }
        path = self._path(episode)
        _atomic_write_json(path, snap)
        _atomic_write_json(self._latest_path(), snap)
        # prune old snapshots
        snaps = sorted([f for f in os.listdir(self.directory)
                        if f.startswith(self.run_id) and f.endswith(".json")
                        and "_latest" not in f])
        while len(snaps) > self.keep_last:
            os.remove(os.path.join(self.directory, snaps.pop(0)))
        self.episodes_since_checkpoint = 0
        return path

    def load_latest(self):
        path = self._latest_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def verify(self, snapshot: dict) -> bool:
        return (snapshot.get("config_fingerprint") == self.fingerprint
                and snapshot.get("run_id") == self.run_id)
