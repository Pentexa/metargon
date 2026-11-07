"""EVREN teacher — persistence for approved tasks, provenance, and usage.

Everything the teacher touches is recorded:
  * approved tasks (config + provenance: model id, generation parameters,
    seeds, token usage, validation outcome, timestamps);
  * rejected candidates (with reasons — kept for audit/reproducibility);
  * token/request usage (from the provider's usage log).

All files are written atomically (temp + ``os.replace``), matching the
Academy's crash-safe checkpoint conventions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from ..envs.gridworld import EnvConfig
from .validator import ValidationResult, config_to_proposal


def _atomic_write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


@dataclass
class ApprovedTask:
    task_id: str
    proposal: dict          # canonical JSON proposal (config only)
    config: EnvConfig
    model: str
    model_id: str
    temperature: float
    max_tokens: int
    seed: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    difficulty_intent: str
    rationale: str
    approved_at: float
    provenance_note: str = ""

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "proposal": self.proposal,
            "model": self.model,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "difficulty_intent": self.difficulty_intent,
            "rationale": self.rationale,
            "approved_at": self.approved_at,
            "provenance_note": self.provenance_note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ApprovedTask":
        prop = d["proposal"]
        from .validator import proposal_to_config
        return cls(
            task_id=d["task_id"],
            proposal=prop,
            config=proposal_to_config(prop),
            model=d["model"],
            model_id=d["model_id"],
            temperature=d["temperature"],
            max_tokens=d["max_tokens"],
            seed=d["seed"],
            prompt_tokens=d["prompt_tokens"],
            completion_tokens=d["completion_tokens"],
            latency_ms=d["latency_ms"],
            difficulty_intent=d["difficulty_intent"],
            rationale=d.get("rationale", ""),
            approved_at=d["approved_at"],
            provenance_note=d.get("provenance_note", ""),
        )


class TaskRegistry:
    """Persists approved tasks, rejected attempts, and usage records."""

    def __init__(self, directory: str, run_id: str = "evren"):
        self.directory = directory
        self.run_id = run_id
        self.tasks_path = os.path.join(directory, "approved_tasks.json")
        self.rejected_path = os.path.join(directory, "rejected_candidates.json")
        self.usage_path = os.path.join(directory, "usage_log.json")
        self.manifest_path = os.path.join(directory, "manifest.json")
        self._approved = _load_json(self.tasks_path, [])
        self._rejected = _load_json(self.rejected_path, [])
        self._usage = _load_json(self.usage_path, [])
        self._manifest = _load_json(self.manifest_path, {})
        # eagerly materialize the registry files on first use
        if not os.path.exists(self.tasks_path):
            _atomic_write_json(self.tasks_path, self._approved)
        if not os.path.exists(self.rejected_path):
            _atomic_write_json(self.rejected_path, self._rejected)
        if not os.path.exists(self.usage_path):
            _atomic_write_json(self.usage_path, self._usage)
        if not os.path.exists(self.manifest_path):
            _atomic_write_json(self.manifest_path, self._manifest)

    # -- approved tasks --------------------------------------------------
    def approved_tasks(self) -> list:
        return [ApprovedTask.from_dict(d) for d in self._approved]

    def approved_configs(self) -> list:
        return [t.config for t in self.approved_tasks()]

    def add_approved(self, task: ApprovedTask) -> None:
        self._approved.append(task.as_dict())
        _atomic_write_json(self.tasks_path, self._approved)

    def task_ids(self) -> set:
        return {d["task_id"] for d in self._approved}

    # -- rejected candidates ---------------------------------------------
    def rejected(self) -> list:
        return self._rejected

    def add_rejected(self, result: ValidationResult, meta: dict = None) -> None:
        row = result.as_dict()
        row["meta"] = meta or {}
        self._rejected.append(row)
        if len(self._rejected) > 2000:
            self._rejected = self._rejected[-2000:]
        _atomic_write_json(self.rejected_path, self._rejected)

    # -- usage ------------------------------------------------------------
    def usage(self) -> list:
        return list(self._usage)

    def add_usage(self, rows: list) -> None:
        self._usage.extend(rows)
        _atomic_write_json(self.usage_path, self._usage)

    def token_summary(self) -> dict:
        ok = [r for r in self._usage if r.get("status") == "ok"]
        totals = {
            "requests": len(ok),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0)) for r in ok),
            "completion_tokens": sum(int(r.get("completion_tokens", 0)) for r in ok),
            "total_tokens": sum(int(r.get("total_tokens", 0)) for r in ok),
        }
        by_model = {}
        for r in ok:
            m = r.get("model", "?")
            by_model.setdefault(m, {"requests": 0, "prompt_tokens": 0,
                                    "completion_tokens": 0, "total_tokens": 0})
            by_model[m]["requests"] += 1
            by_model[m]["prompt_tokens"] += int(r.get("prompt_tokens", 0))
            by_model[m]["completion_tokens"] += int(r.get("completion_tokens", 0))
            by_model[m]["total_tokens"] += int(r.get("total_tokens", 0))
        errors = [r for r in self._usage if r.get("status") != "ok"]
        return {
            "totals": totals,
            "by_model": by_model,
            "errors": [
                {"ts": r.get("ts"), "status": r.get("status"),
                 "error": r.get("error", "")[:300],
                 "model": r.get("model")} for r in errors
            ],
            "error_count": len(errors),
        }

    # -- manifest ---------------------------------------------------------
    def update_manifest(self, **fields) -> None:
        self._manifest.update(fields)
        self._manifest["updated_at"] = time.time()
        _atomic_write_json(self.manifest_path, self._manifest)

    def manifest(self) -> dict:
        return dict(self._manifest)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "directory": self.directory,
            "approved_count": len(self._approved),
            "rejected_count": len(self._rejected),
            "usage": self.token_summary(),
            "manifest": self._manifest,
        }
