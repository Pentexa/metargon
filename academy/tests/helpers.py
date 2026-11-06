"""Shared fixtures for EVREN integration tests."""

from __future__ import annotations

import os

from academy.envs.tasks import load_splits, DEFAULT_SPLITS_PATH


def cached_splits():
    """Use the immutable cached splits (fast; no recomputation)."""
    path = os.environ.get("ACADEMY_SPLITS_PATH", DEFAULT_SPLITS_PATH)
    return load_splits(path)


def temp_registry(tmpdir: str, run_id: str = "test-evren"):
    from academy.evren.task_registry import TaskRegistry
    return TaskRegistry(tmpdir, run_id=run_id)
