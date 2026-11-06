"""METARGON Academy — configuration loader (TOML, stdlib only)."""

from __future__ import annotations

import os
import tomllib

from .ckpt.checkpoint import config_fingerprint

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "academy_v01.toml")
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "academy_v01")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_fingerprint"] = config_fingerprint({k: v for k, v in cfg.items()
                                              if k != "_fingerprint"})
    return cfg


def merge_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply flat CLI overrides like 'evaluation.seeds=8'."""
    for key, value in overrides.items():
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg
