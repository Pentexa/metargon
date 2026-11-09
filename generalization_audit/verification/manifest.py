"""METARGON experiment manifests: load, validate, hash.

Standard-library only. The schema is validated with a small recursive
validator (no external dependency) so this infrastructure runs anywhere
Python runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "experiment_manifest_v01.json"


class ManifestError(Exception):
    """Raised when a manifest is missing, malformed, or fails validation."""


def _err(path: str, msg: str) -> str:
    return f"{path}: {msg}" if path else msg


def _validate(value: Any, schema: Any, path: str = "", errors: list[str] | None = None) -> None:
    """Minimal JSON-Schema subset validator (type, enum, const, required,
    properties, items, minItems, minimum, maximum, pattern, description)."""
    errors = errors if errors is not None else []
    if not isinstance(schema, dict):
        return

    # const
    if "const" in schema and value != schema["const"]:
        errors.append(_err(path, f"expected const {schema['const']!r}, got {value!r}"))
        return

    # enum
    if "enum" in schema and value not in schema["enum"]:
        errors.append(_err(path, f"value {value!r} not in enum {schema['enum']}"))
        return

    # type
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        ok = False
        for t in types:
            if t == "object" and isinstance(value, dict):
                ok = True
            elif t == "array" and isinstance(value, list):
                ok = True
            elif t == "string" and isinstance(value, str):
                ok = True
            elif t == "integer" and isinstance(value, int) and not isinstance(value, bool):
                ok = True
            elif t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
                ok = True
            elif t == "boolean" and isinstance(value, bool):
                ok = True
            elif t == "null" and value is None:
                ok = True
        if not ok:
            errors.append(_err(path, f"expected type {schema['type']}, got {type(value).__name__}"))
            return

    # numeric bounds
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(_err(path, f"value {value} < minimum {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(_err(path, f"value {value} > maximum {schema['maximum']}"))

    # pattern
    if isinstance(value, str) and "pattern" in schema:
        import re
        if re.search(schema["pattern"], value) is None:
            errors.append(_err(path, f"string {value!r} does not match pattern {schema['pattern']!r}"))

    if schema.get("type") == "object" or isinstance(value, dict):
        if "required" in schema:
            for req in schema["required"]:
                if req not in value:
                    errors.append(_err(path, f"missing required property {req!r}"))
        if "properties" in schema:
            for k, sub in schema["properties"].items():
                if k in value:
                    _validate(value[k], sub, f"{path}.{k}" if path else k, errors)

    if schema.get("type") == "array" or isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(_err(path, f"array shorter than minItems {schema['minItems']}"))
        if "items" in schema and isinstance(schema["items"], dict):
            for i, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{i}]", errors)


def load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict) -> list[str]:
    """Return a list of schema violations (empty when valid)."""
    errors: list[str] = []
    _validate(manifest, load_schema(), errors=errors)
    return errors


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"manifest not found: {p}")
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError(
            f"manifest schema violations ({len(errors)}):\n  " + "\n  ".join(errors[:20]))
    return manifest


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(path: str | Path) -> tuple[str, int | None]:
    """Canonical content hash for an artifact.

    Inside a git repository this hashes the git blob (the exact canonical
    bytes) so results are identical on every platform regardless of
    checkout line-ending conversions. Outside a repository, or when the
    file is untracked, it falls back to the on-disk bytes.

    Returns ``(hexdigest, size_bytes)``; size is None when unavailable.
    """
    import subprocess

    p = Path(path)
    try:
        # repo-relative path from git's own view (robust to filesystem
        # normalization differences such as NFC/NFD on Windows)
        prefix = subprocess.run(
            ["git", "-C", str(p.parent), "rev-parse", "--show-prefix"],
            capture_output=True, text=True, check=False)
        if prefix.returncode == 0:
            rel = prefix.stdout.strip().replace("\\", "/") + p.name
            blob = subprocess.run(
                ["git", "cat-file", "blob", f":{rel}"],
                capture_output=True, check=False)
            if blob.returncode == 0 and blob.stdout:
                return hashlib.sha256(blob.stdout).hexdigest(), len(blob.stdout)
    except Exception:
        pass
    try:
        return sha256_file(p), p.stat().st_size
    except OSError:
        return "", None


def resolve_artifact_path(manifest_path: str | Path, rel: str) -> Path:
    """Resolve an artifact path relative to the manifest's directory."""
    base = Path(manifest_path).resolve().parent
    p = Path(rel)
    if p.is_absolute():
        return p
    return base / p
