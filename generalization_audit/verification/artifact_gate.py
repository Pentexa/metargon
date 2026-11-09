"""Provenance-aware artifact gate for frozen evaluation targets.

Background
----------
The frozen evaluation criteria (`configs/frozen_eval_criteria_v02.json`)
were defined BEFORE v0.13 / Memory v0.6 results existed. The original
regression test enforced that by asserting the target result directories
do not exist — a permanent prohibition that would block Research and
Academy from ever publishing reproducible evidence.

This module replaces the *prohibition* with a *publication gate*: the
criteria stay immutable, but once an experiment runs, its raw evidence may
be committed if and only if it is covered by a valid provenance manifest
that verifies with zero FAIL checks. Missing metadata yields INCONCLUSIVE
(never invented values).

Gate semantics per frozen target
-------------------------------
- No artifacts present           -> PASS (gate open; criteria still frozen).
- Artifacts present, covered by
  one or more valid manifests
  whose verification has 0 FAIL -> PASS (evidence published with provenance).
- Artifacts present with NO
  manifest coverage               -> FAIL (raw evidence without provenance).
- Artifacts present, manifest(s)
  invalid or verification FAILs  -> FAIL (with messages).
- Orphan artifact files not
  declared by any manifest        -> FAIL (incomplete provenance).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .checks import CheckResult, FAIL, PASS, VerifierContext, run_checks
from .manifest import ManifestError, load_manifest, resolve_artifact_path

# Frozen targets that may only publish evidence through the provenance gate.
FROZEN_TARGETS: tuple[tuple[str, str], ...] = (
    ("results/v13", "v0.13 mechanism-lab experiments"),
    ("results/academy_v01/memory_v06", "Academy Memory v0.6 experiments"),
)

_DATA_SUFFIXES = (".json", ".jsonl", ".txt", ".md")


@dataclass
class GateReport:
    target: str
    label: str
    checks: list[CheckResult]

    @property
    def overall(self) -> str:
        if any(c.status == FAIL for c in self.checks):
            return FAIL
        if self.checks and all(c.status == PASS for c in self.checks):
            return PASS
        return "INCONCLUSIVE"

    def render_text(self) -> str:
        lines = [f"artifact gate — {self.label} ({self.target})",
                 f"overall: {self.overall}"]
        for c in self.checks:
            lines.append(f"  [{c.status:11s}] {c.name}: {c.message}")
            for d in c.detail[:4]:
                lines.append(f"      - {d}")
        return "\n".join(lines)


def _iter_data_files(directory: Path) -> Iterator[Path]:
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in _DATA_SUFFIXES:
            yield p


def _is_under(path: Path, parent: Path) -> bool:
    import os
    try:
        return os.path.commonpath([os.path.abspath(path),
                                   os.path.abspath(parent)]) == \
            os.path.abspath(parent)
    except ValueError:
        return False


def _load_manifests(manifests_dir: Path | None) -> list[tuple[Path, dict]]:
    if manifests_dir is None or not manifests_dir.is_dir():
        return []
    out = []
    for mf in sorted(manifests_dir.glob("*.manifest.json")):
        try:
            out.append((mf, load_manifest(mf)))
        except (ManifestError, OSError):
            continue
    return out


def check_frozen_target_artifacts(
        root: Path,
        manifests_dir: Path | None = None) -> list[GateReport]:
    """Run the provenance gate for every frozen target under ``root``.

    ``manifests_dir`` defaults to ``<root>/generalization_audit/manifests``.
    """
    if manifests_dir is None:
        manifests_dir = root / "generalization_audit" / "manifests"
    reports: list[GateReport] = []
    for rel_dir, label in FROZEN_TARGETS:
        tgt = Path(root) / rel_dir
        checks: list[CheckResult] = []
        if not tgt.exists():
            checks.append(CheckResult(
                f"gate.{rel_dir}", PASS,
                "no artifacts under this frozen target; gate open"))
            reports.append(GateReport(rel_dir, label, checks))
            continue

        artifacts = list(_iter_data_files(tgt))
        if not artifacts:
            checks.append(CheckResult(
                f"gate.{rel_dir}", PASS,
                "target directory exists but contains no data artifacts"))
            reports.append(GateReport(rel_dir, label, checks))
            continue

        manifests = _load_manifests(manifests_dir)
        covered: dict[str, tuple[Path, dict, dict]] = {}
        for mf, man in manifests:
            for a in man.get("artifacts", []):
                p = resolve_artifact_path(mf, a["path"])
                if _is_under(p, tgt):
                    covered[str(p.resolve())] = (mf, man, a)

        orphans = sorted(set(str(p) for p in artifacts) - set(covered))
        if orphans:
            checks.append(CheckResult(
                f"gate.{rel_dir}.provenance",
                FAIL,
                "raw evidence present without provenance manifest coverage",
                detail=[f"orphan artifacts ({len(orphans)}): "
                        + ", ".join(Path(o).name for o in orphans[:8])]))

        covering_manifests = sorted({mf for mf, _, _ in covered.values()})
        if not covering_manifests:
            # covered may be empty if no manifests; orphans already reported
            if not orphans:
                checks.append(CheckResult(
                    f"gate.{rel_dir}.provenance", FAIL,
                    "raw evidence present with no covering manifest"))
        for mf in covering_manifests:
            try:
                man = load_manifest(mf)
            except ManifestError as exc:
                checks.append(CheckResult(
                    f"gate.{rel_dir}.manifest:{mf.name}", FAIL,
                    f"covering manifest is invalid: {exc}"))
                continue
            ctx = VerifierContext(man, mf)
            fails = [c for c in run_checks(ctx) if c.status == FAIL]
            if fails:
                checks.append(CheckResult(
                    f"gate.{rel_dir}.manifest:{mf.name}", FAIL,
                    "covering manifest verification has FAIL checks",
                    detail=[c.message for c in fails[:8]]))
            else:
                checks.append(CheckResult(
                    f"gate.{rel_dir}.manifest:{mf.name}", PASS,
                    "covering manifest verifies with zero FAIL"))
        reports.append(GateReport(rel_dir, label, checks))
    return reports
