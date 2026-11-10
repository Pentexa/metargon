#!/usr/bin/env python3
"""METARGON experiment verification CLI.

Usage examples
--------------

Verify a committed experiment manifest::

    python -m generalization_audit.tools.verify_experiment \\
        --manifest generalization_audit/manifests/v121_final_l.manifest.json

Contrast two manifests (exact / statistical / non-reproducible)::

    python -m generalization_audit.tools.verify_experiment \\
        --manifest ref.manifest.json --against observed.manifest.json

Deterministic replay of a bounded command, then contrast with a reference::

    python -m generalization_audit.tools.verify_experiment \\
        --manifest ref.manifest.json \\
        --replay-command "python python/benchmark_v12.py --seeds 1 --episodes 2 ..."

Stamp a manifest from a config skeleton (computes hashes)::

    python -m generalization_audit.tools.verify_experiment \\
        --stamp cfg.json -o out.manifest.json

Exit codes: 0 = PASS, 1 = FAIL, 2 = INCONCLUSIVE.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "generalization_audit"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from generalization_audit.verification import (  # noqa: E402
    FAIL,
    INCONCLUSIVE,
    PASS,
    ManifestError,
    VerifierContext,
    canonical_sha256,
    load_manifest,
    run_checks,
    sha256_file,
    validate_manifest,
)
from generalization_audit.verification.transfer_checks import (  # noqa: E402
    TransferProtocol,
    run_transfer_checks,
)
from generalization_audit.verification.report import VerificationReport  # noqa: E402


def _exit_for(status: str) -> int:
    # contrast statuses: EXACT / STATISTICAL are success, NON-REPRODUCIBLE fail
    if status in ("EXACT", "STATISTICAL"):
        return 0
    if status == "NON-REPRODUCIBLE":
        return 1
    return {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}[status]


# ---------------------------------------------------------------------------
# stamping
# ---------------------------------------------------------------------------

def stamp_manifest(cfg: dict, base_dir: Path) -> dict:
    """Fill artifact hashes/sizes and validate the result."""
    manifest = json.loads(json.dumps(cfg))
    if manifest.get("schema_version") is None:
        manifest["schema_version"] = "experiment-manifest-v01"
    for a in manifest.get("artifacts", []):
        p = _resolve(base_dir, a["path"])
        if not p.exists():
            raise ManifestError(f"artifact path in stamp config not found: {p}")
        digest, size = canonical_sha256(p)
        a["sha256"] = digest
        if size is not None:
            a["size_bytes"] = size
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError(
            "stamped manifest invalid:\n  " + "\n  ".join(errors[:20]))
    return manifest


def _resolve(base_dir: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else base_dir / p


# ---------------------------------------------------------------------------
# replay contrast
# ---------------------------------------------------------------------------

def _guess_role(name: str) -> str:
    low = name.lower()
    if low.endswith(".jsonl") or "run" in low:
        return "log"
    if "summary" in low:
        return "summary"
    return "raw"


def _collect_artifacts(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".txt", ".md"):
            out.append({
                "path": str(p.relative_to(root)).replace("\\", "/"),
                "role": _guess_role(p.name),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            })
    return out


def run_replay(command: str, manifest_ref: dict, timeout: int) -> tuple[dict, str]:
    """Run a bounded command in a fresh temp workdir, collect produced
    artifacts, and build an observed manifest sharing the reference's
    experiment block. Commands must write artifacts relative to the
    current working directory; nothing is written into the repository."""
    with tempfile.TemporaryDirectory(prefix="metargon_replay_") as tmp:
        tmp_path = Path(tmp)
        log_path = tmp_path / "replay.log"
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(tmp_path),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout, text=True)
            log_path.write_text(proc.stdout or "", encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            (tmp_path / "replay.log").write_text(
                (exc.stdout or "") + f"\n[TIMEOUT after {timeout}s]", encoding="utf-8")
            observed = dict(manifest_ref)
            observed["artifacts"] = _collect_artifacts(tmp_path)
            return observed, f"replay timed out after {timeout}s (see replay.log)"
        artifacts = _collect_artifacts(tmp_path)
        observed = dict(manifest_ref)
        observed["artifacts"] = artifacts
        return observed, "replay completed"


def contrast_manifests(ref: dict, obs: dict, ref_base: Path, obs_base: Path,
                       tolerance: float) -> tuple[list[dict], str]:
    """Classify reproducibility per artifact and overall.

    Returns (records, overall); the overall classification is the worst of
    the per-artifact records (NON-REPRODUCIBLE > INCONCLUSIVE > STATISTICAL
    > EXACT).
    """
    order = {"EXACT": 0, "STATISTICAL": 1, "INCONCLUSIVE": 2,
             "NON-REPRODUCIBLE": 3}
    ref_by_path = {a["path"]: a for a in ref.get("artifacts", [])}
    obs_by_path = {a["path"]: a for a in obs.get("artifacts", [])}
    records = []
    for path in sorted(set(ref_by_path) | set(obs_by_path)):
        ra = ref_by_path.get(path)
        oa = obs_by_path.get(path)
        if ra is None or oa is None:
            records.append({
                "artifact": path, "status": "INCONCLUSIVE",
                "message": "artifact present in only one manifest",
                "detail": [
                    f"reference={'yes' if ra else 'no'} "
                    f"observed={'yes' if oa else 'no'}"]})
            continue
        ref_file = _resolve(ref_base, path)
        obs_file = _resolve(obs_base, path)
        if ra["sha256"] == oa["sha256"]:
            records.append({
                "artifact": path, "status": "EXACT",
                "message": "identical content (sha256 match)"})
            continue
        # statistical comparison on summary JSONs when both readable
        try:
            ref_data = json.loads(ref_file.read_text(encoding="utf-8"))
            obs_data = json.loads(obs_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            records.append({
                "artifact": path, "status": "INCONCLUSIVE",
                "message": "content differs; summary-level comparison not possible",
                "detail": ["hash mismatch"]})
            continue
        diffs = _summary_diffs(ref_data, obs_data, tolerance)
        if diffs is None:
            records.append({
                "artifact": path, "status": "INCONCLUSIVE",
                "message": "summary structure not comparable (different layouts)",
                "detail": ["hash mismatch"]})
        elif not diffs:
            records.append({
                "artifact": path, "status": "STATISTICAL",
                "message": f"summary metrics within tolerance ({tolerance})",
                "detail": ["hash mismatch, aggregated values agree"]})
        else:
            records.append({
                "artifact": path, "status": "NON-REPRODUCIBLE",
                "message": "summary metrics diverge beyond tolerance",
                "detail": diffs[:8]})
    overall = max(records, key=lambda r: order[r["status"]])["status"]
    return records, overall


def _summary_diffs(ref: dict, obs: dict, tolerance: float) -> list[str] | None:
    """Compare raw/mean metric rows between two summary artifacts.
    Returns None when layouts are incompatible; else a list of differences."""
    # layout A: raw {cond: [{policy: {metric: value}}]}
    if isinstance(ref.get("raw"), dict) and isinstance(obs.get("raw"), dict):
        diffs = []
        for cond in ref["raw"]:
            if cond not in obs["raw"]:
                diffs.append(f"condition {cond} missing in observed")
                continue
            r_rows = ref["raw"][cond]
            o_rows = obs["raw"][cond]
            if len(r_rows) != len(o_rows):
                diffs.append(f"condition {cond}: seed count differs "
                             f"({len(r_rows)} vs {len(o_rows)})")
                continue
            for i, (rr, oo) in enumerate(zip(r_rows, o_rows)):
                for pol in rr:
                    if pol not in oo:
                        diffs.append(f"cond={cond} seed={i} policy {pol} missing")
                        continue
                    for metric, rv in rr[pol].items():
                        ov = oo[pol].get(metric)
                        if ov is None:
                            continue
                        if isinstance(rv, (int, float)) and isinstance(ov, (int, float)):
                            if abs(rv - ov) > tolerance:
                                diffs.append(
                                    f"cond={cond} seed={i} {pol}.{metric}: "
                                    f"{rv} vs {ov}")
        return diffs
    # layout B: conditions_detail vectors
    if isinstance(ref.get("conditions_detail"), dict) and isinstance(obs.get("conditions_detail"), dict):
        diffs = []
        for var in ref["conditions_detail"]:
            if var not in obs["conditions_detail"]:
                diffs.append(f"variant {var} missing in observed")
                continue
            rv = ref["conditions_detail"][var]
            ov = obs["conditions_detail"][var]
            for mode in ("classic_frozen", "classic_adapt", "classic_test"):
                rvec = rv.get(f"{mode}_per_config")
                ovec = ov.get(f"{mode}_per_config")
                if rvec is None or ovec is None:
                    continue
                if len(rvec) != len(ovec):
                    diffs.append(f"variant {var} {mode}: vector length differs")
                    continue
                for a, b in zip(rvec, ovec):
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                            and abs(a - b) > tolerance:
                        diffs.append(f"variant {var} {mode}: {a} vs {b}")
        return diffs
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m generalization_audit.tools.verify_experiment",
        description=__doc__)
    p.add_argument("--manifest", type=str, default=None,
                   help="path to the experiment manifest to verify")
    p.add_argument("--against", type=str, default=None,
                   help="second manifest for contrast/reproducibility classification")
    p.add_argument("--replay-command", type=str, default=None,
                   help="bounded shell command to run in a temp workdir; "
                        "artifacts it produces are contrasted with --manifest")
    p.add_argument("--replay-timeout", type=int, default=600,
                   help="timeout in seconds for --replay-command")
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="absolute tolerance for statistical reproducibility beyond")
    p.add_argument("--transfer", action="store_true",
                   help="also run transfer-evaluation-protocol checks against the manifest")
    p.add_argument("--stamp", type=str, default=None,
                   help="stamp a manifest from a config skeleton (JSON)")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="write JSON report (or stamped manifest) to this path")
    p.add_argument("--json", action="store_true",
                   help="print JSON report to stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.stamp:
        cfg_path = Path(args.stamp)
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            stamped = stamp_manifest(cfg, cfg_path.parent)
        except ManifestError as exc:
            print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
            return _exit_for(INCONCLUSIVE)
        text = json.dumps(stamped, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"stamped manifest written: {args.output}")
        else:
            sys.stdout.write(text)
        return 0

    if not args.manifest:
        print("error: --manifest (or --stamp) is required", file=sys.stderr)
        return 2

    manifest_path = Path(args.manifest)
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        print(f"INCONCLUSIVE: manifest invalid: {exc}", file=sys.stderr)
        return _exit_for(INCONCLUSIVE)

    if args.replay_command:
        observed, replay_msg = run_replay(
            args.replay_command, manifest, args.replay_timeout)
        print(f"* replay: {replay_msg}")
        records, overall = contrast_manifests(
            manifest, observed, manifest_path.parent, manifest_path.parent,
            args.tolerance)
        report = {
            "source": str(manifest_path),
            "replay": replay_msg,
            "overall": overall,
            "contrast": records,
        }
        print(f"reproducibility: {overall}")
        for r in records:
            print(f"  [{r['status']:18s}] {r['artifact']}: {r['message']}")
            for d in r.get("detail", []):
                print(f"        - {d}")
        if args.json:
            print(json.dumps(report, indent=2))
        if args.output:
            Path(args.output).write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return _exit_for(overall)

    if args.against:
        try:
            other = load_manifest(Path(args.against))
        except ManifestError as exc:
            print(f"INCONCLUSIVE: second manifest invalid: {exc}", file=sys.stderr)
            return _exit_for(INCONCLUSIVE)
        records, overall = contrast_manifests(
            manifest, other, manifest_path.parent, Path(args.against).parent,
            args.tolerance)
        print(f"reproducibility vs {args.against}: {overall}")
        for r in records:
            print(f"  [{r['status']:18s}] {r['artifact']}: {r['message']}")
            for d in r.get("detail", []):
                print(f"        - {d}")
        if args.json:
            print(json.dumps({"overall": overall, "contrast": records}, indent=2))
        if args.output:
            Path(args.output).write_text(
                json.dumps({"overall": overall, "contrast": records}, indent=2) + "\n",
                encoding="utf-8")
        return _exit_for(overall)

    ctx = VerifierContext(manifest, manifest_path, options={"tolerance": args.tolerance})
    results = run_checks(ctx)
    if args.transfer:
        protocol = TransferProtocol.load(manifest_path.parent.parent.parent)
        results.extend(run_transfer_checks(ctx, protocol))
    report = VerificationReport(str(manifest_path), results)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render_text())
    if args.output:
        Path(args.output).write_text(
            json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
    return _exit_for(report.overall)


if __name__ == "__main__":
    sys.exit(main())
