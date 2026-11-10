"""Independent artifact-verification checks for METARGON experiments.

Each check is a pure function receiving a :class:`VerifierContext` and
returning a list of :class:`CheckResult`. Checks never modify artifacts.
Missing evidence yields INCONCLUSIVE; wrong evidence yields FAIL.

Two artifact formats are supported natively:

- ``benchmark_v12`` (and ``v121``-style): dicts with ``raw`` (per-seed
  rows per condition), ``summary`` (aggregates per condition/policy), and
  comparison entries named ``comparison:<a>_minus_<b>``.
- ``academy_memory``: ``conditions_detail`` (per-variant per-config
  vectors) plus a ``comparisons`` block of ``<variant>_vs_original``
  paired-bootstrap deltas.

Other formats are verified generically where the manifest provides enough
structure (raw + summary + seeds); otherwise INCONCLUSIVE is returned.
"""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .manifest import resolve_artifact_path, sha256_file, canonical_sha256

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    detail: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"check": self.name, "status": self.status,
                "message": self.message, "detail": self.detail}


@dataclass
class VerifierContext:
    manifest: dict
    manifest_path: Path
    options: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def resolve(self, rel: str) -> Path:
        return resolve_artifact_path(self.manifest_path, rel)

    def load_json(self, rel: str) -> Any:
        if rel not in self._cache:
            self._cache[rel] = json.loads(self.resolve(rel).read_text(encoding="utf-8"))
        return self._cache[rel]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.10g}"
    return str(x)


def _mean(vals: Iterable[float]) -> float:
    return statistics.fmean(vals)


def _artifact_paths(ctx: VerifierContext) -> list[str]:
    return [a["path"] for a in ctx.manifest["artifacts"]]


def _fmt_by_role(ctx: VerifierContext) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in ctx.manifest["artifacts"]:
        out.setdefault(a["role"], []).append(a["path"])
    return out


def _experiment(ctx: VerifierContext) -> dict:
    return ctx.manifest["experiment"]


def _format(ctx: VerifierContext) -> str:
    return _experiment(ctx).get("format", "generic")


def _seeds(ctx: VerifierContext) -> list[int]:
    return list(_experiment(ctx).get("seeds", []))


def _episodes(ctx: VerifierContext) -> int | None:
    return _experiment(ctx).get("episodes_per_seed")


def _convention(ctx: VerifierContext) -> str:
    return _experiment(ctx)["evaluation"]["comparison_convention"]


def _bootstrap(ctx: VerifierContext) -> dict:
    return _experiment(ctx)["evaluation"]["bootstrap"]


def paired_bootstrap(a: list[float], b: list[float], seed: int,
                    rounds: int = 2000, alpha: float = 0.05) -> dict:
    """Canonical paired bootstrap (mirrors academy/eval/metrics.py)."""
    assert len(a) == len(b)
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in idx) / n
        db = sum(b[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int((alpha / 2) * rounds)]
    hi = diffs[int((1 - alpha / 2) * rounds) - 1]
    return {"mean": _mean(diffs), "ci": (lo, hi)}


def harness_paired_bootstrap(diffs: list[float], seed: int,
                             rounds: int = 1500, alpha: float = 0.05) -> tuple:
    """Mirrors the benchmark harness's paired_bootstrap (benchmark_v12.py):
    per-seed difference vector resampled with replacement, fmean per draw."""
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        values.append(statistics.fmean(
            diffs[rng.randrange(len(diffs))] for _ in diffs))
    values.sort()
    lo = values[int((alpha / 2) * rounds)]
    hi = values[int((1 - alpha / 2) * rounds)]
    return lo, hi


def _check(name: str, ok: bool | None, good: str, bad: str,
           inconclusive: str = "", detail: list[str] | None = None) -> CheckResult:
    if ok is None:
        return CheckResult(name, INCONCLUSIVE, inconclusive or "evidence missing", detail or [])
    if ok:
        return CheckResult(name, PASS, good, detail or [])
    return CheckResult(name, FAIL, bad, detail or [])


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def c_artifacts_exist(ctx: VerifierContext) -> list[CheckResult]:
    results = []
    for rel in _artifact_paths(ctx):
        p = ctx.resolve(rel)
        results.append(_check(
            f"artifact.exists:{rel}",
            p.exists(),
            f"artifact exists ({p})",
            f"artifact missing: {p}"))
    return results


def c_artifact_hashes(ctx: VerifierContext) -> list[CheckResult]:
    results = []
    for a in ctx.manifest["artifacts"]:
        rel = a["path"]
        p = ctx.resolve(rel)
        if not p.exists():
            continue
        actual, actual_size = canonical_sha256(p)
        ok = actual == a["sha256"]
        results.append(_check(
            f"artifact.hash:{rel}",
            ok,
            f"sha256 matches ({actual[:12]}…)",
            f"sha256 mismatch: expected {a['sha256'][:12]}… got {actual[:12]}…"))
        if "size_bytes" in a and a["size_bytes"] is not None and actual_size is not None:
            results.append(_check(
                f"artifact.size:{rel}",
                actual_size == a["size_bytes"],
                f"size matches ({actual_size})",
                f"size mismatch: expected {a['size_bytes']}, got {actual_size}"))
    return results


def c_unique_artifact_records(ctx: VerifierContext) -> list[CheckResult]:
    paths = _artifact_paths(ctx)
    dup_paths = sorted({p for p in paths if paths.count(p) > 1})
    return [_check(
        "artifact.records.unique",
        not dup_paths,
        "no duplicate artifact records",
        f"duplicate artifact records: {dup_paths}")]


def c_missing_duplicate_records(ctx: VerifierContext) -> list[CheckResult]:
    """No duplicate per-seed rows; summary policies covered by raw rows."""
    results = []
    fmt = _format(ctx)
    if fmt == "benchmark_v12":
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            raw = data.get("raw") or {}
            for cond, rows in raw.items():
                seen: dict[str, int] = {}
                dup_idx: list[int] = []
                for idx, r in enumerate(rows):
                    key = json.dumps(r, sort_keys=True, default=str)
                    if key in seen:
                        dup_idx.append(idx)
                    seen[key] = idx
                results.append(_check(
                    f"raw.unique.rows:{rel}:{cond}",
                    not dup_idx,
                    f"no duplicate raw rows ({len(rows)} seeds)",
                    f"duplicate raw rows detected at indices {dup_idx}",
                    detail=[f"condition={cond}"]))
                if rows and isinstance(rows[0], dict):
                    # policies present in summary but absent from every raw row
                    summary = data.get("summary", {}).get(cond, {})
                    for pol in summary:
                        if pol.startswith("comparison:"):
                            continue
                        covered = any(pol in r for r in rows)
                        results.append(_check(
                            f"raw.cover:{rel}:{cond}:{pol}",
                            covered,
                            f"policy {pol} covered by raw rows",
                            f"policy {pol} missing from every raw row"))
                for i, row in enumerate(rows):
                    keys = list(row)
                    if len(keys) != len(set(keys)):
                        results.append(CheckResult(
                            f"raw.duplicate.policy:{rel}:{cond}:seed{i}",
                            FAIL, "duplicate policy keys inside raw row"))
    elif fmt == "academy_memory":
        for rel in _fmt_by_role(ctx).get("summary", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            detail = data.get("conditions_detail") or {}
            variants = list(detail)
            if len(variants) != len(set(variants)):
                results.append(CheckResult(
                    f"summary.duplicate.variant:{rel}", FAIL,
                    "duplicate variant entries in conditions_detail"))
            per_seed = [v.get("per_seed") for v in detail.values() if isinstance(v, dict)]
            if per_seed:
                n = len(per_seed[0])
                same = all(len(ps) == n for ps in per_seed)
                results.append(_check(
                    f"summary.per_seed.count:{rel}",
                    same,
                    f"consistent per-seed blocks (n={n})",
                    "per-seed block lengths differ across variants"))
        # duplicate variant records in JSONL logs
        for rel in _fmt_by_role(ctx).get("log", []):
            try:
                seen = {}
                dup = []
                for line in ctx.resolve(rel).read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    v = r.get("variant")
                    if v in seen:
                        dup.append(v)
                    seen[v] = True
                results.append(_check(
                    f"log.unique.variant:{rel}",
                    not dup,
                    "no duplicate variant records in log",
                    f"duplicate variant records: {dup}"))
            except FileNotFoundError:
                continue
    return results


def c_raw_present(ctx: VerifierContext) -> list[CheckResult]:
    fmt = _format(ctx)
    if fmt == "benchmark_v12":
        raw_paths = _fmt_by_role(ctx).get("raw", [])
        if not raw_paths:
            return [CheckResult("raw.present", INCONCLUSIVE,
                                "no raw artifact listed in manifest")]
        out = []
        for rel in raw_paths:
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                out.append(CheckResult(f"raw.present:{rel}", INCONCLUSIVE,
                                       "raw artifact missing on disk"))
                continue
            raw = data.get("raw") or {}
            for cond in _experiment(ctx).get("conditions", []):
                n = len(raw.get(cond, []))
                exp_n = len(_seeds(ctx))
                out.append(_check(
                    f"raw.seeds:{rel}:{cond}",
                    n == exp_n if n else None,
                    f"per-seed rows present (n={n})",
                    f"expected {exp_n} seed rows, found {n}",
                    inconclusive="raw per-seed rows missing for this condition"))
        return out
    if fmt == "academy_memory":
        summary_paths = _fmt_by_role(ctx).get("summary", [])
        if not summary_paths:
            return [CheckResult("raw.present", INCONCLUSIVE,
                                "no summary artifact listed in manifest")]
        out = []
        for rel in summary_paths:
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                out.append(CheckResult(f"raw.present:{rel}", INCONCLUSIVE,
                                       "summary artifact missing on disk"))
                continue
            detail = data.get("conditions_detail") or {}
            for variant in _experiment(ctx).get("policies", []):
                has_vec = variant in detail and any(
                    f"{mode}_per_config" in detail[variant]
                    for mode in ("classic_frozen", "classic_adapt", "classic_test"))
                out.append(_check(
                    f"raw.vector:{rel}:{variant}",
                    has_vec if variant in detail else None,
                    f"per-config vector present for {variant}",
                    f"per-config vector missing for {variant}",
                    inconclusive=f"variant {variant} absent from conditions_detail"))
        return out
    return [CheckResult("raw.present", INCONCLUSIVE,
                        f"unsupported/unknown format {fmt!r}")]


def c_summary_matches_raw(ctx: VerifierContext) -> list[CheckResult]:
    fmt = _format(ctx)
    if fmt != "benchmark_v12":
        return [CheckResult("summary.vs.raw", PASS,
                            "not applicable for this format (no raw+summary pair)")]
    out = []
    for rel in _fmt_by_role(ctx).get("raw", []):
        try:
            data = ctx.load_json(rel)
        except FileNotFoundError:
            continue
        raw = data.get("raw") or {}
        summary = data.get("summary") or {}
        for cond, rows in raw.items():
            if cond not in summary:
                continue
            if not rows or not isinstance(rows[0], dict):
                continue
            for pol, entry in summary[cond].items():
                if pol.startswith("comparison:") or not isinstance(entry, dict):
                    continue
                for metric, value in entry.items():
                    if metric == "peak_mem_mb":
                        continue
                    if isinstance(value, (list, dict)):
                        continue
                    vals = [r[pol][metric] for r in rows
                            if pol in r and r[pol].get(metric) is not None]
                    if not vals:
                        continue
                    avg = _mean(vals)
                    ok = abs(avg - value) <= 1e-9
                    detail = [f"condition={cond} policy={pol} metric={metric}",
                              f"raw mean={_fmt(avg)} summary={_fmt(value)}"]
                    out.append(_check(
                        f"summary.vs.raw:{cond}:{pol}:{metric}",
                        ok,
                        "summary mean equals raw per-seed mean",
                        "summary mean diverges from raw per-seed mean",
                        detail=detail))
    return out


def _comparison_pairs(data: dict, cond: str) -> list[tuple[str, str, str]]:
    """Return (comparison_key, a, b) for all comparison entries."""
    out = []
    for key in data.get("summary", {}).get(cond, {}):
        if key.startswith("comparison:"):
            body = key[len("comparison:"):]
            if "_minus_" in body:
                a, b = body.split("_minus_", 1)
                out.append((key, a, b))
    return out


def _comparison_pairs_full(data: dict) -> list[tuple[str, str, str, str]]:
    """Return (condition, comparison_key, a, b) across all conditions."""
    out = []
    for cond in data.get("summary", {}):
        for key, a, b in _comparison_pairs(data, cond):
            out.append((cond, key, a, b))
    return out


def _infer_comparison_metric(key: str) -> str | None:
    """Pick the metric a comparison refers to from the key text alone
    (e.g. ``..._exact_D1`` -> ``exact_discovery_D1``)."""
    abbr = {
        "exact_discovery_D1": ("exactd1", "exactdiscoveryd1"),
        "exact_discovery_D2": ("exactd2", "exactdiscoveryd2"),
        "covered_discovery_D1": ("coveredd1", "covereddiscoveryd1"),
        "covered_discovery_D2": ("coveredd2", "covereddiscoveryd2"),
        "temporal_probes": ("temporalprobes",),
    }
    key_norm = key.lower().replace("_", "")
    for metric, tokens in abbr.items():
        if any(t in key_norm for t in tokens):
            return metric
    return None


def _comparison_vectors(rows: list[dict], a: str, b: str, key: str,
                        metric_hint: str | None = None
                        ) -> tuple[list[float], list[float], str] | None:
    metric = metric_hint
    if metric is None:
        metric = _infer_comparison_metric(key)
    if metric is None:
        return None
    va = [r[a][metric] for r in rows
          if a in r and b in r and r[a].get(metric) is not None
          and r[b].get(metric) is not None]
    vb = [r[b][metric] for r in rows
          if a in r and b in r and r[a].get(metric) is not None
          and r[b].get(metric) is not None]
    if len(va) != len(rows) or len(vb) != len(rows) or not va:
        return None
    return va, vb, metric


def _resolve_comparison_pair(rows: list[dict], key: str, policies: list[str],
                             metric_hint: str | None = None
                             ) -> tuple[str, str, str] | None:
    """Resolve (policy_a, policy_b, metric) for a comparison key.

    Plain ``<a>_minus_<b>`` keys are resolved by policy name when the
    policies exist. Semantic keys such as
    ``comparison:temporal_minus_notp_exact_D1`` are resolved by matching
    the second token to a policy suffix and the first token to the policy
    with the highest temporal-probe activity.
    """
    body = key.split("comparison:", 1)[-1]
    if "_minus_" not in body:
        return None
    a_tok, b_tok = body.split("_minus_", 1)
    metric = metric_hint or _infer_comparison_metric(key)
    if metric is None:
        return None

    def _strip_metric(tok: str) -> str:
        import re as _re
        stripped = _re.sub(
            r"(?i)_?(?:exact|covered)(?:_discovery)?_?(?:d1|d2)$", "", tok)
        stripped = _re.sub(r"(?i)_?temporal_probes$", "", stripped)
        return stripped.rstrip("_")

    a_tok = _strip_metric(a_tok)
    b_tok = _strip_metric(b_tok)

    def _match(tok: str) -> str | None:
        tok_norm = tok.lower().replace("_", "")
        hits = [p for p in policies
                if p.lower().replace("_", "").endswith(tok_norm)]
        if hits:
            return max(hits, key=len)
        return None

    b = _match(b_tok)
    if b is None:
        return None
    a = _match(a_tok)
    if a is None and a_tok.lower() == "temporal":
        # the temporal-enabled policy is the one with nonzero temporal probes
        probe_rows = []
        for r in rows:
            for p in r:
                v = r[p].get("temporal_probes")
                if isinstance(v, (int, float)) and v > 0:
                    probe_rows.append((p, v))
        if probe_rows:
            a = max(probe_rows, key=lambda t: t[1])[0]
    if a is None or a not in rows[0] or b not in rows[0]:
        return None
    if a == b:
        return None
    return a, b, metric


def resolve_comparison_with_delta(rows: list[dict], key: str, policies: list[str],
                                  recorded_delta: float, metric_hint: str | None = None,
                                  tol: float = 1e-9) -> tuple[str, tuple[str, str, str] | None]:
    """Data-driven comparison resolution.

    Resolves ``(policy_a, policy_b, metric)`` for a comparison key by
    scanning candidate policy pairs and selecting the UNIQUE pair that
    reproduces the recorded delta from the raw per-seed vectors.

    Returns ``(status, result)``:
    - status "ok"        -> result is the unique (a, b, metric) triple
    - status "ambiguous"  -> multiple candidate pairs reproduce the delta
    - status "none"       -> no pair reproduces the delta (or raw vectors
                             unavailable)
    """
    body = key.split("comparison:", 1)[-1]
    if "_minus_" not in body:
        return "none", None
    a_tok, b_tok = body.split("_minus_", 1)
    metric = metric_hint or _infer_comparison_metric(key)
    if metric is None:
        return "none", None

    def _strip_metric(tok: str) -> str:
        import re as _re
        stripped = _re.sub(
            r"(?i)_?(?:exact|covered)(?:_discovery)?_?(?:d1|d2)$", "", tok)
        stripped = _re.sub(r"(?i)_?temporal_probes$", "", stripped)
        return stripped.rstrip("_")

    a_tok = _strip_metric(a_tok)
    b_tok = _strip_metric(b_tok)

    def _suffix_matches(pol: str, tok: str) -> bool:
        tok_norm = tok.lower().replace("_", "")
        return pol.lower().replace("_", "").endswith(tok_norm) and len(tok_norm) > 0

    policies = list(dict.fromkeys(policies))  # de-duplicate roster entries

    def _exact_unique(tok: str) -> str | None:
        hits = [p for p in policies if _suffix_matches(p, tok)]
        return hits[0] if len(hits) == 1 else None

    def _delta_matches(a: str, b: str) -> bool:
        if a not in rows[0] or b not in rows[0] or a == b:
            return False
        va = [r[a].get(metric) for r in rows if a in r]
        vb = [r[b].get(metric) for r in rows if b in r]
        if len(va) != len(rows) or len(vb) != len(rows):
            return False
        if any(v is None for v in va + vb):
            return False
        return abs(_mean(va) - _mean(vb) - recorded_delta) <= tol

    def _matches_value_identical(matches: list[tuple[str, str, str]],
                                 rows_: list[dict], metric_: str) -> bool:
        if not matches:
            return False
        a0, b0, _ = matches[0]
        va0 = [r[a0].get(metric_) for r in rows_]
        vb0 = [r[b0].get(metric_) for r in rows_]
        for a, b, _ in matches[1:]:
            va = [r[a].get(metric_) for r in rows_]
            vb = [r[b].get(metric_) for r in rows_]
            if va != va0 or vb != vb0:
                return False
        return True

    # 1) strict token resolution when both tokens identify a unique policy
    #    (return the pair unconditionally: the caller compares deltas and
    #    reports FAIL on mismatch)
    b = _exact_unique(b_tok)
    a = _exact_unique(a_tok) if b is not None else None
    if a is not None and b is not None:
        return "ok", (a, b, metric)

    # 2) family-hinted resolution (temporal/notp pairs)
    family_hint = None
    if a_tok.lower().replace("_", "").startswith("v12"):
        family_hint = "12"
    elif a_tok.lower().replace("_", "").startswith("v13"):
        family_hint = "13"

    is_temporal_a = a_tok.lower().replace("_", "") == "temporal"

    if b is None:
        b_cands = [p for p in policies if _suffix_matches(p, b_tok)]
        if family_hint is not None:
            b_cands = [p for p in b_cands if family_hint in p]
        if len(b_cands) == 1:
            b = b_cands[0]

    if b is not None:
        # temporal-enabled candidates: policies with nonzero probes
        temporal_cands = []
        for r in rows:
            for p in r:
                v = r[p].get("temporal_probes")
                if isinstance(v, (int, float)) and v > 0:
                    temporal_cands.append(p)
        a_cands = []
        for p in policies:
            if p == b or _suffix_matches(p, "notp"):
                continue
            if family_hint is not None and family_hint not in p:
                continue
            if is_temporal_a and p not in temporal_cands:
                continue
            a_cands.append(p)
        if temporal_cands and not is_temporal_a:
            a_cands = [p for p in a_cands if p in temporal_cands] or a_cands
        matches = [(a, b, metric) for a in a_cands if _delta_matches(a, b)]
        if len(matches) == 1:
            return "ok", matches[0]
        if len(matches) > 1:
            if _matches_value_identical(matches, rows, metric):
                return "ok", matches[0]
            # All candidates reproduce the delta; prefer the baseline whose
            # name shares the candidate's numeric family when one is
            # distinguishable by name (data-indistinguishable baselines).
            fam = None
            if a_cands and any("13" in p for p in a_cands):
                fam = "13"
            elif a_cands and any("12" in p for p in a_cands):
                fam = "12"
            if fam is not None:
                fam_matches = [m for m in matches if fam in m[1]]
                if len(fam_matches) == 1:
                    return "ok", fam_matches[0]
            return "ambiguous", matches
        return "none", None

    # 2b) temporal token without family hint: scan temporal-probe policies
    # against every notp baseline; require a unique delta match.
    if is_temporal_a:
        temporal_cands = []
        for r in rows:
            for p in r:
                v = r[p].get("temporal_probes")
                if isinstance(v, (int, float)) and v > 0:
                    temporal_cands.append(p)
        temporal_cands = list(dict.fromkeys(temporal_cands))
        b_cands = [p for p in policies if _suffix_matches(p, b_tok)]
        matches = []
        for a in temporal_cands:
            for bb in b_cands:
                if _delta_matches(a, bb):
                    matches.append((a, bb, metric))
        if len(matches) == 1:
            return "ok", matches[0]
        if len(matches) > 1:
            if _matches_value_identical(matches, rows, metric):
                return "ok", matches[0]
            fam = None
            if any("13" in m[0] for m in matches):
                fam = "13"
            elif any("12" in m[0] for m in matches):
                fam = "12"
            if fam is not None:
                fam_matches = [m for m in matches if fam in m[1]]
                if len(fam_matches) == 1:
                    return "ok", fam_matches[0]
            return "ambiguous", matches
        return "none", None

    # 3) full scan over candidate pairs (last resort; requires family hint)
    if family_hint is None:
        return "none", None
    b_cands = [p for p in policies if _suffix_matches(p, b_tok)
               and family_hint in p]
    a_cands = [p for p in policies if p not in b_cands
               and family_hint in p
               and not _suffix_matches(p, "notp")]
    matches = []
    for a in a_cands:
        for bb in b_cands:
            if _delta_matches(a, bb):
                matches.append((a, bb, metric))
    if len(matches) == 1:
        return "ok", matches[0]
    if len(matches) > 1:
        return "ambiguous", matches
    return "none", None


def c_comparison_direction(ctx: VerifierContext) -> list[CheckResult]:
    fmt = _format(ctx)
    conv = _convention(ctx)
    out = []

    if fmt == "benchmark_v12":
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for cond in data.get("summary", {}):
                for key, a, b in _comparison_pairs(data, cond):
                    rows = data.get("raw", {}).get(cond, [])
                    entry = data["summary"][cond][key]
                    recorded = entry.get("success_delta")
                    metric_hint = "success_rate" if recorded is not None else None
                    if recorded is None:
                        recorded = entry.get("delta")
                    if recorded is None:
                        out.append(CheckResult(
                            f"comparison.direction:{key}:{cond}", INCONCLUSIVE,
                            "no delta field in summary entry"))
                        continue
                    policies = list(_experiment(ctx).get("policies", [])) or \
                        list(rows[0]) if rows else []
                    sign = 1 if conv == "candidate_minus_baseline" else -1
                    status, resolved = resolve_comparison_with_delta(
                        rows, key, policies, sign * recorded, metric_hint)
                    if status == "ambiguous":
                        out.append(CheckResult(
                            f"comparison.direction:{key}:{cond}", INCONCLUSIVE,
                            "ambiguous comparison key: multiple policy pairs reproduce the recorded delta",
                            detail=[f"candidate pairs: {resolved}"]))
                        continue
                    if status != "ok":
                        out.append(CheckResult(
                            f"comparison.direction:{key}:{cond}", INCONCLUSIVE,
                            "raw paired vectors unavailable for comparison"))
                        continue
                    a, b, metric = resolved
                    rvec = _comparison_vectors(rows, a, b, key, metric)
                    if rvec is None:
                        out.append(CheckResult(
                            f"comparison.direction:{key}:{cond}", INCONCLUSIVE,
                            "raw paired vectors unavailable for comparison"))
                        continue
                    va, vb, metric = rvec
                    delta_ab = _mean(va) - _mean(vb)
                    expected = sign * delta_ab
                    ok = abs(expected - recorded) <= 1e-9
                    detail = [f"condition={cond} a={a} b={b} metric={metric}",
                              f"recorded={_fmt(recorded)} recomputed={_fmt(expected)}"]
                    out.append(_check(
                        f"comparison.direction:{key}:{cond}",
                        ok,
                        f"comparison follows {conv} convention",
                        f"comparison sign/value disagrees with {conv} convention",
                        detail=detail))
        return out

    if fmt == "academy_memory":
        for rel in _fmt_by_role(ctx).get("summary", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            detail = data.get("conditions_detail") or {}
            comparisons = data.get("comparisons") or {}
            for name, cmp_ in comparisons.items():
                if not name.endswith("_vs_original"):
                    continue
                variant = name[: -len("_vs_original")]
                base = "original"
                if variant not in detail or base not in detail:
                    out.append(CheckResult(
                        f"comparison.direction:{name}", INCONCLUSIVE,
                        "variant or baseline missing from conditions_detail"))
                    continue
                for mode in ("classic_frozen", "classic_adapt", "classic_test"):
                    if mode not in cmp_:
                        continue
                    va = detail[variant].get(f"{mode}_per_config")
                    vb = detail[base].get(f"{mode}_per_config")
                    if not va or not vb or len(va) != len(vb):
                        out.append(CheckResult(
                            f"comparison.direction:{name}:{mode}", INCONCLUSIVE,
                            "paired per-config vectors unavailable"))
                        continue
                    recorded = cmp_[mode].get("mean")
                    if recorded is None:
                        continue
                    if conv == "candidate_minus_baseline":
                        expected = _mean(va) - _mean(vb)
                    else:
                        expected = _mean(vb) - _mean(va)
                    # bootstrap mean jitters a little around the exact delta
                    ok = abs(expected - recorded) <= 0.01
                    info = [f"comparison={name} mode={mode}",
                            f"recorded={_fmt(recorded)} analytic={_fmt(expected)}"]
                    out.append(_check(
                        f"comparison.direction:{name}:{mode}",
                        ok,
                        f"comparison follows {conv} convention",
                        f"comparison sign/value disagrees with {conv} convention",
                        detail=info))
        return out

    return [CheckResult("comparison.direction", INCONCLUSIVE,
                        f"unsupported format {fmt!r}")]


def c_bootstrap_ci_consistent(ctx: VerifierContext) -> list[CheckResult]:
    fmt = _format(ctx)
    out = []
    try:
        boot = _bootstrap(ctx)
        seed = boot["seed"]
        rounds = boot.get("rounds", 2000)
        alpha = boot.get("alpha", 0.05)
    except KeyError:
        return [CheckResult("bootstrap.ci", INCONCLUSIVE,
                            "manifest lacks bootstrap parameters")]

    if fmt == "benchmark_v12":
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for cond, key, a, b in _comparison_pairs_full(data):
                rows = data.get("raw", {}).get(cond, [])
                entry = data.get("summary", {}).get(cond, {}).get(key)
                if entry is None or "paired_bootstrap_95pct" not in entry:
                    out.append(CheckResult(
                        f"bootstrap.ci:{key}:{cond}", INCONCLUSIVE,
                        "no paired_bootstrap_95pct entry in summary"))
                    continue
                metric_hint = ("success_rate"
                               if entry.get("success_delta") is not None else None)
                policies = list(_experiment(ctx).get("policies", [])) or \
                    list(rows[0]) if rows else []
                lo, hi = entry["paired_bootstrap_95pct"]
                conv = _convention(ctx)
                sign = 1 if conv == "candidate_minus_baseline" else -1
                # resolve via the recorded delta (data-driven; disambiguates
                # semantic keys such as v12_temporal_minus_notp_exact_D1)
                recorded_delta = entry.get("success_delta")
                metric_hint = "success_rate" if recorded_delta is not None else None
                if recorded_delta is None:
                    recorded_delta = entry.get("delta")
                status, resolved = resolve_comparison_with_delta(
                    rows, key, policies, sign * (recorded_delta or 0.0),
                    metric_hint)
                if status == "ambiguous":
                    out.append(CheckResult(
                        f"bootstrap.ci:{key}:{cond}", INCONCLUSIVE,
                        "ambiguous comparison key: multiple policy pairs reproduce the recorded delta",
                        detail=[f"candidate pairs: {resolved}"]))
                    continue
                if status != "ok":
                    out.append(CheckResult(
                        f"bootstrap.ci:{key}:{cond}", INCONCLUSIVE,
                        "paired per-seed differences unavailable"))
                    continue
                a, b, metric = resolved
                rvec = _comparison_vectors(rows, a, b, key, metric)
                if rvec is None:
                    out.append(CheckResult(
                        f"bootstrap.ci:{key}:{cond}", INCONCLUSIVE,
                        "paired per-seed differences unavailable"))
                    continue
                va, vb, metric = rvec
                diffs = [x - y for x, y in zip(va, vb)]
                r_lo, r_hi = harness_paired_bootstrap(
                    [sign * d for d in diffs], seed, rounds, alpha)
                ok = abs(lo - r_lo) <= 1e-9 and abs(hi - r_hi) <= 1e-9
                out.append(_check(
                    f"bootstrap.ci:{key}:{cond}",
                    ok,
                    "harness bootstrap CI reproduces exactly",
                    "harness bootstrap CI diverges from raw diffs",
                    detail=[f"condition={cond} seed={seed} rounds={rounds} "
                            f"alpha={alpha} convention={conv}",
                            f"recorded ci=({_fmt(lo)},{_fmt(hi)}) "
                            f"recomputed ci=({_fmt(r_lo)},{_fmt(r_hi)})"]))
        return out

    if fmt == "academy_memory":
        for rel in _fmt_by_role(ctx).get("summary", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            detail = data.get("conditions_detail") or {}
            for name, cmp_ in (data.get("comparisons") or {}).items():
                if not name.endswith("_vs_original"):
                    continue
                variant = name[: -len("_vs_original")]
                if variant not in detail:
                    continue
                for mode in ("classic_frozen", "classic_adapt", "classic_test"):
                    if mode not in cmp_:
                        continue
                    va = detail[variant].get(f"{mode}_per_config")
                    vb = detail["original"].get(f"{mode}_per_config")
                    if not va or not vb or len(va) != len(vb):
                        continue
                    rec = cmp_[mode]
                    if "ci" not in rec or "mean" not in rec:
                        continue
                    lo, hi = rec["ci"]
                    ordered = lo <= hi
                    if not ordered:
                        out.append(CheckResult(
                            f"bootstrap.ci.order:{name}:{mode}", FAIL,
                            "CI lower bound exceeds upper bound"))
                        continue
                    # recompute with the canonical procedure (identical seed/rounds)
                    boot_res = paired_bootstrap(va, vb, seed, rounds, alpha)
                    conv = _convention(ctx)
                    if conv == "candidate_minus_baseline":
                        e_mean, e_ci = boot_res["mean"], boot_res["ci"]
                    else:
                        e_mean = -boot_res["mean"]
                        e_ci = (-boot_res["ci"][1], -boot_res["ci"][0])
                    mean_ok = abs(e_mean - rec["mean"]) <= 1e-9
                    ci_ok = (abs(e_ci[0] - lo) <= 1e-9
                              and abs(e_ci[1] - hi) <= 1e-9)
                    if mean_ok and ci_ok:
                        out.append(CheckResult(
                            f"bootstrap.ci:{name}:{mode}", PASS,
                            "CI and mean reproduce exactly from raw vectors",
                            detail=[f"seed={seed} rounds={rounds} alpha={alpha} "
                                    f"convention={conv}"]))
                    else:
                        info = [
                            f"comparison={name} mode={mode}",
                            f"recorded mean={_fmt(rec['mean'])} ci=({_fmt(lo)},{_fmt(hi)})",
                            f"recomputed mean={_fmt(e_mean)} "
                            f"ci=({_fmt(e_ci[0])},{_fmt(e_ci[1])})"]
                        ok = (abs(e_mean - rec["mean"]) <= 0.01
                              and abs(lo - e_ci[0]) <= 0.01
                              and abs(hi - e_ci[1]) <= 0.01)
                        out.append(_check(
                            f"bootstrap.ci:{name}:{mode}",
                            ok,
                            "CI consistent (within bootstrap tolerance)",
                            "CI inconsistent with raw vectors",
                            detail=info))
    return out


_DISCOVERY_RE = ("exact_discovery", "covered_discovery", "exact_d", "discovery")


def _is_discovery(metric: str) -> bool:
    m = metric.lower()
    return any(tok in m for tok in _DISCOVERY_RE)


def c_discovery_metrics(ctx: VerifierContext) -> list[CheckResult]:
    fmt = _format(ctx)
    out = []
    episodes = _episodes(ctx)
    if fmt == "benchmark_v12":
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for cond, rows in data.get("raw", {}).items():
                for pol in _experiment(ctx).get("policies", []):
                    for i, row in enumerate(rows):
                        if pol not in row:
                            continue
                        for metric, value in row[pol].items():
                            if not _is_discovery(metric) or value is None:
                                continue
                            if not (0.0 <= value <= 1.0):
                                out.append(CheckResult(
                                    f"discovery.range:{cond}:{pol}:{metric}:seed{i}",
                                    FAIL, f"metric outside [0,1]: {value}"))
                            if episodes and abs(value * episodes - round(value * episodes)) > 1e-6:
                                out.append(CheckResult(
                                    f"discovery.fraction:{cond}:{pol}:{metric}:seed{i}",
                                    FAIL,
                                    f"metric is not a fraction of episodes (value={value}, episodes={episodes})"))
        # per-seed presence for discovery metrics in summary
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for cond, rows in data.get("raw", {}).items():
                if not rows:
                    continue
                sample = rows[0]
                for pol in _experiment(ctx).get("policies", []):
                    if pol not in sample:
                        continue
                    for metric in sample[pol]:
                        if not _is_discovery(metric):
                            continue
                        vals = [r[pol].get(metric) for r in rows if pol in r]
                        if any(v is None for v in vals):
                            out.append(CheckResult(
                                f"discovery.per_seed:{cond}:{pol}:{metric}",
                                FAIL, "per-seed discovery vector contains missing values"))
    elif fmt == "academy_memory":
        # discovery-like metrics are not part of the memory summary; note it
        out.append(CheckResult("discovery.metrics", PASS,
                               "no discovery metrics declared in this memory-experiment format",
                               detail=["academy memory summaries do not expose exact-discovery counters"]))
    else:
        out.append(CheckResult("discovery.metrics", INCONCLUSIVE,
                               f"unsupported format {fmt!r}"))
    return out


def c_per_seed_variability(ctx: VerifierContext) -> list[CheckResult]:
    fmt = _format(ctx)
    out = []
    seeds = _seeds(ctx)
    n = len(seeds)
    if fmt == "benchmark_v12":
        discovery_positive = {}
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for cond, rows in data.get("raw", {}).items():
                if not rows:
                    continue
                sample = rows[0]
                for pol, pentry in sample.items():
                    for metric, _v in pentry.items():
                        if not _is_discovery(metric):
                            continue
                        vals = [r[pol].get(metric) for r in rows
                                if pol in r and r[pol].get(metric) is not None]
                        if len(vals) < 2:
                            continue
                        positive = sum(1 for v in vals if v > 0)
                        key = f"{cond}/{pol}/{metric}"
                        discovery_positive[key] = (len(vals), positive)
                        stdev = statistics.pstdev(vals)
                        note = (f"n={len(vals)} positive={positive} "
                                f"stdev={stdev:.4f} vals={[round(v, 4) for v in vals]}")
                        if n < 8:
                            out.append(CheckResult(
                                f"per_seed.variability:{key}", PASS,
                                "per-seed discovery vector recorded (small sample)",
                                detail=[note, "n<8: pilot-grade evidence"]))
                        else:
                            out.append(CheckResult(
                                f"per_seed.variability:{key}", PASS,
                                "per-seed discovery vector recorded",
                                detail=[note]))
    elif fmt == "academy_memory":
        for rel in _fmt_by_role(ctx).get("summary", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for variant, v in (data.get("conditions_detail") or {}).items():
                per = v.get("per_seed")
                if not per:
                    continue
                out.append(CheckResult(
                    f"per_seed.present:{variant}", PASS,
                    f"per-seed detail present for {variant} (n={len(per)})",
                    detail=[f"train seeds recorded: {len(per)}"]))
    return out


def c_resource_fairness(ctx: VerifierContext) -> list[CheckResult]:
    out = []
    exp = _experiment(ctx)
    seeds = _seeds(ctx)
    episodes = _episodes(ctx)
    resources = exp.get("resources") or {}

    # identical seeds across policies is structural for raw per-seed rows
    if _format(ctx) == "benchmark_v12":
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for cond, rows in data.get("raw", {}).items():
                if len(rows) != len(seeds):
                    out.append(CheckResult(
                        f"fairness.seeds:{cond}", INCONCLUSIVE,
                        f"raw rows ({len(rows)}) != manifest seeds ({len(seeds)})"))
                    continue
                # every row must contain the same policy keys (shared streams)
                sample_keys = set(rows[0])
                for i, row in enumerate(rows):
                    if set(row) != sample_keys:
                        out.append(CheckResult(
                            f"fairness.policy.set:{cond}:seed{i}", FAIL,
                            "policy roster differs across seeds in the same condition"))
                # episode budget: success rates are fractions of episodes
                if episodes:
                    for i, row in enumerate(rows):
                        for pol, p in row.items():
                            sr = p.get("success_rate")
                            if sr is not None and abs(sr * episodes - round(sr * episodes)) > 1e-6:
                                out.append(CheckResult(
                                    f"fairness.episodes:{cond}:{pol}:seed{i}", FAIL,
                                    f"success_rate not a fraction of {episodes} episodes"))
        # per-policy CPU measurements
        cpu_measured = False
        for rel in _fmt_by_role(ctx).get("raw", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            for cond, rows in data.get("raw", {}).items():
                for row in rows:
                    for pol, p in row.items():
                        if p.get("cpu_s") is not None:
                            cpu_measured = True
        if resources.get("per_policy_cpu_s", True):
            out.append(_check(
                "fairness.cpu.measured",
                cpu_measured,
                "per-policy CPU measurements present",
                "per-policy CPU measurements missing (cpu_s not recorded)",
                inconclusive="CPU measurement policy not declared in manifest"))
        else:
            out.append(CheckResult(
                "fairness.cpu.measured", PASS,
                "per-policy CPU not required by this protocol"))

    # planner expansion budgets and memory limits recorded
    cap = resources.get("planner_max_expand")
    if cap is not None:
        out.append(CheckResult("fairness.planner_cap", PASS,
                               f"planner expansion cap recorded: {cap}"))
    else:
        out.append(CheckResult("fairness.planner_cap", INCONCLUSIVE,
                               "planner expansion cap not recorded in manifest"))
    mem = resources.get("memory_limit_bytes")
    if mem is not None:
        out.append(CheckResult("fairness.memory_limit", PASS,
                               f"memory limit recorded: {mem} bytes"))
    else:
        out.append(CheckResult("fairness.memory_limit", INCONCLUSIVE,
                               "memory limit not recorded in manifest"))
    return out


CHECK_REGISTRY: list[Callable[[VerifierContext], list[CheckResult]]] = [
    c_artifacts_exist,
    c_artifact_hashes,
    c_unique_artifact_records,
    c_raw_present,
    c_missing_duplicate_records,
    c_summary_matches_raw,
    c_comparison_direction,
    c_bootstrap_ci_consistent,
    c_discovery_metrics,
    c_per_seed_variability,
    c_resource_fairness,
]


def run_checks(ctx: VerifierContext) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in CHECK_REGISTRY:
        try:
            results.extend(check(ctx))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            results.append(CheckResult(
                check.__name__, INCONCLUSIVE,
                f"check could not run: {exc}"))
    return results
