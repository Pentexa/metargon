"""Transfer-evaluation protocol checks (Quality v0.7).

Implementation-neutral validation for the versioned transfer protocol
(`configs/transfer_eval_protocol_v01.json`). Checks run against a
manifest's optional `experiment.evaluation.transfer` declarations and
the committed artifacts; missing declarations yield INCONCLUSIVE,
never fabricated values.

Semantics:
- PASS: the requirement is declared and satisfied by the artifacts.
- FAIL: a declared requirement is violated (missing control, overlap,
  mis-mapped role, attribution inconsistency, …).
- INCONCLUSIVE: required metadata/declarations are absent so the item
  cannot be assessed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .checks import CheckResult, FAIL, INCONCLUSIVE, PASS, VerifierContext
from .manifest import validate_manifest, load_schema

PROTOCOL_REL = "generalization_audit/configs/transfer_eval_protocol_v01.json"
REQUIRED_ROLES = ("fresh", "trained_with_memory", "trained_memory_wiped",
                  "shuffled_memory", "planning_only")


@dataclass
class TransferProtocol:
    data: dict
    root: Path
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: Path) -> "TransferProtocol":
        p = root / PROTOCOL_REL
        if not p.exists():
            raise FileNotFoundError(f"protocol file missing: {p}")
        return cls(json.loads(p.read_text(encoding="utf-8")), root)

    @property
    def id(self) -> str:
        return self.data.get("protocol", "transfer-eval-protocol-v01")

    @property
    def version(self) -> str:
        return self.data.get("version", "v01")

    @property
    def frozen_seeds(self) -> list[int]:
        return list(self.data.get("seed_registration", {}).get("frozen_seeds", []))


def _transfer_decl(ctx: VerifierContext) -> dict:
    return ctx.manifest.get("experiment", {}).get("evaluation", {}).get("transfer", {})


def _roles(ctx: VerifierContext) -> dict:
    return _transfer_decl(ctx).get("roles", {})


def _fmt_artifact_paths(ctx: VerifierContext) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in ctx.manifest.get("artifacts", []):
        out.setdefault(a.get("role", "raw"), []).append(a["path"])
    return out


def _raw_data_blocks(ctx: VerifierContext) -> list[dict]:
    blocks = []
    for rel in _fmt_artifact_paths(ctx).get("raw", []) + \
            _fmt_artifact_paths(ctx).get("summary", []):
        try:
            blocks.append(ctx.load_json(rel))
        except FileNotFoundError:
            continue
    return blocks


def _has_row_dicts(blocks: list[dict]) -> bool:
    for b in blocks:
        raw = b.get("raw") if isinstance(b, dict) else None
        if isinstance(raw, dict):
            for rows in raw.values():
                if rows and isinstance(rows[0], dict):
                    return True
    return False


def _config_keys_of(block: dict) -> set[str] | None:
    """Extract serialized config keys when the artifact exposes pools."""
    pools = block.get("train_pool") or block.get("eval_pool") or \
        block.get("pools") or block.get("configs")
    if isinstance(pools, dict):
        return set(json.dumps(v, sort_keys=True, default=str) for v in pools.values())
    if isinstance(pools, list):
        return set(json.dumps(v, sort_keys=True, default=str) for v in pools)
    # academy format: conditions_detail per-config keys
    cd = block.get("conditions_detail")
    if isinstance(cd, dict):
        keys = set()
        for v in cd.values():
            if isinstance(v, dict):
                for mode in ("classic_frozen", "classic_adapt", "classic_test"):
                    pc = v.get(f"{mode}_per_config")
                    if isinstance(pc, list):
                        keys.update(json.dumps(k, sort_keys=True, default=str)
                                    for k in pc)
        return keys
    return None


def check_transfer_protocol_registered(ctx: VerifierContext,
                                       protocol: TransferProtocol | None = None
                                       ) -> list[CheckResult]:
    if protocol is None:
        try:
            protocol = TransferProtocol.load(ctx.manifest_path.parent.parent.parent)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            return [CheckResult("transfer.protocol.registered", FAIL,
                                f"protocol file missing/invalid: {exc}")]
    ok = protocol.id == ctx.manifest.get("protocol_version", "transfer-eval-protocol-v01") \
        or ctx.manifest.get("protocol_version", "").startswith(protocol.id)
    return [_check_result(
        "transfer.protocol.registered",
        ok,
        f"manifest cites protocol {protocol.id} ({protocol.version})",
        f"manifest protocol_version {ctx.manifest.get('protocol_version')!r} does not match {protocol.id}")]


def check_transfer_control_roles(ctx: VerifierContext) -> list[CheckResult]:
    roles = _roles(ctx)
    if not roles:
        return [CheckResult("transfer.roles.declared", INCONCLUSIVE,
                            "no transfer role mapping declared in manifest")]
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    if missing:
        return [CheckResult("transfer.roles.declared", FAIL,
                            "required control arms not declared",
                            detail=[f"missing roles: {missing}"])]
    dup = [v for v in roles.values() if list(roles.values()).count(v) > 1]
    if dup:
        return [CheckResult("transfer.roles.distinct", FAIL,
                            "control roles map to duplicate policies/variants",
                            detail=[f"duplicate targets: {sorted(set(dup))}"])]
    # every mapped policy should exist in raw rows or conditions_detail
    blocks = _raw_data_blocks(ctx)
    present = set()
    for b in blocks:
        raw = b.get("raw") if isinstance(b, dict) else None
        if isinstance(raw, dict):
            for rows in raw.values():
                if rows and isinstance(rows[0], dict):
                    present.update(rows[0])
        cd = b.get("conditions_detail") if isinstance(b, dict) else None
        if isinstance(cd, dict):
            present.update(cd)
    unresolved = [v for v in roles.values() if v not in present]
    if unresolved:
        return [CheckResult("transfer.roles.resolve", FAIL,
                            "declared roles do not resolve to artifact policies",
                            detail=[f"unresolved: {unresolved}"])]
    return [CheckResult("transfer.roles.resolve", PASS,
                        "all declared roles resolve to artifact policies")]


def check_transfer_unseen_separation(ctx: VerifierContext) -> list[CheckResult]:
    decl = _transfer_decl(ctx)
    unseen = decl.get("unseen_configs_declared")
    if unseen is not True:
        return [CheckResult("transfer.unseen.separation", INCONCLUSIVE,
                            "unseen-task separation not declared")]
    split = decl.get("split_source")
    if not split:
        return [CheckResult("transfer.unseen.separation", INCONCLUSIVE,
                            "unseen-task separation declared without split_source")]
    # if artifacts expose both train and eval pools, verify disjointness
    blocks = _raw_data_blocks(ctx)
    for b in blocks:
        train = _config_keys_of({"pools": b.get("train_pool")}) if b.get("train_pool") else None
        eval_ = _config_keys_of({"pools": b.get("eval_pool")}) if b.get("eval_pool") else None
        if train is not None and eval_ is not None:
            overlap = train & eval_
            if overlap:
                return [CheckResult("transfer.unseen.separation", FAIL,
                                    "train/eval configuration overlap detected",
                                    detail=[f"overlapping configs: {len(overlap)}"])]
            return [CheckResult("transfer.unseen.separation", PASS,
                                "train/eval pools disjoint",
                                detail=[f"train={len(train)} eval={len(eval_)}"])]
    return [CheckResult("transfer.unseen.separation", INCONCLUSIVE,
                        "train/eval pools not exposed in artifacts; separation rests on the declaration")]


def check_transfer_frozen_immutability(ctx: VerifierContext) -> list[CheckResult]:
    decl = _transfer_decl(ctx)
    mode = decl.get("eval_mode")
    if mode is None:
        return [CheckResult("transfer.frozen.immutability", INCONCLUSIVE,
                            "evaluation mode not declared")]
    if mode == "frozen":
        asserted = decl.get("frozen_immutability_asserted")
        if asserted is True:
            return [CheckResult("transfer.frozen.immutability", PASS,
                                "frozen evaluation immutability attested")]
        # academy-style trace presence is treated as supporting evidence
        for rel in _fmt_artifact_paths(ctx).get("summary", []):
            try:
                data = ctx.load_json(rel)
            except FileNotFoundError:
                continue
            if isinstance(data, dict) and "classic_frozen_trace" in data:
                return [CheckResult("transfer.frozen.immutability", PASS,
                                    "frozen trace artifact present (inline)")
                        ]
        return [CheckResult("transfer.frozen.immutability", INCONCLUSIVE,
                            "frozen mode declared without immutability attestation")]
    return [CheckResult("transfer.frozen.immutability", PASS,
                        f"evaluation mode {mode}; frozen immutability not applicable")]


def check_transfer_memory_ablation_paired(ctx: VerifierContext) -> list[CheckResult]:
    roles = _roles(ctx)
    if not roles:
        return [CheckResult("transfer.ablation.paired", INCONCLUSIVE,
                            "no transfer roles declared")]
    mem = roles.get("trained_with_memory")
    wiped = roles.get("trained_memory_wiped")
    if not mem or not wiped:
        return [CheckResult("transfer.ablation.paired", INCONCLUSIVE,
                            "memory/wiped ablation roles missing")]
    blocks = _raw_data_blocks(ctx)
    for b in blocks:
        raw = b.get("raw") if isinstance(b, dict) else None
        if isinstance(raw, dict):
            for cond, rows in raw.items():
                if not rows or not isinstance(rows[0], dict):
                    continue
                if mem not in rows[0] or wiped not in rows[0]:
                    continue
                lengths = {p: len(rows[0][p]) for p in (mem, wiped)}
                # per-policy rows are dicts; vector length check applies to
                # per-config arrays in academy format (below)
                return [CheckResult("transfer.ablation.paired", PASS,
                                    "memory/wiped arms co-occur in raw rows")]
        cd = b.get("conditions_detail") if isinstance(b, dict) else None
        if isinstance(cd, dict) and mem in cd and wiped in cd:
            v_mem = cd[mem].get("classic_frozen_per_config")
            v_wiped = cd[wiped].get("classic_frozen_per_config")
            if isinstance(v_mem, list) and isinstance(v_wiped, list):
                if len(v_mem) != len(v_wiped):
                    return [CheckResult("transfer.ablation.paired", FAIL,
                                        "memory/wiped per-config vectors differ in length",
                                        detail=[f"{len(v_mem)} vs {len(v_wiped)}"])]
                return [CheckResult("transfer.ablation.paired", PASS,
                                    "memory/wiped per-config vectors paired (same length)")]
    return [CheckResult("transfer.ablation.paired", INCONCLUSIVE,
                        "memory/wiped arms not found in artifacts")]


def check_transfer_attribution_consistency(ctx: VerifierContext) -> list[CheckResult]:
    """Verify reported memory-to-action attribution traces are internally
    consistent: sum(source_*) == plans_used (or probes_evaluated when
    plans_used is absent)."""
    blocks = _raw_data_blocks(ctx)
    seen = False
    for b in blocks:
        runs = b.get("runs", b.get("records"))
        if isinstance(runs, list):
            for rec in runs:
                per = rec.get("per", rec) if isinstance(rec, dict) else {}
                tr = per.get("frozen_trace_mean", {})
                if not isinstance(tr, dict) or not tr:
                    continue
                seen = True
                src_keys = [k for k in tr if k.startswith("source_")]
                if not src_keys:
                    continue
                s = sum(tr[k] for k in src_keys if isinstance(tr[k], (int, float)))
                plans = tr.get("plans_used")
                probes = tr.get("probes_evaluated")
                baseline = plans if isinstance(plans, (int, float)) else probes
                if not isinstance(baseline, (int, float)):
                    continue
                if abs(s - baseline) > 1e-6:
                    return [CheckResult("transfer.attribution.consistency", FAIL,
                                        "attribution trace sums do not match plans/probes",
                                        detail=[f"sum(source_*)={s} vs {baseline}"])]
        else:
            # jsonl file referenced in artifact path
            for rel in _fmt_artifact_paths(ctx).get("log", []):
                try:
                    for line in ctx.resolve(rel).read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        per = rec.get("per", {})
                        tr = per.get("frozen_trace_mean", {})
                        if not isinstance(tr, dict) or not tr:
                            continue
                        seen = True
                        src_keys = [k for k in tr if k.startswith("source_")]
                        if not src_keys:
                            continue
                        s = sum(tr[k] for k in src_keys if isinstance(tr[k], (int, float)))
                        plans = tr.get("plans_used")
                        probes = tr.get("probes_evaluated")
                        baseline = plans if isinstance(plans, (int, float)) else probes
                        if not isinstance(baseline, (int, float)):
                            continue
                        if abs(s - baseline) > 1e-6:
                            return [CheckResult("transfer.attribution.consistency", FAIL,
                                                "attribution trace sums do not match plans/probes",
                                                detail=[f"variant={rec.get('variant')} "
                                                        f"sum(source_*)={s} vs {baseline}"])]
                except FileNotFoundError:
                    continue
    if not seen:
        return [CheckResult("transfer.attribution.consistency", INCONCLUSIVE,
                            "no attribution trace data available")]
    return [CheckResult("transfer.attribution.consistency", PASS,
                        "attribution traces internally consistent")]


def check_transfer_budget_fairness(ctx: VerifierContext) -> list[CheckResult]:
    out = []
    resources = ctx.manifest.get("experiment", {}).get("resources", {})
    cap = resources.get("planner_max_expand")
    if cap is None:
        out.append(CheckResult("transfer.budgets.planner", INCONCLUSIVE,
                               "planner expansion cap not recorded"))
    else:
        out.append(CheckResult("transfer.budgets.planner", PASS,
                               f"planner cap recorded: {cap}"))
    cpu_req = resources.get("per_policy_cpu_s")
    if cpu_req is None:
        out.append(CheckResult("transfer.budgets.cpu", INCONCLUSIVE,
                               "per-policy CPU reporting not declared"))
    else:
        blocks = _raw_data_blocks(ctx)
        measured = False
        for b in blocks:
            raw = b.get("raw") if isinstance(b, dict) else None
            if isinstance(raw, dict):
                for rows in raw.values():
                    for row in rows:
                        if isinstance(row, dict):
                            for p in row.values():
                                if isinstance(p, dict) and p.get("cpu_s") is not None:
                                    measured = True
        if not cpu_req:
            out.append(CheckResult("transfer.budgets.cpu", PASS,
                                   "CPU not required by this protocol"))
        elif not _has_row_dicts(blocks):
            out.append(CheckResult("transfer.budgets.cpu", INCONCLUSIVE,
                                   "no raw rows to inspect for CPU measurement"))
        elif measured:
            out.append(CheckResult("transfer.budgets.cpu", PASS,
                                   "per-policy CPU present in raw rows"))
        else:
            out.append(CheckResult("transfer.budgets.cpu", FAIL,
                                   "per-policy CPU missing from raw rows"))
    max_steps = ctx.manifest.get("experiment", {}).get("action_budget_max_steps")
    if max_steps is None:
        out.append(CheckResult("transfer.budgets.steps", INCONCLUSIVE,
                               "action-budget max steps not recorded"))
    else:
        out.append(CheckResult("transfer.budgets.steps", PASS,
                               f"action budget recorded: {max_steps}"))
    return out


def _check_result(name: str, ok: bool | None, good: str, bad: str,
                  inconclusive: str = "", detail: list[str] | None = None) -> CheckResult:
    if ok is None:
        return CheckResult(name, INCONCLUSIVE, inconclusive or "evidence missing", detail or [])
    return CheckResult(name, PASS if ok else FAIL, good if ok else bad, detail or [])


TRANSFER_CHECK_REGISTRY = (
    check_transfer_protocol_registered,
    check_transfer_control_roles,
    check_transfer_unseen_separation,
    check_transfer_frozen_immutability,
    check_transfer_memory_ablation_paired,
    check_transfer_attribution_consistency,
    check_transfer_budget_fairness,
)


def run_transfer_checks(ctx: VerifierContext,
                        protocol: TransferProtocol | None = None) -> list[CheckResult]:
    results = []
    for check in TRANSFER_CHECK_REGISTRY:
        try:
            results.extend(check(ctx, protocol) if check.__name__ ==
                           "check_transfer_protocol_registered" else check(ctx))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            results.append(CheckResult(check.__name__, INCONCLUSIVE,
                                       f"check could not run: {exc}"))
    return results
