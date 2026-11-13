"""Render the Cognitive Integration v0.1 scorecard + analysis."""
from __future__ import annotations

from pathlib import Path

from . import config as C

DEFAULT_REPORT = (Path(__file__).resolve().parent / "docs" /
                  "COGNITIVE_INTEGRATION_V01.md")


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _fmt_ci(s):
    m, ci = s.get("mean"), s.get("ci")
    if m is None:
        return "—"
    if ci is None:
        return f"{m:.3f}"
    return f"{m:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]"


def _table(rows, headers=None):
    if not rows:
        return ""
    headers = headers or list(rows[0].keys())
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(h, "")) for h in headers)
                   + " |")
    return "\n".join(out)


def render_report(scorecard, analysis, path=DEFAULT_REPORT):
    meta = scorecard["meta"]
    lines = []
    lines.append("# METARGON Cognitive Integration Evaluation v0.1")
    lines.append("")
    lines.append(f"- Date: {meta['date']}")
    lines.append(f"- Git head: `{meta['git_head']}`")
    lines.append(f"- Seeds (confirmatory): {meta['seeds']}")
    lines.append(f"- Sizes: {meta['sizes']}")
    lines.append(f"- Frozen split counts: {meta['splits']}")
    lines.append(f"- Preregistration: {meta['preregistration']}")
    lines.append("")

    lines.append("## 1. Arms overview (success on primary pool, val_clean)")
    lines.append("")
    rows = []
    for arm in C.ARM_ORDER:
        e = scorecard["arms"].get(arm, {}).get(C.PRIMARY_POOL, {})
        if not e:
            continue
        rows.append({
            "arm": arm,
            "success": _fmt_ci(e.get("success_rate")),
            "zero-shot coverage": _fmt_ci(e.get("zero_shot_coverage")),
            "within-episode discovery": _fmt_ci(
                e.get("within_episode_discovery")),
            "planner ops/step": _fmt_ci(e.get("planner_ops_per_step")),
            "fallback": _fmt_ci(e.get("fallback_rate")),
        })
    lines.append(_table(rows, ["arm", "success", "zero-shot coverage",
                               "within-episode discovery", "planner ops/step",
                               "fallback"]))
    lines.append("")

    lines.append("## 2. All pools (success; memory controls)")
    lines.append("")
    for pool in C.EVAL_POOLS:
        name = pool[0]
        lines.append(f"### {name}")
        lines.append("")
        rows = []
        for arm in C.ARM_ORDER:
            e = scorecard["arms"].get(arm, {}).get(name, {})
            if not e:
                continue
            rows.append({
                "arm": arm,
                "success": _fmt_ci(e.get("success_rate")),
                "zero-shot coverage": _fmt_ci(
                    e.get("zero_shot_coverage")),
                "steps→success": _fmt_ci(
                    e.get("mean_steps_to_success")),
                "planner ops/step": _fmt_ci(
                    e.get("planner_ops_per_step")),
                "fallback": _fmt_ci(e.get("fallback_rate")),
            })
        lines.append(_table(rows, ["arm", "success", "zero-shot coverage",
                                   "steps→success", "planner ops/step",
                                   "fallback"]))
        lines.append("")

    lines.append("## 3. Preregistered comparisons (paired bootstrap 95% CI)")
    lines.append("")
    rows = []
    for label, d in scorecard.get("paired", {}).items():
        rows.append({"comparison": label, "Δ": _fmt(d.get("mean")),
                     "95% CI": (f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
                                if d.get("ci") else "—"),
                     "n": d.get("n", "")})
    lines.append(_table(rows, ["comparison", "Δ", "95% CI", "n"]))
    lines.append("")

    lines.append("## 4. Interaction and attribution analysis")
    lines.append("")
    lines.append("Interaction term per pool: (explore_epoch_reset − "
                 "epoch_reset) − (explore − trained) on success rate.")
    lines.append("")
    rows = []
    for pool, d in analysis["interactions"].items():
        rows.append({"pool": pool, "interaction Δ": _fmt(d.get("mean")),
                     "95% CI": (f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"
                                if d.get("ci") else "—"),
                     "n": d.get("n", "")})
    lines.append(_table(rows, ["pool", "interaction Δ", "95% CI", "n"]))
    lines.append("")
    lines.append("Attribution deltas (memory-wiped variants):")
    lines.append("")
    rows = []
    for label, d in analysis["deltas"].items():
        if "muted" in label or "shuffled" in label:
            rows.append({"comparison": label, "Δ": _fmt(d.get("mean")),
                         "95% CI": (f"[{d['ci'][0]:.3f}, "
                                    f"{d['ci'][1]:.3f}]"
                                    if d.get("ci") else "—"),
                         "n": d.get("n", "")})
    lines.append(_table(rows, ["comparison", "Δ", "95% CI", "n"]))
    lines.append("")

    lines.append("## 5. A/B/A retention (interference)")
    lines.append("")
    rows = []
    for arm in C.ARM_ORDER:
        r = scorecard["arms"].get(arm, {}).get("ab.retention", {})
        if not r:
            continue
        rows.append({
            "arm": arm,
            "A1": _fmt_ci(r.get("a1_success")),
            "B (conflict)": _fmt_ci(r.get("b_success")),
            "A2": _fmt_ci(r.get("a2_success")),
            "A-rule retained final": _fmt_ci(
                r.get("a_rule_retained_final")),
        })
    lines.append(_table(rows, ["arm", "A1", "B (conflict)", "A2",
                               "A-rule retained final"]))
    lines.append("")

    lines.append("## 6. Floors (sanity)")
    lines.append("")
    rows = [{"floor": k, "success": _fmt_ci(v)}
            for k, v in scorecard.get("floors", {}).items()]
    lines.append(_table(rows, ["floor", "success"]))
    lines.append("")

    lines.append("## 7. Methodology")
    lines.append("")
    for line in meta.get("methodology", []):
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("")
    for line in meta.get("limitations", []):
        lines.append(f"- {line}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
