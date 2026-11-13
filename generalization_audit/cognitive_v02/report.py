"""Render the v0.2 scorecard as a compact Markdown report."""
from __future__ import annotations

from pathlib import Path

DEFAULT_REPORT = (Path(__file__).resolve().parent / "docs" /
                  "COGNITIVE_EVALUATION_V02.md")


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


def _fmt_method(s):
    if s.get("method") == "wilson":
        return f"W ({s.get('n_events', 0)} events)"
    return f"t ({s.get('n', 0)} seeds)"


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


ARM_ORDER = ("trained", "epoch_reset", "explore", "explore_epoch_reset",
             "fresh", "muted", "shuffled")
POOL_ORDER = ("val_clean", "test_clean", "test_noisy", "ood")


def render_scorecard(scorecard, path=DEFAULT_REPORT):
    meta = scorecard["meta"]
    lines = []
    lines.append("# METARGON Cognitive Evaluation v0.2 — independent audit")
    lines.append("")
    lines.append(f"- Date: {meta['date']}")
    lines.append(f"- Git head: `{meta['git_head']}`")
    lines.append(f"- Seeds: {meta['seeds']}")
    lines.append(f"- Sizes: {meta['sizes']}")
    lines.append(f"- Phases: {meta['phases']}")
    lines.append(f"- Frozen split counts: {meta['splits']}")
    lines.append("")

    lines.append("## 0. Questions asked")
    lines.append("")
    for q in meta.get("questions", []):
        lines.append(f"- {q}")
    lines.append("")

    lines.append("## 1. Training phase (knowledge + within-episode discovery)")
    lines.append("")
    rows = []
    for arm in ARM_ORDER:
        t = scorecard["arms"].get(arm, {}).get("train", {})
        if not t:
            continue
        rows.append({
            "arm": arm,
            "success": _fmt_ci(t.get("success_rate")),
            "within-episode discovery": _fmt_ci(
                t.get("within_episode_discovery")),
            "coverage": _fmt_ci(t.get("coverage")),
            "precision": _fmt_ci(t.get("precision")),
            "verified-correct": _fmt_ci(t.get("verified_correct")),
            "verified-wrong": _fmt_ci(t.get("verified_wrong")),
        })
    lines.append(_table(rows, ["arm", "success", "within-episode discovery",
                               "coverage", "precision", "verified-correct",
                               "verified-wrong"]))
    lines.append("")

    lines.append("## 2. Eval pools (clean / noisy / OOD)")
    lines.append("")
    for pool in POOL_ORDER:
        lines.append(f"### {pool}")
        lines.append("")
        rows = []
        for arm in ARM_ORDER:
            e = scorecard["arms"].get(arm, {}).get(pool, {})
            if not e:
                continue
            rows.append({
                "arm": arm,
                "success": _fmt_ci(e.get("success_rate")),
                "zero-shot coverage": _fmt_ci(e.get("zero_shot_coverage")),
                "within-episode discovery": _fmt_ci(
                    e.get("within_episode_discovery")),
                "steps→success": _fmt_ci(e.get("mean_steps_to_success")),
                "planner ops/step": _fmt_ci(e.get("planner_ops_per_step")),
                "fallback": _fmt_ci(e.get("fallback_rate")),
            })
        lines.append(_table(rows, ["arm", "success", "zero-shot coverage",
                                   "within-episode discovery", "steps→success",
                                   "planner ops/step", "fallback"]))
        lines.append("")

    lines.append("## 3. Paired comparisons (95% bootstrap CI)")
    lines.append("")
    rows = []
    for label, d in scorecard.get("paired", {}).items():
        rows.append({"comparison": label,
                     "Δ": _fmt(d.get("mean")),
                     "95% CI": f"[{d.get('ci', (None, None))[0]}, "
                               f"{d.get('ci', (None, None))[1]}]"
                     if d.get("ci") else "—",
                     "n": d.get("n", "")})
    lines.append(_table(rows, ["comparison", "Δ", "95% CI", "n"]))
    lines.append("")

    if "ab" in meta.get("phases", []):
        lines.append("## 4. A/B/A repeated learning (conflicting sequence)")
        lines.append("")
        rows = []
        for arm in ARM_ORDER:
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

    lines.append("## 5. Floors (sanity context)")
    lines.append("")
    rows = [{"floor": k, "success": _fmt_ci(v)}
            for k, v in scorecard.get("floors", {}).items()]
    lines.append(_table(rows, ["floor", "success"]))
    lines.append("")

    lines.append("## 6. Methodology")
    lines.append("")
    for line in meta.get("methodology", []):
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## 7. Limitations")
    lines.append("")
    for line in meta.get("limitations", []):
        lines.append(f"- {line}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
