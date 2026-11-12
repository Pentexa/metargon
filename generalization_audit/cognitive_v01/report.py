"""Render the cognitive scorecard as a compact Markdown report."""
from __future__ import annotations

from pathlib import Path

DEFAULT_REPORT = (Path(__file__).resolve().parent / "docs" /
                  "COGNITIVE_REPORT_v01.md")


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


DIM_ORDER = (
    ("causal_discovery", "1. Causal discovery"),
    ("learning_efficiency", "2. Learning efficiency"),
    ("knowledge_retention", "3. Knowledge retention"),
    ("generalization", "4. Generalization"),
    ("adaptability", "5. Adaptability"),
    ("planning_efficiency", "6. Planning efficiency"),
)

INDICATOR_NOTES = {
    "causal_discovery": "coverage = fraction of true (door, rule) pairs "
                        "found as hypotheses; precision = verified correct / verified.",
    "learning_efficiency": "second-half success rate on the training stream; "
                           "episodes-to-first-verified-rule inverted into 0-1.",
    "knowledge_retention": "revisit = success preserved at late vs early "
                           "occurrence; restart parity = restored/direct success.",
    "generalization": "zero-shot coverage = true rules already verified in "
                      "pre-eval memory; test success rate.",
    "adaptability": "success after rule switch; recovery speed = 1/(1+lag); "
                    "fraction of pre-switch wrong beliefs contradicted after.",
    "planning_efficiency": "fallback reduction = 1 − fallback rate on test.",
}


def render_scorecard(scorecard, path=DEFAULT_REPORT):
    meta = scorecard["meta"]
    lines = []
    lines.append("# METARGON Cognitive Intelligence Benchmark v0.1")
    lines.append("")
    lines.append(f"- Date: {meta['date']}")
    lines.append(f"- Git head: `{meta['git_head']}`")
    lines.append(f"- Seeds: {meta['seeds']}")
    lines.append(f"- Episode sizes: {meta['sizes']}")
    lines.append(f"- Frozen split counts: {meta['splits']}")
    lines.append(f"- Distinct true (door, rule) pairs in training stream: "
                 f"{meta['train_truths']}")
    lines.append("")

    # ---- descriptive indicators -------------------------------------------
    ind = scorecard.get("indicators", {})
    if ind:
        lines.append("## Descriptive indicators (0–1; NOT a composite IQ)")
        lines.append("")
        rows = []
        for dim_key, title in DIM_ORDER:
            for name, val in ind.get(dim_key, {}).items():
                rows.append({"dimension": title.split(". ", 1)[1],
                             "indicator": name, "value": _fmt(val)})
        lines.append(_table(rows, ["dimension", "indicator", "value"]))
        lines.append("")
        lines.append("Formula notes:")
        for dim_key, note in INDICATOR_NOTES.items():
            lines.append(f"- **{dim_key}**: {note}")
        lines.append("")

    # ---- dimension metric tables ------------------------------------------
    lines.append("CI methods — **W** = pooled Wilson interval (bounded "
                 "probability metrics); **t** = Student-t on per-seed "
                 "means (unbounded metrics).")
    lines.append("")
    for key, title in DIM_ORDER:
        dim = scorecard["dimensions"].get(key)
        if not dim:
            continue
        lines.append(f"## {title}")
        lines.append("")
        rows = []
        for name, s in dim["metrics"].items():
            rows.append({"metric": name, "mean ± 95% CI": _fmt_ci(s),
                         "CI": _fmt_method(s)})
        lines.append(_table(rows, ["metric", "mean ± 95% CI", "CI"]))
        lines.append("")

    lines.append("## Baseline context (evaluation pools)")
    lines.append("")
    for pool, row in scorecard.get("baselines", {}).items():
        lines.append(f"### {pool}")
        lines.append("")
        rows = []
        for pol, s in row.items():
            rows.append({"policy": pol,
                         "success_rate": _fmt_ci(s["success_rate"]),
                         "success CI": _fmt_method(s["success_rate"]),
                         "mean_steps_to_success": _fmt_ci(
                             s["mean_steps_to_success"]),
                         "within_episode_adaptation": _fmt_ci(
                             s["within_episode_adaptation"])})
        lines.append(_table(rows, ["policy", "success_rate", "success CI",
                                   "mean_steps_to_success",
                                   "within_episode_adaptation"]))
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Subject**: the unmodified Academy student "
                 "(`MetargonAcademyAgent`, default configuration).")
    lines.append("- **Splits**: frozen train/val/test/OOD pools snapshotted "
                 "from the Academy's canonical split function and verified "
                 "(disjointness, solvability, equality with the canonical "
                 "computation).")
    lines.append("- **Matched streams**: every policy runs the identical "
                 "deterministic episode stream per seed (same configs, "
                 "seeds, noise, budgets).")
    lines.append("- **Protocols**: training stream; zero-shot evaluation on "
                 "val/test/OOD; continual revisit (A,B,C,A,B,C,A); rule-"
                 "switch adaptability (10×A then 10×B); save/restore restart.")
    lines.append("- **Controls**: fresh student; memory-muted student "
                 "(identical training, declarative memory wiped before "
                 "evaluation); random / episodic / Q-learning baselines; "
                 "privileged oracle (diagnostic only).")
    lines.append("- **Leakage prevention**: students receive public "
                 "observations only; hidden dynamics are used solely by the "
                 "oracle; a test audits this.")
    lines.append("- **Statistics**: bounded probability metrics (rates, "
                 "coverage, precision, correction fractions) use a pooled "
                 "95% Wilson interval, which stays inside [0, 1]; "
                 "unbounded metrics (steps, episodes-to-event, planner "
                 "counters) use a Student-t interval on per-seed means.")
    lines.append("- **Planning counters**: the earlier `plan_cache_hit_rate` "
                 "(cache hits / plans used) was removed because the two "
                 "counters are not a hit/miss pair; it is replaced by "
                 "`plan_cache_hits_per_step` and `plans_used_per_step`, "
                 "reported separately with accurate names.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Results describe this student on this task distribution "
                 "(Academy gridworld, 80-step budget, no observation noise); "
                 "they are not a general statement about the agent.")
    lines.append("- The memory-muted control still allows within-episode "
                 "learning; it removes only cross-episode declarative "
                 "knowledge.")
    lines.append("- Muted discovery metrics are measured on the training "
                 "stream before wiping (identical by construction to the "
                 "trained arm); the muted arm's *behavioral* effect appears "
                 "in the evaluation protocols.")
    lines.append("- Zero-shot coverage is a knowledge-content measure "
                 "(pre-eval memory vs test truths), distinct from task "
                 "success, which also involves planning and exploration.")
    lines.append("- OOD results have wide confidence intervals; the lower "
                 "trained vs fresh OOD success is a suggestive negative-"
                 "transfer observation, not a definitive claim.")
    lines.append("- Confidence intervals quantify seed variability, not "
                 "environmental breadth; more families and noise regimes are "
                 "future work.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
