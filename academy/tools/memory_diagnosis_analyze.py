"""Analysis helpers for the Persistent Learning Diagnosis v0.1 artifacts.

Computes paired per-seed deltas (candidate − reference) with bootstrap
95% CIs for frozen/adaptive success across variants and cells, and
renders the markdown tables used in the root-cause report.
"""

from __future__ import annotations

import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results", "academy_v01",
                           "memory_diagnosis_attribution_v01")


def bootstrap_ci(values, seed=0, n_boot=2000):
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    m = len(values)
    means = []
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]
        means.append(sum(values[i] for i in idx) / m)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return (lo, hi)


def paired_delta(a_per_seed, b_per_seed, seed=0):
    """a − b per seed; returns (mean, ci)."""
    assert len(a_per_seed) == len(b_per_seed)
    diffs = [a - b for a, b in zip(a_per_seed, b_per_seed)]
    lo, hi = bootstrap_ci(diffs, seed=seed)
    return (sum(diffs) / len(diffs), (lo, hi))


def per_seed_success(cell_records, regime="frozen"):
    """Ordered list over seeds of config-mean success for a variant cell."""
    out = []
    for rec in cell_records:
        out.append(rec[regime]["summary"]["config_mean_success"])
    return out


def load_main(path=None):
    path = path or os.path.join(RESULTS_DIR, "main_summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def variant_table(main, cell, reference="epoch_reset", regime="frozen",
                 seed=1234):
    """Rows: variant | success | Δ vs reference [95% CI]."""
    cells = main["cells"].get(cell, {})
    variants = cells.get("variants", {})
    ref = per_seed_success(variants[reference], regime)
    lines = [f"| variant | {regime} | delta vs {reference} [95% CI] |",
             "|---|---:|---:|"]
    for name, recs in variants.items():
        vals = per_seed_success(recs, regime)
        mean = sum(vals) / len(vals)
        d, (lo, hi) = paired_delta(vals, ref, seed=seed)
        lines.append(f"| {name} | {mean:.3f} | {d:+.3f} "
                     f"[{lo:+.3f}, {hi:+.3f}] |")
    return "\n".join(lines)


def snapshot_table(main, cell):
    """Per-variant aggregate memory snapshot facts (frozen eval start)."""
    cells = main["cells"].get(cell, {})
    variants = cells.get("variants", {})
    lines = ["| variant | exact-usable | false-contradict | verified-total | "
             "size |",
             "|---|---:|---:|---:|---:|"]
    for name, recs in variants.items():
        snap = None
        for rec in recs:
            s = rec["frozen"].get("snapshot_agg")
            if s:
                snap = s if snap is None else _add_nested(snap, s)
        if snap is None:
            continue
        lines.append(
            f"| {name} | {snap['by_class']['exact']['usable']} | "
            f"{snap['false_contradictions']} | "
            f"{snap['verified_total']} | {snap['size']} |")
    return "\n".join(lines)


def _add_nested(a, b):
    for k, v in b.items():
        if isinstance(v, dict):
            a[k] = _add_nested(a.get(k, {}), v)
        else:
            a[k] = a.get(k, 0) + v
    return a


def reuse_table(main, cell):
    cells = main["cells"].get(cell, {})
    variants = cells.get("variants", {})
    lines = ["| variant | hyp-plans (frozen) | reuse-plans | reuse share | "
             "memory_correct | memory_incorrect |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, recs in variants.items():
        hyp = reuse = corr = incorr = 0
        for rec in recs:
            agg = rec["frozen"].get("agg", {})
            hyp += agg.get("hypothesis_plans", 0)
            reuse += agg.get("reuse_plans", 0)
            corr += agg.get("trace", {}).get("memory_correct", 0)
            incorr += agg.get("trace", {}).get("memory_incorrect", 0)
        share = (reuse / hyp) if hyp else None
        lines.append(
            f"| {name} | {hyp} | {reuse} | "
            f"{(f'{share:.2f}' if share is not None else '—')} | "
            f"{corr} | {incorr} |")
    return "\n".join(lines)


def render_all(main_path=None):
    main = load_main(main_path)
    out = []
    for cell in ("classic", "conflict", "noisy", "ood"):
        if cell not in main["cells"]:
            continue
        out.append(f"### {cell} — frozen")
        out.append("")
        out.append(variant_table(main, cell, regime="frozen"))
        out.append("")
        out.append(f"### {cell} — adaptive")
        out.append("")
        out.append(variant_table(main, cell, regime="adaptive"))
        out.append("")
        out.append(f"### {cell} — memory snapshot (frozen eval start)")
        out.append("")
        out.append(snapshot_table(main, cell))
        out.append("")
        out.append(f"### {cell} — reuse attribution (frozen)")
        out.append("")
        out.append(reuse_table(main, cell))
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    print(render_all(p))
