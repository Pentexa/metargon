"""METARGON Academy — report tooling.

Aggregates training curves and evaluation JSONs from results/academy_v01/
into summary CSVs and a readable text summary. Run from the project root:

    python -m academy.tools.report
"""

from __future__ import annotations

import csv
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "academy_v01")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS, exist_ok=True)

    # ---- training curves -------------------------------------------------
    rows = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.startswith("train_") or not f.endswith(".json"):
            continue
        data = load_json(os.path.join(RESULTS, f))
        policy = data["policy"]
        for c in data["curve"]:
            rows.append([policy, c["episode"], c["success"], c["steps"],
                         c["door_events"]])
    if rows:
        with open(os.path.join(RESULTS, "learning_curves.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["policy", "episode", "success", "steps", "door_events"])
            w.writerows(rows)

    # ---- evaluation summaries --------------------------------------------
    summary_lines = []
    for kind in ("val", "test", "ood"):
        path = os.path.join(RESULTS, f"eval_{kind}.json")
        if not os.path.exists(path):
            continue
        data = load_json(path)
        summary_lines.append(f"=== evaluation pool: {kind} ===")
        summary_lines.append(f"n_seeds={data['n_seeds']}")
        for policy, res in data["results"].items():
            s = res["summary"]
            summary_lines.append(
                f"  {policy:20s} success={s['success_rate']:.3f} "
                f"(CI {s['ci'][0]:.3f}-{s['ci'][1]:.3f}) "
                f"steps={s['mean_steps']:.1f} n={s['n']}")
        with open(os.path.join(RESULTS, f"eval_{kind}_summary.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["policy", "success_rate", "ci_lo", "ci_hi",
                        "mean_steps", "n"])
            for policy, res in data["results"].items():
                s = res["summary"]
                w.writerow([policy, f"{s['success_rate']:.4f}",
                            f"{s['ci'][0]:.4f}", f"{s['ci'][1]:.4f}",
                            f"{s['mean_steps']:.2f}", s["n"]])
        if "comparisons" in data:
            for name, cmp_ in data["comparisons"].items():
                d = cmp_["paired_difference"]
                summary_lines.append(
                    f"  paired {name}: mean={d['mean']:.4f} "
                    f"CI {d['ci'][0]:.4f}-{d['ci'][1]:.4f}")
    # ---- block-wise curves (learning progress / forgetting view) --------
    block_rows = []
    BLOCK = 120
    if rows:
        by_policy = {}
        for r in rows:
            by_policy.setdefault(r[0], []).append(r)
        for policy, curve in sorted(by_policy.items()):
            n = len(curve)
            for start in range(0, n, BLOCK):
                seg = [r for r in curve if start <= r[1] < start + BLOCK]
                if not seg:
                    continue
                succ = sum(1 for r in seg if r[2])
                block_rows.append([policy, start, start + len(seg) - 1,
                                   f"{succ / len(seg):.3f}"])
    if block_rows:
        with open(os.path.join(RESULTS, "training_progress_blocks.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["policy", "ep_start", "ep_end", "success_rate"])
            w.writerows(block_rows)

    # ---- model sizes -----------------------------------------------------
    pol_dir = os.path.join(RESULTS, "policies")
    model_rows = []
    if os.path.isdir(pol_dir):
        for f in sorted(os.listdir(pol_dir)):
            if not f.endswith(".json"):
                continue
            size = os.path.getsize(os.path.join(pol_dir, f))
            model_rows.append([f[:-5], size])
            summary_lines.append(f"model {f[:-5]}: {size} bytes (JSON state)")
    if model_rows:
        with open(os.path.join(RESULTS, "model_sizes.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["policy", "state_bytes"])
            w.writerows(model_rows)

    if summary_lines:
        with open(os.path.join(RESULTS, "eval_summary.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(summary_lines))
    print("\n".join(summary_lines) if summary_lines else "no eval data yet")
    if rows:
        print(f"\nlearning curves: {len(rows)} rows -> learning_curves.csv")


if __name__ == "__main__":
    main()
