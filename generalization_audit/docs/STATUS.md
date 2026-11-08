# Generalization Audit — Status (8 Nov 2026, morning)

## Done

- **Protocol audit** (v0.9 / v0.10 / Academy v0.1–v0.2 / EVREN):
  leakage, bias, artifacts, baseline fairness — findings written in
  `docs/AUDIT_REPORT.md` §1.
- **New task families** (standalone, unchanged student agents):
  - Family R — sustained-hold gates (`envs/base.py`, `HOLD_K=3`).
  - Family L — ordered-sequence gates (strict order, 30-step window).
  - Both solvable-by-construction, frozen before benchmarking
    (`configs/frozen_r.json`: 19 configs; `configs/frozen_l.json`: 9).
- **Benchmark runner** (`benchmark_generalization.py`): frozen-config
  load + solvability re-verification, paired seeds, paired bootstrap
  95% CIs, offline discovery/forgetting/adaptation metrics, noise
  parity, no-hidden-access runtime checks, matched-compute arm
  (`-cap` policies with per-decision expansion cap 120) and privileged
  per-family oracles.
- **Tests**: 18/18 pass (`python -m unittest discover -s
  generalization_audit/tests -t .`).
- **CI**: new `generalization-audit` job added to `.github/workflows/ci.yml`.
- **Final benchmarks run** (2 seeds × 15 episodes × 4 conditions per
  family): `results/summary_r.json`, `results/summary_l.json`, rendered
  tables in `results/rendered_tables.md`, findings written in §4/§5 of
  `docs/AUDIT_REPORT.md`.

## Notes / remaining

- Result scale is deliberately small (2 seeds × 15 episodes) because the
  machine was heavily loaded by parallel agents; CIs are wide and
  numeric findings are directional. A full-scale re-run (6+ seeds × 60+
  episodes) can be done later with `benchmark_generalization.py`.
- The cross-family condition intentionally omits the oracle (per-family
  plan models).
- Previous interrupted-run logs are in `results/logs/` (empty).
