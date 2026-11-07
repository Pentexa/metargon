# Generalization Audit — Nightly Status (8 Nov 2026, ~02:00)

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

## Open / next morning

1. **Run the final benchmarks** (they were started several times but the
   machine was too loaded; the last attempt was interrupted at night):
   ```bash
   python generalization_audit/benchmark_generalization.py --family r \
       --seeds 3 --episodes 20 \
       --conditions stable,shift,novel,cross_family \
       --json generalization_audit/results/summary_r.json
   python generalization_audit/benchmark_generalization.py --family l \
       --seeds 3 --episodes 20 \
       --conditions stable,shift,novel,cross_family \
       --json generalization_audit/results/summary_l.json
   ```
   Serial mode is recommended (parallel works, but spawn + heavy CPU
   contention made progress hard to monitor on this machine).
2. **Render report tables** (after the runs):
   ```bash
   python generalization_audit/tools/render_report_tables.py \
       generalization_audit/results/summary_r.json \
       generalization_audit/results/summary_l.json
   ```
   Then fill `docs/AUDIT_REPORT.md` §4 (results) and §5 (findings).
3. **Final pass**: unit tests, freeze determinism, `scripts/check_repo.py`,
   commit and push the completed results.
4. Decide whether to keep the parallel `benchmark_parallel` path as-is
   (it is fixed w.r.t. oracle roster filtering; only monitoring was slow).

## Notes

- Logs from interrupted runs: `results/logs/r_log.txt`, `l_log.txt`
  (empty — runs produced no output before being stopped).
- No final result JSONs exist yet (`results/summary_{r,l}.json` absent).
- Expected/known behaviours observed in smoke runs (not final numbers):
  family R: Bayes (family-F model) still strong; metargon10 beats
  model-free baselines; compute-capped variants lose ground at cap 120.
  family L: metargon10 discovers D1 exact rule often but overall success
  remains limited; model-free baselines near zero.
