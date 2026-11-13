# METARGON — Persistent Learning Diagnosis v0.1

**Branch:** `research/memory-attribution-v01` · **Base:** `origin/main` @ `0d1a71c`

## What this is

A **root-cause measurement study** (not a mechanism change) of why
METARGON sometimes fails to retain and reuse correct causal knowledge
across tasks. It builds on the closed PR #50 / PR #53 findings and the
Memory v0.7 mechanisms. No planner, exploration agent, memory or audit
framework code is modified; only a dedicated diagnostic harness, tests
and artifacts are added.

## Questions and how they are measured

| Q | approach |
|---|---|
| 1. Why do verified hypotheses later become unreliable? | per-phase memory snapshots classified against ground truth (exact/subset/incorrect × status), false-contradiction counts, A1→B status transitions |
| 2. Why does retained memory sometimes reduce success? | negative-transfer controls: trained vs memory-muted vs shuffled-content on identical eval streams |
| 3. Do successful decisions depend on previous knowledge? | reuse attribution: hypothesis-sourced plans whose hypothesis predates eval (retained) vs created during eval; memory_correct/incorrect |
| 4. Is A/B/A recovery retention or rediscovery? | repeated-learning protocol with per-phase hypothesis bookkeeping; A2 success split into retained (created A1, survived B) vs rediscovered |
| 5. When does epoch_reset protect / fail? | variant × family matrix (classic/conflict/noisy/OOD) × regime (frozen/adaptive) |

## Protocol

- Harness: `academy/tools/memory_diagnosis_attribution_v01.py`
- Variants: fresh / original / epoch / epoch_reset / muted (memory-
  muted control) / shuffled (content-broken control)
- Families: classic (held-out), conflict (flipped polarity, isolated),
  noisy (q=0.15), OOD
- Regimes: frozen (integrity-checked) + adaptive
- Independent diagnostic seeds 20270201+ (disjoint from all accredited
  runs); PYTHONHASHSEED=0; canonical hash-seed-independent ordering;
  paired per-config bootstrap CIs
- No tuning against the independent auditor's final evaluation seeds

## Results (summary)

Full tables in `docs/RESEARCH_MEMORY_ATTRIBUTION_V01.md` and
`results/academy_v01/memory_diagnosis_attribution_v01/REPORT.md`.

### Q1 — Why verified hypotheses become unreliable
Global memory accumulates false contradictions across tasks (93–174 per
cell; 0–12 usable exact hypotheses). Episode-scoped memory demotes
instead of destroying: 84–156 usable exact hypotheses survive with zero
false contradictions. Unreliability is a global-contradiction artifact.

### Q2 — Why retained memory sometimes reduces success
Retained memory hurts exactly when it is wrong for the current context
(conflict frozen: original 0.062 vs muted 0.000) or when hypothesis-
driven planning is over-trusted vs exploration under noise (noisy
adaptive: original 0.917 vs epoch_reset 0.722).

### Q3 — Do successful decisions depend on previous knowledge
Episode-scoped variants reuse pre-eval hypotheses for 100% of frozen
hypothesis-sourced plans (classic 1695/1695; noisy 547/547; OOD
741/741). Global memory is never reused under frozen eval (0 usable
hypotheses — 0 plans).

### Q4 — Is A/B/A recovery retention or rediscovery
For epoch_reset: A2 frozen success 1.00, with **100% of A2 plans
retained from A1 across the conflicting phase B** (10 verified→
contradicted but also 10 contradicted→verified re-promotions). For
original: 0% survive — recovery is pure rediscovery (A2 frozen 0.625,
0 plans).

### Q5 — When epoch_reset protects / fails
Protects: classic/conflict/OOD frozen (best or tied), noisy frozen
(+0.222 vs original, CI [0.056, 0.389]), retention across conflict.
Fails: adaptive noisy (0.722 vs original 0.917) — under noise,
exploration should be trusted more than retained hypotheses; adaptive
classic slightly below epoch (0.667 vs 0.729).

## Files

- `academy/tools/memory_diagnosis_attribution_v01.py`
- `academy/tests/test_memory_diagnosis_attribution.py`
- `docs/RESEARCH_MEMORY_ATTRIBUTION_V01.md`
- `results/academy_v01/memory_diagnosis_attribution_v01/*`

Not merged — evidence PR; no auto-merge.
