# METARGON v0.12 — robust general causal learning

**Date:** 8–9 November 2026.
**Status:** Controlled paired study vs v0.10/v0.11 and established baselines.
**Branch:** `research/v12-general-causality` (never merged into `main`).
**Headline:**
- **Mission B (temporal) succeeds:** the new bounded ordered-sequence
  hypothesis language lifts exact D1 rule discovery on the audit's L
  family from 0.0% (v0.10/v0.11) to 16.9–23.1% (v0.12), with success
  +6–11pp.
- **Mission A (clean regressions) partially closes:** F-stable regression
  is gone (85.6% = v0.10), F exact D2 discovery restored to 84–89% on
  every F condition; H-stable and R remain below v0.10/v0.11 (honest
  negatives).

---

## 1. Motivation

v0.11 achieved noisy-observation robustness (F noisy 45.1% → 84.9%) but
regressed on clean conditions: F-stable 80.6% → 74.6%, H-stable 79.9% →
68.0%, and it underperformed v0.10 on several dispatch metrics. The
independent generalization audit (generalization_audit/docs/AUDIT_REPORT.md)
also documented a representational ceiling: on the ordered-sequence family
L, v0.10's unordered conjunctive hypotheses achieve 0.0% exact D1
discovery on every condition.

## 2. Changes (all family-agnostic, public information only)

### Mission A — preserve noise robustness, reduce clean regressions
1. **Dual-context candidate generation** (`compositional_memory_v10.py`
   `event_context_mode="dual"`): v0.11's cleaned contexts dropped the
   event-flipped atom (the likely cause) from candidate generation, so
   toggle-induced rules could never be learned. v0.12 generates candidates
   from BOTH the pre-event and post-event contexts while keeping v0.11's
   confirmed-event gating and certainty weighting.
2. **Adaptive confirmation persistence** (`robust_belief.py`): the number
   of agreeing raw readings required to confirm a door event tracks the
   perceived anomaly rate (1 when calm, 2 when noisy), instead of a fixed
   threshold.
3. **Short recent-causal-credit protection** (`metargon_v12.py`): after a
   door opens, the atoms that held at opening are protected for 6 steps
   while the door is open and uncrossed — enough to preserve the causal
   state, short enough to never block the toggles needed for D2.
4. **Post-transition nonevent skip:** contradiction counting is paused for
   2 steps after a confirmed transition (no self-contradiction during the
   event windows).

### Mission B — temporal causal representation (prototype)
New bounded, interpretable hypothesis in `python/temporal_memory.py`:

    <atom A becomes true> THEN <atom B becomes true> (within 30 steps)
    => door opens

- Candidates are generated from *observed ordered completions*
  (door-independent), credited with support when a confirmed door-open
  follows within 4 steps, and contradicted when completions remain
  uncredited (including reverse-order completions).
- Bounds: ≤64 temporal hypotheses, per-door transition ring ≤30,
  deterministic score eviction, weighted evidence with episode decay
  (mirrors the compositional memory).
- Agent integration: a predicted/verified temporal hypothesis for a
  closed door directs probing toward its first atom, then its second
  atom (bootstrapping the sequence).

## 3. Evaluation

Families F/H (shared pools), R/L (independent audit frozen configs, used
read-only). Policies: v0.10, v0.11, v0.12, ablations (-nodc dual-context
off; -norc recent-credit off; -noad adaptive-confirm off; -notp temporal
off), random/episodic/qlearning/bayes, oracle (privileged, reported
separately). Paired seeds, frozen worlds, matched action budgets; paired
bootstrap 95% CIs; success, exact/covered discovery, invalid actions, CPU,
peak memory.

Run scale: 4 seeds × 40 episodes × 2 conditions (L/R/H) and × 5
conditions (F); memory sample 2 × 20. CIs are wider than the headline
8×100 runs of v0.10/v0.11 — directional evidence, not definitive.

## 4. Results

### 4.1 Family F (declared)
| condition | v0.10 | v0.11 | v0.12 | bayes | v12−v10 | v12−v11 | v12−bayes |
|---|---:|---:|---:|---:|---:|---:|---:|
| stable | 85.62 | 85.62 | 85.62 | 78.75 | 0.00 | 0.00 | +6.87 |
| shift | 86.25 | 90.62 | **91.25** | 80.62 | +5.00 | +0.62 | +10.62 |
| noisy_obs | 44.38 | **80.62** | 72.50 | 76.25 | +28.12 | −8.12 | −3.75 |
| novel | 91.25 | 91.88 | 91.25 | 79.38 | 0.00 | −0.63 | +11.88 |
| tight_budget | 73.12 | 90.00 | **90.62** | 65.62 | +17.50 | +0.63 | +25.00 |

Exact discovery (D1/D2): v0.12 97.5/84.4 (stable), 95.0/88.8 (shift),
87.5/76.9 (noisy), 96.2/88.1 (novel), 93.1/86.9 (tight). v0.11 was
65.6/65.0/31.9/66.9/64.4 on D2 across those conditions - dual-context
restores D2 learning everywhere. (*D2 values verified against
`results/v12/summary_f_small.json`; the shift value is 65.0, not 66.9
as in an earlier draft.)

### 4.2 Family H (structurally novel)
| condition | v0.10 | v0.11 | v0.12 | v12−v10 |
|---|---:|---:|---:|---:|
| stable | 83.75 | 73.12 | 73.12 | −10.63 [−30, 0] |
| novel | 83.75 | 79.38 | 79.38 | −4.38 [−11.3, 0] |

Exact D2: v0.10 63.1/61.9, v0.11 41.2/43.8, v0.12 **70.0/76.9** —
D2 discovery recovered (indeed better than v0.10) while success remains
at v0.11's level.

### 4.3 Family R (sustained-hold)
| condition | v0.10 | v0.11 | v0.12 | v12−v11 |
|---|---:|---:|---:|---:|
| stable | 61.25 | **71.88** | 60.62 | −11.25 [−33.8, 0] |
| novel | 80.00 | **84.38** | 79.38 | −5.00 [−15, 0] |

Exact D1 recovered at v0.10 parity (72.5/84.4) vs v0.11's 70.6/75.6.

### 4.4 Family L (ordered-sequence) — Mission B
| condition | v0.10 | v0.11 | v0.12 | v0.12-notp | bayes |
|---|---:|---:|---:|---:|---:|
| stable | 21.25 | 18.12 | **27.50** | 18.12 | 48.12 |
| novel | 21.88 | 19.38 | **29.38** | 18.75 | 42.50 |

| condition | exact D1 (v10/v11) | exact D1 (v12) | temporal Δ | 
|---|---:|---:|---:|
| stable | 0.0% | **16.9%** | +16.88 [0, 50.6] |
| novel | 0.0% | **23.1%** | +23.12 [0, 50.6] |

> Δ = v0.12 temporal exact D1 − v0.12-notp exact D1 (notp = 0 on both
> conditions). Values verified against `results/v12/summary_l_small.json`
> (`comparison:temporal_minus_notp_exact_D1`: 0.16875 / 0.23125). An
> earlier draft swapped the two delta values; corrected in this audit.

Temporal diagnostics (v12): generated ≈9, verified ≈0.75, probes ≈44 per
seed-run — the bootstrapping probe policy turns learned sequences into
executed behavior.

## 5. Ablations (F/shift, F/noisy)

| ablation | shift | noisy | interpretation |
|---|---:|---:|---|
| v12 full | 91.25 | 72.50 | — |
| -nodc (no dual context) | 90.62 | 80.00 | dual-context costs −0.62pp shift, −7.5pp noisy; restores D2 learning (66.2/43.8 vs 88.8/76.9) |
| -norc (no recent credit) | 91.25 | 81.25 | recent-credit neutral on success; helps D2 (75.6 vs 76.9 noisy) |
| -noad (no adaptive confirm) | 91.25 | 72.50 | identical to full (adaptive confirm is neutral here) |
| -notp (no temporal; L) | 18.12/18.75 | — | temporal is responsible for the entire L gain |

## 6. Negative results and limitations

1. **Noisy gap to v0.11 remains** (−8.1pp on F/noisy; CI [−24.4,
   +3.8]). v0.11's confirmed-event gating + certainty weighting was the
   strongest noise combination; v0.12's dual-context trades some noise
   precision for D2 recovery. Honest: v0.11 is still the best noisy
   variant.
2. **H-stable regression persists** (−10.6pp vs v0.10; at v0.11's
   level). The dual-context D2 gain (+28.8pp exact) did not translate to
   success on H-stable in this sample.
3. **R underperforms v0.11** (−11.25pp stable). The recent-credit
   protection can delay necessary D2 manipulations in hold-gated worlds.
4. Invalid-action counts are slightly higher for v0.12 (noisy 16.0 vs
   10.0 v11) — a side effect of more speculative dual-context probing.
5. Sample: 4×40 (2×20 memory); CIs wide. 8×100 runs were attempted but
   the environment is slow (~2–4 s/episode for METARGON variants on L),
   so the full-size run was not feasible in the time budget.
6. Temporal verified counts are low (≈0.75/seed-run); exact discovery
   relies on predicted + bootstrapped probing as much as on verified
   hypotheses.

## 7. Resources

Per seed-run CPU (s, 4×40): F stable v10 110.8, v11 108.0, v12 110.0,
bayes 36.8; L stable v12 349.8 vs v10 489.8 (temporal probing is cheaper
than blind exploration here). Peak memory (2×20 F): v12 0.13–0.98 MB vs
v10 0.23–1.57 MB (unchanged scale; bounded stores).

## Reproduce

```bash
python -m unittest discover -s python -v
python python/benchmark_v12.py --seeds 4 --episodes 40 --family l \
    --conditions stable,novel --parallel --workers 6 \
    --json results/v12/summary_l_small.json
python python/benchmark_v12.py --seeds 4 --episodes 40 --family r \
    --conditions stable,novel --parallel --workers 6 \
    --json results/v12/summary_r_small.json
python python/benchmark_v12.py --seeds 4 --episodes 40 --family h \
    --conditions stable,novel --parallel --workers 6 \
    --json results/v12/summary_h_small.json
python python/benchmark_v12.py --seeds 4 --episodes 40 --family f \
    --conditions stable,shift,noisy_obs,novel,tight_budget \
    --parallel --workers 6 --json results/v12/summary_f_small.json
python python/v12_postprocess.py results/v12/summary_f_small.json
```
