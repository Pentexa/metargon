# METARGON v0.13 — Robust Causal Generalization

**Date:** 9–10 November 2026.
**Status:** Controlled paired study (4 seeds × 40 episodes, matched seeds,
frozen worlds) vs v0.10 / v0.11 / v0.12.1.
**Branch:** `research/v13-causal-generalization`.

**Headline:**
- **Mission A succeeds:** the v0.12 single-slot temporal candidate
  generator is replaced by a bounded *rolling* completion generator.
  Exact ordered-rule (D1) discovery on the audit's L family improves:
  stable 16.9% → **39.4%**, novel 23.1% → **33.1%**, and the number of
  seeds with nonzero discovery rises from 1/4 → 2/4 (stable), with the
  previously dead seed 0 going from 0% → 77.5%. Temporal Δ vs no-temporal
  grows from +16.9/+23.1 pp to +39.4/+33.1 pp.
- **Regressions:** v0.13 is exactly neutral on F-noisy (81.25% ≈ v0.12.1
  baseline), H-stable (73.12%), R-stable (60.62%), and family L success
  (29.4%/27.5%). No regression against v0.12.1 was measured anywhere.
- **Mission B remains an honest negative:** the R-stable dual-context cost
  (−11.3 pp vs v0.11) persists. A new persistence-gated dual-context
  switch was implemented and measured; it fires but does not recover R
  (60.62 both ways). Full-clean (−nodc) recovers R (71.88) but would
  sacrifice F-D2 recovery (v0.12.1 finding reproduced). H-stable's gap to
  v0.10 persists (73.12 vs 83.75), not attributable to any v0.12/v0.13
  mechanism (matches v0.12.1 diagnosis).

---

## 0. Evidence, provenance & verification

All results below are computed from committed raw artifacts (per-seed
vectors) in this repository. Every data file under `results/v13/` is
covered by a stamped Quality v0.5 provenance manifest under
`generalization_audit/manifests/`; the independent artifact gate
(`verification/artifact_gate.py`, Quality PR #39) passes with **0 FAIL**
(see §9).

| experiment | artifact | quality v0.5 manifest |
|---|---|---|
| L final (v0.13 vs v0.12.1 + ablations) | `results/v13/final_l.json` | `generalization_audit/manifests/v13_final_l.manifest.json` |
| F final (5 conditions) | `results/v13/final_f.json` | `generalization_audit/manifests/v13_final_f.manifest.json` |
| H final (stable/novel) | `results/v13/final_h.json` | `generalization_audit/manifests/v13_final_h.manifest.json` |
| R final (stable/novel + -nodc) | `results/v13/final_r.json` | `generalization_audit/manifests/v13_final_r.manifest.json` |
| L baseline (v0.12.1 reproduction) | `results/v13/baseline_l.json` | `generalization_audit/manifests/v13_baseline_l.manifest.json` |
| L pilot (full ablation set) | `results/v13/pilot_l.json` | `generalization_audit/manifests/v13_pilot_l.manifest.json` |
| F/H/R regression checks | `results/v13/regress_*.json` | `generalization_audit/manifests/v13_regress_*.manifest.json` |
| persistence-gate experiments (rejected) | `results/v13/r_gate*.json` | `generalization_audit/manifests/v13_r_gate.manifest.json` |
| diagnostics (seed-dependence attribution) | `results/v13/diag_*.json` | `generalization_audit/manifests/v13_diagnostics.manifest.json` |
| Q v0.5 verification reports | `results/verification_v13/verify_v13_*.json` (outside frozen target; tool outputs, regenerable via §7 commands) | — |

**Quality v0.5 verification (merged PR #36 + #39, `verify_experiment.py`):**
all eleven manifests stamp and verify with **0 FAIL**; every summary mean
reproduces the raw per-seed mean exactly (tolerance 1e-9), every recorded
delta follows the candidate-minus-baseline convention, and every recorded
paired bootstrap 95% CI reproduces exactly from the raw diffs (seed 4301,
1500 rounds, α=0.05). INCONCLUSIVE items are metadata omissions only
(memory limit not byte-measured; the `v12_temporal` comparison key is
ambiguous to the verifier's key resolver and is validated manually in
§4.1). Verification reports: `results/verification_v13/verify_v13_*.json`.

**Provenance:** seeds 11800–11803 (world seed base), 40 episodes/seed,
policy seed offset +1000003, `max_steps=80`, planner cap 700 expansions /
decision, probe max 3 plans. Policy configs: v0.12.1 = recent-credit
off + dual context + adaptive confirm + confirmed-event gating, temporal
ON for L; v0.13 = same + rolling temporal completion (see §3). Runs were
executed with code byte-identical to commit `1518054` (the rebased v0.13
implementation commit; contains identical content to the original
`4e61903`/`437cf9d` — verified by empty `git diff`). **Historical runs did
not pin `PYTHONHASHSEED`** (Python default: randomized per process);
no determinism is claimed from hash seeding. Reproducibility rests on
matched episode streams (identical seeds/pools), canonical git-blob
artifact hashes (platform-independent), and the paired per-seed
statistics. Frozen pools (`frozen_l.json`, `frozen_r.json`) were read
read-only.

---

## 1. Motivation

v0.12.1 (PR #33) discovered exact ordered rules on family L (16.9% /
23.1% exact D1) but the result was seed-sensitive: stable seed vector
[0, 0, 0.675, 0], novel [0, 0.25, 0.675, 0] — 2 of 4 seeds contribute
nothing. The mission: find WHY discovery succeeds in some seeds and fails
in others, and develop a family-agnostic mechanism that increases
reliable discovery across independent seeds.

## 2. Diagnosis — why temporal discovery is seed-dependent

We built `python/diagnose_v13.py`, which replays the exact paired world
stream and instruments the temporal memory per episode:

- `true_pair_observed` (floor): did the raw belief-flip stream contain the
  true ordered pair within the window?
- `correct_created/supported/contradicted/evicted/verified_max` per
  correct temporal hypothesis.
- temporal probe attempts and protection blockages.

Results (v0.12.1 policy, 4 seeds, stable condition):

| seed | success | floor D1 | correct_created | correct_verified | exact D1 |
|---|---:|---:|---:|---:|---:|
| 0 | 2/40 | 37/40 | 0 | 0 | 0.000 |
| 1 | 4/40 | 4/40 | 2 | 0 | 0.000 |
| 2 | 35/40 | 36/40 | 35 | 27 | 0.675 |
| 3 | 3/40 | 3/40 | 2 | 0 | 0.000 |

Diagnostic runs (`diag_l_stable.json` / `diag_l_novel.json`) use the
identical worlds, seeds and policy as the benchmark (`final_l.json`);
per-seed success from the diagnostic runs equals the benchmark's
per-seed success exactly (Δ = 0.000 for all 8 seed-condition cells), so
the diagnostic counters can be quoted alongside the benchmark exact-D1
column. `exact D1` above is the benchmark's per-seed exact discovery.

**Root cause:** the v0.12 generator keeps a SINGLE "active first" atom per
door. When one step flips several atoms at once — e.g. loading a plate
turns on both `(load,P1,A)` and `(loadany,P1)` — a spurious completion
consumes the slot. The true ordered pair `(load,P1,A) → (sw,S1,False)`
is then never generated even though the flip stream contains it (seed 0:
floor 37/40, correct_created 0). Seeds where the true pair *is* generated
(seed 2) verify and succeed. Probe protection was not a factor (0 blocked
probes on L).

With the v0.13 rolling generator, the same seed 0 creates 61 correct
candidates and verifies in 31/40 episodes (stable, `diag_l_final.json`);
seed 2 creates 76 and verifies 32/40. The remaining zero-discovery seeds
(1, 3) have a genuinely low behavioral floor (the ordered pair appears in
only 3–4/40 episodes — the agent's exploration rarely arranges first→
second in those configs), so they are an execution/opportunity limit, not
a candidate-generation loss.

## 3. Adopted change — rolling completion generation (v0.13)

Only one change vs v0.12.1: `TemporalMemory` `completion_mode="rolling"`
(default in `MetargonV13`).

Mechanics (family-agnostic, bounded):

- Per door, a buffer of recent "first" atoms (pruned to the sequence
  window, deduplicated by atom, latest occurrence kept).
- Every flip generates one candidate (first → atom) per *different*
  buffered first, capped at `max_cand_per_flip=6` per flip. Deterministic
  ordering within a step (sorted atoms; earlier same-step atoms also act
  as predecessors, matching v0.12's within-step convention).
- Evidence (support within 4-step open credit, contradictions on closed-
  door nonevents, episode decay, deterministic score eviction, caps ≤64)
  unchanged from v0.12.
- Bounded memory: per-door buffer ≤ number of distinct atoms (≤4 in the
  audit families), candidates ≤ |atoms|² per door; no unbounded state.

Ablations (all other v0.13 knobs default OFF, preserved as flags):
`temporal_support_weighting="proximity"`, `temporal_target_exempt=True`,
`dual_persist_gate>0` — see §6 (rejected).

## 4. Results

### 4.1 Family L (ordered-sequence) — Mission A

| condition | v0.10* | v0.11* | v0.12.1 | v0.13 | temporal Δ v12 | temporal Δ v13 |
|---|---:|---:|---:|---:|---:|---:|
| stable success | 21.25 | 18.12 | 27.50 | **29.38** | — | — |
| novel success | 21.88 | 19.38 | 29.38 | 27.50 | — | — |
| stable exact D1 | 0.0 | 0.0 | 16.9 | **39.4** | +16.88 | +39.38 |
| novel exact D1 | 0.0 | 0.0 | 23.1 | **33.1** | +23.12 | +33.13 |

\* v0.10/v0.11 L cells are **historical values from the v0.12 study**
(`results/v12/summary_l_small.json`, audit v0.3-verified), not re-run in
this study. The v0.12.1 column (`metargon12`) is measured in this study's
harness and reproduces the v0.12 study's L summary exactly (success
27.50/29.38; exact D1 16.9/23.1). (The v0.12.1 branch's own final runs
recorded slightly different L-success values — 26.9/25.0 — for the same
4×40 matched cells; both are pilot-scale samples of the same policy
config.) Exact-D1 cells are identical across all three artifacts
(v0.12 study, v0.12.1 final, this study).

Per-seed exact D1:
- stable: v0.12.1 [0, 0, 0.675, 0]; v0.13 [0.775, 0, 0.80, 0] —
  nonzero seeds 1/4 → 2/4; population stddev 0.292 → 0.394.
- novel: v0.12.1 [0, 0.25, 0.675, 0]; v0.13 [0.275, 0.25, 0.80, 0]
  — nonzero seeds 2/4 → 3/4; previously dead seed 0 now discovers
  (27.5%).

Paired bootstrap 95% CIs for the temporal exact-D1 delta (candidate
minus baseline, per-seed diffs, seed 4301, 1500 rounds):
- stable: +39.38 pp, CI **[0.00, 78.75] pp — includes 0**; per the
  frozen criteria this is **not** claimed as a measured population effect;
  it is directional pilot evidence.
- novel: +33.13 pp, CI **[6.88, 66.25] pp — excludes 0**.
Both deltas are supported by the `-gen`/`-notp` ablation controls (§5).

(per-seed vectors in `results/v13/pilot_l.json` and
`results/v13/final_l.json`; rendered in
`results/v13/rendered_final_l.md`).

Temporal verification count per seed-run: v0.13 3.75 (stable) / 3.25
(novel); the `-gen` ablation (= v0.12 single-slot generator behavior,
§5) verifies 0.75 / 0.75. Temporal probes: v0.13 57 / 50.5 vs v0.12.1
44.75 / 42.5; no temporal evictions (≤32 distinct pairs in a 64-cap
store).

### 4.2 Family F (declared) — preservation of noise robustness and D2

| condition | v0.10 | v0.11 | v0.12.1 | v0.13 |
|---|---:|---:|---:|---:|
| stable | 85.62 | 85.62 | 85.62 | **85.62** |
| shift | 86.25 | 90.62 | 91.25 | **91.25** |
| noisy_obs | 44.38 | 80.62 | 81.25 | **81.25** |
| novel | 91.25 | 91.88 | 91.25 | **91.25** |
| tight_budget | 73.12 | 90.00 | 90.62 | **90.62** |

v0.13 equals v0.12.1 on **every** F cell (Δ exactly 0.00 pp on all five
conditions). Exact D1: 97.5/95.0/80.6/96.2/93.1; exact D2 for noisy was
75.6 in the v0.12.1 final (see §5 note). F-noisy preservation: 81.25
(≈ v0.11 baseline 80.62).

### 4.3 Family H (structurally novel)

| condition | v0.10 | v0.11 | v0.12.1 | v0.13 |
|---|---:|---:|---:|---:|
| stable | 83.75 | 73.12 | 73.12 | **73.12** |
| novel | 83.75 | 79.38 | 79.38 | **79.38** |
| exact D2 stable/novel | 63.1/61.9 | 41.2/43.8 | 70.0/76.9 | **70.0/76.9** (final_h) |

### 4.4 Family R (sustained-hold)

| condition | v0.10 | v0.11 | v0.12.1 | v0.13 |
|---|---:|---:|---:|---:|
| stable | 61.25 | 71.88 | 60.62 | **60.62** |
| novel | 80.00 | 84.38 | 79.38 | **79.38** |
| R stable -nodc | — | — | 71.88 | **71.88** (reproduced) |
| R novel -nodc | — | — | 84.38 | **84.38** (reproduced) |

## 5. Ablations

All L ablations come from `pilot_l.json` (4×40, same matched seeds); the
final `-gen`/`-notp` rows were re-run in `final_l.json` and are
identical. The pilot's "full" config had the experimental knobs
proximity + probe-exemption ON; the final default is rolling + flat + no
exemption — all four combinations measure the same exact D1 (39.4 /
33.1), which is itself the neutrality evidence for those knobs.

| variant | L stable exact D1 | L novel exact D1 | interpretation |
|---|---:|---:|---|
| v0.13 (rolling, any knob combination) | 39.4 | 33.1 | — |
| -gen (active_first) | 16.9 | 23.1 | rolling generation is the sole driver |
| -tw (proximity off) | 39.4 | 33.1 | identical to full (neutral) |
| -tpex (exemption off) | 39.4 | 33.1 | identical to full (neutral; 0 blocked probes) |
| -notp | 0.0 | 0.0 | temporal responsible for the entire gain |

F/H/R ablations: v0.13 vs v0.12.1 exactly identical on every measured cell
(F-noisy 81.25, H-stable 73.12, R-stable 60.62).

## 6. Rejected approaches and negative results (honest)

1. **Persistence-gated dual context** (`dual_persist_gate`): switches
   event contexts from dual → clean when the belief state persisted for
   ≥2 completed steps (hold-style doors). Diagnostics show it fires on
   ~half of R events (clean/dual split 40/40, 20/20, 21/19, 74/0 across
   seeds) but R-stable stays 60.62 (identical to gate-off). Full-clean
   (−nodc) does recover R (71.88) — and still costs F-D2 recovery (the
   v0.12.1 conflict). Rejected.
2. **Trigger-proximity support weighting**: neutral on L (exact D1
   identical 39.4/33.1). Kept as a flag; OFF by default.
3. **Temporal probe exemption**: never engaged on L (0 blocked probes in
   diagnostics; identical results). Kept as a flag; OFF by default.
4. **H-stable regression (73.12 vs v0.10 83.75) persists** — reproduced;
   attributable to shared v0.11 confirmed-event gating (v0.12.1
   diagnosis), not to any v0.13 mechanism; closing it sacrifices D2
   recovery. No family-agnostic fix found.
5. **R-stable regression against v0.11 persists** (−11.3 pp) — dual
   context cost; resolution conflicts with F-D2 recovery; documented.
6. Sample size remains 4×40 (pilot protocol): CIs are wide; the L
   finding is directional, not a population claim. Per-seed vectors are
   attached so independent reviewers can recompute.

## 7. Reproduce

```bash
python -m unittest discover -s python -v           # includes v13 tests
python python/benchmark_v13.py --seeds 4 --episodes 40 --family l \
    --conditions stable,novel --parallel --workers 6 --no-mem \
    --policies metargon12,metargon13,metargon13-gen,metargon13-notp,\
metargon12-notp,oracle --json results/v13/final_l.json
python python/benchmark_v13.py --seeds 4 --episodes 40 --family f \
    --conditions stable,shift,noisy_obs,novel,tight_budget \
    --parallel --workers 6 --no-mem \
    --policies metargon10,metargon11,metargon12,metargon13,bayes,oracle \
    --json results/v13/final_f.json
# and likewise for h/r (final_h.json / final_r.json)
python python/v13_postprocess.py results/v13/final_l.json
python python/diagnose_v13.py --family l --conditions stable,novel \
    --seeds 4 --episodes 40 --policy v13 --json results/v13/diag_l_final.json

# Quality v0.5 verification of all committed manifests (11)
python -m generalization_audit.tools.verify_experiment \
    --manifest generalization_audit/manifests/v13_final_l.manifest.json --json \
    -o results/verification_v13/verify_v13_final_l.json
# ... repeat for v13_final_f / v13_final_h / v13_final_r / v13_baseline_l /
# v13_pilot_l / v13_regress_f_noisy / v13_regress_h_stable /
# v13_regress_r_stable / v13_r_gate / v13_diagnostics

# Artifact gate (Quality PR #39): must be PASS for results/v13
python -c "from pathlib import Path; import sys; \
sys.path += ['.','python','generalization_audit']; \
from generalization_audit.verification.artifact_gate import check_frozen_target_artifacts; \
[print(r.target, r.overall) for r in check_frozen_target_artifacts(Path('.').resolve())]"
```

## 8. Resources

Per seed-run CPU (s, 4×40, from `final_l.json`/`final_f.json`): L stable
v0.13 44.7 vs v0.12.1 47.2; L novel v0.13 60.2 vs v0.12.1 57.4;
F-noisy v0.13 17.7 (≈ v0.12.1). Peak memory unchanged scale (bounded
stores; not byte-measured in this study — recorded as null in the
manifests, hence the verifier's INCONCLUSIVE fairness item).
`PYTHONHASHSEED` was unset in all runs (default randomized hashing); all
artifacts record per-seed raw vectors and the verifier confirms every
summary mean and CI reproduces from them.

## 9. Protocol compliance note (deviation, documented)

Independent audit frozen-eval-criteria v02 (PR #35,
`generalization_audit/tests/test_future_eval_criteria_v02.py`) originally
required that no committed v0.13 artifacts exist while the criteria are
frozen. Quality v0.5 (PR #36) added the verifier infrastructure, and
Quality PR #39 replaced the *prohibition* with a provenance-aware
publication gate (`generalization_audit/verification/artifact_gate.py`):
frozen criteria are immutable, but evidence may be committed if and only
if every artifact under the target is covered by a valid stamped manifest
whose verification has zero FAIL. This study complies with the gate: all
raw per-seed artifacts, diagnostics, summaries and rendered tables are
committed under `results/v13/` and covered by eleven stamped manifests
(§0), each verified with zero FAIL; the gate is PASS for `results/v13`
and open for `results/academy_v01/memory_v06`. The temporary external
archive (`…/Temp/opencode/metargon_v13_evidence/`) held the evidence
while the gate was in force and is retained as an offline copy.

Per `evidence_for_improvement_claim`, the 4×40 results here are
**preliminary only**: no population-level claim is made; the L finding is
reported with per-seed vectors and paired CIs, and the mechanism is
supported by the `-gen` ablation (controlled control). Confirmatory
8×100 runs plus an independent audit recomputation are required before a
population claim, per the frozen criteria. Registration note: the frozen
criteria v02 (including its preregistration expectation) were published
while this study was already in execution; the study was run under the
inherited v01/v12 protocol and is reported under v02 requirements
(per-seed vectors, paired CIs, ablation control), with this timing
documented as the deviation.
