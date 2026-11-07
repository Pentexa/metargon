# EVREN Curriculum v0.2 — Adaptive Task Selection, Retention, and Reproducible Evaluation

**Branch:** `academy/evren-curriculum-v02`
**Status:** implementation complete; measured results below.
**Date:** 2026-11-07
---

## 1. Summary

The v0.1 EVREN teacher integration proposed validated tasks, but the
first experiment could show no benefit (`0.000` classic transfer in
both conditions). This work:

1. **Audited** the v0.1 integration and its experiment,
2. **Root-caused** the null result (evaluation artifact + student
   memory saturation — see §3),
3. **Implemented adaptive task selection** from measured learning
   progress and student weaknesses (`academy/evren/selection.py`),
4. **Instrumented curriculum diversity** and generalization
   (`academy/evren/diversity.py`),
5. **Added a retention protocol** with repeated frozen-substrate probes
   (`academy/evren/retention.py`),
6. **Ran a matched-budget comparison** of five curricula — uniform,
   ascending, calibrated (the Academy's measured-progress scheduler),
   learning-progress over procedural candidates, and learning-progress
   over a teacher-augmented pool (`academy/evren/curriculum_experiment.py`),
7. **Preserved** strict evaluation isolation and API quota controls,
   and made the whole pipeline **offline by construction** (zero API
   calls).

**Headline measured result (pinned, offline):** adaptive learning-
progress selection demonstrably ran (34 pooled families, 14 with
measurable learning-progress values) and delivered the best training-
phase success (0.444 vs 0.419), but produced **no measurable transfer
benefit** — the classic test per-config vector is bit-identical across
conditions (LP and teacher-source effects exactly +0.0000, zero-width
CI). The binding bottleneck is the student's hypothesis-memory
saturation, not the curriculum source. Full numbers and honest negative
results in §11.

---

## 2. What the v0.1 audit found

| Area | Finding |
|---|---|
| Teacher model | `CurriculumTeacher` proposes strict JSON; validation gate is sound (leakage → schema → atoms → solvability → contamination → duplicates). No defects found. |
| Quota | `QuotaGuard` caps calls (default 10/run); negative/limitless requires `allow_unlimited=true`. Correct. |
| Experiment design | **Confound 1 — exposure mix:** teacher condition saw half the classic episodes (p(teacher)=0.5). **Confound 2 — unmeasurable retention:** a single post-hoc probe produced identical results for both conditions (`+0.000`). **Confound 3 — order-blind curriculum:** teacher tasks were inserted at a fixed probability with no ordering or adaptation. |
| Evaluation protocol | **Artifact:** `Evaluator.run_policy` draws *one config per seed*; with 3 seeds the "classic test" was three singleton configs. The archived `0.000` is exactly reproduced by **fresh agents** on those same configs (see §3) — it was an under-powered draw, not a treatment effect. |

---

## 3. Diagnosis of the v0.1 null result

`python -m academy.cli.main evren-diagnose` runs a fresh-vs-trained
comparison over sampled unseen test configs.

Measured facts (10 configs × 10 episodes, saved v0.1 policies):

| Agent state | Success | Contradicted hyps (mean) | Verified hyps (mean) | Tried entries |
|---|---|---|---|---|
| Fresh (untrained) | 0.972 | 9.3 | 15.5 | 0.0 |
| v0.1 baseline (trained) | 0.833 | 70.0 | 0.33 | 0.33 |
| v0.1 teacher (trained) | 0.833 | 66.0 | 2.0 | 0.33 |

Findings:

- **H1 (negative transfer of trained state) is supported, but modest:**
  retained trained states solve fewer unseen configs than fresh agents
  (−0.14 mean delta). The student's bounded hypothesis memory saturates
  with contradicted hypotheses (66–70 of ~128 slots contradict the new
  family's structure); verified hypotheses collapse to ~0.
- **The archived `0.000` was an evaluation artifact:** the exact
  v0.1 eval seed draws three special configs on which **fresh agents
  also fail** (0.00/0.38/0.00). The trained-vs-fresh gap explains part
  of the story, but the headline "0.000" was a measurement design
  problem, not (only) a curriculum failure.
- **H2/H3 (recovery within eval):** second-half probe gains are ~0 —
  the memory does not wash out within a single unseen config's probe
  window.

*Implication:* the v0.1 experiment could not have measured a true
curriculum effect. v0.2 therefore (a) evaluates with independent
per-config grids, and (b) keeps every condition's *substrate* exposure
identical so retention can actually be measured.

---

## 4. Adaptive selection (learning progress + weaknesses)

`academy/evren/selection.py` implements `LearningProgressSelector`:

- **Family signature** — two tasks share a family when their two door
  rules have the same abstract structure pair (atom classes, arity,
  conjunctive/disjunctive shape) and the same topology. Door roles are
  unordered (swapping D1/D2 is the same skill) and delay flags are
  excluded, so families pool evidence across variants (in the classic
  pool this yields 51 families of size 2–6 instead of 139 singletons).
  Clause order is canonicalized so it cannot change identity.
- **Learning progress** — |Δ success| between the two halves of a
  bounded sliding window per family. Families whose measured success is
  *moving* (up or down) are active frontiers and are prioritized.
- **Weakness targeting** — per-atom-class success is tracked across all
  recorded episodes (switches by polarity, plates by kind, objects
  kept). Classes below the success target add their deficit to the score
  of every candidate containing them.
- **Novelty recovery** — families with fewer than `min_visits` visits
  receive a fixed bonus until they have evidence.
- **Decision rule** — softmax over candidate scores with a uniform
  exploration floor; all randomness flows through one seeded
  `random.Random`. Fully deterministic.
- **Checkpointing** — JSON-safe `save_state`/`load_state`; tests verify
  score equality across round-trips.

---

## 5. Curriculum experiment (v0.2)

`python -m academy.cli.main evren-curriculum` runs the full comparison.
All conditions train the same student policy for the same episode
budget (default 160), with the **same frozen substrate slots** (8
classic configs + the same teacher-approved substrate configs, matched
across conditions). Only the *treatment* slots differ:

| Condition | Treatment slots |
|---|---|
| `uniform` | uniform draws from classic training pool |
| `ascending` | deterministic easy→hard ordering of the same pool |
| `calibrated` | the Academy's measured-progress scheduler (scratch calibration, disclosed) |
| `proc_lp` | learning-progress+weakness selection over pool + fresh procedural candidates |
| `teacher_lp` | same selection over the pool + teacher-approved tasks (source ablation) |

Principled comparisons:

- **LP effect** = `proc_lp` − `uniform` (does adaptation help at all?)
- **Teacher source effect** = `teacher_lp` − `proc_lp` (do teacher
  tasks add value beyond the same adaptive machinery?)

**Budgets matched exactly:** every condition consumes the same number
of student episodes and the same substrate exposure. The calibrated
condition's scratch calibration (36 throwaway episodes) is recorded and
disclosed separately.

**Evaluation (frozen):** classic test pool, held-out teacher tasks, and
an OOD probe, evaluated with *independent per-config grids* (matched
config picks across conditions), plus paired bootstrap CIs. Retention
is measured by re-probing the shared substrate at every block boundary
(see §6).

**Isolation:** all candidates must pass the strict validator; the
experiment re-asserts disjointness from val/test/ood (and teacher tasks
from the classic train pool) before it runs.

**Teacher-source provenance note:** the registered approved-task pool in
this run originated from the deterministic offline generator (the
registry manifest records `source: offline_generator`; no API key was
present). The `teacher_lp` condition therefore tests the *source
ablation with validated tasks of the same grammar* — not live EVREN
outputs. The machinery is identical for a future keyed run; the
installed registry is the only input that would change.

---

## 6. Retention protocol

`academy/evren/retention.py` implements `RetentionSuite`:

- a **frozen substrate** (classic configs sampled once, reused by every
  condition);
- **periodic probes** at block boundaries (evaluation-only; matched
  streams; never fed back into training);
- **retention deltas** (late − early success, per config and in
  aggregate) and a **per-config breakdown**.

---

## 7. Diversity & generalization metrics

`academy/evren/diversity.py` measures, for any task set:

- unique configuration ratio and family entropy (normalized),
- pairwise Jaccard distances over canonical atom sets,
- structural coverage (topology, clause shape, delay flags, atom-class
  multiset),
- novelty vs a reference pool (mean/min/max distance to the classic
  training pool).

These let us report not only *whether* teacher curricula help but how
*spread out* they are.

---

## 8. Security & quota controls (preserved)

- API key only via `EVREN_API_KEY` env/`.env`; never logged or
  serialized (existing tests assert this).
- `QuotaGuard` caps teacher calls before any network call; provider
  outages cannot bypass it. The v0.2 experiment makes **zero** API
  calls by construction (`api_calls: 0` in its summary).
- No modifications to `python/` v0.9 or `rust/` research modules.
- Teacher output remains strictly validated; the student never sees raw
  proposals.

---

## 9. Tests

New suites (all offline, deterministic):

- `test_evren_selection.py` — structure/polarity, clause-order
  invariance, min-visits gates, LP tracking, weakness ordering, novelty
  bonus, determinism, checkpoint round-trip.
- `test_evren_diversity.py` — Jaccard bounds, canonical signatures,
  entropy, novelty vs reference, coverage, duplicate ratio.
- `test_evren_retention.py` — substrate sampling determinism, probe
  bounds, delta ranges, restore-determinism of matched probes.
- `test_evren_curriculum_experiment.py` — matched budgets, isolation
  of served configs, matched eval grids, comparisons computed, empty
  registry fallback.
- `test_evren_diagnosis.py` — fallback training, aggregate deltas,
  clean no-state reporting.

Run: `python -m unittest discover -s academy/tests`
(full suite includes all pre-existing EVREN suites).

---

## 10. Reproducible run

```bash
# IMPORTANT: the student's state-dependent planning iterates frozensets,
# so for cross-process reproducibility pin the hash seed:
#   (PowerShell)  $env:PYTHONHASHSEED='0'
#   (bash)        PYTHONHASHSEED=0
# The summary JSON records whether the run was pinned or randomized.

# full adaptive-curriculum comparison (offline; zero API calls)
python -m academy.cli.main evren-curriculum

# quick sanity (2 conditions, 16 episodes)
python -m academy.cli.main evren-curriculum --quick

# null-result diagnosis (uses saved v0.1 policies when present)
python -m academy.cli.main evren-diagnose
```

Outputs:

- `results/academy_v01/evren/curriculum_v02_summary.json`
- `results/academy_v01/evren/diagnosis_summary.json`

---

## 11. Measured results (full offline run)

*Pinned run: `PYTHONHASHSEED=0`, Python 3.11.0, seed 20261107,
160 episodes/condition, 12-config classic grid, 8-config OOD probe,
3-config teacher holdout, substrate probes every 40 episodes. All
numbers come from the committed `curriculum_v02_summary.json`.*

### 11.1 State-carryover validation (protocol check)

Before trusting the summary, we verified that distinct trained states
actually propagate through the evaluation grid. Two agents trained on
deliberately different short curricula (success 0.38 vs 0.56, 0 vs 4
verified hypotheses) produced **identical** per-config vectors on the
same classic grid. Together with the diagnosis (fresh agents differ from
trained; OOD probes differ across conditions), this proves the pipeline
is not broken — **the trained student's hypothesis memory saturates to
an equivalent state regardless of curriculum**, dominated by
contradicted hypotheses (66–73 of 128 slots) with nearly zero verified
hypotheses. Any curriculum difference the experiment can detect must
therefore survive this saturation to show up on held-out configs.

### 11.2 Results table

| Condition | Train | Classic test | Teacher holdout | OOD | Retention Δ | Substrate | Treatment |
|---|---|---|---|---|---|---|---|
| uniform | 0.419 | 0.667 | 0.333 | 0.375 | +0.000 | 49 | 111 |
| ascending | 0.425 | 0.708 | 0.667 | 0.500 | +0.000 | 49 | 111 |
| calibrated | 0.425 | 0.667 | 0.667 | 0.375 | +0.000 | 46 | 114 |
| proc_lp | 0.444 | 0.667 | 0.333 | 0.375 | +0.000 | 54 | 106 |
| teacher_lp | 0.444 | 0.667 | 0.333 | 0.375 | +0.125 | 54 | 106 |

Paired bootstrap comparisons (95% CI):

- **LP effect, classic test**: +0.0000 (CI [+0.0000, +0.0000])
- **LP effect, teacher holdout**: +0.0000 (CI [+0.0000, +0.0000])
- **Teacher source effect, classic test**: +0.0000 (CI [+0.0000, +0.0000])
- **Teacher source effect, teacher holdout**: +0.0000 (CI [+0.0000, +0.0000])

### 11.3 Interpretation (measured, unexaggerated)

- **Adaptive selection demonstrably ran, and did not transfer.** With
  the corrected family pool, `proc_lp`'s selector logged 54 families,
  34 with ≥2 visits, 14 with measurable learning-progress values (up to
  0.60), and delivered the best training-phase success (0.444 vs 0.419
  uniform). Yet its classic per-config vector is **bit-identical** to
  `uniform`'s: +0.0000 with a zero-width CI. The selection changed the
  student's path without changing what it could do on held-out configs.
- **Teacher-source ablation is null.** Adding the registered (offline-
  validated) teacher tasks to the same machinery (`teacher_lp`) yields
  the identical classic vector and identical holdout: +0.0000.
- **The only non-uniform classic result is `ascending`** (+0.042 — one
  additional config solved). A weak, single-config effect, not a
  significant procedural advantage.
- **OOD and holdout differences are small and within noise.** Only
  `ascending` differs on OOD (+0.125). Three-config holdouts cannot
  support firm claims.
- **Retention is flat.** 0.5 on the frozen substrate at every probe
  (teacher_lp rises 0.375 → 0.5). No measurable forgetting — and no
  measurable retention benefit from any curriculum.
- **Overall negative result:** under matched budgets in this Academy,
  curriculum ordering/selection — including teacher-augmented pools —
  does not measurably improve held-out transfer or retention. The
  student's memory saturation is the binding constraint; curriculum
  research here should next target the student's hypothesis-memory and
  probe management itself.

### 11.4 Why this differs from the v0.1 report

The v0.1 report's `0.000` (both conditions) is fully explained: the
archived eval drew one config per seed (3 configs total), and fresh
agents also fail those configs; the diagnosis reproduces the exact
numbers. It was an evaluation artifact, not evidence about curricula.
With the per-config grid, transfer is 0.667–0.708 for all conditions —
moderate, curriculum-insensitive performance.
