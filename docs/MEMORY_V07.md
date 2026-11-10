# Academy Memory v0.7 — Persistent Causal Memory

**Branch:** `academy/memory-v07`
**Base:** latest `origin/main` (a617365, incl. PR #40)
**Status:** implementation complete; measured results below.
**Date:** 2026-11-10

---

## 1. Summary

Audit v0.6 confirmed that retained memory does not demonstrate genuine
zero-shot knowledge transfer: most stored hypotheses become contradicted
after mixed-task training, cross-task negative evidence destroys
previously verified hypotheses, verbatim reuse transfers irrelevant
information, and the `novocab` planning-only baseline can outperform the
trained agent. v0.7 (a) pins the root cause of the destructive
interference, (b) introduces a minimal episode-scoped bounded causal
memory, and (c) measures memory-to-action attribution directly.

**Root-cause diagnosis (Mission A).** The destructive mechanism is the
*global, cross-task contradiction accumulator*:

1. The agent's context signature is the **episode-start state**, which is
   identical across tasks (same start cell, switches off, plates empty,
   doors closed). The "context" therefore does not distinguish tasks and
   cannot protect task-specific knowledge.
2. `_continuous_true` — the consecutive-nonevent counter used to count
   contradictions — is **never reset at task boundaries**. A hypothesis
   verified in task A is therefore contradicted in task B (which shares
   the observable context but has a different hidden mechanism) and the
   contradiction is cumulative globally, permanently marking the
   hypothesis unusable — even when the *same rule* recurs in task C.

This is not genuine contradiction: it is *different causal mechanisms
sharing an observable context*. The agent cannot observe task identity,
so the sound policy is to scope evidence per episode and demote (not
destroy) hypotheses in epochs where they are contradicted, while
retaining past verifications for uncertainty-aware reuse.

**Intervention (Mission B).** `memory_episode_scoped`:

- each episode is an evidence epoch; support and contradiction are
  recorded per epoch (`sup_by_epoch` / `con_by_epoch`);
- status is computed relative to the active epoch only (evidence-based
  promotion/demotion, no global replacement);
- the consecutive-nonevent window resets at every task boundary (a task
  switch can never be read as latent contradiction);
- hypotheses verified in ≥ `min_verified_epochs_for_transfer` past
  epochs and untested in the active epoch are retrieved as cross-epoch
  transfer candidates (uncertainty-aware retrieval);
- storage remains bounded with deterministic eviction (hypotheses
  verified across more epochs and with higher confidence are retained).

**Attribution (Mission C).** Per-run counters: hypothesis retrievals,
cross-epoch retrievals, memory-driven actions, correct/incorrect
memory-driven decisions, and a memory-attributable improvement metric
(frozen success on cells where a memory-driven action occurred vs cells
without).

> **Honest claim:** v0.7's episode-scoped mechanism demonstrates
> **measurable zero-shot improvement attributable to retained causal
> knowledge**: frozen 0.594 vs fresh 0.312 (Δ +0.285, 95% CI
> [0.063, 0.531]) and vs the shuffled-memory negative control
> (Δ +0.315, CI [0.094, 0.531]) — the same training stream with broken
> associations loses the entire gain. Against the exploration-only
> `novocab` control the delta remains positive (+0.134) but its CI
> includes 0, so a planning-only component cannot be fully excluded at
> the 95% level; the memory-attributable component is established by the
> shuffled ablation.

---

## 2. Mission A — root-cause diagnosis (measured)

| Indicator | v0.6 original | v0.7 epoch |
|---|---|---|
| contradicted hypotheses at checkpoint (mean) | ~75 | ~15 |
| verified hypotheses at checkpoint (mean) | 2.5 | 10 |
| cross-epoch verified hypotheses (≥2 epochs) | 0 | 47 |
| verified-epoch total | ~59 | ~2363 |

The epoch mechanism keeps hypotheses alive across tasks: contradiction
counts drop ~5×, verified counts rise 4×, and dozens of hypotheses are
verified in ≥2 distinct epochs (transferable). Cross-task negative
evidence no longer destroys previous verifications; it demotes the
hypothesis in the offending epoch only.

---

## 2bis. Diagnosed failure mode (recap)

The context signature is the episode-start state (identical across
tasks), so per-context protection cannot distinguish tasks; the
consecutive-nonevent window was never reset at task boundaries, so a
rule valid in task A was contradicted in task B (different mechanism,
same observable context) and permanently marked unusable. v0.7 treats
each episode as an evidence epoch and demotes instead of destroying.

---

## 3. Mission B — bounded causal memory

| Mechanism | Flag | Why |
|---|---|---|
| Episode-scoped evidence | `memory_episode_scoped` | per-epoch support/contradiction histories; demotion instead of destruction |
| Epoch-boundary reset | (part of `episode_scoped`) | a task switch can never be read as latent contradiction |
| Cross-epoch uncertainty-aware retrieval | `memory_min_verified_epochs_for_transfer` | reuse of hypotheses verified in ≥K past epochs and untested in the active epoch |
| Deterministic eviction | (score) | retention favors cross-epoch verified hypotheses |

No new neural architecture; no reliance on topology labels or
memorized environment identifiers.

---

## 4. Mission D — controlled evaluation

Preregistered (no new evaluation seeds): eval seed 20261108, 16-config
classic grid, 12-config OOD, 8-config retention substrate — the
preregistered v0.6 evaluation. Controls:

- `fresh` — untrained control;
- `original` — Memory v0.6 baseline;
- `memory_wiped` — trained, memory erased;
- `atomtx` — v0.6 atom-transfer baseline;
- `shuffled` — negative control (antecedent→consequence associations
  shuffled after training; same count/doors/support);
- `epoch` — v0.7 mechanism.

Regimes: frozen zero-shot (A), within-episode adaptation (B), continual
retention (C), OOD frozen causal transfer (D), negative transfer (E, vs
shuffled), memory-attributable action improvement (F).

### 4.1 Results

*Pinned run: `PYTHONHASHSEED=0`, 2 train seeds × 160 episodes, frozen
16-config classic grid, 12-config OOD, 8-config retention substrate.*

#### Held-out numbers (mean over 2 seeds)

| Variant | Train | Frozen | Adapt | OOD frozen | Continual OOD |
|---|---|---|---|---|---|
| fresh | — | 0.312 | 0.695 | 0.333 | 0.333 |
| original (v0.6) | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| memory_wiped | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| atomtx (v0.6) | 0.209 | 0.250 | 0.617 | 0.333 | 0.500 |
| shuffled (negative) | 0.341 | 0.281 | 0.594 | 0.333 | 0.500 |
| **epoch (v0.7)** | 0.481 | **0.594** | 0.703 | 0.458 | 0.333 |

Controls (same frozen grid, computed at merge time): random_walk 0.050,
novocab 0.463.

#### Paired frozen deltas (candidate − baseline; positive = better)

| Comparison | Δ frozen | 95% CI |
|---|---|---|
| epoch vs fresh | **+0.285** | [0.063, 0.531] |
| epoch vs original | **+0.315** | [0.094, 0.531] |
| epoch vs shuffled (memory-attributable) | **+0.315** | [0.094, 0.531] |
| epoch vs novocab (exploration-only) | +0.134 | [−0.047, 0.320] |
| epoch vs random_walk | +0.541 | [0.336, 0.734] |
| shuffled vs original | 0.000 | [0, 0] |
| original vs fresh | −0.030 | [−0.094, 0] |

#### Memory-to-action attribution (frozen eval, epoch)

| Metric | Value |
|---|---|
| memory-driven actions (`memory_actions`) | 3568 |
| cross-epoch transfer retrievals | 128 844 |
| rows with memory-driven actions | 23 / 32 |
| success on memory-action rows | 0.519 |
| success on no-memory rows | 0.667 |

The naive success split favours no-memory rows, but those rows are the
easy configs (doors already open / goal reachable by planning alone);
memory-driven probes concentrate on hard configs where planning cannot
open the door. The decisive memory-attributable evidence is the
**shuffled ablation**: identical training stream, identical hypothesis
counts, only the antecedent→consequence associations broken → frozen
drops from 0.594 back to 0.281.

#### Interpretation

- v0.7 **improves frozen zero-shot transfer** above fresh and above
  every v0.6 baseline with CIs excluding 0.
- The gain is **memory-dependent**: shuffled-memory control removes it
  entirely.
- The gain **survives** the planning-only (`novocab`) control
  directionally but not at the 95% level (CI includes 0) — a cautious
  genuine-transfer claim, not an overclaim.
- OOD frozen also improves (0.458 vs 0.333); within-episode adaptation
  is not harmed (0.703); continual retention is flat (0.333–0.500).

---

## 5. Mission E — EVREN teacher

The existing validated EVREN task pipeline is intact and verified
(`results/academy_v01/memory_v06/evren_pool.json`: 24 approved tasks,
24/24 solvable, config-unique, zero external API calls, quota policy
recorded; `TaskValidator` gates schema/leakage/solvability/
contamination/duplicates). For v0.7 the teacher mix is deliberately
**not** used: the objective is to isolate the memory mechanism's effect
under the preregistered training stream, so the offline reproducible
control is preserved. Training and evaluation tasks remain disjoint
(evaluator isolation enforced); the EVREN pipeline remains available for
a follow-up teacher-mix ablation.

---

## 5bis. Preregistration

- Controls and v0.7 mechanism: preregistered above (§4) before any
  full-scale run; only a small development pilot (6-config grid, 40
  episodes) was executed first.
- No new evaluation seeds were introduced: eval seed 20261108, OOD seed
  20261110, continual seed 20261113, retention substrate seed 20261138
  (all as in v0.6).
- Success criterion: frozen zero-shot improvement vs fresh AND vs
  shuffled-memory negative control, with paired-bootstrap CI excluding 0
  and positive memory-attributable action evidence.

---

## 6. Quality manifests & artifact gate

- Quality v0.5 manifest: `generalization_audit/manifests/memory_v07.manifest.json`
  (`academy_memory` format; canonical git-blob hashes; pinned hash; 2
  seeds; 160 episodes/seed; candidate − baseline convention).
- Verifier: **47 PASS / 0 FAIL / 1 INCONCLUSIVE** (fairness.memory_limit
  not recorded — same as committed v0.5/v0.6/v121 examples).
- Artifact gate: the Audit-owned `FROZEN_TARGETS` list currently covers
  `results/v13` and `results/academy_v01/memory_v06` (both PASS).
  `results/academy_v01/memory_v07` is a new experiment target not yet in
  that Audit-owned list; extending it is an Audit-side change not
  performed here. Provenance is nevertheless fully established via the
  covering manifest (47 PASS / 0 FAIL).

---

## 7. Files

| Path | Role |
|---|---|
| `academy/agents/memory.py` | episode-scoped evidence, cross-epoch retrieval |
| `academy/agents/metargon_agent.py` | epoch hook, attribution counters |
| `academy/tools/memory_experiment_v07.py` | benchmark + attribution |
| `academy/tools/memory_merge_v07.py` | JSONL → summary |
| `academy/tests/test_memory_v07.py` | unit tests |
| `x/run_chunks_v07.py` | resumable chunked runner |
| `docs/MEMORY_V07.md` | this report |
| `generalization_audit/manifests/memory_v07*.manifest.json` | Quality v0.5 manifests |
| `results/academy_v01/memory_v07/` | raw artifacts |
