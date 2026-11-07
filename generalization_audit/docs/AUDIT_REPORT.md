# METARGON Generalization Audit — Report

**Auditor:** Independent Generalization Evaluation Agent
**Date:** 8 November 2026
**Scope:** v0.9, v0.10, Academy v0.1/v0.2, EVREN protocol audit + new
independent benchmark on structurally novel families R and L.
**Artifacts:** `results/summary_r.json`, `results/summary_l.json`,
`configs/frozen_{r,l}.json`.

---

## 1. Protocol audit (leakage, bias, artifacts, baseline fairness)

### 1.1 v0.9 / v0.10 (Mechanism Lab)

**Leakage — none found in the student path.**
- Student policies consume only public `Observation` (13 fields: agent,
  object, switch, plate, door readings). No hidden config is exposed.
- The oracle is explicitly privileged and excluded from deployable
  rankings (`oracle_has_hidden_config: true`).
- Train/val/test pools are disjoint and held out (`novel` uses the test
  pool from the midpoint on).
- Noise streams are deterministic per (episode, step index), so all
  policies see identical corrupted readings.
- The `StepResult.door_events` field is available to every policy; it is
  derivable from the observation stream and is ignored by the student
  agents. A hostile agent could use it as a privileged event signal — a
  minor interface caveat, not an observed leak.
- **Interface caveat (Academy):** `EpisodeSpec` passes the full hidden
  `EnvConfig` object to every policy via `episode_start(spec, geom)`;
  opacity relies on convention (student agents ignore it), not on
  enforcement. The docs' claim "policies never receive hidden rules" is
  factually false at the interface level. No implemented policy violates
  it, but a future policy could.

**Baseline fairness.**
- Bayes receives the *declared generative family F* (a correct model
  class on F; intentionally misspecified on H). This is a documented,
  deliberate strong-comparator choice.
- Compute is asymmetric: Bayes expands up to 4000 nodes per plan while
  METARGON is capped at 700; CPU is measured and reported but not
  equalized. The audit adds a matched-compute arm (Section 3).
- Model-free baselines (random/episodic/Q-learning) were present in the
  v0.9 benchmark but **absent from the v0.10 benchmark**. The audit
  restores them (Section 3).

**Evaluation artifacts.**
- Committed `results/v10/summary.json` predates the `-rp` ablation and
  contains a stale (all-zero) stall detector column.
- `results/v10/per_seed_results_v10.json` is actually from the stall
  validation run (8 seeds × 20 episodes × 3 conditions), not from the
  main summary run — the fixed output filename overwrites whatever ran
  last.
- `true_rules_discovered()` in the benchmark is dead code (guarded by
  `if False`); discovery numbers in the v0.10 report come only from the
  offline `discovery_analysis_v10.py`.

### 1.2 Academy v0.1/v0.2

- **Matched exposure is real:** one frozen stream per seed consumed
  identically by all policies; same seeds, steps, noise.
- **Bayes prior is TP1-only** (`CORE_RULESETS["TP1"]`), so it is
  structurally disadvantaged on the ~half of the pool whose D2 rules are
  not in the TP1 prior. The headline "METARGON beats Bayes" comparison
  partly reflects prior mismatch, not student superiority.
- **Headline comparisons lack paired-difference artifacts:** the 30-seed
  run exists only as a log line (`log_eval_test30.txt`); the JSON
  artifacts record no paired-difference statistics.
- Artifacts do not record `episodes_per_seed`, `max_steps`, `noise_q`,
  or the active hash seed; `seeds.ood` is configured but unused.
- `noise_q`/`tight_steps` are dead config (always 0.0/80 in practice).
- `cmd_eval` silently falls back to *untrained* agents when a policy
  checkpoint is missing.
- **v0.2 doc/artifact discrepancy:** `EVREN_CURRICULUM_V02.md` §11
  tables (fresh 0.972, contradicted 9.3/70.0) do not match the committed
  `diagnosis_summary.json` (fresh 0.53, contradicted 6.5/70/66); two
  runs were conflated in the doc.
- Results are hash-seed sensitive in internal hypothesis bookkeeping
  (summary metrics reproduce under fixed `PYTHONHASHSEED`).

### 1.3 EVREN

- Validation chain is sound: leakage scan → strict JSON schema → atom
  grammar → canonicalization → independent BFS solvability →
  contamination check → duplicate check.
- Teacher proposes tasks, never answers; tasks never reach the student
  directly. Novelty is bounded by the prompt's grammar (documented).
- v0.2 selection showed zero transfer effect (all effects +0.0000);
  overfitting-by-selection was not observed.

### 1.4 Findings to carry into the benchmark

1. Always include model-free baselines and random.
2. Add a matched-compute arm (per-decision expansion cap applied to all
   policies).
3. Freeze test configurations before benchmarking (this audit).
4. Record paired bootstrap 95% CIs on per-seed deltas.
5. Record config fingerprints, seeds, and hash-seed policy in artifacts.
6. Keep oracle privileged and excluded from deployable rankings.

---

## 2. New task families (structurally different from F and H)

| Axis | F | H | R (new) | L (new) |
|---|---|---|---|---|
| Rule shape | conjunctive, ≤2 atoms | conjunctive, polarity-flipped | single/conjunctive + **hold threshold** | **ordered pair** (strict sequence) |
| Opening semantics | instantaneous on satisfaction | instantaneous | after `HOLD_K=3` continuous holds | after first-then-second within window |
| History dependence | none | none | sustained-condition counter | per-door phase + window |
| Reference to other doors | no | no | no | no |
| Learnable with v0.10 vocabulary | yes | yes | yes (single atoms) | partially (pair hypotheses; order learned behaviorally) |
| Bayes family-F model | correct | misspecified | misspecified | misspecified |

Both families are ground-truth solvable (independent search), use the
same 7×5 geometry and public atom vocabulary, and never expose hidden
rules.

---

## 3. Benchmark methodology

- **Same pre-generated worlds per policy:** matched seeds, episode
  counts, action budgets (80 steps; invalid actions consume steps), and
  deterministic per-step noise streams.
- **Conditions:** `stable`, `shift`, `novel`, `iid`, `noisy_obs`, and
  `cross_family` (mechanism change: family F for the first half, the
  audited family for the second half).
- **Policies:** `metargon10`, `random`, `episodic`, `qlearning`,
  `bayes`, plus the matched-compute arm `metargon10-cap`, `bayes-cap`,
  `qlearning-cap` (per-decision expansion cap 120), and a privileged
  per-family `oracle`.
- **Offline metrics:** exact/covered discovery, false-verified ratio,
  forgetting across mechanism shifts (Jaccard complement on verified
  antecedents), adaptation (post-shift success, first recovery episode),
  success, steps, invalid actions, events, CPU seconds.
- **Pairing:** per-seed deltas; paired bootstrap 95% CIs (1500
  resamples, seed 4301).
- **Shortcut checks:** every frozen config re-verified solvable at run
  time; runtime scan that no student policy ever holds a hidden config
  reference; noise-parity check.

---

## 4. Results

### 4.1 Family R (sustained-hold)

> Filled from `results/summary_r.json` after the run.

### 4.2 Family L (ordered-sequence)

> Filled from `results/summary_l.json` after the run.

---

## 5. Positive and negative findings

> Completed after runs.

---

## 6. Limitations

- Sample sizes are 6 paired seeds (runtime-constrained); CIs reflect
  that.
- The cross-family oracle is excluded (its model is per-family).
- Academy results remain separate and are not compared with research
  benchmarks (different environments/protocols).
- Hash-seed sensitive bookkeeping: runs are bit-reproducible under a
  fixed `PYTHONHASHSEED`.

---

## 7. Reproduction

```bash
# freeze configs (one-time; committed)
python generalization_audit/freeze.py --family r
python generalization_audit/freeze.py --family l

# unit tests
python -m unittest discover -s generalization_audit/tests -v

# benchmarks
python generalization_audit/benchmark_generalization.py --family r \
    --seeds 6 --episodes 60 \
    --conditions stable,shift,novel,iid,noisy_obs,cross_family \
    --parallel --workers 5 --json generalization_audit/results/summary_r.json
python generalization_audit/benchmark_generalization.py --family l \
    --seeds 6 --episodes 60 \
    --conditions stable,shift,novel,iid,noisy_obs,cross_family \
    --parallel --workers 5 --json generalization_audit/results/summary_l.json
```
