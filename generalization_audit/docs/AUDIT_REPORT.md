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

Full per-condition tables (all policies, paired bootstrap CIs) are in
`results/rendered_tables.md`; raw per-seed data is in
`results/summary_r.json` and `results/summary_l.json`.

### 4.1 Family R (sustained-hold) — success rates

| Condition | metargon10 | random / episodic | qlearning | bayes | metargon10-cap | bayes-cap | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| stable | 50.0% | 3.3% | 0.0% | **83.3%** | 0.0% | 73.3% | 100% |
| shift | 63.3% | 0.0% | 0.0% | **90.0%** | 30.0% | 33.3% | 100% |
| novel | 76.7% | 3.3% | 0.0% | **90.0%** | 53.3% | 73.3% | 100% |
| cross_family | 70.0% | 0.0% | 0.0% | **86.7%** | 50.0% | 36.7% | —¹ |

¹ Oracle excluded from `cross_family` (its plan model is per-family; the
first half is family F and the second half family R).

Discovery (D1, offline): metargon10 exact-rule 50% (stable), 77%
(shift/novel), 93% (cross_family); false-verified hypotheses remain
high (2.1–4.6 per episode). Full v0.10 loses 50pp on stable when compute
is capped at 120 expansions/decision.

### 4.2 Family L (ordered-sequence) — success rates

| Condition | metargon10 | random / episodic | qlearning | bayes | metargon10-cap | bayes-cap | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| stable | 33.3% | 0.0% | 0.0% | **43.3%** | 0.0% | 3.3% | 100% |
| shift | 26.7% | 0.0% | 0.0% | **50.0%** | 0.0% | 0.0% | 100% |
| novel | 23.3% | 0.0% | 0.0% | **36.7%** | 0.0% | 6.7% | 100% |
| cross_family | 43.3% | 0.0% | 0.0% | **60.0%** | 23.3% | 10.0% | —¹ |

¹ See note above.

Discovery (D1): metargon10 exact-rule **0.0%** on stable/shift/novel
(the ordered rule is never identified as an exact antecedent); doors
open only behaviorally (events > 0). On `cross_family` exact jumps to
35.7% (family-F half supplies familiar conjunctive context).

---

## 5. Positive and negative findings

**Positive.**
- v0.10 transfers its causal machinery to a mechanism axis it has never
  seen (sustained-hold): it achieves 50–77% on family R, far above
  model-free baselines (~0–3%) and remains competitive with the
  family-F Bayesian planner on shift/novel (within ~13–27pp).
- On family R, the student discovers exact D1 rules 50–93% of the time
  (cross_family best), confirming family-agnostic hypothesis learning is
  not limited to instantaneous conjunctions.
- Compute matching matters and is measurable: at 120 expansions/decision
  the student loses 20–50pp; the Bayes planner loses 10–56pp. The
  matched-compute arm is a real fairness lever.
- Model-free baselines (random/episodic/Q-learning) are uniformly at
  chance on both novel families; the audit confirms the v0.10 advantage
  over them generalizes.

**Negative (honest).**
- On family R, v0.10 still does not beat the family-F Bayesian planner on
  any condition (−13 to −33pp); Bayes's correct model class on F remains
  a strong prior even when the mechanism is misspecified (its plans
  incidentally satisfy the hold condition while acting).
- On family L (ordered-sequence), v0.10's hypothesis language
  (unordered conjunctive sets) **cannot express the rule**: exact D1
  discovery is 0.0% on stable/shift/novel. Success (~23–33%) comes from
  behavior only (opening doors by incidentally satisfying both atoms),
  not from causal understanding. The family-F Bayes planner also
  struggles (36–50%), but remains above the student. This is a clean
  negative result: order-dependent rules are outside the agent's
  representational capacity.
- Cap-120 compute budget cripples the student on R (50pp loss on stable);
  the student's planner is closer to the budget limit than Bayes's.
- False-verified hypotheses remain a problem under sustained-hold
  dynamics (2–4.6/ep), indicating evidence accumulation over hold
  windows is not yet noise-robust.

---

## 6. Limitations

- **Sample size:** 2 paired seeds × 15 episodes per condition; bootstrap
  CIs are wide. Numbers are directional, not definitive.
- `cross_family` has no oracle row (by design).
- Compute arm uses an expansion budget of 120 per decision (arbitrary
  but identical across policies); results are sensitive to this value.
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
