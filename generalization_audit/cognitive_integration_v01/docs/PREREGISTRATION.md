# Cognitive Integration Evaluation v0.1 — Preregistration

This protocol is frozen **before** the confirmatory run is executed. The
confirmatory evaluation uses the seeds listed below; smoke tests use a
separate seed (0) and are not part of confirmatory inference.

- Branch: `audit/cognitive-integration-v01`
- Framework: `generalization_audit/cognitive_v02` (merged via PR #52),
  reused unmodified; this package only registers additional arms and
  post-processes results.
- Status: **preregistered** (written before confirmatory execution).

## 1. Question

Do **memory** (`epoch_reset`) and **Cognitive Exploration v0.1**
(`explore`) **together** improve METARGON's genuine learning and
generalization more than either alone — and can any joint benefit be
attributed to retained causal knowledge rather than to planning/search
effort?

## 2. Arms (all matched budgets, identical streams per seed)

| Arm | Definition | Role |
|---|---|---|
| `fresh` | untrained student | floor |
| `trained` | default student, trained | reference |
| `muted` | trained → declarative memory wiped | memory control |
| `shuffled` | trained → associations randomized | memory control |
| `epoch_reset` | v0.7 episode-scoped evidence + per-episode probe reset | memory improvement |
| `explore` | Cognitive Exploration v0.1 on | exploration improvement |
| `explore_epoch_reset` | both together | **integration** |
| `muted_epoch_reset` | epoch_reset trained → wiped | memory-attribution control for epoch_reset |
| `muted_explore_epoch_reset` | explore_epoch_reset trained → wiped | memory-attribution control for integration |
| `shuffled_explore_epoch_reset` | explore_epoch_reset trained → shuffled | association control for integration |

## 3. Environments / phases

- Train stream: 12 episodes (matched across arms).
- Zero-shot eval pools: `val_clean`, `test_clean`, `test_noisy`
  (q=0.08), `ood` (6 episodes each).
- Adaptive learning: within-episode discovery (per-episode) + A/B/A
  conflicting sequence (8 train per phase; 4 held-out eval per phase, D1
  rule-sharing configs).

## 4. Seeds

- Confirmatory: **101, 103, 107, 109** (untouched by any previous
  committed evaluation; smoke uses seed 0 only).
- Streams are pure functions of (seed, frozen splits); seeds are never
  selected based on outcomes.

## 5. Metrics (computed per seed, then aggregated)

- Bounded probabilities → pooled 95% Wilson interval.
- Unbounded scalars → mean ± 95% t interval.
- Comparisons → paired per-seed delta with bootstrap 95% CI.

Primary per-pool: `success_rate`; `zero_shot_coverage` (pre-eval memory);
`within_episode_discovery`; `verified_correct`/`verified_wrong`;
`mean_steps_to_success`; `planner_ops_per_step`; `fallback_rate`;
`invalid_rate`; A/B/A `a1/a2/b_success`, `a_rule_retained_final`.

## 6. Primary hypotheses (confirmatory tests)

- H1 (integration vs memory): `explore_epoch_reset` − `epoch_reset`
  success delta on **val_clean** has bootstrap CI excluding 0 (positive).
- H2 (integration vs exploration): `explore_epoch_reset` − `explore`
  success delta on **val_clean** CI excludes 0 (positive).
- H3 (interaction): per-seed interaction term
  (explore_epoch_reset − epoch_reset) − (explore − trained) on
  **val_clean** has CI excluding 0 → exploration adds **more** on top of
  epoch_reset than on top of default memory.
- H4 (causal-knowledge attribution): `muted_explore_epoch_reset` −
  `muted_epoch_reset` (both memory-wiped) **does not** reproduce the H1
  delta → the joint benefit is carried by retained causal knowledge, not
  by planning/search alone.

Secondary: same comparisons on `test_clean`, `test_noisy`, `ood`; zero-shot
coverage and within-episode discovery deltas; A/B/A retention (A2−A1) and
recovery (B) per arm.

## 7. Confound guards

- Identical episode streams, budgets and max steps per seed across arms.
- `_floors` (random, oracle) reported for sanity context.
- Planning-vs-knowledge separation: `planner_ops_per_step`,
  `fallback_rate`, `invalid_rate` reported alongside success; muted/
  shuffled variants hold planning machinery constant while removing
  declarative content.

## 8. Stopping / reporting rules

- Fixed seed set (101,103,107,109); no sequential addition.
- Results reported per metric with CIs; no composite IQ score.
- Any violation of matched-budget equality (stream signature mismatch
  across arms) fails the run.
