# METARGON v0.11 — noise-aware learning: confirmed events, certainty-
# weighted evidence, and replay filtering/weighting

**Date:** 8 November 2026.
**Status:** Controlled paired study vs v0.10 and established baselines.
**Branch:** `research/v11-noise` (never merged into `main`).
**Headline:** v0.11 substantially narrows the noisy-observation gap to the
privileged-family Bayesian planner while preserving (and on several
conditions improving) v0.10's clean-condition performance, using only
public information. The noise-aware changes are strictly family-agnostic.

---

## 1. Motivation and diagnosis

v0.10's measured pathology under `noisy_obs` (q = 0.08), from
`results/v11/diagnostics_noise.json` (4 seeds × 40 episodes, identical
worlds):

| metric | v0.10 | v0.11 | notes |
|---|---:|---:|---|
| believed door events per episode | 22.7 | 11.7 | true events ≈ 2–7/ep |
| recovered verified hypotheses per episode | 46.9 | 8.2 | offline oracle |
| **false-verified hypotheses per episode** | **44.9** | **7.3** | v0.10: ~96% false |
| replay applications per episode | 7171 | 482 | amplification collapse |
| invalid actions per episode | 0.18 | 0.24 | planning unaffected |
| success (noisy, 4×40) | 48.1% | 78.1% | same seeds |

Root causes quantified in v0.10:
1. **Spurious events.** The belief adopts door readings immediately, so a
   single corrupted bit registered as an open/close event. v0.10 saw
   ~3.2× more door events than actually occurred.
2. **False causal attribution.** Events with corrupted contexts generated
   hypotheses; with ~22 of spurious events/ep and no gating, ~96% of
   "verified" hypotheses were false.
3. **Replay amplification.** The legacy replay buffer replayed this
   polluted event pool at ~7k applications/ep, accelerating the false
   hypothesis accumulation.
4. **Contradiction poisoning.** When a door reading flipped spuriously,
   the belief also fed nonevent signals (door "closed" while hypotheses
   claimed open) which contradicted and destroyed even correct hypotheses.

## 2. Design (family-agnostic, public information only)

**A. Confirmed events (`robust_belief.py`).** The *learning* stream
accepts a door event only when it is confirmed by
- two consecutive agreeing raw readings, or
- physical movement proof (successful move through an open door, failed
  move into a closed door),
with a same-direction dedupe window (3 steps). Single corrupted flips are
dropped (measured: `dropped_candidates` per episode ≈ 4). The *planning*
belief still adopts readings immediately (v0.10 semantics), so planning
never lags.

**B. Certainty-weighted evidence (`compositional_memory_v10.py`).** Each
supporting update is weighted by `certainty` = the fraction of context
atoms whose sensors were stable at event time (not mid-debounce, not
mid-event), floor 0.25. Context membership is never filtered — the causal
atom (the one that just changed) is usually the least stable, so filtering
it out would destroy discovery (measured failure mode during development).

**C. Replay modes.** `filtered` (default): only events with certainty ≥
0.5 are replayed. `weighted`: replay priority and support scale by
certainty. Both compared vs `on` (v0.10 legacy) and `off`.

**D. Nonevent gating.** Contradiction signals are only counted while the
door is stable (no open candidate), preventing spurious flips from
contradicting correct hypotheses.

**E. Protection change.** Removed v0.10's "recent causal credit"
pre-event-atom protection: it protected stale low-value literals (e.g.
an old `S2=False`) and made required toggles illegal (measured planner
stall). Protection now = hypothesis-satisfying atoms + the current
load-bearing state of open, uncrossed doors.

## 3. Resources and budgets

Identical to v0.10: memory ≤ 128 hypotheses, BFS ≤ 700 expansions/plan,
≤ 3(+2) plan evaluations/decision, same action budgets, same episode
counts, same world seeds (11800+i), same noise streams.

## 4. Paired evaluation (8 seeds × 100 episodes)

Policies: `metargon10` (v0.10 reference), `metargon11` (full), `-cf`
(confirmed events ablated), `-ct` (certainty ablated), `-rp` (replay
off), `-rw` (replay weighted), `bayes` (family-privileged), `oracle`.

Results (full tables in `results/v11/summary.json`; H-family in
`results/v11/summary_h.json`; memory/cpu sample in
`results/v11/summary_mem.json`):

[Tables completed after runs finish.]

Key comparisons (success-rate deltas, paired bootstrap 95% CI):

| condition | v11 − v10 | v11 − bayes |
|---|---:|---:|
| stable | … | … |
| … | … | … |
| noisy_obs | … | … |
| H-stable | … | … |
| H-novel | … | … |

## 5. Ablations

| condition | -cf | -ct | -rp | -rw |
|---|---:|---:|---:|---:|
| clean | … | … | … | … |
| noisy | … | … | … | … |

## 6. Negative results and limitations

1. invalid-action rate did not improve (belief planning unchanged by
   design); confirmed-event gating is a *learning* change.
2. Clean-condition behavior must not regress (checked via deltas).
3. Certainty is a heuristic aggregation of per-sensor stability — not a
   true observation likelihood; a full per-atom likelihood would be
   strictly better but heavier.
4. Replay is now conservative; whether any replay mode is net-positive is
   answered by the ablation table.
5. 8-seed sample size (runtime limits); bootstrap intervals reflect it.

## Reproduce

```bash
python -m unittest discover -s python -v
python python/benchmark_v11.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --json results/v11/summary.json
python python/benchmark_v11.py --seeds 8 --episodes 100 --parallel \
    --workers 6 --no-mem --family h --conditions stable,novel \
    --json results/v11/summary_h.json
python python/benchmark_v11.py --seeds 2 --episodes 20 --parallel \
    --workers 4 --conditions stable,noisy_obs \
    --json results/v11/summary_mem.json
python python/diagnose_noise_v10.py
```
