# METARGON Cognitive Evaluation v0.2 — independent intelligence audit

Independent evaluation of whether METARGON's cognitive improvements are
genuine learning, memory transfer and adaptation, rather than increased
search effort or benchmark-specific behavior. The student agent
(`MetargonAcademyAgent`) is used unmodified; no learning algorithm is
touched.

## Questions

1. Do improvements reflect genuine learning / memory transfer / adaptation?
2. Does a **trained** agent outperform a **fresh** agent with identical
   action and computation budgets (matched streams, same max steps)?
3. Do gains disappear when learned memory is **removed** (muted) or its
   associations are **randomized** (shuffled negative control)?
4. Separating **within-episode discovery** from **cross-episode transfer**.
5. **Catastrophic forgetting** after a conflicting A/B/A task sequence.
6. Distinguishing **improved planning** from **improved causal knowledge**.

## Arms

| Arm | Definition |
|---|---|
| `trained` | default Academy student, trained on the standard stream |
| `epoch_reset` | v0.7 best-known memory: episode-scoped evidence + per-episode probe reset |
| `explore` | Cognitive Exploration v0.1 on (master switch) |
| `explore_epoch_reset` | both mechanisms together |
| `fresh` | untrained student (never trains) |
| `muted` | trained → declarative memory wiped (counts preserved) |
| `shuffled` | trained → hypothesis associations randomized (counts/support preserved) |

## Protocols

- **Standard**: train 12 episodes → zero-shot eval on val/test (clean),
  test (noisy q=0.08) and OOD pools (6 episodes each). Records success,
  verified-correct/wrong hypotheses, zero-shot coverage (pre-eval memory
  vs true rules), within-episode discovery, planner ops, fallback,
  invalid actions, steps.
- **A/B/A**: train rule A → eval A′ → train conflicting rule B (flipped
  D1 polarity, same other-door rule, solvability-verified) → eval B′ →
  retrain A → eval A″. A″ vs A′ = forgetting; B′ = recovery after the
  mechanism change. Eval sets are held-out configs sharing A's/B's D1
  rule when available (fallback: the config itself).
- **Floors** (context only): random policy and a privileged oracle that
  solves each eval config once (cached).

## Statistics

- Bounded probability metrics → pooled 95% Wilson interval (inside
  [0, 1]).
- Unbounded metrics → mean ± 95% Student-t on per-seed values.
- Paired comparisons (trained vs fresh, muted/shuffled vs trained, etc.)
  → mean paired delta with bootstrap 95% CI.

## Running

```bash
python -m generalization_audit.cognitive_v02.runner \
    --seeds 0 1 2 3 --out generalization_audit/cognitive_v02/results
python -m generalization_audit.cognitive_v02.runner --quick   # CI smoke
```

Outputs: `results/summary.json` (scorecard), `results/per_seed.json`
(compact raw records), `docs/COGNITIVE_EVALUATION_V02.md` (report).

## Integration interface for upcoming v0.2 candidate branches

Cognitive Memory v0.2 / Exploration v0.2 branches can be measured without
modifying this repository:

- Drop a JSON spec into `generalization_audit/cognitive_v02/candidate_specs/`:
  - trained arm: `{"name": "...", "kwargs": {<agent kwargs>}}`
  - control: `{"name": "...", "kind": "transform", "base": "...", "transform": "wipe"|"shuffle"}`
- Or call `arms.register_candidate(name, kwargs=...)` from the candidate's
  own harness before invoking the runner.
- The runner prints registered arms at startup. Candidate policy code
  itself stays on the candidate branch — only the spec is read here.

## Tests

```bash
python -m unittest discover -s generalization_audit/cognitive_v02/tests -t . -v
```

Coverage: frozen-split integrity, stream determinism, conflict-pair
construction, arm factory/transforms/candidate loader, driver + discovery
flag, raw-metric builders, aggregation (Wilson/t/paired), end-to-end
runner smoke (arms × phases completeness, scorecard render, raw-size
bound).
