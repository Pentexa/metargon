# METARGON Cognitive Integration Evaluation v0.1

Independent evaluation of whether **memory** (`epoch_reset`) and
**Cognitive Exploration v0.1** (`explore`) **together** improve the
Academy student's genuine learning and generalization more than either
alone — and whether a joint benefit is carried by retained causal
knowledge rather than planning/search effort.

Reuses the unmodified [Cognitive Evaluation v0.2](../cognitive_v02/)
framework (merged via PR #52). This package only:

- registers additional control arms through the v0.2 public candidate
  interface (no Academy code touched);
- computes interaction and attribution deltas on top of the shared
  per-seed results.

## Preregistration

`docs/PREREGISTRATION.md` is frozen before the confirmatory run:

- Confirmatory seeds: **101, 103, 107, 109** (untouched by prior
  evaluations; smoke uses seed 0 only).
- Hypotheses H1–H4 (integration vs memory, vs exploration, interaction
  term, causal-knowledge attribution via memory-wiped variants).
- Fixed sizes, pools, metrics and analysis conventions.

## Arms

fresh · trained · muted · shuffled · epoch_reset · muted_epoch_reset ·
explore · explore_epoch_reset · muted_explore_epoch_reset ·
shuffled_explore_epoch_reset

## Reproduce

```bash
# confirmatory (preregistered seeds)
python -m generalization_audit.cognitive_integration_v01.runner \
    --seeds 101 103 107 109 \
    --out generalization_audit/cognitive_integration_v01/results

# smoke (seed 0, separate output; never confirmatory)
python -m generalization_audit.cognitive_integration_v01.runner \
    --quick --out /tmp/cog_integration_smoke
```

Artifacts:

- `results/summary.json` — comparative scorecard (arms × pools, paired
  bootstrap comparisons, floors).
- `results/analysis.json` — interaction terms, attribution deltas,
  per-seed rate table.
- `results/per_seed.json` — compact raw per-seed records.
- `docs/COGNITIVE_INTEGRATION_V01.md` — rendered report.

## Tests

```bash
python -m unittest discover -s generalization_audit/cognitive_integration_v01/tests -t . -v
```

Coverage: preregistration invariants, interaction math, attribution
delta summaries, runner smoke (seed 0).
