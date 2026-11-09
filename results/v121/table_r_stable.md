# METARGON v0.12 — r family

Run: 4 seeds × 40 episodes × 1 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## stable

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 61.25% | 72.5% | 85.0% | 50.66 | 0.00 | 20.01 |
| metargon11 | 71.88% | 70.6% | 86.9% | 40.14 | 0.00 | 8.77 |
| metargon12 | 60.62% | 72.5% | 83.8% | 46.68 | 0.00 | 13.54 |
| metargon12-norc | 60.62% | 72.5% | 83.8% | 46.68 | 0.00 | 12.54 |
| metargon12-noad | 60.62% | 72.5% | 83.8% | 46.68 | 0.00 | 12.14 |
| random | 0.62% | 0.0% | 0.0% | 79.93 | 0.00 | 0.34 |
| bayes | 77.50% | 0.0% | 0.0% | 45.20 | 0.00 | 5.10 |
| oracle | 100.00% | 0.0% | 0.0% | 17.75 | 0.00 | 0.07 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | -0.62 | [-1.87, 0.00] |
| comparison:metargon12_minus_metargon11 | -11.25 | [-33.75, 0.00] |
| comparison:metargon12_minus_bayes | -16.88 | [-61.25, 23.75] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |
