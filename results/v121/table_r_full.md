# METARGON v0.12 — r family

Run: 4 seeds × 40 episodes × 2 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## stable

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 61.25% | 72.5% | 85.0% | 50.66 | 0.00 | 34.18 |
| metargon11 | 71.88% | 70.6% | 86.9% | 40.14 | 0.00 | 17.10 |
| metargon12 | 60.62% | 72.5% | 83.8% | 46.68 | 0.00 | 32.92 |
| metargon12-norc | 60.62% | 72.5% | 83.8% | 46.68 | 0.00 | 23.53 |
| metargon12-noad | 60.62% | 72.5% | 83.8% | 46.68 | 0.00 | 20.21 |
| random | 0.62% | 0.0% | 0.0% | 79.93 | 0.00 | 0.79 |
| bayes | 77.50% | 0.0% | 0.0% | 45.20 | 0.00 | 10.42 |
| oracle | 100.00% | 0.0% | 0.0% | 17.75 | 0.00 | 0.18 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | -0.62 | [-1.87, 0.00] |
| comparison:metargon12_minus_metargon11 | -11.25 | [-33.75, 0.00] |
| comparison:metargon12_minus_bayes | -16.88 | [-61.25, 23.75] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |

## novel

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 80.00% | 84.4% | 90.0% | 36.99 | 0.00 | 19.94 |
| metargon11 | 84.38% | 75.6% | 83.1% | 30.12 | 0.00 | 12.74 |
| metargon12 | 79.38% | 84.4% | 89.4% | 33.30 | 0.00 | 21.42 |
| metargon12-norc | 79.38% | 84.4% | 89.4% | 33.30 | 0.00 | 18.91 |
| metargon12-noad | 79.38% | 84.4% | 89.4% | 33.30 | 0.00 | 17.52 |
| random | 2.50% | 0.0% | 0.0% | 79.47 | 0.00 | 0.77 |
| bayes | 84.38% | 0.0% | 0.0% | 43.69 | 0.00 | 12.51 |
| oracle | 100.00% | 0.0% | 0.0% | 16.62 | 0.00 | 0.21 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | -0.62 | [-1.87, 0.00] |
| comparison:metargon12_minus_metargon11 | -5.00 | [-15.00, 0.00] |
| comparison:metargon12_minus_bayes | -5.00 | [-24.38, 10.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |
