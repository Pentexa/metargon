# METARGON v0.12 — l family

Run: 4 seeds × 40 episodes × 2 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## stable

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 21.25% | 0.0% | 0.0% | 78.06 | 0.00 | 489.84 |
| metargon11 | 18.12% | 0.0% | 0.0% | 75.54 | 0.00 | 428.57 |
| metargon12 | 27.50% | 16.9% | 0.0% | 68.01 | 0.00 | 349.80 |
| metargon12-norc | 18.12% | 0.0% | 0.0% | 75.31 | 0.00 | 376.99 |
| metargon12-noad | 18.12% | 0.0% | 0.0% | 75.31 | 0.00 | 323.58 |
| metargon12-notp | 18.12% | 0.0% | 0.0% | 75.31 | 0.00 | 354.25 |
| random | 0.00% | 0.0% | 0.0% | 80.00 | 0.00 | 2.14 |
| episodic | 0.00% | 0.0% | 0.0% | 80.00 | 0.00 | 2.40 |
| qlearning | 0.00% | 0.0% | 0.0% | 80.00 | 0.00 | 2.94 |
| bayes | 48.12% | 0.0% | 0.0% | 69.45 | 0.00 | 58.09 |
| oracle | 100.00% | 0.0% | 0.0% | 24.25 | 0.00 | 1.22 |

temporal diag (metargon12): generated=8.75, verified=0.75, probes=44.75


### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +6.25 | [-23.75, 48.75] |
| comparison:metargon12_minus_metargon11 | +9.37 | [-11.25, 41.87] |
| comparison:metargon12_minus_bayes | -20.62 | [-33.75, -4.37] |
| comparison:metargon12_minus_metargon12-norc | +9.38 | [-9.38, 40.00] |
| comparison:metargon12_minus_metargon12-noad | +9.38 | [-9.38, 40.00] |
| comparison:temporal_minus_notp_exact_D1 | +16.88 | [0.00, 50.63] |

## novel

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 21.88% | 0.0% | 0.0% | 77.14 | 0.00 | 409.64 |
| metargon11 | 19.38% | 0.0% | 0.0% | 74.21 | 0.00 | 365.26 |
| metargon12 | 29.38% | 23.1% | 6.2% | 67.21 | 0.00 | 282.07 |
| metargon12-norc | 18.75% | 0.0% | 0.0% | 74.01 | 0.00 | 283.51 |
| metargon12-noad | 18.75% | 0.0% | 0.0% | 74.01 | 0.00 | 278.51 |
| metargon12-notp | 18.75% | 0.0% | 0.0% | 74.01 | 0.00 | 248.53 |
| random | 0.00% | 0.0% | 0.0% | 80.00 | 0.00 | 1.58 |
| episodic | 0.00% | 0.0% | 0.0% | 80.00 | 0.00 | 2.09 |
| qlearning | 0.00% | 0.0% | 0.0% | 80.00 | 0.00 | 2.27 |
| bayes | 42.50% | 0.0% | 0.0% | 69.67 | 0.00 | 45.16 |
| oracle | 100.00% | 0.0% | 0.0% | 23.75 | 0.00 | 1.27 |

temporal diag (metargon12): generated=9.0, verified=0.75, probes=42.5


### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | +7.50 | [-12.50, 26.25] |
| comparison:metargon12_minus_metargon11 | +10.00 | [-3.75, 26.25] |
| comparison:metargon12_minus_bayes | -13.12 | [-26.25, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +10.63 | [-3.75, 26.25] |
| comparison:metargon12_minus_metargon12-noad | +10.63 | [-3.75, 26.25] |
| comparison:temporal_minus_notp_exact_D1 | +23.12 | [0.00, 50.63] |
