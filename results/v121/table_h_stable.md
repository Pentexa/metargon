# METARGON v0.12 — h family

Run: 4 seeds × 40 episodes × 1 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301).


## stable

| policy | success | exact D1 | exact D2 | steps | invalid | cpu s |
|---|---:|---:|---:|---:|---:|---:|
| metargon10 | 83.75% | 88.8% | 63.1% | 36.61 | 0.00 | 17.66 |
| metargon11 | 73.12% | 92.5% | 41.2% | 34.09 | 0.00 | 22.24 |
| metargon12 | 73.12% | 95.6% | 70.0% | 34.09 | 0.00 | 20.37 |
| metargon12-nodc | 73.12% | 92.5% | 41.2% | 34.09 | 0.00 | 21.09 |
| metargon12-norc | 73.12% | 95.6% | 70.0% | 34.09 | 0.00 | 20.68 |
| metargon12-noad | 73.12% | 95.6% | 70.0% | 34.09 | 0.00 | 18.68 |
| bayes | 71.88% | 0.0% | 0.0% | 46.00 | 0.00 | 7.41 |
| oracle | 100.00% | 0.0% | 0.0% | 16.50 | 0.00 | 0.07 |

### Paired comparisons (success-rate deltas, pp)

| comparison | Δ pp | 95% CI pp |
|---|---:|---:|
| comparison:metargon12_minus_metargon10 | -10.63 | [-30.00, 0.00] |
| comparison:metargon12_minus_metargon11 | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_bayes | +1.25 | [-22.50, 23.75] |
| comparison:metargon12_minus_metargon12-nodc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-norc | +0.00 | [0.00, 0.00] |
| comparison:metargon12_minus_metargon12-noad | +0.00 | [0.00, 0.00] |
