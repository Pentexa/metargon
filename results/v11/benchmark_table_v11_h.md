# METARGON v0.11 — benchmark table

Run: 8 seeds × 100 episodes × 2 conditions (family h).

Uncertainty: paired bootstrap 95% intervals (1500 resamples).


## stable

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 79.88% | 37.80 | 0.00 | 161.0 | 4.01 | 67.48 | — |
| metargon11 | 68.00% | 37.10 | 0.00 | 136.0 | 4.11 | 96.14 | — |
| metargon11-cf | 74.75% | 33.51 | 0.00 | 150.0 | 4.04 | 77.14 | — |
| metargon11-ct | 68.00% | 37.10 | 0.00 | 136.0 | 4.11 | 93.17 | — |
| metargon11-rp | 68.00% | 37.10 | 0.00 | 136.0 | 4.11 | 92.09 | — |
| metargon11-rw | 68.00% | 37.10 | 0.00 | 136.0 | 4.11 | 94.35 | — |
| bayes | 66.88% | 50.09 | 0.00 | 258.9 | 6.45 | 23.65 | — |
| oracle | 100.00% | 16.50 | 0.00 | 200.0 | 3.62 | 0.21 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 11.75 | 15.0 | 2566.125 | None | None |
| metargon11 | 9.875 | 8.0 | 1080.0 | 136.0 | 0.0 |
| metargon11-cf | 9.375 | 10.75 | 1192.0 | 0.0 | 0.0 |
| metargon11-ct | 9.875 | 8.0 | 1080.0 | 136.0 | 0.0 |
| metargon11-rp | 6.125 | 8.0 | 0.0 | 136.0 | 0.0 |
| metargon11-rw | 9.875 | 8.0 | 1080.0 | 136.0 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| stable | v11 vs v10 | -11.88 | [-28.87, -0.38] |
| stable | v11 vs bayes | +1.13 | [-14.37, 15.75] |
| stable | v11 vs -cf (no confirmed events) | -6.75 | [-16.88, 0.00] |
| stable | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| stable | v11 vs -rp (replay off) | +0.00 | [0.00, 0.00] |
| stable | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |

## novel

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 79.88% | 33.34 | 0.00 | 161.0 | 3.81 | 49.57 | — |
| metargon11 | 74.25% | 32.98 | 0.00 | 149.2 | 3.80 | 60.10 | — |
| metargon11-cf | 77.50% | 31.26 | 0.00 | 156.2 | 3.81 | 46.22 | — |
| metargon11-ct | 74.25% | 32.98 | 0.00 | 149.2 | 3.80 | 53.72 | — |
| metargon11-rp | 74.25% | 32.98 | 0.00 | 149.2 | 3.80 | 53.80 | — |
| metargon11-rw | 74.25% | 32.98 | 0.00 | 149.2 | 3.80 | 49.47 | — |
| bayes | 68.12% | 48.78 | 0.00 | 252.2 | 6.40 | 17.65 | — |
| oracle | 100.00% | 18.00 | 0.00 | 200.0 | 3.50 | 0.19 | — |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 19.875 | 16.375 | 2566.125 | None | None |
| metargon11 | 15.375 | 9.875 | 1180.0 | 148.5 | 0.0 |
| metargon11-cf | 15.375 | 14.375 | 1242.0 | 0.0 | 0.0 |
| metargon11-ct | 15.375 | 9.875 | 1180.0 | 148.5 | 0.0 |
| metargon11-rp | 13.125 | 9.875 | 0.0 | 148.5 | 0.0 |
| metargon11-rw | 15.375 | 9.875 | 1180.0 | 148.5 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| novel | v11 vs v10 | -5.62 | [-13.25, -0.38] |
| novel | v11 vs bayes | +6.13 | [-7.25, 19.00] |
| novel | v11 vs -cf (no confirmed events) | -3.25 | [-8.12, 0.00] |
| novel | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| novel | v11 vs -rp (replay off) | +0.00 | [0.00, 0.00] |
| novel | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |
