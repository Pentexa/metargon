# METARGON v0.11 — benchmark table

Run: 2 seeds × 20 episodes × 2 conditions (family f).

Uncertainty: paired bootstrap 95% intervals (1500 resamples).


## stable

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 72.50% | 34.52 | 0.00 | 31.5 | 4.90 | 44.83 | 0.23 |
| metargon11 | 72.50% | 34.33 | 0.00 | 29.0 | 5.11 | 33.27 | 0.13 |
| metargon11-cf | 72.50% | 34.27 | 0.00 | 29.0 | 5.11 | 34.68 | 0.28 |
| metargon11-ct | 72.50% | 34.33 | 0.00 | 29.0 | 5.11 | 34.16 | 0.13 |
| metargon11-rp | 72.50% | 34.33 | 0.00 | 29.0 | 5.11 | 30.24 | 0.13 |
| metargon11-rw | 72.50% | 34.33 | 0.00 | 29.0 | 5.11 | 27.70 | 0.13 |
| bayes | 82.50% | 42.78 | 0.00 | 57.5 | 4.12 | 10.48 | 0.48 |
| oracle | 100.00% | 16.50 | 0.00 | 40.0 | 3.00 | 0.15 | 0.09 |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 17.5 | 14.0 | 492.5 | None | None |
| metargon11 | 13.0 | 2.0 | 221.5 | 29.0 | 0.0 |
| metargon11-cf | 9.5 | 2.0 | 221.5 | 0.0 | 0.0 |
| metargon11-ct | 13.0 | 2.0 | 221.5 | 29.0 | 0.0 |
| metargon11-rp | 10.0 | 2.0 | 0.0 | 29.0 | 0.0 |
| metargon11-rw | 13.0 | 2.0 | 221.5 | 29.0 | 0.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| stable | v11 vs v10 | +0.00 | [0.00, 0.00] |
| stable | v11 vs bayes | -10.00 | [-25.00, 5.00] |
| stable | v11 vs -cf (no confirmed events) | +0.00 | [0.00, 0.00] |
| stable | v11 vs -ct (no certainty) | +0.00 | [0.00, 0.00] |
| stable | v11 vs -rp (replay off) | +0.00 | [0.00, 0.00] |
| stable | v11 vs -rw (replay weighted) | +0.00 | [0.00, 0.00] |

## noisy_obs

| policy | success | steps | invalid | events | firstEv | cpu s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| metargon10 | 47.50% | 55.70 | 3.50 | 44.0 | 8.62 | 59.46 | 1.57 |
| metargon11 | 60.00% | 47.80 | 11.50 | 47.0 | 8.91 | 25.29 | 0.98 |
| metargon11-cf | 67.50% | 42.92 | 1.50 | 44.5 | 6.45 | 38.57 | 0.92 |
| metargon11-ct | 87.50% | 31.00 | 5.50 | 42.0 | 6.76 | 11.75 | 0.65 |
| metargon11-rp | 67.50% | 38.55 | 10.00 | 32.5 | 6.20 | 27.61 | 0.71 |
| metargon11-rw | 85.00% | 31.20 | 7.00 | 39.0 | 8.75 | 12.46 | 0.50 |
| bayes | 72.50% | 44.10 | 9.50 | 49.5 | 3.06 | 12.05 | 0.64 |
| oracle | 97.50% | 19.95 | 1.00 | 43.0 | 3.00 | 0.66 | 0.12 |

Learning diagnostics (mean per seed-run; offline oracle false-verification not available in summary):

| policy | verified | contradicted | replay applied | confirmed events | dropped candidates |
|---|---:|---:|---:|---:|---:|
| metargon10 | 30.0 | 10.5 | 5441.0 | None | None |
| metargon11 | 7.5 | 29.0 | 536.5 | 68.5 | 6.0 |
| metargon11-cf | 30.5 | 25.5 | 2077.0 | 0.0 | 4.5 |
| metargon11-ct | 19.0 | 12.0 | 464.5 | 59.5 | 3.5 |
| metargon11-rp | 13.5 | 3.0 | 0.0 | 50.5 | 6.5 |
| metargon11-rw | 18.0 | 19.0 | 408.5 | 52.5 | 3.0 |

### Paired comparisons (success-rate deltas, pp)

| condition | comparison | Δ pp | 95% CI pp |
|---|---:|---:|---:|
| noisy_obs | v11 vs v10 | +12.50 | [10.00, 15.00] |
| noisy_obs | v11 vs bayes | -12.50 | [-35.00, 10.00] |
| noisy_obs | v11 vs -cf (no confirmed events) | -7.50 | [-10.00, -5.00] |
| noisy_obs | v11 vs -ct (no certainty) | -27.50 | [-55.00, 0.00] |
| noisy_obs | v11 vs -rp (replay off) | -7.50 | [-15.00, 0.00] |
| noisy_obs | v11 vs -rw (replay weighted) | -25.00 | [-50.00, 0.00] |
