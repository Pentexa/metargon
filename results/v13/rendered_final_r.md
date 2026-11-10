# METARGON v0.13 — family r
seeds 4 × 40 episodes; conditions ['stable', 'novel']; policies ['metargon10', 'metargon11', 'metargon12', 'metargon13', 'metargon13-nodc', 'bayes', 'oracle', 'random', 'random']

## stable
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 61.3% | 72.5% | 85.0% | 72.5% | 85.0% | 50.66 | 0.00 | 19.9 | — | — | — | — |
| metargon11 | 71.9% | 70.6% | 86.9% | 70.6% | 86.9% | 40.14 | 0.00 | 9.6 | — | — | — | — |
| metargon12 | 60.6% | 72.5% | 83.8% | 72.5% | 83.8% | 46.68 | 0.00 | 21.0 | 0.00 | — | — | — |
| metargon13 | 60.6% | 72.5% | 83.8% | 72.5% | 83.8% | 46.68 | 0.00 | 20.8 | 0.00 | — | — | — |
| metargon13-nodc | 71.9% | 70.6% | 86.9% | 70.6% | 86.9% | 40.14 | 0.00 | 10.1 | 0.00 | — | — | — |
| bayes | 77.5% | 0.0% | 0.0% | 0.0% | 0.0% | 45.20 | 0.00 | 6.8 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 17.75 | 0.00 | 0.1 | — | — | — | — |
| random | 0.6% | 0.0% | 0.0% | 0.0% | 0.0% | 79.93 | 0.00 | 0.5 | — | — | — | — |
| random | 0.6% | 0.0% | 0.0% | 0.0% | 0.0% | 79.93 | 0.00 | 0.5 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 | metargon13-nodc |
|---|---|---|---|---|---|
| 0 | 100.0% | 97.5% | 100.0% | 100.0% | 97.5% |
| 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 95.0% | 92.5% | 95.0% | 95.0% | 92.5% |
| 3 | 95.0% | 92.5% | 95.0% | 95.0% | 92.5% |
- metargon10: exact D1 nonzero seeds 3/4, population stddev 0.419
- metargon11: exact D1 nonzero seeds 3/4, population stddev 0.408
- metargon12: exact D1 nonzero seeds 3/4, population stddev 0.419
- metargon13: exact D1 nonzero seeds 3/4, population stddev 0.419
- metargon13-nodc: exact D1 nonzero seeds 3/4, population stddev 0.408
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta -0.62 pp CI (-1.9, 0.0)
- comparison:metargon13_minus_metargon11: success delta -11.25 pp CI (-33.8, 0.0)
- comparison:metargon12_minus_metargon10: success delta -0.62 pp CI (-1.9, 0.0)
- comparison:metargon12_minus_metargon11: success delta -11.25 pp CI (-33.8, 0.0)

## novel
| policy | success_rate | exact_discovery_D1 | exact_discovery_D2 | covered_discovery_D1 | covered_discovery_D2 | mean_steps | invalid_actions | cpu_s | temporal_probes | temporal_generated | temporal_verified | temporal_evictions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metargon10 | 80.0% | 84.4% | 90.0% | 84.4% | 90.6% | 36.99 | 0.00 | 11.5 | — | — | — | — |
| metargon11 | 84.4% | 75.6% | 83.1% | 75.6% | 83.1% | 30.12 | 0.00 | 6.9 | — | — | — | — |
| metargon12 | 79.4% | 84.4% | 89.4% | 84.4% | 89.4% | 33.30 | 0.00 | 12.0 | 0.00 | — | — | — |
| metargon13 | 79.4% | 84.4% | 89.4% | 84.4% | 89.4% | 33.30 | 0.00 | 11.4 | 0.00 | — | — | — |
| metargon13-nodc | 84.4% | 75.6% | 83.1% | 75.6% | 83.1% | 30.12 | 0.00 | 6.2 | 0.00 | — | — | — |
| bayes | 84.4% | 0.0% | 0.0% | 0.0% | 0.0% | 43.69 | 0.00 | 6.6 | — | — | — | — |
| oracle | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 16.62 | 0.00 | 0.1 | — | — | — | — |
| random | 2.5% | 0.0% | 0.0% | 0.0% | 0.0% | 79.47 | 0.00 | 0.5 | — | — | — | — |
| random | 2.5% | 0.0% | 0.0% | 0.0% | 0.0% | 79.47 | 0.00 | 0.5 | — | — | — | — |

### per-seed exact D1 (frozen criteria: seed sensitivity)
| seed | metargon10 | metargon11 | metargon12 | metargon13 | metargon13-nodc |
|---|---|---|---|---|---|
| 0 | 100.0% | 82.5% | 100.0% | 100.0% | 82.5% |
| 1 | 47.5% | 35.0% | 47.5% | 47.5% | 35.0% |
| 2 | 95.0% | 92.5% | 95.0% | 95.0% | 92.5% |
| 3 | 95.0% | 92.5% | 95.0% | 95.0% | 92.5% |
- metargon10: exact D1 nonzero seeds 4/4, population stddev 0.214
- metargon11: exact D1 nonzero seeds 4/4, population stddev 0.238
- metargon12: exact D1 nonzero seeds 4/4, population stddev 0.214
- metargon13: exact D1 nonzero seeds 4/4, population stddev 0.214
- metargon13-nodc: exact D1 nonzero seeds 4/4, population stddev 0.238
- comparison:metargon13_minus_metargon12: success delta +0.00 pp CI (0.0, 0.0)
- comparison:metargon13_minus_metargon10: success delta -0.62 pp CI (-1.9, 0.0)
- comparison:metargon13_minus_metargon11: success delta -5.00 pp CI (-15.0, 0.0)
- comparison:metargon12_minus_metargon10: success delta -0.62 pp CI (-1.9, 0.0)
- comparison:metargon12_minus_metargon11: success delta -5.00 pp CI (-15.0, 0.0)
