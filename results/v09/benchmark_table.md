# METARGON v0.9 — benchmark table

Environment: Mechanism Lab (`python/lab_env.py`). Paired pre-generated worlds; same seeds per policy; oracle is privileged (labelled, excluded from deployable rankings).

Run: 8 seeds × 100 episodes × 7 conditions.

Uncertainty: paired bootstrap 95% intervals (1500 resamples, seed 4301/4302).


## stable

| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.25% | 0.50% | 0.00% | 79.97 | 2.04 | — |
| episodic | 0.25% | 0.50% | 0.00% | 79.97 | 2.04 | — |
| qlearning | 0.00% | 0.00% | 0.00% | 80.00 | 2.26 | — |
| bayes | 77.50% | 76.00% | 79.00% | 45.63 | 27.71 | — |
| metargon | 35.12% | 38.25% | 32.00% | 61.12 | 32.54 | — |
| oracle | 100.00% | 100.00% | 100.00% | 17.75 | 0.31 | — |

- metargon − bayes: -42.38 pp (95% CI [-68.12, -17.00] pp)

- metargon − qlearning: +35.12 pp (95% CI [+9.75, +60.00] pp)

- metargon − episodic: +34.88 pp (95% CI [+9.25, +59.88] pp)

- bayes − qlearning: +77.50 pp (95% CI [+68.25, +86.12] pp)

## shift

| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.62% | 0.50% | 0.75% | 79.87 | 1.87 | — |
| episodic | 0.75% | 0.50% | 1.00% | 79.76 | 2.18 | — |
| qlearning | 0.12% | 0.00% | 0.25% | 79.96 | 2.30 | — |
| bayes | 80.62% | 76.00% | 85.25% | 41.41 | 22.96 | — |
| metargon | 46.38% | 38.25% | 54.50% | 55.22 | 25.34 | — |
| oracle | 100.00% | 100.00% | 100.00% | 20.62 | 0.41 | — |

- metargon − bayes: -34.25 pp (95% CI [-51.50, -15.87] pp)

- metargon − qlearning: +46.25 pp (95% CI [+28.75, +65.62] pp)

- metargon − episodic: +45.62 pp (95% CI [+28.12, +65.00] pp)

- bayes − qlearning: +80.50 pp (95% CI [+73.62, +87.00] pp)

## rapid_shift

| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.50% | 0.25% | 0.75% | 79.90 | 2.35 | — |
| episodic | 0.38% | 0.25% | 0.50% | 79.90 | 2.59 | — |
| qlearning | 0.00% | 0.00% | 0.00% | 80.00 | 2.24 | — |
| bayes | 76.88% | 74.25% | 79.50% | 44.66 | 32.68 | — |
| metargon | 40.75% | 40.00% | 41.50% | 59.21 | 29.65 | — |
| oracle | 91.25% | 95.00% | 87.50% | 26.55 | 0.77 | — |

- metargon − bayes: -36.12 pp (95% CI [-55.75, -16.12] pp)

- metargon − qlearning: +40.75 pp (95% CI [+20.12, +61.12] pp)

- metargon − episodic: +40.38 pp (95% CI [+19.75, +60.88] pp)

- bayes − qlearning: +76.88 pp (95% CI [+71.50, +81.38] pp)

## iid

| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.38% | 0.25% | 0.50% | 79.98 | 2.50 | — |
| episodic | 0.38% | 0.25% | 0.50% | 79.98 | 2.54 | — |
| qlearning | 0.25% | 0.00% | 0.50% | 79.97 | 2.89 | — |
| bayes | 79.00% | 78.00% | 80.00% | 43.04 | 39.32 | — |
| metargon | 53.62% | 60.25% | 47.00% | 53.80 | 82.21 | — |
| oracle | 78.88% | 78.50% | 79.25% | 39.98 | 1.20 | — |

- metargon − bayes: -25.37 pp (95% CI [-31.12, -19.62] pp)

- metargon − qlearning: +53.38 pp (95% CI [+46.88, +60.00] pp)

- metargon − episodic: +53.25 pp (95% CI [+46.75, +59.63] pp)

- bayes − qlearning: +78.75 pp (95% CI [+74.63, +82.88] pp)

## novel

| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.62% | 0.50% | 0.75% | 79.90 | 3.17 | — |
| episodic | 0.75% | 0.50% | 1.00% | 79.89 | 2.92 | — |
| qlearning | 0.50% | 0.00% | 1.00% | 79.84 | 2.78 | — |
| bayes | 78.50% | 76.00% | 81.00% | 43.69 | 33.62 | — |
| metargon | 31.25% | 38.25% | 24.25% | 62.47 | 27.34 | — |
| oracle | 93.75% | 100.00% | 87.50% | 25.25 | 0.43 | — |

- metargon − bayes: -47.25 pp (95% CI [-68.50, -23.62] pp)

- metargon − qlearning: +30.75 pp (95% CI [+8.12, +56.50] pp)

- metargon − episodic: +30.50 pp (95% CI [+8.00, +56.50] pp)

- bayes − qlearning: +78.00 pp (95% CI [+71.38, +84.00] pp)

## noisy_obs

| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.25% | 0.25% | 0.25% | 79.98 | 2.20 | — |
| episodic | 0.25% | 0.25% | 0.25% | 79.98 | 2.23 | — |
| qlearning | 0.00% | 0.00% | 0.00% | 80.00 | 2.66 | — |
| bayes | 73.12% | 74.00% | 72.25% | 48.10 | 27.17 | — |
| metargon | 33.75% | 33.75% | 33.75% | 68.63 | 118.61 | — |
| oracle | 98.38% | 98.75% | 98.00% | 21.38 | 1.13 | — |

- metargon − bayes: -39.38 pp (95% CI [-52.12, -28.25] pp)

- metargon − qlearning: +33.75 pp (95% CI [+19.50, +49.00] pp)

- metargon − episodic: +33.50 pp (95% CI [+19.50, +48.25] pp)

- bayes − qlearning: +73.12 pp (95% CI [+63.75, +82.12] pp)

## tight_budget

| policy | success | 1st half | 2nd half | mean steps | cpu s | peak mem MB |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.00% | 0.00% | 0.00% | 44.00 | 0.75 | — |
| episodic | 0.00% | 0.00% | 0.00% | 44.00 | 0.77 | — |
| qlearning | 0.12% | 0.00% | 0.25% | 44.00 | 0.88 | — |
| bayes | 64.88% | 60.75% | 69.00% | 32.05 | 14.22 | — |
| metargon | 45.00% | 35.75% | 54.25% | 35.36 | 11.05 | — |
| oracle | 100.00% | 100.00% | 100.00% | 20.62 | 0.24 | — |

- metargon − bayes: -19.88 pp (95% CI [-39.12, +1.38] pp)

- metargon − qlearning: +44.88 pp (95% CI [+26.88, +64.00] pp)

- metargon − episodic: +45.00 pp (95% CI [+26.88, +64.12] pp)

- bayes − qlearning: +64.75 pp (95% CI [+58.50, +71.25] pp)
