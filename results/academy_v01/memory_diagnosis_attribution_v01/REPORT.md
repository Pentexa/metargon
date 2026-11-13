# Persistent Learning Diagnosis v0.1 — root-cause report

Run: `memory-diagnosis-attribution-v01-main`; seeds 20270201; train 120 eps/seed; PYTHONHASHSEED=0.

## Cell: classic

| variant | frozen | adaptive | exact-usable | false-contradictions | reuse share |
|---|---:|---:|---:|---:|---:|
| epoch | 0.500 | 0.729 | 117 | 0 | 1.00 |
| epoch_reset | 0.500 | 0.667 | 120 | 0 | 1.00 |
| fresh | 0.375 | 0.562 | 0 | 0 | — |
| muted | 0.438 | 0.583 | 0 | 0 | — |
| original | 0.479 | 0.625 | 0 | 135 | — |
| shuffled | 0.479 | 0.625 | 0 | 135 | — |

## Cell: conflict

| variant | frozen | adaptive | exact-usable | false-contradictions | reuse share |
|---|---:|---:|---:|---:|---:|
| epoch | 0.188 | 0.583 | 84 | 0 | 1.00 |
| epoch_reset | 0.188 | 0.562 | 87 | 0 | 1.00 |
| fresh | 0.000 | 0.250 | 0 | 0 | — |
| muted | 0.000 | 0.292 | 0 | 0 | — |
| original | 0.062 | 0.312 | 12 | 93 | — |
| shuffled | 0.062 | 0.312 | 12 | 93 | — |

## Cell: noisy

| variant | frozen | adaptive | exact-usable | false-contradictions | reuse share |
|---|---:|---:|---:|---:|---:|
| epoch | 0.639 | 0.722 | 93 | 0 | 1.00 |
| epoch_reset | 0.833 | 0.722 | 96 | 0 | 1.00 |
| fresh | 0.639 | 0.806 | 0 | 0 | — |
| muted | 0.639 | 0.806 | 0 | 0 | — |
| original | 0.611 | 0.917 | 0 | 111 | — |
| shuffled | 0.611 | 0.917 | 0 | 111 | — |

## Cell: ood

| variant | frozen | adaptive | exact-usable | false-contradictions | reuse share |
|---|---:|---:|---:|---:|---:|
| epoch | 0.688 | 0.750 | 141 | 0 | 1.00 |
| epoch_reset | 0.688 | 0.750 | 156 | 0 | 1.00 |
| fresh | 0.625 | 0.500 | 0 | 0 | — |
| muted | 0.562 | 0.500 | 0 | 0 | — |
| original | 0.562 | 0.562 | 6 | 174 | — |
| shuffled | 0.562 | 0.562 | 6 | 174 | — |

## A/B/A repeated learning

| variant | A1 | B | A2 | A2−A1 | retained share | A1-verified-after-B |
|---|---:|---:|---:|---:|---:|---:|
| epoch_reset | 0.475 | 0.733 | 0.433 | -0.042 | 1.00 | 10 |
| original | 0.350 | 0.867 | 0.467 | +0.117 | — | 0 |

A1→B status transitions (A1 hypotheses that persist into B):

| variant | transition | count |
|---|---:|---:|
| epoch_reset | contradicted->verified | 10 |
| epoch_reset | hypothesis->hypothesis | 5 |
| epoch_reset | predicted->hypothesis | 5 |
| epoch_reset | verified->contradicted | 10 |
| epoch_reset | verified->hypothesis | 4 |
| original | contradicted->contradicted | 15 |
| original | verified->contradicted | 19 |
