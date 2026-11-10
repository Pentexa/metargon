# Academy Memory v0.6 — Experiment Manifest (Quality v0.5)

Machine-readable provenance for the Memory v0.6 benchmark, prepared with
the **Quality v0.5** infrastructure
(`generalization_audit/verification/manifest.py`, schema
`experiment-manifest-v01`, verifier
`generalization_audit/tools/verify_experiment.py`).

## 1. Manifests

| Manifest | Format | Scope |
|---|---|---|
| `memory_v06.manifest.json` | `academy_memory` | main three-regime benchmark (11 variants) |
| `memory_v06_evren.manifest.json` | `generic` | Phase 4 EVREN teacher-experience benchmark |

Verified results (verifier):

| Manifest | PASS | FAIL | INCONCLUSIVE |
|---|---|---|---|
| `memory_v06.manifest.json` | 77 | 0 | 1 (`fairness.memory_limit` not recorded) |
| `memory_v06_evren.manifest.json` | 9 | 0 | 4 (generic-format checks unsupported; memory-limit) |

The single INCONCLUSIVE on the main manifest matches the committed
example manifests (Memory v0.5, v121): memory byte limits are not part
of this experiment's protocol.

## 2. Published artifacts

The audit artifact-gate hotfix (PR #39) has merged: the frozen criteria
(v02) remain immutable but the permanent prohibition is replaced by the
provenance-aware publication gate. The raw evidence is therefore
**published in-repository** at `results/academy_v01/memory_v06/`, with
the manifests at `generalization_audit/manifests/`. The gate verifies
**PASS** for the frozen target (manifest coverage, 0 FAIL).

Content hashes (SHA-256 over the canonical git-blob bytes — line-ending
independent, computed by `canonical_sha256` on the committed files):

| File (in-repo) | SHA-256 | Bytes |
|---|---|---|
| `results/academy_v01/memory_v06/memory_v06_summary.json` | `56f16c2838d3e0aed5c6d4c072161f856617968b40f9269514d19fb02c01dc70` | 289 024 |
| `results/academy_v01/memory_v06/memory_v06_runs.jsonl` | `596ebecb33c6f88b0730f8494c0f51d4e25943bc4ffd21f94f4179e4fcaa0894` | 230 124 |
| `results/academy_v01/memory_v06/fresh_control.json` | `9313894fb53c9a2b422bd5f19bb0931b9da76ba2c7c3168f1c20d6baa0c104b3` | 5 698 |
| `results/academy_v01/memory_v06/evren_pool.json` | `578f0cd2d7f1245db250209e8c20c365d631d28bc7fff515c776bc9cd26cb797` | 19 464 |
| `results/academy_v01/memory_v06/evren_v06_summary.json` | `1062a3e7dc4ce5effca198f7d15b54ea0befab58d0b66f63434b89927c396ac4` | 61 205 |
| `generalization_audit/manifests/memory_v06.manifest.json` | stamped (see artifact entry) | 4 218 |
| `generalization_audit/manifests/memory_v06_evren.manifest.json` | stamped (see artifact entry) | 2 924 |

## 3. Exact experiment configuration

- **Python / hash:** 3.11.0; `PYTHONHASHSEED=0` (both training and
  evaluation; the chunk runner sets it per subprocess; control
  computation in `memory_merge_v06.py` likewise requires it).
- **Train seeds:** [20261107, 20261108]; 160 episodes per seed,
  deterministic streams (`make_stream`), 0 noise.
- **Evaluation grid (classic held-out):** 16 configs sampled
  deterministically (`sample_substrate`, seed 20261108), 4 episodes per
  config × 2 eval seeds per config; matched across variants.
- **OOD probe:** 12 configs (seed 20261110), 3 episodes per seed.
- **Continual probe:** first 6 OOD configs (seed 20261113), 3 episodes
  per seed, learning enabled.
- **Retention substrate:** 8 train-pool configs (seed 20261138), 2
  episodes per config, 2 snapshots.
- **Frozen integrity:** frozen cells disable `learn()` / replay and
  assert memory `generated` + `step` + agent `steps` are unchanged.
- **Comparisons:** paired bootstrap on per-config vectors, seed
  20261108, 2000 rounds, α=0.05, **candidate − baseline** convention
  (v0.5's reversed-convention defect corrected; historical v0.5 raw
  evidence untouched).
- **Variants & kwargs:** see `VARIANTS` in
  `academy/tools/memory_experiment_v06.py` (fresh, original,
  memory_wiped, ctx, ctx_full, bank, bank_ctx, bank_ctx_full, reset,
  atomtx, atomtx_bank).
- **Controls:** `random_walk` (pure random, no planning) and
  `novocab` (planning intact, vocabulary probing disabled), computed
  in `memory_merge_v06.py` on the same frozen grid.

### 3.1 EVREN phase

- Teacher-pool artifact: 24 approved tasks (OfflineTaskGenerator +
  TaskValidator; zero API calls; quota policy documented in
  `evren_pool.json`).
- Training: 30% teacher-task episodes mixed into the same per-seed
  streams (seed-matched); evaluation identical to the main grid +
  held-out teacher tasks (6) + train-reuse diagnostic (6).

## 4. Regeneration

```bash
PYTHONHASHSEED=0 python x/run_chunks_v06.py            # main benchmark
PYTHONHASHSEED=0 python academy/tools/memory_experiment_v06_evren.py
PYTHONHASHSEED=0 python academy/tools/memory_merge_v06.py
# verify the committed manifests (requires Quality v0.5 tooling on main):
python -m generalization_audit.tools.verify_experiment \
    --manifest generalization_audit/manifests/memory_v06.manifest.json --json
python -m generalization_audit.tools.verify_experiment \
    --manifest generalization_audit/manifests/memory_v06_evren.manifest.json --json
# provenance-aware publication gate (hotfix #39):
python -c "from pathlib import Path; from generalization_audit.verification.artifact_gate import check_frozen_target_artifacts; [print(r.target, r.overall) for r in check_frozen_target_artifacts(Path('.').resolve(), (Path('.')/'generalization_audit/manifests').resolve())]"
```

The verifier recomputes raw→summary means, comparison directions, and
paired-bootstrap CIs from the committed vectors (77 PASS on the main
manifest; gate PASS).
