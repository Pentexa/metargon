# Run the remaining confirmatory preservation cells after the L primary
# completes. Checkpointed JSONL logs + final JSON per cell.
# Usage (in repo root): powershell -File results/v13_confirmatory/run_remaining.ps1

$env:PYTHONHASHSEED = "0"

# F noisy_obs (preservation)
python python/benchmark_v13_confirmatory.py `
    --family f --conditions noisy_obs `
    --seeds 8 --episodes 100 --seed-base 21800 `
    --policies metargon12,metargon13 `
    --workers 3 `
    --jsonl results/v13_confirmatory/final_f_8x100.jsonl `
    --json results/v13_confirmatory/final_f_8x100.json

# H stable (preservation)
python python/benchmark_v13_confirmatory.py `
    --family h --conditions stable `
    --seeds 8 --episodes 100 --seed-base 21800 `
    --policies metargon12,metargon13 `
    --workers 3 `
    --jsonl results/v13_confirmatory/final_h_8x100.jsonl `
    --json results/v13_confirmatory/final_h_8x100.json

# R stable (preservation)
python python/benchmark_v13_confirmatory.py `
    --family r --conditions stable `
    --seeds 8 --episodes 100 --seed-base 21800 `
    --policies metargon12,metargon13 `
    --workers 3 `
    --jsonl results/v13_confirmatory/final_r_8x100.jsonl `
    --json results/v13_confirmatory/final_r_8x100.json

# R stable -nodc (exploratory, Mission D)
python python/benchmark_v13_confirmatory.py `
    --family r --conditions stable `
    --seeds 8 --episodes 100 --seed-base 21800 `
    --policies metargon12-nodc,metargon13-nodc `
    --workers 3 `
    --jsonl results/v13_confirmatory/final_r_nodc_8x100.jsonl `
    --json results/v13_confirmatory/final_r_nodc_8x100.json

Write-Host "ALL REMAINING CELLS DONE"
