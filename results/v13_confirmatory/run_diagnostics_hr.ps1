# Mission C/D diagnostics on confirmatory H/R seeds (8 x 40, instrumented
# v13 policy) — light add-on for regression classification.
# Usage (in repo root): powershell -File results/v13_confirmatory/run_diagnostics_hr.ps1

$env:PYTHONHASHSEED = "0"

python python/diagnose_v13.py --family h --conditions stable --seeds 8 `
    --episodes 40 --policy v13 --seed-base 21800 `
    --json results/v13_confirmatory/diag_h_stable_8x40.json
python python/diagnose_v13.py --family r --conditions stable --seeds 8 `
    --episodes 40 --policy v13 --seed-base 21800 `
    --json results/v13_confirmatory/diag_r_stable_8x40.json

Write-Host "ALL H/R DIAGNOSTICS DONE"
