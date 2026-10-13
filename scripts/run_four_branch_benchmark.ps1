$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

& .\.venv\Scripts\deep-pharma.exe run-four-branch-benchmark `
    --config configs\four_branch_benchmark.yaml
