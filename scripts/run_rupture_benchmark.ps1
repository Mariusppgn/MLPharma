param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$config = if ($Quick) { "configs\rupture_benchmark_quick.yaml" } else { "configs\rupture_benchmark.yaml" }

& .\.venv\Scripts\python.exe -m deep_pharma.cli run-rupture-benchmark --config $config
