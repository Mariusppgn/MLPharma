param(
    [switch]$FinalizeIfImproved
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$arguments = @(
    "run-lockbox-candidate",
    "--config", "configs\lockbox_candidate.yaml"
)
if ($FinalizeIfImproved) {
    $arguments += "--finalize-if-improved"
}

& .\.venv\Scripts\python.exe -m deep_pharma.cli @arguments
