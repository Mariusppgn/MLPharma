[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:KAGGLE_CONFIG_DIR = Join-Path $ProjectRoot "work\.kaggle"

Push-Location $ProjectRoot
try {
    $Arguments = @(
        "competitions",
        "download",
        "deep-pharma-multimodal-intelligence-challenge-data-hack-3-0",
        "--path",
        "data\raw",
        "--unzip"
    )
    if ($Force) {
        $Arguments += "--force"
    }
    & ".\.venv\Scripts\kaggle.exe" @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Le téléchargement Kaggle a échoué. Avez-vous accepté les règles et lancé scripts\auth_kaggle.ps1 ?"
    }
    & ".\.venv\Scripts\deep-pharma.exe" inspect-data --write-report reports\data_contract.json
    if ($LASTEXITCODE -ne 0) {
        throw "Le téléchargement a réussi, mais le contrat de données est invalide."
    }
}
finally {
    Pop-Location
}
