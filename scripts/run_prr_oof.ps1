$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"

Push-Location $ProjectRoot
try {
    & ".\.venv\Scripts\deep-pharma.exe" run-prr-oof --config configs/prr_oof.yaml
    if ($LASTEXITCODE -ne 0) {
        throw "L'évaluation OOF PRR a échoué."
    }
    Write-Host "OOF terminé. Le jeu de test n'a pas été chargé."
}
finally {
    Pop-Location
}
