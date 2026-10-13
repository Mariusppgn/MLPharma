$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"

Push-Location $ProjectRoot
try {
    & ".\.venv\Scripts\deep-pharma.exe" compare-prr-pytorch
    if ($LASTEXITCODE -ne 0) {
        throw "La comparaison PyTorch PRR a échoué."
    }
    Write-Host "Comparaison terminée. Le test n'a été lu qu'après sélection de la méthode."
}
finally {
    Pop-Location
}
