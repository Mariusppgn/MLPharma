$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "work\.uv-cache"

Push-Location $ProjectRoot
try {
    & ".\.venv\Scripts\deep-pharma.exe" model-audit
    if ($LASTEXITCODE -ne 0) {
        throw "L'audit approfondi a échoué."
    }

    & ".\.venv\Scripts\deep-pharma.exe" train-candidate
    if ($LASTEXITCODE -eq 3) {
        Write-Host "Le modèle n'améliore pas la référence : aucun fichier de soumission créé."
        exit 0
    }
    if ($LASTEXITCODE -ne 0) {
        throw "L'entraînement du candidat a échoué."
    }

    & ".\.venv\Scripts\deep-pharma.exe" build-hybrid-candidate
    if ($LASTEXITCODE -ne 0) {
        throw "La sélection du candidat hybride a échoué."
    }
    Write-Host "Candidat local construit et validé. Aucun envoi Kaggle n'a été effectué."
}
finally {
    Pop-Location
}
