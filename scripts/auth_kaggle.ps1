$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:KAGGLE_CONFIG_DIR = Join-Path $ProjectRoot "work\.kaggle"

Push-Location $ProjectRoot
try {
    & ".\.venv\Scripts\kaggle.exe" auth login
    if ($LASTEXITCODE -ne 0) {
        throw "L'authentification Kaggle a échoué."
    }
}
finally {
    Pop-Location
}

