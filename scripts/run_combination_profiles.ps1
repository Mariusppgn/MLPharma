param([int]$Port = 8766)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    & "$projectRoot\.venv\Scripts\python.exe" scripts/serve_combination_profiles.py --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Le comparateur s'est arrêté avec une erreur." }
} finally {
    Pop-Location
}
