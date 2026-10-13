[CmdletBinding()]
param(
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "work\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot "work\.uv-python"

if (-not $PythonExecutable) {
    $PythonExecutable = "3.12"
}

Push-Location $ProjectRoot
try {
    uv sync --python $PythonExecutable
    uv run python -m pytest
    uv run ruff check .
    Write-Host "Environnement prêt. Voir README.md et docs/reproduction.md."
}
finally {
    Pop-Location
}

