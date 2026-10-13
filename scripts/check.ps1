$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "work\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot "work\.uv-python"

Push-Location $ProjectRoot
try {
    uv run python -m pytest
    uv run ruff check .
}
finally {
    Pop-Location
}

