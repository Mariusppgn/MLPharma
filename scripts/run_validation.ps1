$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "work\.uv-cache"
$RegistryPath = Join-Path $ProjectRoot "artifacts\submission_registry.json"
$BaselineMetrics = Join-Path $ProjectRoot "artifacts\runs\smoke_baseline_v1\metrics.json"

Push-Location $ProjectRoot
try {
    & ".\.venv\Scripts\deep-pharma.exe" build-validation
    if ($LASTEXITCODE -ne 0) {
        throw "La construction ou l'audit des splits a échoué."
    }

    & ".\.venv\Scripts\deep-pharma.exe" run-smoke-oof
    if ($LASTEXITCODE -ne 0) {
        throw "Le baseline OOF de contrôle a échoué."
    }

    if (Test-Path -LiteralPath $RegistryPath) {
        $ExistingRegistry = Get-Content -LiteralPath $RegistryPath -Raw | ConvertFrom-Json
        $CurrentMetrics = Get-Content -LiteralPath $BaselineMetrics -Raw | ConvertFrom-Json
        if (($ExistingRegistry.schema_version -ne 2) -or ($ExistingRegistry.protocol_id -ne $CurrentMetrics.protocol_id)) {
            $Timestamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
            $ArchivedRegistry = Join-Path $ProjectRoot "artifacts\submission_registry.protocol-change-$Timestamp.json"
            Move-Item -LiteralPath $RegistryPath -Destination $ArchivedRegistry
            Write-Host "Ancien registre archivé car son schéma ou protocole a changé : $ArchivedRegistry"
        }
    }

    if (-not (Test-Path -LiteralPath $RegistryPath)) {
        & ".\.venv\Scripts\deep-pharma.exe" init-gate-reference $BaselineMetrics
        if ($LASTEXITCODE -ne 0) {
            throw "L'initialisation du verrou de soumission a échoué."
        }
    }
    else {
        Write-Host "Registre de soumission existant conservé : $RegistryPath"
    }
}
finally {
    Pop-Location
}
