param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tex = "reports/final_pipeline/rapport_pipeline_deep_pharma.tex"
$tmp = "tmp/pdfs/final_pipeline"
$output = "output/pdf/rapport_pipeline_deep_pharma_2026-10-01.pdf"
$compiler = (Get-Command xelatex -ErrorAction SilentlyContinue).Source
if (-not $compiler) {
    $compiler = "C:\texlive\2022\bin\win32\xelatex.exe"
}
if (-not (Test-Path -LiteralPath $compiler)) {
    throw "XeLaTeX introuvable. Installer TeX Live ou ajouter xelatex au PATH."
}

Push-Location $root
try {
    New-Item -ItemType Directory -Force $tmp, "output/pdf" | Out-Null
    for ($pass = 1; $pass -le 2; $pass++) {
        & $compiler -halt-on-error -interaction=nonstopmode "-output-directory=$tmp" $tex |
            Select-Object -Last 8
        if ($LASTEXITCODE -ne 0) {
            throw "Échec de la compilation LaTeX, passe $pass."
        }
    }
    Copy-Item -LiteralPath "$tmp/rapport_pipeline_deep_pharma.pdf" -Destination $output -Force
    Write-Host "Rapport PDF : $output"
} finally {
    Pop-Location
}
