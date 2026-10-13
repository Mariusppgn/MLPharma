$ErrorActionPreference = "Stop"

& .\.venv\Scripts\python.exe -m deep_pharma.cli run-prr-rupture-v2 `
    --config configs\prr_rupture_v2.yaml

