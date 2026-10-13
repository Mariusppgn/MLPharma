"""Fetch the pinned Cytoscape dependency once; dashboards remain fully offline."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

VERSION = "3.33.1"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src/deep_pharma/graph/assets"

if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    receipts = []
    for remote, local in [
        ("dist/cytoscape.min.js", "cytoscape.min.js"),
        ("LICENSE", "CYTOSCAPE_LICENSE"),
    ]:
        url = f"https://raw.githubusercontent.com/cytoscape/cytoscape.js/v{VERSION}/{remote}"
        with urlopen(url, timeout=45) as response:
            data = response.read()
        (DESTINATION / local).write_bytes(data)
        receipts.append({"url": url, "file": local, "sha256": hashlib.sha256(data).hexdigest()})
    (DESTINATION / "vendor.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    print(f"Cytoscape {VERSION}: downloaded and pinned locally")
