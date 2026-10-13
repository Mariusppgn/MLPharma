"""Bundle the existing UI and a train-only snapshot into one double-clickable HTML."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..graph.preprocessing import chemical_properties
from ..text_features import TEXT_FIELDS, normalize_biomedical_text
from .datastore import ProfileStore
from .labels import LABELS
from .statistics import number
from .structures import molecular_structure

ASSETS = Path(__file__).parent / "assets"


def prepare_snapshot(store: ProfileStore) -> dict:
    if store.imported:
        raise ValueError(
            "L'export autonome est réservé au train et à ses OOF, sans profils externes."
        )
    pool, text_indices, normalized = [], {}, {}

    def intern(raw):
        key = (type(raw).__name__, str(raw))
        if key not in normalized:
            normalized[key] = normalize_biomedical_text(raw)
        text = normalized[key]
        if text not in text_indices:
            text_indices[text] = len(pool)
            pool.append(text)
        return text_indices[text]

    field_arrays = {
        side: [store.frame[f"{field}_{side}"].tolist() for field in TEXT_FIELDS]
        for side in ("A", "B")
    }
    texts, variants = [], defaultdict(set)
    for i, ids in enumerate(store.row_ids):
        sides = []
        for side, mid in zip(("A", "B"), ids, strict=True):
            indices = [intern(values[i]) for values in field_arrays[side]]
            sides.append(indices)
            # For A+A, existing pharmacology() uses the A-side record consistently.
            if side == "A" or ids[0] != ids[1]:
                variants[mid].add(tuple(indices))
        texts.append(sides)
    rows = [
        {
            "pair_id": str(store.frame.iloc[i].Pair_ID),
            "ids": ids,
            "binary": store.binary[i].astype(int).tolist(),
            "prr": [number(v) for v in store.prr[i]],
            "oof": store.oof[i].tolist() if store.oof is not None else None,
        }
        for i, ids in enumerate(store.row_ids)
    ]
    pairs = {}
    totals = {
        mid: {
            "n": np.zeros(len(store.effects), dtype=int),
            "k": np.zeros(len(store.effects), dtype=int),
        }
        for mid in store.molecules
    }
    for key, indices in store.pair_rows.items():
        block = store.binary[indices]
        agrees = block.min(axis=0) == block.max(axis=0)
        vote = [int(v) if valid else None for v, valid in zip(block[0], agrees, strict=True)]
        pairs["\x1f".join(key)] = {"rows": indices, "vote": vote}
        for mid in set(key):
            totals[mid]["n"] += agrees.astype(int)
            totals[mid]["k"] += np.where(agrees, block[0], 0).astype(int)
    catalog = store.catalog()
    a, b = catalog["default"]
    limitations = store.compare({"a": a, "b": b})["limitations"]
    return {
        "schema_version": 1,
        "catalog": catalog,
        "effects": [{"id": e, "label": LABELS.get(e, e.replace("_", " "))} for e in store.effects],
        "field_names": list(TEXT_FIELDS),
        "text_pool": pool,
        "texts": texts,
        "rows": rows,
        "pairs": pairs,
        "totals": {
            mid: {key: values.tolist() for key, values in totals[mid].items()} for mid in totals
        },
        "molecules": {
            mid: {
                **meta,
                "text_variants": len(variants[mid]),
                "properties": chemical_properties(meta["smiles"]),
                "structure": molecular_structure(meta["smiles"]),
            }
            for mid, meta in store.molecules.items()
        },
        "limitations": limitations,
    }


def generate_offline(store: ProfileStore, output: Path) -> dict:
    snapshot = prepare_snapshot(store)
    data = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    compressed = gzip.compress(data.encode("utf-8"), compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    js = (ASSETS / "app.js").read_text(encoding="utf-8")
    offline = (ASSETS / "offline.js").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="/style.css">', f"<style>{css}</style>")
    html = html.replace('<script src="/app.js" defer></script>', "")
    html = html.replace("Local • lecture seule", "HTML autonome • hors ligne")
    html = html.replace("Chargement des données locales…", "Décompression des données embarquées…")
    html = html.replace(
        "</head>",
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src blob: data:; "
        "connect-src 'none'; object-src 'none'; base-uri 'none'\">\n</head>",
    )
    scripts = offline + "\n" + js
    scripts = scripts.replace("</script", "<\\/script")
    html = html.replace(
        "</body>",
        f'<script id="deep-pharma-snapshot" type="application/octet-stream">{encoded}</script>'
        f"<script>{scripts}</script></body>",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return {
        "html": str(output),
        "html_bytes": output.stat().st_size,
        "uncompressed_data_bytes": len(data.encode("utf-8")),
        "rows": len(rows := snapshot["rows"]),
        "canonical_pairs": len(snapshot["pairs"]),
        "text_entries": len(snapshot["text_pool"]),
        "oof_available": rows[0]["oof"] is not None,
        "train_sha256": store.train_hash,
    }


def main():
    parser = argparse.ArgumentParser(description="Build a standalone combination explorer HTML")
    parser.add_argument("--config", type=Path, default=Path("configs/combination_profiles.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/combination_profiles/DeepPharma_Combinaisons.html"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent.parent

    def path(key):
        return root / config[key] if config.get(key) else None

    store = ProfileStore(path("train"), oof_dir=path("oof_dir"), manifest=path("manifest"))
    result = generate_offline(store, args.output)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
