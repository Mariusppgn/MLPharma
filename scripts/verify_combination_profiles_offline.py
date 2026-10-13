"""Prepare sampled Python ground truth to check the standalone JS adapter, never fit models."""

import json
import random
from pathlib import Path

from deep_pharma.combination_profiles.datastore import ProfileStore


def main():
    store = ProfileStore(
        Path("data/raw/train.csv").resolve(),
        oof_dir=Path("outputs/graph/effect_graph_v1").resolve(),
        manifest=Path("artifacts/validation/pair_grouped_folds.csv").resolve(),
    )
    catalog = store.catalog()
    keys = [tuple(catalog["default"])]
    keys += random.Random(2026).sample(list(store.pair_rows), 12)
    keys += [key for key, rows in store.pair_rows.items() if len(rows) > 1][:3]
    keys += [key for key in store.pair_rows if key[0] == key[1]][:2]
    cases = []
    for a, b in keys:
        choices = [
            {"source": "binary", "context": True},
            {
                "source": "binary",
                "context": True,
                "min_support": 5,
                "threshold": 0.1,
                "filters": {"minimum": 1, "sort": "value"},
            },
            {"source": "binary", "context": False},
            {"source": "prr", "threshold": 0, "filters": {"maximum": 10, "sort": "value"}},
            {"source": "oof", "threshold": 0.6, "filters": {"search": "céph"}},
            {"source": "binary", "context": True, "filters": {"delta_sign": "positive"}},
        ]
        for config in choices:
            request = {"a": a, "b": b, **config}
            expected = store.compare(request)
            cases.append(
                {
                    "request": request,
                    "expected": {key: expected[key] for key in ("rows", "summary", "selection")},
                }
            )
        # Verify source row, side-specific texts and duplicate variant explicitly.
        rows = store.pair_rows[tuple(sorted((a, b)))]
        request = {
            "a": b,
            "b": a,
            "source": "binary",
            "context": True,
            "pair_id": str(store.frame.iloc[rows[-1]].Pair_ID),
        }
        expected = store.compare(request)
        cases.append({"request": request, "expected": expected})
    output = Path("work/combination_profiles_offline_parity.json")
    output.write_text(json.dumps(cases, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "parity_cases": len(cases),
                "sampled_pairs": len(keys),
                "train_sha256": store.train_hash,
            }
        )
    )


if __name__ == "__main__":
    main()
