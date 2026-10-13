"""Read-only real-data smoke check; writes only local research smoke artifacts."""

import json
import time
from pathlib import Path

from deep_pharma.combination_profiles.datastore import ProfileStore


def main():
    started = time.perf_counter()
    store = ProfileStore(
        Path("data/raw/train.csv"),
        oof_dir=Path("outputs/graph/effect_graph_v1"),
        manifest=Path("artifacts/validation/pair_grouped_folds.csv"),
    )
    catalog = store.catalog()
    a, b = catalog["default"]
    context = store.compare({"a": a, "b": b, "context": True})
    prr = store.compare({"a": a, "b": b, "source": "prr"})
    oof = store.compare({"a": a, "b": b, "source": "oof"})
    filtered = store.compare(
        {"a": a, "b": b, "context": True, "filters": {"minimum": 1, "sort": "value"}}
    )
    assert all(r["ab"]["value"] == 1 for r in filtered["rows"])
    assert len(context["rows"]) == len(prr["rows"]) == len(oof["rows"]) == 50
    assert all(r["reference"] is None and r["delta"] is None for r in context["rows"])
    assert all(r["a"]["value"] is None for r in prr["rows"])
    assert store.oof is not None, store.oof_status
    absent_pair = next(
        (a, m) for m in store.molecules if tuple(sorted((a, m))) not in store.pair_rows
    )
    absent = store.compare({"a": absent_pair[0], "b": absent_pair[1], "context": True})
    assert all(r["ab"]["value"] is None for r in absent["rows"])
    for response in (context, prr, oof, filtered, absent):
        response["request"] = {"filters": {}}
    artifact = {
        "catalog": catalog,
        "context": context,
        "prr": prr,
        "oof": oof,
        "filtered": filtered,
        "absent": absent,
    }
    output = Path("work/combination_profiles_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "effects": 50,
                "molecules": len(store.molecules),
                "filtered_effects": len(filtered["rows"]),
                "oof": store.oof_status,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
