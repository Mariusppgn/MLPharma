from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from deep_pharma.constants import ID_COLUMN, PRR_PREFIX
from deep_pharma.hierarchical_prr import HierarchicalPRRConfig, cross_fit_hierarchical_prr
from deep_pharma.prr_pytorch import masked_prr_metrics
from deep_pharma.submission_gate import load_json, write_json
from deep_pharma.validation import audit_pair_grouped_folds, split_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge", type=float, default=20.0)
    parser.add_argument("--svd-rank", type=int, default=10)
    parser.add_argument("--svd-mix", type=float, default=0.25)
    args = parser.parse_args()
    train = pd.read_csv("data/raw/train.csv", low_memory=False)
    manifest = pd.read_csv("artifacts/validation/pair_grouped_folds.csv")
    manifest = manifest.sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    stored = load_json(Path("artifacts/validation/pair_grouped_audit.json"))
    fingerprint = split_fingerprint(manifest)
    if not audit["passed"] or fingerprint != stored["fingerprint"]:
        raise ValueError("Validation groupee invalide.")
    columns = [column for column in train if column.startswith(PRR_PREFIX)]
    targets = train[columns].to_numpy(np.float32)
    config = HierarchicalPRRConfig(
        ridge=args.ridge, svd_rank=args.svd_rank, svd_mix=args.svd_mix
    )
    predictions = cross_fit_hierarchical_prr(
        train, targets, manifest["fold"].to_numpy(), config=config
    )
    output = Path("outputs/hierarchical_prr")
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [train[[ID_COLUMN]], pd.DataFrame(predictions, columns=columns)], axis=1
    ).to_csv(output / "oof_predictions.csv", index=False)
    incumbent = pd.read_csv("outputs/prr_rupture_v2/oof_predictions_incumbent_blend.csv")
    adaptive = pd.read_csv("outputs/prr_rupture_v2/oof_predictions_adaptive_moe.csv")
    if not incumbent[ID_COLUMN].astype(str).equals(train[ID_COLUMN].astype(str)):
        raise ValueError("Incumbent OOF desaligne.")
    if not adaptive[ID_COLUMN].astype(str).equals(train[ID_COLUMN].astype(str)):
        raise ValueError("Adaptive OOF desaligne.")
    old = incumbent[columns].to_numpy(np.float32)
    new = adaptive[columns].to_numpy(np.float32)
    cases = {
        "hierarchical": predictions,
        "incumbent": old,
        "adaptive": new,
        "equal_old_new": 0.5 * old + 0.5 * new,
        "equal_with_hierarchical_10pct": 0.45 * old + 0.45 * new + 0.1 * predictions,
        "equal_with_hierarchical_20pct": 0.4 * old + 0.4 * new + 0.2 * predictions,
    }
    rows = [
        {"candidate": name, **asdict(masked_prr_metrics(targets, value))}
        for name, value in cases.items()
    ]
    pd.DataFrame(rows).sort_values("sprr", ascending=False).to_csv(
        output / "comparison.csv", index=False
    )
    result = {
        "validation_fingerprint": fingerprint,
        "config": asdict(config),
        "comparison": rows,
        "leakage_audit": {
            "passed": True,
            "pair_group_overlap_max": 0,
            "all_predictions_oof": True,
            "test_loaded": False,
        },
        "automatic_submission": False,
    }
    write_json(result, output / "summary.json")
    print(pd.DataFrame(rows).sort_values("sprr", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
