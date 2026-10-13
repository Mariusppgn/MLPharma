from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .metric import hardcore_score
from .validation import audit_pair_grouped_folds, split_fingerprint


def _positive_prr_means(frame: pd.DataFrame, prr_columns: list[str]) -> np.ndarray:
    values = frame[prr_columns].to_numpy(dtype=float)
    positive = values > 0
    sums = np.where(positive, values, 0.0).sum(axis=0)
    counts = positive.sum(axis=0)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def run_smoke_oof(
    train: pd.DataFrame,
    manifest: pd.DataFrame,
    split_audit: dict[str, object],
    *,
    output_dir: Path,
    binary_prevalence_threshold: float = 0.20,
) -> dict[str, object]:
    """Exercise the OOF plumbing with fold-fitted constant predictors.

    This is deliberately a smoke baseline, not a competitive model. Every fitted
    statistic is computed from the training portion of its fold only.
    """

    if not 0.0 <= binary_prevalence_threshold <= 1.0:
        raise ValueError("Le seuil de prévalence doit être compris entre 0 et 1.")
    current_audit = audit_pair_grouped_folds(train, manifest)
    if not current_audit["passed"] or not split_audit.get("passed", False):
        raise ValueError("Audit de fuite invalide; le calcul OOF est bloqué.")
    fingerprint = split_fingerprint(manifest)
    if fingerprint != split_audit.get("fingerprint"):
        raise ValueError("Le manifeste de folds ne correspond pas à son empreinte d'audit.")

    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    oof = pd.concat(
        [
            train[[ID_COLUMN]].astype(str).reset_index(drop=True),
            pd.DataFrame({SEVERITY_COLUMN: pd.Series(pd.NA, index=train.index, dtype="string")}),
            pd.DataFrame(np.nan, index=train.index, columns=binary_columns),
            pd.DataFrame(np.nan, index=train.index, columns=prr_columns),
        ],
        axis=1,
    )
    predicted = np.zeros(len(train), dtype=bool)
    fold_metrics: list[dict[str, object]] = []

    fold_values = manifest.sort_values("row_index")["fold"].to_numpy()
    for fold in sorted(np.unique(fold_values)):
        validation_mask = fold_values == fold
        fit_mask = ~validation_mask
        fit_frame = train.loc[fit_mask]
        validation_frame = train.loc[validation_mask]

        severity_prediction = str(fit_frame[SEVERITY_COLUMN].mode().iloc[0])
        binary_prediction = (
            fit_frame[binary_columns].mean().to_numpy() >= binary_prevalence_threshold
        ).astype(np.int8)
        prr_prediction = _positive_prr_means(fit_frame, prr_columns)

        oof.loc[validation_mask, SEVERITY_COLUMN] = severity_prediction
        oof.loc[validation_mask, binary_columns] = np.tile(
            binary_prediction, (int(validation_mask.sum()), 1)
        )
        oof.loc[validation_mask, prr_columns] = np.tile(
            prr_prediction, (int(validation_mask.sum()), 1)
        )
        predicted[validation_mask] = True

        fold_score = hardcore_score(
            validation_frame[SEVERITY_COLUMN],
            oof.loc[validation_mask, SEVERITY_COLUMN],
            validation_frame[binary_columns],
            oof.loc[validation_mask, binary_columns].astype(np.int8),
            validation_frame[prr_columns],
            oof.loc[validation_mask, prr_columns],
        )
        fold_metrics.append({"fold": int(fold), **asdict(fold_score)})

    if not predicted.all() or oof.drop(columns=[ID_COLUMN]).isna().any().any():
        raise RuntimeError(
            "La couverture OOF n'est pas complète ou contient des valeurs manquantes."
        )
    oof[binary_columns] = oof[binary_columns].astype(np.int8)
    overall_score = hardcore_score(
        train[SEVERITY_COLUMN],
        oof[SEVERITY_COLUMN],
        train[binary_columns],
        oof[binary_columns],
        train[prr_columns],
        oof[prr_columns],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "oof_predictions.csv"
    metrics_path = output_dir / "metrics.json"
    oof.to_csv(predictions_path, index=False)
    metrics: dict[str, object] = {
        "schema_version": 1,
        "run_id": "smoke_baseline_v1",
        "run_type": "infrastructure_smoke_baseline",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": split_audit["protocol_id"],
        "validation_fingerprint": fingerprint,
        "score": asdict(overall_score),
        "fold_scores": fold_metrics,
        "leakage_audit": {
            "passed": bool(current_audit["passed"]),
            "pair_group_overlap_max": max(
                int(fold["pair_group_overlap"]) for fold in current_audit["folds"]
            ),
            "coverage_complete": bool(current_audit["coverage_complete"]),
            "fit_statistics_computed_within_each_fold": True,
        },
        "data_scope": {
            "train_only_for_fit": True,
            "test_used_for_fit": False,
            "test_used_for_local_scoring": False,
        },
        "oof": {
            "expected_rows": len(train),
            "predicted_rows": int(predicted.sum()),
            "coverage_complete": bool(predicted.all()),
            "predictions_file": predictions_path.name,
        },
        "parameters": {
            "binary_prevalence_threshold": binary_prevalence_threshold,
            "note": "Baseline constante destinée à valider le pipeline, pas à être soumise.",
        },
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics
