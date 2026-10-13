from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .metric import hardcore_score
from .submission import validate_submission
from .submission_gate import decision_as_dict, evaluate_submission_gate, load_json, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_means(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=float)
    positive = values > 0
    return np.divide(
        np.where(positive, values, 0.0).sum(axis=0),
        positive.sum(axis=0),
        out=np.zeros(len(columns), dtype=float),
        where=positive.sum(axis=0) > 0,
    )


def build_constant_prr_hybrid(
    *,
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    folds_path: Path,
    parent_run_dir: Path,
    output_dir: Path,
    submission_path: Path,
    registry_path: Path,
) -> dict[str, object]:
    started_at = time.perf_counter()
    train = pd.read_csv(train_path, low_memory=False)
    folds = pd.read_csv(folds_path).sort_values("row_index").reset_index(drop=True)
    parent_oof = pd.read_csv(parent_run_dir / "oof_predictions.csv", low_memory=False)
    parent_metrics = load_json(parent_run_dir / "metrics.json")
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]

    constant_oof = np.zeros((len(train), len(prr_columns)), dtype=np.float32)
    for fold in sorted(folds["fold"].unique()):
        validation_mask = folds["fold"].to_numpy() == fold
        constant_oof[validation_mask] = _positive_means(train.loc[~validation_mask], prr_columns)

    selected_score = hardcore_score(
        train[SEVERITY_COLUMN],
        parent_oof[SEVERITY_COLUMN],
        train[binary_columns],
        parent_oof[binary_columns],
        train[prr_columns],
        constant_oof,
    )
    parent_score = float(parent_metrics["score"]["total"])
    if selected_score.total <= parent_score:
        raise ValueError("Le remplacement PRR constant n'améliore pas le modèle parent.")

    fold_scores = []
    for fold in sorted(folds["fold"].unique()):
        validation_mask = folds["fold"].to_numpy() == fold
        fold_score = hardcore_score(
            train.loc[validation_mask, SEVERITY_COLUMN],
            parent_oof.loc[validation_mask, SEVERITY_COLUMN],
            train.loc[validation_mask, binary_columns],
            parent_oof.loc[validation_mask, binary_columns],
            train.loc[validation_mask, prr_columns],
            constant_oof[validation_mask],
        )
        fold_scores.append({"fold": int(fold), **asdict(fold_score)})

    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid_oof = pd.concat(
        [
            parent_oof[[ID_COLUMN, SEVERITY_COLUMN]].reset_index(drop=True),
            parent_oof[binary_columns].reset_index(drop=True),
            pd.DataFrame(constant_oof, columns=prr_columns),
        ],
        axis=1,
    )
    hybrid_oof_path = output_dir / "oof_predictions.csv"
    hybrid_oof.to_csv(hybrid_oof_path, index=False)

    metrics = copy.deepcopy(parent_metrics)
    metrics["run_id"] = "multimodal_lgbm_extratrees_constant_prr_v1"
    metrics["run_type"] = "selected_hybrid_candidate"
    metrics["created_at"] = datetime.now(UTC).isoformat()
    metrics["score"] = asdict(selected_score)
    metrics["fold_scores"] = fold_scores
    metrics["parent_model_score"] = parent_metrics["score"]
    metrics["parent_run_id"] = parent_metrics["run_id"]
    metrics["prr_selection"] = {
        "selected": "fold_local_positive_mean",
        "rejected": "extra_trees_log1p_regression",
        "selection_basis": "same pair-grouped OOF rows",
        "candidate_scores": {
            "fold_local_positive_mean": selected_score.prr_score,
            "extra_trees_log1p_regression": parent_metrics["score"]["prr_score"],
        },
    }
    metrics["oof"]["predictions_file"] = hybrid_oof_path.name
    metrics["oof"]["predictions_sha256"] = _sha256(hybrid_oof_path)
    metrics["automatic_submission"] = False
    metrics["parent_training_duration_seconds"] = parent_metrics["duration_seconds"]
    metrics["hybrid_postprocessing_seconds"] = time.perf_counter() - started_at
    metrics.pop("cold_drug_score", None)
    metrics["cold_drug_parent_model_score"] = parent_metrics["cold_drug_score"]
    metrics["cold_drug_hybrid_not_run"] = True
    metrics_path = output_dir / "metrics.json"
    write_json(metrics, metrics_path)

    gate = decision_as_dict(evaluate_submission_gate(metrics, load_json(registry_path)))
    write_json(gate, output_dir / "gate_decision.json")
    if not gate["eligible"]:
        return {"metrics": metrics, "gate": gate, "submission": None}

    test = pd.read_csv(test_path, low_memory=False)
    parent_submission = pd.read_csv(submission_path, low_memory=False)
    full_train_prr = _positive_means(train, prr_columns)
    parent_submission[prr_columns] = np.tile(full_train_prr, (len(test), 1))
    sample = pd.read_csv(sample_path, low_memory=False)
    parent_submission = parent_submission[list(sample.columns)]
    parent_submission.to_csv(submission_path, index=False)
    errors = validate_submission(submission_path, sample_path)
    if errors:
        raise ValueError("Soumission hybride invalide : " + "; ".join(errors))

    constants = {
        column: float(value) for column, value in zip(prr_columns, full_train_prr, strict=True)
    }
    write_json(constants, output_dir / "prr_positive_means.json")
    validation = {
        "valid": True,
        "rows": len(parent_submission),
        "columns": len(parent_submission.columns),
        "submission_sha256": _sha256(submission_path),
        "selected_prr_strategy": "full_train_positive_mean",
        "prr_min": float(parent_submission[prr_columns].min().min()),
        "prr_max": float(parent_submission[prr_columns].max().max()),
        "uploaded_to_kaggle": False,
    }
    write_json(validation, output_dir / "submission_validation.json")
    return {"metrics": metrics, "gate": gate, "submission": validation}
