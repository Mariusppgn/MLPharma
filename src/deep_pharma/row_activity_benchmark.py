from __future__ import annotations

import hashlib
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .four_branch_benchmark import _load_or_build_features
from .metric import hardcore_score
from .molecule_priors import (
    build_cross_fitted_molecule_priors,
    fit_molecule_prior_tables,
    transform_molecule_priors,
)
from .prr_pytorch import _feature_builder
from .submission import validate_submission
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_aligned(path: str, train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if not frame[ID_COLUMN].astype(str).equals(train[ID_COLUMN].astype(str)):
        raise ValueError(f"Predictions OOF desalignees: {path}")
    return frame


def _activity_features(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    active: np.ndarray,
    folds: np.ndarray,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = active[:, None].astype(np.float32)
    fit_raw = build_cross_fitted_molecule_priors(
        fit, targets, folds, smoothing=smoothing
    )
    tables = fit_molecule_prior_tables(fit, targets, smoothing=smoothing)
    validation_raw = transform_molecule_priors(validation, tables)
    scaler = StandardScaler().fit(fit_raw)
    return (
        scaler.transform(fit_raw).astype(np.float32),
        scaler.transform(validation_raw).astype(np.float32),
    )


def run_row_activity_benchmark(config_path: Path) -> dict[str, object]:
    """Evaluate a pair-level activity gate using grouped OOF predictions only."""

    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(Path(paths["audit"]))
    audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    if not audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Le manifeste OOF ne correspond pas a son audit.")
    folds = manifest["fold"].to_numpy()
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    targets = train[binary_columns].to_numpy(np.int8)
    active = targets.any(axis=1).astype(np.int8)
    prr_targets = train[prr_columns].to_numpy(np.float32)
    probabilities = np.full(len(train), np.nan, dtype=np.float32)
    for fold in sorted(int(value) for value in np.unique(folds)):
        held_out = folds == fold
        fit = train.loc[~held_out].reset_index(drop=True)
        validation = train.loc[held_out].reset_index(drop=True)
        fit_base, validation_base, _, _ = _load_or_build_features(
            train, prr_targets, folds, fold, fingerprint, config
        )
        fit_prior, validation_prior = _activity_features(
            fit,
            validation,
            active[~held_out],
            folds[~held_out],
            float(config["model"]["prior_smoothing"]),
        )
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=int(config["model"]["n_estimators"]),
            learning_rate=float(config["model"]["learning_rate"]),
            num_leaves=int(config["model"]["num_leaves"]),
            min_child_samples=int(config["model"]["min_child_samples"]),
            reg_lambda=float(config["model"]["reg_lambda"]),
            random_state=int(config["seed"]) + fold,
            n_jobs=int(config["model"]["n_jobs"]),
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(np.hstack([fit_base, fit_prior]), active[~held_out])
        probabilities[held_out] = model.predict_proba(
            np.hstack([validation_base, validation_prior])
        )[:, 1]
        print(f"Activity fold {fold} termine", flush=True)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Probabilites OOF incompletes.")

    binary_oof = _read_aligned(paths["binary_oof"], train)
    severity_oof = _read_aligned(paths["severity_oof"], train)
    old_prr = _read_aligned(paths["old_prr_oof"], train)
    new_prr = _read_aligned(paths["new_prr_oof"], train)
    reference_binary = binary_oof[binary_columns].to_numpy(np.int8)
    fixed_prr = 0.5 * (
        old_prr[prr_columns].to_numpy(np.float32)
        + new_prr[prr_columns].to_numpy(np.float32)
    )
    fixed_severity = severity_oof[SEVERITY_COLUMN].astype(str).to_numpy()
    rows = []
    for threshold in [None, *config["evaluation"]["thresholds"]]:
        binary = (
            reference_binary
            if threshold is None
            else reference_binary * (probabilities >= float(threshold))[:, None]
        )
        score = hardcore_score(
            train[SEVERITY_COLUMN],
            fixed_severity,
            train[binary_columns],
            binary,
            train[prr_columns],
            fixed_prr,
        )
        rows.append(
            {
                "threshold": "none" if threshold is None else float(threshold),
                "binary_micro_f1": score.side_effect_micro_f1,
                "total": score.total,
                "predicted_active_rows": int(binary.any(axis=1).sum()),
                "true_positive_cells": int((binary * targets).sum()),
                "false_positive_cells": int((binary * (1 - targets)).sum()),
            }
        )
    output_dir = Path(paths["output_dir"])
    run_dir = Path(paths["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], "active_probability": probabilities}).to_csv(
        output_dir / "oof_activity_probability.csv", index=False
    )
    pd.DataFrame(rows).to_csv(output_dir / "threshold_diagnostics.csv", index=False)
    summary = {
        "run_id": config["run_id"],
        "train_sha256": _sha256(train_path),
        "validation_fingerprint": fingerprint,
        "activity_roc_auc": float(roc_auc_score(active, probabilities)),
        "activity_average_precision": float(average_precision_score(active, probabilities)),
        "threshold_diagnostics": rows,
        "selected_threshold": None,
        "leakage_audit": {
            "passed": True,
            "pair_group_overlap_max": 0,
            "training_activity_priors_cross_fitted": True,
            "validation_activity_priors_fit_only": True,
            "test_loaded": False,
        },
        "automatic_submission": False,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def finalize_row_activity_candidate(config_path: Path) -> dict[str, object]:
    """Apply a fixed 0.5 activity gate to a validated Severity candidate."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(Path(paths["audit"]))
    audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    summary = load_json(Path(paths["run_dir"]) / "summary.json")
    if not audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Validation groupee invalide.")
    if summary["train_sha256"] != _sha256(train_path):
        raise ValueError("Le train a change depuis le benchmark d'activite.")
    if summary["validation_fingerprint"] != fingerprint:
        raise ValueError("Les folds ont change depuis le benchmark d'activite.")
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    targets = train[binary_columns].to_numpy(np.int8)
    active = targets.any(axis=1).astype(np.int8)
    folds = manifest["fold"].to_numpy()
    reference_oof = _read_aligned(paths["incumbent_binary_oof"], train)[
        binary_columns
    ].to_numpy(np.int8)
    activity_oof = _read_aligned(
        str(Path(paths["output_dir"]) / "oof_activity_probability.csv"), train
    )["active_probability"].to_numpy(np.float32)
    threshold = float(config["selection"]["activity_threshold"])
    gated_oof = reference_oof * (activity_oof >= threshold)[:, None]
    incumbent_f1 = float(f1_score(targets.ravel(), reference_oof.ravel()))
    candidate_f1 = float(f1_score(targets.ravel(), gated_oof.ravel()))
    fold_gains = []
    for fold in np.unique(folds):
        held_out = folds == fold
        fold_gains.append(
            float(
                f1_score(targets[held_out].ravel(), gated_oof[held_out].ravel())
                - f1_score(targets[held_out].ravel(), reference_oof[held_out].ravel())
            )
        )
    improvement = candidate_f1 - incumbent_f1
    if improvement < float(config["selection"]["minimum_binary_f1_gain"]):
        raise ValueError("La porte d'activite ne passe pas le minimum OOF.")
    if min(fold_gains) <= 0:
        raise ValueError("La porte d'activite ne gagne pas sur chaque fold.")

    # Selection is complete before test features or the existing CSV are read.
    test = pd.read_csv(paths["test"], low_memory=False)
    sample_path = Path(paths["sample_submission"])
    reference_path = Path(paths["severity_candidate_submission"])
    reference = pd.read_csv(reference_path, low_memory=False)
    if not reference[ID_COLUMN].astype(str).equals(test[ID_COLUMN].astype(str)):
        raise ValueError("Le candidat Severity et le test sont desalignees.")
    if validate_submission(reference_path, sample_path):
        raise ValueError("Le candidat Severity est invalide.")
    feature_config = yaml.safe_load(Path(paths["feature_cache_config"]).read_text("utf-8"))
    builder = _feature_builder(feature_config, int(config["seed"]))
    train_base_raw = builder.fit_transform(train)
    test_base_raw = builder.transform(test)
    base_scaler = StandardScaler().fit(train_base_raw)
    train_base = base_scaler.transform(train_base_raw).astype(np.float32)
    test_base = base_scaler.transform(test_base_raw).astype(np.float32)
    train_prior, test_prior = _activity_features(
        train,
        test,
        active,
        folds,
        float(config["model"]["prior_smoothing"]),
    )
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(config["model"]["n_estimators"]),
        learning_rate=float(config["model"]["learning_rate"]),
        num_leaves=int(config["model"]["num_leaves"]),
        min_child_samples=int(config["model"]["min_child_samples"]),
        reg_lambda=float(config["model"]["reg_lambda"]),
        random_state=int(config["seed"]),
        n_jobs=int(config["model"]["n_jobs"]),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(np.hstack([train_base, train_prior]), active)
    activity_probability = model.predict_proba(
        np.hstack([test_base, test_prior])
    )[:, 1]
    candidate = reference.copy()
    candidate[binary_columns] = (
        reference[binary_columns].to_numpy(np.int8)
        * (activity_probability >= threshold)[:, None]
    )
    candidate_path = Path(paths["candidate_submission"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_path, index=False)
    errors = validate_submission(candidate_path, sample_path)
    if errors:
        candidate_path.unlink(missing_ok=True)
        raise ValueError("Soumission candidate invalide : " + "; ".join(errors))
    if not candidate.drop(columns=binary_columns).equals(
        reference.drop(columns=binary_columns)
    ):
        raise AssertionError("Une colonne hors Binary a ete modifiee.")
    run_dir = Path(paths["run_dir"])
    model_path = run_dir / "final_activity_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_builder": builder,
            "base_scaler": base_scaler,
            "fold_fingerprint": fingerprint,
            "train_sha256": _sha256(train_path),
        },
        model_path,
        compress=3,
    )
    result = {
        "run_id": config["run_id"],
        "activity_threshold": threshold,
        "binary_micro_f1_incumbent_oof": incumbent_f1,
        "binary_micro_f1_gated_oof": candidate_f1,
        "binary_micro_f1_oof_gain": improvement,
        "fold_binary_f1_gains": fold_gains,
        "estimated_global_gain_from_binary": 0.3 * improvement,
        "reference_submission": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "candidate_submission": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "rows": len(candidate),
        "gated_test_rows": int((activity_probability < threshold).sum()),
        "removed_binary_positives": int(
            reference[binary_columns].to_numpy(np.int8).sum()
            - candidate[binary_columns].to_numpy(np.int8).sum()
        ),
        "severity_and_prr_identical": True,
        "model_artifact": str(model_path),
        "automatic_upload": False,
    }
    write_json(result, run_dir / "candidate_summary.json")
    return result
