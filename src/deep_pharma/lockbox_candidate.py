from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .four_branch_benchmark import _run_branch_c
from .metric import HardcoreScore, hardcore_score
from .modeling import _build_models
from .modeling import _feature_builder as baseline_feature_builder
from .molecule_priors import (
    build_cross_fitted_molecule_priors,
    fit_molecule_prior_tables,
    transform_molecule_priors,
)
from .prr_oof import _fit_convex_blend_weights
from .prr_pytorch import (
    _device as prr_device,
)
from .prr_pytorch import (
    _feature_builder as prr_feature_builder,
)
from .prr_pytorch import (
    _target_scales,
    fit_prr_fixed_epochs,
    predict_prr_arrays,
    train_prr_method,
)
from .submission import validate_submission
from .submission_gate import write_json
from .validation import build_pair_grouped_folds, frame_pair_keys, split_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(
    frame: pd.DataFrame,
    severity: np.ndarray,
    binary: np.ndarray,
    prr: np.ndarray,
    binary_columns: list[str],
    prr_columns: list[str],
) -> HardcoreScore:
    return hardcore_score(
        frame[SEVERITY_COLUMN],
        severity,
        frame[binary_columns],
        binary,
        frame[prr_columns],
        prr,
    )


def _fit_severity(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    config: dict[str, object],
    seed: int,
) -> np.ndarray:
    builder = baseline_feature_builder(config, seed)
    fit_features = builder.fit_transform(fit)
    validation_features = builder.transform(validation)
    severity_model, _, _ = _build_models(config, seed)
    severity_model.fit(fit_features, fit[SEVERITY_COLUMN])
    return severity_model.predict(validation_features)


def _prr_features(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    fit_targets: np.ndarray,
    fit_folds: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    builder = prr_feature_builder(config, int(config["seed"]))
    fit_base_raw = builder.fit_transform(fit)
    validation_base_raw = builder.transform(validation)
    base_scaler = StandardScaler().fit(fit_base_raw)
    fit_base = base_scaler.transform(fit_base_raw).astype(np.float32)
    validation_base = base_scaler.transform(validation_base_raw).astype(np.float32)

    smoothing = float(config["molecule_priors"]["smoothing"])
    fit_priors_raw = build_cross_fitted_molecule_priors(
        fit,
        fit_targets,
        fit_folds,
        smoothing=smoothing,
    )
    tables = fit_molecule_prior_tables(fit, fit_targets, smoothing=smoothing)
    validation_priors_raw = transform_molecule_priors(validation, tables)
    prior_scaler = StandardScaler().fit(fit_priors_raw)
    fit_priors = prior_scaler.transform(fit_priors_raw).astype(np.float32)
    validation_priors = prior_scaler.transform(validation_priors_raw).astype(np.float32)
    scales = _target_scales(
        fit_targets,
        float(config["training"]["target_scale_quantile"]),
    )
    return (
        fit_base,
        validation_base,
        np.hstack([fit_base, fit_priors]).astype(np.float32),
        np.hstack([validation_base, validation_priors]).astype(np.float32),
        scales,
    )


def _select_prr_epochs(
    development: pd.DataFrame,
    targets: np.ndarray,
    inner_folds: np.ndarray,
    config: dict[str, object],
    run_dir: Path,
) -> dict[str, int]:
    selection_fold = int(sorted(np.unique(inner_folds))[0])
    validation_mask = inner_folds == selection_fold
    core = development.loc[~validation_mask].reset_index(drop=True)
    validation = development.loc[validation_mask].reset_index(drop=True)
    core_targets = targets[~validation_mask]
    validation_targets = targets[validation_mask]
    matrices = _prr_features(
        core,
        validation,
        core_targets,
        inner_folds[~validation_mask],
        config,
    )
    device = prr_device(config["training"])
    epochs: dict[str, int] = {}
    variants = {
        "base_huber": (matrices[0], matrices[1]),
        "molecule_priors_huber": (matrices[2], matrices[3]),
    }
    for name, (fit_features, validation_features) in variants.items():
        result = train_prr_method(
            "huber",
            fit_features,
            core_targets,
            validation_features,
            validation_targets,
            matrices[4],
            config,
            run_dir / "epoch_selection" / name,
            device,
        )
        epochs[name] = int(result["best_epoch"])
    return epochs


def _fit_blended_prr(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    fit_targets: np.ndarray,
    fit_folds: np.ndarray,
    config: dict[str, object],
    selected_epochs: dict[str, int],
    run_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    matrices = _prr_features(fit, validation, fit_targets, fit_folds, config)
    device = prr_device(config["training"])
    fit_predictions: dict[str, np.ndarray] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    variants = {
        "base_huber": (matrices[0], matrices[1]),
        "molecule_priors_huber": (matrices[2], matrices[3]),
    }
    for name, (fit_features, validation_features) in variants.items():
        result = fit_prr_fixed_epochs(
            "huber",
            fit_features,
            fit_targets,
            matrices[4],
            config,
            run_dir / name,
            device,
            epochs=selected_epochs[name],
        )
        fit_predictions[name] = predict_prr_arrays(
            result, fit_features, matrices[4], config, device
        )["selected"]
        validation_predictions[name] = predict_prr_arrays(
            result, validation_features, matrices[4], config, device
        )["selected"]
    weights = _fit_convex_blend_weights(
        fit_predictions["molecule_priors_huber"],
        fit_predictions["base_huber"],
        fit_targets,
    )
    prediction = (
        weights * validation_predictions["molecule_priors_huber"]
        + (1.0 - weights) * validation_predictions["base_huber"]
    )
    return np.maximum(prediction.astype(np.float32), 0.0), {
        "selected_epochs": selected_epochs,
        "mean_prior_weight": float(weights.mean()),
        "device": str(device),
    }


def _pair_overlap(fit: pd.DataFrame, validation: pd.DataFrame) -> int:
    return len(set(frame_pair_keys(fit)) & set(frame_pair_keys(validation)))


def _is_improved(
    candidate: HardcoreScore,
    control: HardcoreScore,
    minimum: float,
) -> tuple[bool, float]:
    improvement = candidate.total - control.total
    return improvement > minimum + 1e-12, improvement


def run_lockbox_candidate(
    config_path: Path,
    *,
    finalize_if_improved: bool = False,
) -> dict[str, object]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    paths = config["paths"]
    run_dir = Path(paths["run_dir"])
    output_dir = Path(paths["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # No test path is read before the lockbox decision below.
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    prr_targets = train[prr_columns].to_numpy(dtype=np.float32)

    lockbox_manifest, lockbox_audit = build_pair_grouped_folds(
        train,
        n_splits=int(config["lockbox"]["n_splits"]),
        seed=int(config["lockbox"]["seed"]),
    )
    lockbox_fold = int(config["lockbox"]["fold"])
    lockbox_mask = lockbox_manifest["fold"].to_numpy() == lockbox_fold
    development = train.loc[~lockbox_mask].reset_index(drop=True)
    lockbox = train.loc[lockbox_mask].reset_index(drop=True)
    development_targets = prr_targets[~lockbox_mask]
    lockbox_manifest.assign(
        role=np.where(lockbox_mask, "lockbox", "development")
    ).to_csv(run_dir / "lockbox_manifest.csv", index=False)

    inner_manifest, inner_audit = build_pair_grouped_folds(
        development,
        n_splits=int(config["inner_validation"]["n_splits"]),
        seed=int(config["inner_validation"]["seed"]),
    )
    inner_folds = inner_manifest["fold"].to_numpy()
    inner_manifest.to_csv(run_dir / "inner_manifest.csv", index=False)

    baseline_config = yaml.safe_load(Path(paths["baseline_config"]).read_text(encoding="utf-8"))
    prr_config = yaml.safe_load(Path(paths["prr_config"]).read_text(encoding="utf-8"))
    severity = _fit_severity(development, lockbox, baseline_config, seed)
    _, c_binary, c_prr, c_metadata = _run_branch_c(
        development,
        lockbox,
        development[binary_columns].to_numpy(dtype=np.int8),
        development_targets,
        inner_folds,
        lockbox_fold,
        config["branch_c"],
        seed,
    )
    selected_epochs = _select_prr_epochs(
        development,
        development_targets,
        inner_folds,
        prr_config,
        run_dir,
    )
    candidate_prr, prr_metadata = _fit_blended_prr(
        development,
        lockbox,
        development_targets,
        inner_folds,
        prr_config,
        selected_epochs,
        run_dir / "lockbox_refit",
    )
    candidate_score = _score(
        lockbox, severity, c_binary, candidate_prr, binary_columns, prr_columns
    )
    control_score = _score(lockbox, severity, c_binary, c_prr, binary_columns, prr_columns)
    eligible, improvement = _is_improved(
        candidate_score,
        control_score,
        float(config["selection"]["minimum_total_improvement"]),
    )
    pair_overlap = _pair_overlap(development, lockbox)
    leakage_passed = bool(
        lockbox_audit["passed"] and inner_audit["passed"] and pair_overlap == 0
    )
    eligible = eligible and leakage_passed
    decision = {
        "eligible": eligible,
        "candidate_total": candidate_score.total,
        "control_total": control_score.total,
        "improvement": improvement,
        "minimum_improvement": float(config["selection"]["minimum_total_improvement"]),
        "reason": "candidate_improves_grouped_lockbox"
        if eligible
        else "candidate_does_not_pass_grouped_lockbox",
    }
    summary: dict[str, object] = {
        "run_id": config["run_id"],
        "train_sha256": _sha256(train_path),
        "lockbox_fingerprint": split_fingerprint(lockbox_manifest),
        "lockbox_rows": int(lockbox_mask.sum()),
        "development_rows": int((~lockbox_mask).sum()),
        "candidate_score": asdict(candidate_score),
        "control_relational_score": asdict(control_score),
        "decision": decision,
        "leakage_audit": {
            "passed": leakage_passed,
            "pair_group_overlap": pair_overlap,
            "lockbox_fold_audit": lockbox_audit,
            "inner_fold_audit": inner_audit,
            "lockbox_never_used_for_epoch_threshold_or_blend_selection": True,
            "test_loaded_before_selection": False,
        },
        "branch_c": c_metadata,
        "prr": prr_metadata,
        "finalize_requested": finalize_if_improved,
        "submission": None,
    }
    write_json(summary, run_dir / "lockbox_summary.json")
    pd.DataFrame(
        {
            ID_COLUMN: lockbox[ID_COLUMN].astype(str),
            SEVERITY_COLUMN: severity,
            **{column: c_binary[:, i] for i, column in enumerate(binary_columns)},
            **{column: candidate_prr[:, i] for i, column in enumerate(prr_columns)},
        }
    ).to_csv(output_dir / "lockbox_predictions.csv", index=False)

    if not finalize_if_improved or not eligible:
        summary["duration_seconds"] = time.perf_counter() - started
        write_json(summary, run_dir / "lockbox_summary.json")
        return summary

    # Selection is frozen. This is intentionally the first read of test.csv.
    test_path = Path(paths["test"])
    sample_path = Path(paths["sample_submission"])
    test = pd.read_csv(test_path, low_memory=False)
    sample = pd.read_csv(sample_path, low_memory=False)
    forbidden = set(binary_columns + prr_columns + [SEVERITY_COLUMN]) & set(test.columns)
    if forbidden:
        raise ValueError(f"Le test contient des cibles inattendues : {sorted(forbidden)}")
    final_inner_manifest, final_inner_audit = build_pair_grouped_folds(
        train,
        n_splits=int(config["inner_validation"]["n_splits"]),
        seed=int(config["inner_validation"]["seed"]),
    )
    if not final_inner_audit["passed"]:
        raise RuntimeError("L'audit des folds du refit final a echoue.")
    final_folds = final_inner_manifest["fold"].to_numpy()
    test_severity = _fit_severity(train, test, baseline_config, seed)
    _, test_binary, _, final_c_metadata = _run_branch_c(
        train,
        test,
        train[binary_columns].to_numpy(dtype=np.int8),
        prr_targets,
        final_folds,
        lockbox_fold,
        config["branch_c"],
        seed,
    )
    test_prr, final_prr_metadata = _fit_blended_prr(
        train,
        test,
        prr_targets,
        final_folds,
        prr_config,
        selected_epochs,
        run_dir / "full_refit",
    )
    submission = pd.concat(
        [
            test[[ID_COLUMN]].reset_index(drop=True),
            pd.Series(test_severity, name=SEVERITY_COLUMN),
            pd.DataFrame(test_binary, columns=binary_columns),
            pd.DataFrame(test_prr, columns=prr_columns),
        ],
        axis=1,
    )[list(sample.columns)]
    submission_path = Path(paths["submission"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    errors = validate_submission(submission_path, sample_path)
    if errors:
        submission_path.unlink(missing_ok=True)
        raise ValueError("Soumission invalide : " + "; ".join(errors))
    summary["submission"] = {
        "path": str(submission_path),
        "sha256": _sha256(submission_path),
        "rows": len(submission),
        "columns": len(submission.columns),
        "automatic_upload": False,
        "test_read_only_after_lockbox_selection": True,
        "branch_c": final_c_metadata,
        "prr": final_prr_metadata,
    }
    summary["duration_seconds"] = time.perf_counter() - started
    write_json(summary, run_dir / "lockbox_summary.json")
    return summary
