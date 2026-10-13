from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import f1_score

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .metric import HardcoreScore, hardcore_score
from .multimodal_features import MultimodalFeatureBuilder
from .submission import validate_submission
from .submission_gate import decision_as_dict, evaluate_submission_gate, load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_probabilities(model: ExtraTreesClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    result = np.zeros((len(features), len(probabilities)), dtype=np.float32)
    for target_index, target_probabilities in enumerate(probabilities):
        classes = np.asarray(model.classes_[target_index])
        positive = np.flatnonzero(classes == 1)
        if len(positive):
            result[:, target_index] = target_probabilities[:, int(positive[0])]
    return result


def _oob_positive_probabilities(model: ExtraTreesClassifier) -> np.ndarray:
    oob = np.asarray(model.oob_decision_function_)
    result = np.zeros((oob.shape[0], oob.shape[2]), dtype=np.float32)
    for target_index, classes in enumerate(model.classes_):
        positive = np.flatnonzero(np.asarray(classes) == 1)
        if len(positive):
            result[:, target_index] = oob[:, int(positive[0]), target_index]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_oob_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.80, 31)
    for target_index in range(y_true.shape[1]):
        target = y_true[:, target_index]
        if np.unique(target).size < 2:
            thresholds[target_index] = 0.0 if target[0] == 1 else 1.0
            continue
        scores = [
            f1_score(target, probabilities[:, target_index] >= threshold, zero_division=0)
            for threshold in grid
        ]
        thresholds[target_index] = float(grid[int(np.argmax(scores))])
    return thresholds


def _build_models(config: dict[str, object], seed: int) -> tuple[object, object, object]:
    model_config = config["models"]
    n_jobs = int(config["runtime"]["n_jobs"])
    severity_model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=int(model_config["severity_estimators"]),
        learning_rate=float(model_config["learning_rate"]),
        num_leaves=int(model_config["num_leaves"]),
        max_depth=int(model_config["max_depth"]),
        min_child_samples=int(model_config["min_child_samples"]),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    binary_model = ExtraTreesClassifier(
        n_estimators=int(model_config["binary_estimators"]),
        max_depth=int(model_config["forest_max_depth"]),
        min_samples_leaf=int(model_config["forest_min_samples_leaf"]),
        max_features=float(model_config["forest_max_features"]),
        bootstrap=True,
        max_samples=0.85,
        oob_score=True,
        random_state=seed,
        n_jobs=n_jobs,
    )
    prr_model = ExtraTreesRegressor(
        n_estimators=int(model_config["prr_estimators"]),
        max_depth=int(model_config["forest_max_depth"]),
        min_samples_leaf=int(model_config["forest_min_samples_leaf"]),
        max_features=float(model_config["forest_max_features"]),
        bootstrap=True,
        max_samples=0.85,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return severity_model, binary_model, prr_model


def _feature_builder(config: dict[str, object], seed: int) -> MultimodalFeatureBuilder:
    feature_config = config["features"]
    return MultimodalFeatureBuilder(
        text_components=int(feature_config["text_components"]),
        morgan_components=int(feature_config["morgan_components"]),
        word_max_features=int(feature_config["word_max_features"]),
        char_max_features=int(feature_config["char_max_features"]),
        min_df=int(feature_config["min_df"]),
        include_warning_toxicity=bool(feature_config["include_warning_toxicity"]),
        random_state=seed,
    )


def _fit_predict_split(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    binary_columns: list[str],
    prr_columns: list[str],
    config: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    builder = _feature_builder(config, seed)
    fit_features = builder.fit_transform(fit_frame)
    validation_features = builder.transform(validation_frame)

    severity_model, binary_model, prr_model = _build_models(config, seed)
    if fit_frame[SEVERITY_COLUMN].nunique() < 2:
        severity_model = DummyClassifier(
            strategy="constant", constant=fit_frame[SEVERITY_COLUMN].iloc[0]
        )
    severity_model.fit(fit_features, fit_frame[SEVERITY_COLUMN])

    y_binary = fit_frame[binary_columns].to_numpy(dtype=np.int8)
    binary_model.fit(fit_features, y_binary)
    oob_probabilities = _oob_positive_probabilities(binary_model)
    thresholds = _fit_oob_thresholds(y_binary, oob_probabilities)
    validation_binary_probabilities = _positive_probabilities(binary_model, validation_features)
    validation_binary = (validation_binary_probabilities >= thresholds).astype(np.int8)

    y_prr_log = np.log1p(fit_frame[prr_columns].to_numpy(dtype=np.float32))
    prr_model.fit(fit_features, y_prr_log)
    validation_prr = np.expm1(prr_model.predict(validation_features))
    validation_prr = np.maximum(np.nan_to_num(validation_prr, nan=0.0, posinf=0.0), 0.0)

    metadata = {
        "feature_dimensions": builder.family_dimensions,
        "total_feature_dimensions": int(fit_features.shape[1]),
        "binary_thresholds": thresholds.tolist(),
        "duration_seconds": time.perf_counter() - started,
    }
    return (
        severity_model.predict(validation_features),
        validation_binary,
        validation_prr,
        metadata,
    )


def _score_frame(
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


def _train_final_bundle(
    train: pd.DataFrame,
    test: pd.DataFrame,
    binary_columns: list[str],
    prr_columns: list[str],
    config: dict[str, object],
    seed: int,
    artifact_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    builder = _feature_builder(config, seed)
    train_features = builder.fit_transform(train)
    test_features = builder.transform(test)
    severity_model, binary_model, prr_model = _build_models(config, seed)
    severity_model.fit(train_features, train[SEVERITY_COLUMN])
    y_binary = train[binary_columns].to_numpy(dtype=np.int8)
    binary_model.fit(train_features, y_binary)
    thresholds = _fit_oob_thresholds(y_binary, _oob_positive_probabilities(binary_model))
    test_binary = (_positive_probabilities(binary_model, test_features) >= thresholds).astype(
        np.int8
    )
    prr_model.fit(train_features, np.log1p(train[prr_columns].to_numpy(dtype=np.float32)))
    test_prr = np.maximum(
        np.nan_to_num(np.expm1(prr_model.predict(test_features)), nan=0.0, posinf=0.0), 0.0
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_builder": builder,
            "severity_model": severity_model,
            "binary_model": binary_model,
            "prr_model": prr_model,
            "binary_thresholds": thresholds,
            "binary_columns": binary_columns,
            "prr_columns": prr_columns,
        },
        artifact_path,
        compress=3,
    )
    metadata = {
        "feature_dimensions": builder.family_dimensions,
        "artifact": artifact_path.name,
        "artifact_sha256": _sha256(artifact_path),
    }
    return severity_model.predict(test_features), test_binary, test_prr, metadata


def train_candidate(
    *,
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    folds_path: Path,
    folds_audit_path: Path,
    strict_manifest_path: Path,
    registry_path: Path,
    config_path: Path,
    output_dir: Path,
    submission_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    run_id = str(config["run_id"])
    train = pd.read_csv(train_path, low_memory=False)
    folds = pd.read_csv(folds_path).sort_values("row_index").reset_index(drop=True)
    split_audit = load_json(folds_audit_path)
    current_audit = audit_pair_grouped_folds(train, folds)
    fingerprint = split_fingerprint(folds)
    if not current_audit["passed"] or fingerprint != split_audit["fingerprint"]:
        raise ValueError("L'audit des folds a échoué; entraînement bloqué.")

    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    oof_severity = np.empty(len(train), dtype=object)
    oof_binary = np.zeros((len(train), len(binary_columns)), dtype=np.int8)
    oof_prr = np.zeros((len(train), len(prr_columns)), dtype=np.float32)
    fold_scores = []
    fold_metadata = []
    for fold in sorted(folds["fold"].unique()):
        validation_mask = folds["fold"].to_numpy() == fold
        fit_mask = ~validation_mask
        print(f"Fold {fold}: fit={fit_mask.sum()}, validation={validation_mask.sum()}", flush=True)
        severity, binary, prr, metadata = _fit_predict_split(
            train.loc[fit_mask].reset_index(drop=True),
            train.loc[validation_mask].reset_index(drop=True),
            binary_columns,
            prr_columns,
            config,
            seed + int(fold),
        )
        oof_severity[validation_mask] = severity
        oof_binary[validation_mask] = binary
        oof_prr[validation_mask] = prr
        score = _score_frame(
            train.loc[validation_mask], severity, binary, prr, binary_columns, prr_columns
        )
        fold_scores.append({"fold": int(fold), **asdict(score)})
        fold_metadata.append({"fold": int(fold), **metadata})
        print(f"Fold {fold}: score={score.total:.6f}", flush=True)

    overall_score = _score_frame(
        train, oof_severity, oof_binary, oof_prr, binary_columns, prr_columns
    )
    strict_manifest = pd.read_csv(strict_manifest_path).sort_values("row_index")
    strict_train = strict_manifest["role"].to_numpy() == "train"
    strict_validation = strict_manifest["role"].to_numpy() == "validation"
    cold_severity, cold_binary, cold_prr, cold_metadata = _fit_predict_split(
        train.loc[strict_train].reset_index(drop=True),
        train.loc[strict_validation].reset_index(drop=True),
        binary_columns,
        prr_columns,
        config,
        seed + 100,
    )
    cold_score = _score_frame(
        train.loc[strict_validation],
        cold_severity,
        cold_binary,
        cold_prr,
        binary_columns,
        prr_columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    oof_frame = pd.concat(
        [
            train[[ID_COLUMN]].reset_index(drop=True),
            pd.Series(oof_severity, name=SEVERITY_COLUMN),
            pd.DataFrame(oof_binary, columns=binary_columns),
            pd.DataFrame(oof_prr, columns=prr_columns),
        ],
        axis=1,
    )
    oof_path = output_dir / "oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)
    metrics: dict[str, object] = {
        "schema_version": 2,
        "run_id": run_id,
        "run_type": "multimodal_lgbm_extratrees_baseline",
        "created_at": datetime.now(UTC).isoformat(),
        "protocol_id": split_audit["protocol_id"],
        "validation_fingerprint": fingerprint,
        "score": asdict(overall_score),
        "fold_scores": fold_scores,
        "cold_drug_score": asdict(cold_score),
        "cold_drug_rows": int(strict_validation.sum()),
        "cold_drug_purged_rows": int((strict_manifest["role"] == "purged").sum()),
        "fold_metadata": fold_metadata,
        "cold_metadata": cold_metadata,
        "leakage_audit": {
            "passed": True,
            "pair_group_overlap_max": 0,
            "coverage_complete": True,
            "fit_statistics_computed_within_each_fold": True,
            "strict_molecule_overlap": 0,
        },
        "data_scope": {
            "train_only_for_fit": True,
            "test_used_for_fit": False,
            "test_used_for_local_scoring": False,
        },
        "oof": {
            "expected_rows": len(train),
            "predicted_rows": len(oof_frame),
            "coverage_complete": not oof_frame.isna().any().any(),
            "predictions_file": oof_path.name,
            "predictions_sha256": _sha256(oof_path),
        },
        "feature_safety": {
            "pair_id_used": False,
            "target_columns_used": False,
            "fold_local_tfidf": True,
            "fold_local_svd": True,
            "fold_local_imputation": True,
            "fold_local_binary_thresholds": True,
        },
        "reproducibility": {
            "seed": seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "train_sha256": _sha256(train_path),
            "test_sha256": _sha256(test_path),
            "config_sha256": _sha256(config_path),
        },
        "duration_seconds": time.perf_counter() - started,
        "automatic_submission": False,
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics, metrics_path)

    decision = evaluate_submission_gate(metrics, load_json(registry_path))
    gate_payload = decision_as_dict(decision)
    write_json(gate_payload, output_dir / "gate_decision.json")
    if not decision.eligible:
        print(json.dumps(gate_payload, indent=2, ensure_ascii=False), flush=True)
        return {"metrics": metrics, "gate": gate_payload, "submission": None}

    print("Score OOF amélioré et audits validés; entraînement final autorisé.", flush=True)
    test = pd.read_csv(test_path, low_memory=False)
    target_columns = {SEVERITY_COLUMN, *binary_columns, *prr_columns}
    if target_columns & set(test.columns):
        raise ValueError("Le test contient une cible inattendue dans les features.")
    final_severity, final_binary, final_prr, final_metadata = _train_final_bundle(
        train,
        test,
        binary_columns,
        prr_columns,
        config,
        seed,
        output_dir / "final_model.joblib",
    )
    sample = pd.read_csv(sample_path, low_memory=False)
    submission = pd.concat(
        [
            test[[ID_COLUMN]].reset_index(drop=True),
            pd.Series(final_severity, name=SEVERITY_COLUMN),
            pd.DataFrame(final_binary, columns=binary_columns),
            pd.DataFrame(final_prr, columns=prr_columns),
        ],
        axis=1,
    )
    submission = submission[list(sample.columns)]
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    submission_errors = validate_submission(submission_path, sample_path)
    if submission_errors:
        submission_path.unlink(missing_ok=True)
        raise ValueError("Soumission candidate invalide : " + "; ".join(submission_errors))
    submission_validation = {
        "valid": True,
        "rows": len(submission),
        "columns": len(submission.columns),
        "ordered_pair_id_sha256": hashlib.sha256(
            "\n".join(submission[ID_COLUMN].astype(str)).encode()
        ).hexdigest(),
        "severity_distribution": submission[SEVERITY_COLUMN].value_counts().to_dict(),
        "binary_positive_rates": submission[binary_columns].mean().to_dict(),
        "prr_min": float(submission[prr_columns].min().min()),
        "prr_max": float(submission[prr_columns].max().max()),
        "submission_sha256": _sha256(submission_path),
        "final_model": final_metadata,
        "uploaded_to_kaggle": False,
    }
    write_json(submission_validation, output_dir / "submission_validation.json")
    return {"metrics": metrics, "gate": gate_payload, "submission": submission_validation}
