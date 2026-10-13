from __future__ import annotations

import hashlib
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
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


def _severity_features(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    fit_labels: np.ndarray,
    fit_folds: np.ndarray,
    classes: list[str],
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    one_hot = np.column_stack([(fit_labels == label).astype(np.float32) for label in classes])
    fit_encoded = build_cross_fitted_molecule_priors(
        fit, one_hot, fit_folds, smoothing=smoothing
    )
    tables = fit_molecule_prior_tables(fit, one_hot, smoothing=smoothing)
    validation_encoded = transform_molecule_priors(validation, tables)
    scaler = StandardScaler().fit(fit_encoded)
    return (
        scaler.transform(fit_encoded).astype(np.float32),
        scaler.transform(validation_encoded).astype(np.float32),
    )


def _read_aligned(path: str, train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if not frame[ID_COLUMN].astype(str).equals(train[ID_COLUMN].astype(str)):
        raise ValueError(f"Predictions OOF desalignees: {path}")
    return frame


def run_severity_priors_benchmark(config_path: Path) -> dict[str, object]:
    """Evaluate severity drug priors using pair-grouped OOF only."""

    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(Path(paths["audit"]))
    current_audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    if not current_audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Le manifeste de validation ne correspond pas a son audit.")
    folds = manifest["fold"].to_numpy()
    labels = train[SEVERITY_COLUMN].astype(str).to_numpy()
    prr_targets = train[
        [column for column in train if column.startswith(PRR_PREFIX)]
    ].to_numpy(np.float32)
    classes = sorted(np.unique(labels))
    candidates = {
        "priors_unweighted": np.empty(len(train), dtype=object),
        "priors_balanced": np.empty(len(train), dtype=object),
        "priors_multitask_balanced": np.empty(len(train), dtype=object),
    }
    metadata: list[dict[str, object]] = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        held_out = folds == fold
        fit = train.loc[~held_out].reset_index(drop=True)
        validation = train.loc[held_out].reset_index(drop=True)
        fit_base, validation_base, fit_prr, validation_prr = _load_or_build_features(
            train,
            prr_targets,
            folds,
            fold,
            fingerprint,
            config,
        )
        fit_prior, validation_prior = _severity_features(
            fit,
            validation,
            labels[~held_out],
            folds[~held_out],
            classes,
            float(config["model"]["prior_smoothing"]),
        )
        fit_features = np.hstack([fit_base, fit_prior]).astype(np.float32)
        validation_features = np.hstack([validation_base, validation_prior]).astype(np.float32)
        for name, class_weight, fold_fit, fold_validation in (
            ("priors_unweighted", None, fit_features, validation_features),
            ("priors_balanced", "balanced", fit_features, validation_features),
            (
                "priors_multitask_balanced",
                "balanced",
                np.hstack([fit_prr, fit_prior]).astype(np.float32),
                np.hstack([validation_prr, validation_prior]).astype(np.float32),
            ),
        ):
            model = lgb.LGBMClassifier(
                objective="multiclass",
                n_estimators=int(config["model"]["n_estimators"]),
                learning_rate=float(config["model"]["learning_rate"]),
                num_leaves=int(config["model"]["num_leaves"]),
                min_child_samples=int(config["model"]["min_child_samples"]),
                reg_lambda=float(config["model"]["reg_lambda"]),
                class_weight=class_weight,
                random_state=int(config["seed"]) + fold,
                n_jobs=int(config["model"]["n_jobs"]),
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            model.fit(fold_fit, labels[~held_out])
            candidates[name][held_out] = model.predict(fold_validation)
        metadata.append(
            {
                "fold": fold,
                "train_rows": int((~held_out).sum()),
                "validation_rows": int(held_out.sum()),
                "prior_dimensions": int(fit_prior.shape[1]),
            }
        )
        print(f"Severity fold {fold} termine", flush=True)

    incumbent = _read_aligned(paths["incumbent_oof"], train)
    binary = _read_aligned(paths["binary_oof"], train)
    old_prr = _read_aligned(paths["old_prr_oof"], train)
    new_prr = _read_aligned(paths["new_prr_oof"], train)
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    blended_prr = 0.5 * (
        old_prr[prr_columns].to_numpy(np.float32)
        + new_prr[prr_columns].to_numpy(np.float32)
    )
    candidates["incumbent"] = incumbent[SEVERITY_COLUMN].astype(str).to_numpy()
    rows = []
    output_dir = Path(paths["output_dir"])
    run_dir = Path(paths["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, predicted in candidates.items():
        score = hardcore_score(
            labels,
            predicted,
            train[binary_columns],
            binary[binary_columns],
            train[prr_columns],
            blended_prr,
        )
        rows.append(
            {
                "candidate": name,
                "severity_macro_f1": score.severity_macro_f1,
                "total_with_fixed_binary_prr": score.total,
                "per_class_f1": {
                    label: float(value)
                    for label, value in zip(
                        classes,
                        f1_score(labels, predicted, labels=classes, average=None),
                        strict=True,
                    )
                },
            }
        )
        pd.DataFrame({ID_COLUMN: train[ID_COLUMN], SEVERITY_COLUMN: predicted}).to_csv(
            output_dir / f"oof_severity_{name}.csv", index=False
        )
    comparison = pd.DataFrame(rows).sort_values(
        "total_with_fixed_binary_prr", ascending=False
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    summary = {
        "run_id": config["run_id"],
        "train_sha256": _sha256(train_path),
        "validation_fingerprint": fingerprint,
        "comparison": rows,
        "folds": metadata,
        "fixed_binary_source": paths["binary_oof"],
        "fixed_prr_sources": [paths["old_prr_oof"], paths["new_prr_oof"]],
        "leakage_audit": {
            "passed": True,
            "pair_group_overlap_max": 0,
            "training_severity_priors_cross_fitted": True,
            "validation_severity_priors_fit_only": True,
            "test_loaded": False,
        },
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def finalize_severity_priors_candidate(config_path: Path) -> dict[str, object]:
    """Refit the OOF winner and replace only Severity in an existing valid submission."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(Path(paths["audit"]))
    current_audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    summary = load_json(Path(paths["run_dir"]) / "summary.json")
    if not current_audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Validation groupee invalide.")
    if summary["train_sha256"] != _sha256(train_path):
        raise ValueError("Le train a change depuis le benchmark OOF.")
    if summary["validation_fingerprint"] != fingerprint:
        raise ValueError("Les folds ont change depuis le benchmark OOF.")
    scores = {row["candidate"]: row for row in summary["comparison"]}
    selected = "priors_balanced"
    improvement = (
        scores[selected]["severity_macro_f1"] - scores["incumbent"]["severity_macro_f1"]
    )
    folds = manifest["fold"].to_numpy()
    labels = train[SEVERITY_COLUMN].astype(str).to_numpy()
    selected_oof = _read_aligned(
        str(Path(paths["output_dir"]) / f"oof_severity_{selected}.csv"), train
    )[SEVERITY_COLUMN].astype(str).to_numpy()
    incumbent_oof = _read_aligned(paths["incumbent_oof"], train)[
        SEVERITY_COLUMN
    ].astype(str).to_numpy()
    fold_gains = []
    for fold in np.unique(folds):
        held_out = folds == fold
        fold_gains.append(
            float(
                f1_score(labels[held_out], selected_oof[held_out], average="macro")
                - f1_score(labels[held_out], incumbent_oof[held_out], average="macro")
            )
        )
    minimum_gain = float(config["selection"]["minimum_macro_f1_gain"])
    if improvement < minimum_gain or min(fold_gains) <= 0:
        raise ValueError("Le candidat Severity ne passe pas la porte OOF.")

    # No test data or previous submission is read before the OOF decision.
    test = pd.read_csv(paths["test"], low_memory=False)
    sample_path = Path(paths["sample_submission"])
    incumbent_path = Path(paths["incumbent_submission"])
    incumbent = pd.read_csv(incumbent_path, low_memory=False)
    if not incumbent[ID_COLUMN].astype(str).equals(test[ID_COLUMN].astype(str)):
        raise ValueError("La soumission de reference et le test sont desalignees.")
    if validate_submission(incumbent_path, sample_path):
        raise ValueError("La soumission de reference est invalide.")

    feature_config = yaml.safe_load(Path(paths["feature_cache_config"]).read_text("utf-8"))
    builder = _feature_builder(feature_config, int(config["seed"]))
    train_base_raw = builder.fit_transform(train)
    test_base_raw = builder.transform(test)
    base_scaler = StandardScaler().fit(train_base_raw)
    train_base = base_scaler.transform(train_base_raw).astype(np.float32)
    test_base = base_scaler.transform(test_base_raw).astype(np.float32)
    classes = sorted(np.unique(labels))
    train_prior, test_prior = _severity_features(
        train,
        test,
        labels,
        folds,
        classes,
        float(config["model"]["prior_smoothing"]),
    )
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=int(config["model"]["n_estimators"]),
        learning_rate=float(config["model"]["learning_rate"]),
        num_leaves=int(config["model"]["num_leaves"]),
        min_child_samples=int(config["model"]["min_child_samples"]),
        reg_lambda=float(config["model"]["reg_lambda"]),
        class_weight="balanced",
        random_state=int(config["seed"]),
        n_jobs=int(config["model"]["n_jobs"]),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(np.hstack([train_base, train_prior]), labels)
    candidate = incumbent.copy()
    candidate[SEVERITY_COLUMN] = model.predict(np.hstack([test_base, test_prior]))
    candidate_path = Path(paths["candidate_submission"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_path, index=False)
    errors = validate_submission(candidate_path, sample_path)
    if errors:
        candidate_path.unlink(missing_ok=True)
        raise ValueError("Soumission candidate invalide : " + "; ".join(errors))
    if not candidate.drop(columns=SEVERITY_COLUMN).equals(
        incumbent.drop(columns=SEVERITY_COLUMN)
    ):
        raise AssertionError("Une autre colonne que Severity a ete modifiee.")
    run_dir = Path(paths["run_dir"])
    model_path = run_dir / "final_severity_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_builder": builder,
            "base_scaler": base_scaler,
            "classes": classes,
            "fold_fingerprint": fingerprint,
            "train_sha256": _sha256(train_path),
        },
        model_path,
        compress=3,
    )
    result = {
        "run_id": config["run_id"],
        "selected": selected,
        "macro_f1_oof_gain": improvement,
        "fold_macro_f1_gains": fold_gains,
        "estimated_global_gain_from_severity": 0.4 * improvement,
        "incumbent_submission": str(incumbent_path),
        "incumbent_sha256": _sha256(incumbent_path),
        "candidate_submission": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "rows": len(candidate),
        "changed_severity_rows": int(
            (candidate[SEVERITY_COLUMN] != incumbent[SEVERITY_COLUMN]).sum()
        ),
        "other_columns_identical": True,
        "model_artifact": str(model_path),
        "automatic_upload": False,
    }
    write_json(result, run_dir / "candidate_summary.json")
    return result
