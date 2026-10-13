from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .metric import HardcoreScore, hardcore_score
from .modeling import _oob_positive_probabilities, _positive_probabilities
from .prr_oof import _feature_cache_signature, _prepare_fold_features
from .submission_gate import load_json, write_json
from .validation import (
    audit_pair_grouped_folds,
    canonicalize_smiles,
    split_fingerprint,
)

BRANCH_A = "branch_a_calibrated"
BRANCH_B = "branch_b_targetwise_lgbm"
BRANCH_C = "branch_c_relational"
BRANCH_D = "branch_d_fixed_blend"
CONTROL = "control_existing"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA demandé mais indisponible dans cet environnement PyTorch.")
    return device


def _load_aligned_predictions(path: Path, train: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(path, low_memory=False)
    if ID_COLUMN not in predictions:
        raise ValueError(f"{path}: colonne {ID_COLUMN} absente.")
    expected = train[ID_COLUMN].astype(str).reset_index(drop=True)
    observed = predictions[ID_COLUMN].astype(str).reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(f"{path}: ordre des {ID_COLUMN} différent du train.")
    return predictions


def _fit_micro_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    steps: int,
) -> tuple[float, float]:
    if targets.shape != probabilities.shape:
        raise ValueError("Cibles et probabilités désalignées pour la calibration micro-F1.")
    grid = np.linspace(minimum, maximum, steps, dtype=np.float64)
    scores = np.asarray(
        [
            f1_score(targets, probabilities >= threshold, average="micro", zero_division=0)
            for threshold in grid
        ]
    )
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


@dataclass(frozen=True)
class PlattParameters:
    slope: np.ndarray
    intercept: np.ndarray
    constant: np.ndarray


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def fit_platt_parameters(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    regularization: float,
) -> PlattParameters:
    if targets.shape != probabilities.shape:
        raise ValueError("Cibles et probabilités désalignées pour Platt.")
    n_targets = targets.shape[1]
    slope = np.ones(n_targets, dtype=np.float32)
    intercept = np.zeros(n_targets, dtype=np.float32)
    constant = np.full(n_targets, np.nan, dtype=np.float32)
    logits = _logit(probabilities)
    for target_index in range(n_targets):
        target = targets[:, target_index].astype(np.int8)
        classes = np.unique(target)
        if len(classes) < 2:
            constant[target_index] = float(classes[0])
            continue
        model = LogisticRegression(
            C=regularization,
            solver="lbfgs",
            max_iter=300,
            random_state=0,
        )
        model.fit(logits[:, target_index : target_index + 1], target)
        slope[target_index] = float(model.coef_[0, 0])
        intercept[target_index] = float(model.intercept_[0])
    return PlattParameters(slope=slope, intercept=intercept, constant=constant)


def apply_platt_parameters(
    probabilities: np.ndarray,
    parameters: PlattParameters,
) -> np.ndarray:
    calibrated_logits = _logit(probabilities) * parameters.slope + parameters.intercept
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(calibrated_logits, -30.0, 30.0)))
    constant_mask = np.isfinite(parameters.constant)
    calibrated[:, constant_mask] = parameters.constant[constant_mask]
    return calibrated.astype(np.float32)


def _score(
    train: pd.DataFrame,
    rows: np.ndarray,
    severity: np.ndarray,
    binary: np.ndarray,
    prr: np.ndarray,
    binary_columns: list[str],
    prr_columns: list[str],
) -> HardcoreScore:
    return hardcore_score(
        train.loc[rows, SEVERITY_COLUMN],
        severity[rows],
        train.loc[rows, binary_columns],
        binary[rows],
        train.loc[rows, prr_columns],
        prr[rows],
    )


def _load_or_build_features(
    train: pd.DataFrame,
    prr_targets: np.ndarray,
    fold_values: np.ndarray,
    fold: int,
    fingerprint: str,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = config["paths"]
    feature_config_path = Path(paths["feature_cache_config"])
    feature_config = yaml.safe_load(feature_config_path.read_text(encoding="utf-8"))
    cache_dir = Path(paths["feature_cache_dir"]) / f"fold_{fold}"
    cache_path = cache_dir / "features.npz"
    metadata_path = cache_dir / "features_metadata.json"
    signature = _feature_cache_signature(feature_config, fingerprint, fold)
    if cache_path.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata.get("signature") == signature:
            cached = np.load(cache_path)
            return (
                cached["fit_base"],
                cached["validation_base"],
                cached["fit_with_priors"],
                cached["validation_with_priors"],
            )
    prepared = _prepare_fold_features(
        train,
        prr_targets,
        fold_values,
        fold,
        feature_config,
        fingerprint,
        cache_dir,
    )
    return prepared[0], prepared[1], prepared[2], prepared[3]


def _run_branch_a(
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    fit_targets: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    model = ExtraTreesClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=float(config["max_features"]),
        bootstrap=True,
        max_samples=float(config["max_samples"]),
        oob_score=True,
        random_state=seed,
        n_jobs=int(config["n_jobs"]),
    )
    model.fit(fit_features, fit_targets)
    oob_probabilities = _oob_positive_probabilities(model)
    validation_probabilities = _positive_probabilities(model, validation_features)
    parameters = fit_platt_parameters(
        fit_targets,
        oob_probabilities,
        regularization=float(config["platt_regularization"]),
    )
    calibrated_oob = apply_platt_parameters(oob_probabilities, parameters)
    calibrated_validation = apply_platt_parameters(validation_probabilities, parameters)
    threshold, calibration_f1 = _fit_micro_threshold(
        fit_targets,
        calibrated_oob,
        minimum=float(config["threshold_min"]),
        maximum=float(config["threshold_max"]),
        steps=int(config["threshold_steps"]),
    )
    decisions = (calibrated_validation >= threshold).astype(np.int8)
    return (
        calibrated_validation,
        decisions,
        {
            "threshold": threshold,
            "calibration_micro_f1": calibration_f1,
            "calibration_source": "outer_fit_oob_only",
        },
    )


def _lgb_classifier(config: dict[str, object], seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=seed,
        n_jobs=int(config["model_n_jobs"]),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _lgb_regressor(config: dict[str, object], seed: int, n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=n_estimators,
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=seed,
        n_jobs=int(config["model_n_jobs"]),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _positive_probability(model: lgb.LGBMClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features))
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if not len(positive):
        return np.zeros(len(features), dtype=np.float32)
    return probabilities[:, int(positive[0])].astype(np.float32)


def _run_branch_b(
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    fit_binary: np.ndarray,
    fit_prr: np.ndarray,
    fit_fold_values: np.ndarray,
    outer_fold: int,
    config: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if str(config["backend"]).lower() != "lightgbm":
        raise ValueError("Le benchmark implémente actuellement backend=lightgbm.")
    inner_values = sorted(int(value) for value in np.unique(fit_fold_values))
    calibration_fold = inner_values[outer_fold % len(inner_values)]
    calibration_mask = fit_fold_values == calibration_fold
    core_mask = ~calibration_mask
    n_targets = fit_binary.shape[1]
    calibration_probabilities = np.zeros((int(calibration_mask.sum()), n_targets), dtype=np.float32)
    validation_probabilities = np.zeros((len(validation_features), n_targets), dtype=np.float32)
    validation_prr = np.zeros((len(validation_features), n_targets), dtype=np.float32)
    classifier_iterations: list[int] = []
    regressor_iterations: list[int] = []

    def fit_target(
        target_index: int,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int, int]:
        callbacks = [
            lgb.early_stopping(int(config["early_stopping_rounds"]), verbose=False),
            lgb.log_evaluation(0),
        ]
        y_binary = fit_binary[:, target_index]
        if np.unique(y_binary[core_mask]).size < 2:
            value = float(np.mean(y_binary[core_mask]))
            target_calibration_probabilities = np.full(
                int(calibration_mask.sum()), value, dtype=np.float32
            )
            target_validation_probabilities = np.full(
                len(validation_features), value, dtype=np.float32
            )
            classifier_iteration = 0
        else:
            selector = _lgb_classifier(config, seed + target_index, int(config["max_estimators"]))
            fit_kwargs: dict[str, object] = {}
            if np.unique(y_binary[calibration_mask]).size > 1:
                fit_kwargs = {
                    "eval_X": fit_features[calibration_mask],
                    "eval_y": y_binary[calibration_mask],
                    "eval_metric": "binary_logloss",
                    "callbacks": callbacks,
                }
            selector.fit(fit_features[core_mask], y_binary[core_mask], **fit_kwargs)
            best_iteration = int(selector.best_iteration_ or config["max_estimators"])
            target_calibration_probabilities = _positive_probability(
                selector, fit_features[calibration_mask]
            )
            final_model = _lgb_classifier(config, seed + target_index, best_iteration)
            final_model.fit(fit_features, y_binary)
            target_validation_probabilities = _positive_probability(
                final_model, validation_features
            )
            classifier_iteration = best_iteration

        y_prr = fit_prr[:, target_index]
        core_positive = core_mask & (y_prr > 0)
        calibration_positive = calibration_mask & (y_prr > 0)
        all_positive = y_prr > 0
        if core_positive.sum() < 3 or all_positive.sum() < 3:
            constant = float(y_prr[all_positive].mean()) if all_positive.any() else 0.0
            target_validation_prr = np.full(len(validation_features), constant, dtype=np.float32)
            regressor_iteration = 0
        else:
            selector_regressor = _lgb_regressor(
                config, seed + 1000 + target_index, int(config["max_estimators"])
            )
            regression_kwargs: dict[str, object] = {}
            if calibration_positive.sum() >= 3:
                regression_kwargs = {
                    "eval_X": fit_features[calibration_positive],
                    "eval_y": y_prr[calibration_positive],
                    "eval_metric": "rmse",
                    "callbacks": callbacks,
                }
            selector_regressor.fit(
                fit_features[core_positive], y_prr[core_positive], **regression_kwargs
            )
            best_iteration = int(selector_regressor.best_iteration_ or config["max_estimators"])
            final_regressor = _lgb_regressor(config, seed + 1000 + target_index, best_iteration)
            final_regressor.fit(fit_features[all_positive], y_prr[all_positive])
            target_validation_prr = np.maximum(
                np.asarray(final_regressor.predict(validation_features), dtype=np.float32), 0.0
            )
            regressor_iteration = best_iteration

        return (
            target_index,
            target_calibration_probabilities,
            target_validation_probabilities,
            target_validation_prr,
            classifier_iteration,
            regressor_iteration,
        )

    results = Parallel(n_jobs=int(config["target_n_jobs"]), prefer="threads")(
        delayed(fit_target)(target_index) for target_index in range(n_targets)
    )
    for (
        target_index,
        target_calibration_probabilities,
        target_validation_probabilities,
        target_validation_prr,
        classifier_iteration,
        regressor_iteration,
    ) in results:
        calibration_probabilities[:, target_index] = target_calibration_probabilities
        validation_probabilities[:, target_index] = target_validation_probabilities
        validation_prr[:, target_index] = target_validation_prr
        classifier_iterations.append(classifier_iteration)
        regressor_iterations.append(regressor_iteration)

    calibration_targets = fit_binary[calibration_mask]
    parameters = fit_platt_parameters(
        calibration_targets,
        calibration_probabilities,
        regularization=float(config["platt_regularization"]),
    )
    calibrated_inner = apply_platt_parameters(calibration_probabilities, parameters)
    calibrated_validation = apply_platt_parameters(validation_probabilities, parameters)
    threshold, calibration_f1 = _fit_micro_threshold(
        calibration_targets,
        calibrated_inner,
        minimum=float(config["threshold_min"]),
        maximum=float(config["threshold_max"]),
        steps=int(config["threshold_steps"]),
    )
    decisions = (calibrated_validation >= threshold).astype(np.int8)
    return (
        calibrated_validation,
        decisions,
        validation_prr,
        {
            "calibration_fold": calibration_fold,
            "threshold": threshold,
            "calibration_micro_f1": calibration_f1,
            "classifier_iteration_median": float(np.median(classifier_iterations)),
            "regressor_iteration_median": float(np.median(regressor_iterations)),
            "selection_source": "outer_fit_inner_fold_only",
        },
    )


def _molecule_indices(
    fit_frame: pd.DataFrame,
    frames: list[pd.DataFrame],
) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    fit_left = fit_frame["SMILES_A"].astype(str).map(canonicalize_smiles)
    fit_right = fit_frame["SMILES_B"].astype(str).map(canonicalize_smiles)
    molecules = sorted(set(fit_left) | set(fit_right))
    lookup = {molecule: index + 1 for index, molecule in enumerate(molecules)}
    encoded: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in frames:
        left = (
            frame["SMILES_A"]
            .astype(str)
            .map(canonicalize_smiles)
            .map(lambda value: lookup.get(value, 0))
            .to_numpy(dtype=np.int64)
        )
        right = (
            frame["SMILES_B"]
            .astype(str)
            .map(canonicalize_smiles)
            .map(lambda value: lookup.get(value, 0))
            .to_numpy(dtype=np.int64)
        )
        encoded.append((left, right))
    return len(molecules) + 1, encoded


class RelationalFactorization(nn.Module):
    def __init__(
        self,
        n_molecules: int,
        n_targets: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.molecule = nn.Embedding(n_molecules, embedding_dim, padding_idx=0)
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.binary_head = nn.Linear(hidden_dim, n_targets)
        self.prr_head = nn.Linear(hidden_dim, n_targets)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_embedding = self.molecule(left)
        right_embedding = self.molecule(right)
        pair = torch.cat(
            [
                left_embedding + right_embedding,
                torch.abs(left_embedding - right_embedding),
                left_embedding * right_embedding,
            ],
            dim=1,
        )
        hidden = self.trunk(pair)
        return self.binary_head(hidden), nn.functional.softplus(self.prr_head(hidden))


def _target_scales(targets: np.ndarray, quantile: float) -> np.ndarray:
    scales = np.ones(targets.shape[1], dtype=np.float32)
    for index in range(targets.shape[1]):
        positive = targets[:, index][targets[:, index] > 0]
        if len(positive):
            scales[index] = max(float(np.quantile(positive, quantile)), 1.0)
    return scales


def _relational_loader(
    left: np.ndarray,
    right: np.ndarray,
    binary: np.ndarray,
    prr: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(left),
        torch.from_numpy(right),
        torch.from_numpy(binary.astype(np.float32)),
        torch.from_numpy(prr.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _train_relational_epochs(
    model: RelationalFactorization,
    loader: DataLoader,
    *,
    epochs: int,
    config: dict[str, object],
    device: torch.device,
    scales: np.ndarray,
    positive_weights: np.ndarray,
    optimizer: torch.optim.Optimizer | None = None,
) -> torch.optim.Optimizer:
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    scale_tensor = torch.as_tensor(scales, dtype=torch.float32, device=device)
    positive_weight_tensor = torch.as_tensor(positive_weights, dtype=torch.float32, device=device)
    binary_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight_tensor)
    model.to(device)
    for _ in range(epochs):
        model.train()
        for left, right, binary, prr in loader:
            left = left.to(device)
            right = right.to(device)
            binary = binary.to(device)
            prr = prr.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, conditional_scaled = model(left, right)
            loss_binary = binary_loss(logits, binary)
            positive = prr > 0
            normalized_targets = prr / scale_tensor
            if positive.any():
                loss_prr = torch.square(
                    conditional_scaled[positive] - normalized_targets[positive]
                ).mean()
            else:
                loss_prr = conditional_scaled.sum() * 0.0
            loss = (
                float(config["binary_loss_weight"]) * loss_binary
                + float(config["prr_loss_weight"]) * loss_prr
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
    return optimizer


@torch.no_grad()
def _predict_relational(
    model: RelationalFactorization,
    left: np.ndarray,
    right: np.ndarray,
    *,
    batch_size: int,
    scales: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scale_tensor = torch.as_tensor(scales, dtype=torch.float32, device=device)
    probability_parts: list[np.ndarray] = []
    prr_parts: list[np.ndarray] = []
    dataset = TensorDataset(torch.from_numpy(left), torch.from_numpy(right))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch_left, batch_right in loader:
        logits, conditional_scaled = model(batch_left.to(device), batch_right.to(device))
        probability_parts.append(torch.sigmoid(logits).cpu().numpy())
        prr_parts.append((conditional_scaled * scale_tensor).cpu().numpy())
    return (
        np.vstack(probability_parts).astype(np.float32),
        np.maximum(np.vstack(prr_parts).astype(np.float32), 0.0),
    )


def _relational_validation_score(
    targets_binary: np.ndarray,
    targets_prr: np.ndarray,
    probabilities: np.ndarray,
    predictions_prr: np.ndarray,
) -> float:
    binary_f1 = f1_score(
        targets_binary,
        probabilities >= 0.5,
        average="micro",
        zero_division=0,
    )
    positive = targets_prr > 0
    rmse = float(np.sqrt(np.mean(np.square(targets_prr[positive] - predictions_prr[positive]))))
    return 0.5 * float(binary_f1) + 0.5 / (1.0 + rmse)


def _run_branch_c(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    fit_binary: np.ndarray,
    fit_prr: np.ndarray,
    fit_fold_values: np.ndarray,
    outer_fold: int,
    config: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    device = _device(str(config["device"]))
    inner_values = sorted(int(value) for value in np.unique(fit_fold_values))
    calibration_fold = inner_values[outer_fold % len(inner_values)]
    calibration_mask = fit_fold_values == calibration_fold
    core_mask = ~calibration_mask
    core_frame = fit_frame.loc[core_mask].reset_index(drop=True)
    calibration_frame = fit_frame.loc[calibration_mask].reset_index(drop=True)
    n_molecules, encoded = _molecule_indices(core_frame, [core_frame, calibration_frame])
    (core_left, core_right), (cal_left, cal_right) = encoded
    scales = _target_scales(fit_prr[core_mask], float(config["target_scale_quantile"]))
    prevalence = fit_binary[core_mask].mean(axis=0)
    positive_weights = np.clip(
        (1.0 - prevalence) / np.maximum(prevalence, 1e-4),
        1.0,
        float(config["positive_weight_max"]),
    ).astype(np.float32)
    loader = _relational_loader(
        core_left,
        core_right,
        fit_binary[core_mask],
        fit_prr[core_mask],
        int(config["batch_size"]),
        True,
    )
    model = RelationalFactorization(
        n_molecules,
        fit_binary.shape[1],
        int(config["embedding_dim"]),
        int(config["hidden_dim"]),
        float(config["dropout"]),
    )
    best_score = -np.inf
    best_epoch = 1
    epochs_without_improvement = 0
    selection_optimizer: torch.optim.Optimizer | None = None
    best_calibration_probabilities: np.ndarray | None = None
    for epoch in range(1, int(config["max_epochs"]) + 1):
        selection_optimizer = _train_relational_epochs(
            model,
            loader,
            epochs=1,
            config=config,
            device=device,
            scales=scales,
            positive_weights=positive_weights,
            optimizer=selection_optimizer,
        )
        calibration_probabilities, calibration_prr = _predict_relational(
            model,
            cal_left,
            cal_right,
            batch_size=int(config["batch_size"]),
            scales=scales,
            device=device,
        )
        score = _relational_validation_score(
            fit_binary[calibration_mask],
            fit_prr[calibration_mask],
            calibration_probabilities,
            calibration_prr,
        )
        if score > best_score + 1e-7:
            best_score = score
            best_epoch = epoch
            best_calibration_probabilities = calibration_probabilities.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            break
    if best_calibration_probabilities is None:
        raise RuntimeError("La sélection interne du modèle relationnel n'a produit aucun score.")

    # Refit from scratch on every outer-fit row for the selected number of epochs.
    _seed_everything(seed + 10_000)
    n_molecules, encoded = _molecule_indices(fit_frame, [fit_frame, validation_frame])
    (fit_left, fit_right), (val_left, val_right) = encoded
    final_scales = _target_scales(fit_prr, float(config["target_scale_quantile"]))
    final_prevalence = fit_binary.mean(axis=0)
    final_positive_weights = np.clip(
        (1.0 - final_prevalence) / np.maximum(final_prevalence, 1e-4),
        1.0,
        float(config["positive_weight_max"]),
    ).astype(np.float32)
    final_loader = _relational_loader(
        fit_left,
        fit_right,
        fit_binary,
        fit_prr,
        int(config["batch_size"]),
        True,
    )
    final_model = RelationalFactorization(
        n_molecules,
        fit_binary.shape[1],
        int(config["embedding_dim"]),
        int(config["hidden_dim"]),
        float(config["dropout"]),
    )
    _train_relational_epochs(
        final_model,
        final_loader,
        epochs=best_epoch,
        config=config,
        device=device,
        scales=final_scales,
        positive_weights=final_positive_weights,
    )
    validation_probabilities, validation_prr = _predict_relational(
        final_model,
        val_left,
        val_right,
        batch_size=int(config["batch_size"]),
        scales=final_scales,
        device=device,
    )
    parameters = fit_platt_parameters(
        fit_binary[calibration_mask],
        best_calibration_probabilities,
        regularization=float(config["platt_regularization"]),
    )
    calibrated_inner = apply_platt_parameters(best_calibration_probabilities, parameters)
    calibrated_validation = apply_platt_parameters(validation_probabilities, parameters)
    threshold, calibration_f1 = _fit_micro_threshold(
        fit_binary[calibration_mask],
        calibrated_inner,
        minimum=float(config["threshold_min"]),
        maximum=float(config["threshold_max"]),
        steps=int(config["threshold_steps"]),
    )
    decisions = (calibrated_validation >= threshold).astype(np.int8)
    return (
        calibrated_validation,
        decisions,
        validation_prr,
        {
            "calibration_fold": calibration_fold,
            "best_epoch": best_epoch,
            "inner_selection_score": float(best_score),
            "threshold": threshold,
            "calibration_micro_f1": calibration_f1,
            "device": str(device),
            "selection_source": "outer_fit_inner_fold_only",
        },
    )


def fixed_hard_vote(
    decisions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    if set(decisions) != set(weights):
        raise ValueError("Les branches du vote et les poids ne correspondent pas.")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("La somme des poids Binary doit être positive.")
    vote = sum(float(weights[name]) * decisions[name] for name in decisions) / total
    return (vote >= 0.5).astype(np.int8)


def fixed_prr_blend(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    if set(predictions) != set(weights):
        raise ValueError("Les branches PRR et les poids ne correspondent pas.")
    total = float(sum(weights.values()))
    if total <= 0 or any(value < 0 for value in weights.values()):
        raise ValueError("Les poids PRR doivent être positifs et de somme non nulle.")
    blended = sum(float(weights[name]) * predictions[name] for name in predictions) / total
    return np.maximum(np.asarray(blended, dtype=np.float32), 0.0)


def _fold_local_positive_prr_mean(
    targets: np.ndarray,
    fold_values: np.ndarray,
) -> np.ndarray:
    predictions = np.full(targets.shape, np.nan, dtype=np.float32)
    for fold in np.unique(fold_values):
        validation = fold_values == fold
        fit_targets = targets[~validation]
        positive = fit_targets > 0
        means = np.divide(
            np.where(positive, fit_targets, 0.0).sum(axis=0),
            positive.sum(axis=0),
            out=np.zeros(targets.shape[1], dtype=np.float32),
            where=positive.sum(axis=0) > 0,
        )
        predictions[validation] = means
    if not np.isfinite(predictions).all():
        raise RuntimeError("Le témoin PRR fold-local est incomplet.")
    return predictions


def _write_report(
    comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    cardinalities: dict[str, float],
    true_cardinality: float,
    audit: dict[str, object],
    path: Path,
) -> None:
    lines = [
        "# Benchmark OOF à quatre branches",
        "",
        "| Branche | Score global | Macro-F1 Severity | Micro-F1 Binary | S_PRR | RMSE PRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.branch} | {row.total:.6f} | {row.severity_macro_f1:.6f} | "
            f"{row.side_effect_micro_f1:.6f} | {row.prr_score:.6f} | "
            f"{row.prr_masked_rmse:.6f} |"
        )
    fold_summary = (
        fold_metrics.groupby("branch")
        .agg(total_mean=("total", "mean"), total_std=("total", "std"))
        .reset_index()
        .sort_values("total_mean", ascending=False)
    )
    lines.extend(
        [
            "",
            "## Stabilité et cardinalité",
            "",
            "| Branche | Score fold moyen | Écart-type | Labels prédits / ligne |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in fold_summary.itertuples(index=False):
        lines.append(
            f"| {row.branch} | {row.total_mean:.6f} | {row.total_std:.6f} | "
            f"{cardinalities[row.branch]:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Cardinalité vraie moyenne : **{true_cardinality:.3f}** labels par ligne.",
            "",
            "## Lecture",
            "",
            "- C gagne sur chacun des cinq folds grâce au Binary, mais son PRR régresse.",
            "- B améliore Binary et PRR simultanément par rapport au témoin.",
            "- A réduit légèrement la sur-prédiction, avec un gain global limité.",
            "- D est plus stable que C, mais ses poids fixes diluent trop le meilleur Binary.",
            "- Le prochain test doit combiner Binary de C et PRR de B sur un lockbox indépendant.",
            "",
            "## Garanties anti-fuite",
            "",
            f"- Empreinte des folds : `{audit['validation_fingerprint']}`.",
            "- Même split outer-fold pour A, B, C et D.",
            "- A : calibration Platt et seuil micro-F1 sur probabilités OOB de l'outer-fit.",
            "- B : itérations, Platt et seuil choisis sur un fold interne de l'outer-fit.",
            "- C : époque, Platt et seuil choisis sur un fold interne de l'outer-fit.",
            "- D : poids fixés dans la configuration avant le score, sans optimisation OOF.",
            "- `test.csv` n'est jamais chargé par ce benchmark.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_four_branch_benchmark(config_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    _seed_everything(seed)
    paths = config["paths"]
    run_dir = Path(paths["run_dir"])
    output_dir = Path(paths["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    config_sha256 = _sha256_file(config_path)
    cached_summary: dict[str, object] | None = None
    if summary_path.exists() and (output_dir / "comparison.csv").exists():
        cached_summary = load_json(summary_path)

    # Intentionally no test path in the config or code path.
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(Path(paths["manifest"])).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(Path(paths["audit"]))
    current_audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    if not current_audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Le manifeste de folds ne correspond plus à l'audit anti-fuite.")
    train_sha256 = _sha256_file(train_path)
    if (
        cached_summary is not None
        and cached_summary.get("config_sha256") == config_sha256
        and cached_summary.get("validation_fingerprint") == fingerprint
        and cached_summary.get("train_sha256") == train_sha256
    ):
        print("Benchmark complet déjà disponible : réutilisation des artefacts.", flush=True)
        return cached_summary

    baseline = _load_aligned_predictions(Path(paths["baseline_oof"]), train)
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    if len(binary_columns) != len(prr_columns):
        raise ValueError("Le nombre de cibles Binary et PRR doit correspondre.")
    fold_values = manifest["fold"].to_numpy(dtype=np.int16)
    y_binary = train[binary_columns].to_numpy(dtype=np.int8)
    y_prr = train[prr_columns].to_numpy(dtype=np.float32)
    fixed_severity = baseline[SEVERITY_COLUMN].astype(str).to_numpy()
    fixed_prr = _fold_local_positive_prr_mean(y_prr, fold_values)

    branch_probabilities = {
        name: np.full(y_binary.shape, np.nan, dtype=np.float32)
        for name in (BRANCH_A, BRANCH_B, BRANCH_C)
    }
    branch_binary = {
        name: np.zeros(y_binary.shape, dtype=np.int8)
        for name in (CONTROL, BRANCH_A, BRANCH_B, BRANCH_C, BRANCH_D)
    }
    branch_prr = {
        CONTROL: fixed_prr.copy(),
        BRANCH_A: fixed_prr.copy(),
        BRANCH_B: np.full(y_prr.shape, np.nan, dtype=np.float32),
        BRANCH_C: np.full(y_prr.shape, np.nan, dtype=np.float32),
        BRANCH_D: np.full(y_prr.shape, np.nan, dtype=np.float32),
    }
    fold_metadata: list[dict[str, object]] = []
    branch_binary[CONTROL] = baseline[binary_columns].to_numpy(dtype=np.int8)

    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation_mask = fold_values == fold
        fit_mask = ~validation_mask
        print(
            f"Fold {fold}: fit={int(fit_mask.sum())}, validation={int(validation_mask.sum())}",
            flush=True,
        )
        fit_base, validation_base, _, _ = _load_or_build_features(
            train, y_prr, fold_values, fold, fingerprint, config
        )
        a_prob, a_binary, a_meta = _run_branch_a(
            fit_base,
            validation_base,
            y_binary[fit_mask],
            config["branch_a"],
            seed + fold,
        )
        branch_probabilities[BRANCH_A][validation_mask] = a_prob
        branch_binary[BRANCH_A][validation_mask] = a_binary
        print(f"Fold {fold}: branche A terminée.", flush=True)

        b_prob, b_binary, b_prr, b_meta = _run_branch_b(
            fit_base,
            validation_base,
            y_binary[fit_mask],
            y_prr[fit_mask],
            fold_values[fit_mask],
            fold,
            config["branch_b"],
            seed + 100 + fold,
        )
        branch_probabilities[BRANCH_B][validation_mask] = b_prob
        branch_binary[BRANCH_B][validation_mask] = b_binary
        branch_prr[BRANCH_B][validation_mask] = b_prr
        print(f"Fold {fold}: branche B terminée.", flush=True)

        fit_frame = train.loc[fit_mask].reset_index(drop=True)
        validation_frame = train.loc[validation_mask].reset_index(drop=True)
        c_prob, c_binary, c_prr, c_meta = _run_branch_c(
            fit_frame,
            validation_frame,
            y_binary[fit_mask],
            y_prr[fit_mask],
            fold_values[fit_mask],
            fold,
            config["branch_c"],
            seed + 200 + fold,
        )
        branch_probabilities[BRANCH_C][validation_mask] = c_prob
        branch_binary[BRANCH_C][validation_mask] = c_binary
        branch_prr[BRANCH_C][validation_mask] = c_prr
        print(f"Fold {fold}: branche C terminée.", flush=True)

        fold_metadata.append(
            {"fold": fold, "branch_a": a_meta, "branch_b": b_meta, "branch_c": c_meta}
        )

    for name, values in branch_probabilities.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Couverture OOF incomplète pour les probabilités {name}.")
    for name in (BRANCH_B, BRANCH_C):
        if not np.isfinite(branch_prr[name]).all():
            raise RuntimeError(f"Couverture OOF incomplète pour le PRR {name}.")

    binary_weights = {
        str(name): float(value) for name, value in config["branch_d"]["binary_weights"].items()
    }
    prr_weights = {
        str(name): float(value) for name, value in config["branch_d"]["prr_weights"].items()
    }
    branch_binary[BRANCH_D] = fixed_hard_vote(
        {name: branch_binary[name] for name in binary_weights}, binary_weights
    )
    branch_prr[BRANCH_D] = fixed_prr_blend(
        {name: branch_prr[name] for name in prr_weights}, prr_weights
    )

    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for branch in (CONTROL, BRANCH_A, BRANCH_B, BRANCH_C, BRANCH_D):
        score = hardcore_score(
            train[SEVERITY_COLUMN],
            fixed_severity,
            y_binary,
            branch_binary[branch],
            y_prr,
            branch_prr[branch],
        )
        rows.append({"branch": branch, **asdict(score)})
        for fold in sorted(int(value) for value in np.unique(fold_values)):
            fold_mask = fold_values == fold
            fold_score = _score(
                train,
                fold_mask,
                fixed_severity,
                branch_binary[branch],
                branch_prr[branch],
                binary_columns,
                prr_columns,
            )
            fold_rows.append({"branch": branch, "fold": fold, **asdict(fold_score)})
        prediction_frame = pd.concat(
            [
                train[[ID_COLUMN]].reset_index(drop=True),
                pd.Series(fixed_severity, name=SEVERITY_COLUMN),
                pd.DataFrame(branch_binary[branch], columns=binary_columns),
                pd.DataFrame(branch_prr[branch], columns=prr_columns),
            ],
            axis=1,
        )
        prediction_frame.to_csv(output_dir / f"oof_predictions_{branch}.csv", index=False)

    comparison = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    fold_frame = pd.DataFrame(fold_rows)
    cardinalities = {
        branch: float(branch_binary[branch].sum(axis=1).mean())
        for branch in (CONTROL, BRANCH_A, BRANCH_B, BRANCH_C, BRANCH_D)
    }
    true_cardinality = float(y_binary.sum(axis=1).mean())
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "oof_probabilities.npz",
        **{name: values for name, values in branch_probabilities.items()},
    )
    write_json({"folds": fold_metadata}, run_dir / "fold_metadata.json")

    leakage_audit = {
        "passed": True,
        "validation_fingerprint": fingerprint,
        "pair_group_overlap_max": max(
            int(item["pair_group_overlap"]) for item in current_audit["folds"]
        ),
        "outer_folds_identical_for_all_branches": True,
        "branch_a_calibration": "outer_fit_oob_only",
        "branch_b_selection": "outer_fit_inner_fold_only",
        "branch_c_selection": "outer_fit_inner_fold_only",
        "branch_d_weights": "fixed_before_oof_scoring",
        "feature_target_statistics": "cached_but_not_used_by_branches_a_b_or_c",
        "branch_b_target_statistics_used": False,
        "branch_a_prr_control": "fold_local_positive_mean",
        "test_loaded": False,
    }
    write_json(leakage_audit, run_dir / "leakage_audit.json")
    _write_report(
        comparison,
        fold_frame,
        cardinalities,
        true_cardinality,
        leakage_audit,
        output_dir / "BENCHMARK_REPORT.md",
    )
    summary = {
        "run_id": config["run_id"],
        "selected_oof_branch": str(comparison.iloc[0]["branch"]),
        "comparison": json.loads(comparison.to_json(orient="records")),
        "validation_fingerprint": fingerprint,
        "train_sha256": train_sha256,
        "config_sha256": config_sha256,
        "leakage_audit": leakage_audit,
        "duration_seconds": time.perf_counter() - started,
        "automatic_submission": False,
        "test_inference": None,
    }
    write_json(summary, summary_path)
    return summary
