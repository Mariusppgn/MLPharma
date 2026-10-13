from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch import nn

from .constants import ID_COLUMN, PRR_PREFIX
from .four_branch_benchmark import _load_or_build_features
from .molecule_priors import (
    build_cross_fitted_molecule_priors,
    fit_molecule_prior_tables,
    transform_molecule_priors,
)
from .prr_pytorch import masked_prr_metrics, seed_everything
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, canonicalize_smiles, split_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA demande mais indisponible.")
    return result


def _batches(n_rows: int, batch_size: int, seed: int) -> list[np.ndarray]:
    rows = np.arange(n_rows)
    np.random.default_rng(seed).shuffle(rows)
    return [rows[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    safe = values.clamp_min(0.05)
    return torch.where(safe > 20.0, safe, torch.log(torch.expm1(safe).clamp_min(1e-7)))


def _positive_update(prior: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return nn.functional.softplus(_inverse_softplus(prior) + delta)


class ResidualPRRNetwork(nn.Module):
    """Predict a correction around a leakage-safe molecular prior."""

    def __init__(self, input_dim: int, n_targets: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.delta = nn.Linear(previous, n_targets)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, features: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return _positive_update(prior, self.delta(self.backbone(features)))


class MaskedFactorizationPRR(nn.Module):
    """Low-rank event factorization without an identity shortcut gate."""

    def __init__(
        self,
        input_dim: int,
        n_identities: int,
        n_targets: int,
        *,
        identity_dim: int,
        rank: int,
        dropout: float,
        identity_mask_rate: float,
    ) -> None:
        super().__init__()
        self.identity_mask_rate = identity_mask_rate
        self.identity = nn.Embedding(n_identities, identity_dim, padding_idx=0)
        self.structure = nn.Sequential(
            nn.Linear(input_dim, rank),
            nn.LayerNorm(rank),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.identity_pair = nn.Sequential(
            nn.Linear(identity_dim * 3, rank),
            nn.LayerNorm(rank),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.target = nn.Parameter(torch.randn(n_targets, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_targets))

    @staticmethod
    def symmetric_pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat([left + right, torch.abs(left - right), left * right], dim=1)

    def _masked_ids(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.identity_mask_rate <= 0:
            return values
        keep = torch.rand(values.shape, device=values.device) >= self.identity_mask_rate
        return values * keep.to(values.dtype)

    def forward(
        self,
        features: torch.Tensor,
        left_id: torch.Tensor,
        right_id: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        left = self.identity(self._masked_ids(left_id))
        right = self.identity(self._masked_ids(right_id))
        latent = self.structure(features) + self.identity_pair(self.symmetric_pair(left, right))
        delta = latent @ self.target.T / math.sqrt(self.target.shape[1]) + self.bias
        return _positive_update(prior, delta)


def _masked_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    use_mse: bool,
    huber_beta: float,
    log_aux_weight: float,
) -> torch.Tensor:
    positive = target > 0
    if not positive.any():
        return prediction.sum() * 0.0
    if use_mse:
        return nn.functional.mse_loss(prediction[positive], target[positive])
    raw = nn.functional.smooth_l1_loss(
        prediction[positive], target[positive], beta=huber_beta
    )
    log = nn.functional.mse_loss(
        torch.log1p(prediction[positive]), torch.log1p(target[positive])
    )
    return raw + log_aux_weight * log


def _fit_residual_network(
    train_features: np.ndarray,
    train_prior: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_prior: np.ndarray,
    config: dict[str, object],
    device: torch.device,
    seed: int,
) -> np.ndarray:
    seed_everything(seed)
    model = ResidualPRRNetwork(
        train_features.shape[1],
        train_targets.shape[1],
        [int(value) for value in config["hidden_dims"]],
        float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x = torch.from_numpy(train_features.astype(np.float32, copy=False)).to(device)
    prior = torch.from_numpy(train_prior.astype(np.float32, copy=False)).to(device)
    targets = torch.from_numpy(train_targets.astype(np.float32, copy=False)).to(device)
    epochs = int(config["epochs"])
    mse_epochs = int(config["mse_epochs"])
    for epoch in range(epochs):
        model.train()
        for rows in _batches(len(x), int(config["batch_size"]), seed + epoch):
            prediction = model(x[rows], prior[rows])
            loss = _masked_loss(
                prediction,
                targets[rows],
                use_mse=epoch >= epochs - mse_epochs,
                huber_beta=float(config["huber_beta"]),
                log_aux_weight=float(config["log_aux_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(validation_features), int(config["batch_size"])):
            stop = start + int(config["batch_size"])
            batch_x = torch.from_numpy(validation_features[start:stop]).to(device)
            batch_prior = torch.from_numpy(validation_prior[start:stop]).to(device)
            predictions.append(model(batch_x, batch_prior).cpu().numpy())
    return np.vstack(predictions).astype(np.float32)


def _molecule_ids(
    fit: pd.DataFrame, validation: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    molecules = sorted(
        {
            canonicalize_smiles(str(value))
            for column in ("SMILES_A", "SMILES_B")
            for value in fit[column]
        }
    )
    lookup = {value: index + 1 for index, value in enumerate(molecules)}
    fit_values = pd.concat([fit.SMILES_A, fit.SMILES_B]).astype(str).map(canonicalize_smiles)
    frequencies = fit_values.value_counts().to_dict()

    def encode(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_smiles = frame.SMILES_A.astype(str).map(canonicalize_smiles)
        right_smiles = frame.SMILES_B.astype(str).map(canonicalize_smiles)
        left = left_smiles.map(lambda value: lookup.get(value, 0)).to_numpy(np.int64)
        right = right_smiles.map(lambda value: lookup.get(value, 0)).to_numpy(np.int64)
        seen_count = (left > 0).astype(np.int8) + (right > 0).astype(np.int8)
        max_frequency = max(max(frequencies.values(), default=1), 1)
        left_frequency = left_smiles.map(
            lambda value: math.log1p(frequencies.get(value, 0)) / math.log1p(max_frequency)
        ).to_numpy(np.float32)
        right_frequency = right_smiles.map(
            lambda value: math.log1p(frequencies.get(value, 0)) / math.log1p(max_frequency)
        ).to_numpy(np.float32)
        familiarity = np.column_stack(
            [seen_count, 0.5 * (left_frequency + right_frequency)]
        ).astype(np.float32)
        return left, right, familiarity

    fit_left, fit_right, _ = encode(fit)
    validation_left, validation_right, familiarity = encode(validation)
    return fit_left, fit_right, validation_left, validation_right, familiarity


def _fit_factorization(
    train_features: np.ndarray,
    train_prior: np.ndarray,
    train_targets: np.ndarray,
    fit_frame: pd.DataFrame,
    validation_features: np.ndarray,
    validation_prior: np.ndarray,
    validation_frame: pd.DataFrame,
    config: dict[str, object],
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(seed)
    fit_left, fit_right, val_left, val_right, familiarity = _molecule_ids(
        fit_frame, validation_frame
    )
    n_identities = int(max(fit_left.max(initial=0), fit_right.max(initial=0))) + 1
    model = MaskedFactorizationPRR(
        train_features.shape[1],
        n_identities,
        train_targets.shape[1],
        identity_dim=int(config["identity_dim"]),
        rank=int(config["rank"]),
        dropout=float(config["dropout"]),
        identity_mask_rate=float(config["identity_mask_rate"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x = torch.from_numpy(train_features.astype(np.float32, copy=False)).to(device)
    prior = torch.from_numpy(train_prior.astype(np.float32, copy=False)).to(device)
    targets = torch.from_numpy(train_targets.astype(np.float32, copy=False)).to(device)
    left = torch.from_numpy(fit_left).to(device)
    right = torch.from_numpy(fit_right).to(device)
    epochs = int(config["epochs"])
    mse_epochs = int(config["mse_epochs"])
    for epoch in range(epochs):
        model.train()
        for rows in _batches(len(x), int(config["batch_size"]), seed + epoch):
            prediction = model(x[rows], left[rows], right[rows], prior[rows])
            loss = _masked_loss(
                prediction,
                targets[rows],
                use_mse=epoch >= epochs - mse_epochs,
                huber_beta=float(config["huber_beta"]),
                log_aux_weight=float(config["log_aux_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(validation_features), int(config["batch_size"])):
            stop = start + int(config["batch_size"])
            batch_x = torch.from_numpy(validation_features[start:stop]).to(device)
            batch_prior = torch.from_numpy(validation_prior[start:stop]).to(device)
            batch_left = torch.from_numpy(val_left[start:stop]).to(device)
            batch_right = torch.from_numpy(val_right[start:stop]).to(device)
            predictions.append(
                model(batch_x, batch_left, batch_right, batch_prior).cpu().numpy()
            )
    return np.vstack(predictions).astype(np.float32), familiarity


def select_high_sse_targets(targets: np.ndarray, top_k: int) -> np.ndarray:
    """Select hard targets using fit-only dispersion, never validation residuals."""

    scores = np.zeros(targets.shape[1], dtype=np.float64)
    for target in range(targets.shape[1]):
        positive = targets[:, target][targets[:, target] > 0]
        if len(positive) > 1:
            scores[target] = float(np.square(positive - positive.mean()).sum())
    return np.argsort(scores)[::-1][: min(top_k, targets.shape[1])]


def _targeted_gbdt(
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    fit_targets: np.ndarray,
    fallback: np.ndarray,
    target_indices: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> np.ndarray:
    def fit_one(target: int) -> tuple[int, np.ndarray]:
        values = fit_targets[:, target]
        positive = values > 0
        if positive.sum() < int(config["minimum_positive"]):
            return target, fallback[:, target]
        model = lgb.LGBMRegressor(
            objective="regression_l2",
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            num_leaves=int(config["num_leaves"]),
            min_child_samples=int(config["min_child_samples"]),
            reg_lambda=float(config["reg_lambda"]),
            random_state=seed + int(target),
            deterministic=True,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(fit_features[positive], values[positive])
        prediction = np.maximum(model.predict(validation_features), 0.0).astype(np.float32)
        return target, prediction

    result = fallback.astype(np.float32, copy=True)
    fitted = Parallel(n_jobs=int(config["target_n_jobs"]), prefer="threads")(
        delayed(fit_one)(int(target)) for target in target_indices
    )
    for target, prediction in fitted:
        result[:, target] = prediction
    return result


def knn_positive_prr(
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    fit_targets: np.ndarray,
    validation_prior: np.ndarray,
    *,
    neighbors: int,
    temperature: float,
    prior_strength: float,
    dimensions: int,
    n_jobs: int,
) -> np.ndarray:
    dimensions = min(dimensions, fit_features.shape[1])
    count = min(neighbors, len(fit_features))
    model = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=n_jobs)
    model.fit(fit_features[:, :dimensions])
    distance, indices = model.kneighbors(validation_features[:, :dimensions])
    weights = np.exp(-distance / max(temperature, 1e-6)).astype(np.float32)
    neighbor_targets = fit_targets[indices]
    positive = neighbor_targets > 0
    weighted = weights[:, :, None]
    numerator = (weighted * np.where(positive, neighbor_targets, 0.0)).sum(axis=1)
    denominator = (weighted * positive).sum(axis=1)
    prediction = (numerator + prior_strength * validation_prior) / (
        denominator + prior_strength
    )
    return np.maximum(prediction, 0.0).astype(np.float32)


def regularized_simplex_weights(
    experts: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    if experts.ndim != 2:
        raise ValueError("experts doit etre une matrice 2D.")
    n_experts = experts.shape[1]
    if len(experts) < n_experts:
        return np.full(n_experts, 1.0 / n_experts, dtype=np.float32)
    uniform = np.full(n_experts, 1.0 / n_experts, dtype=np.float64)
    augmented_x = np.vstack([experts.astype(np.float64), math.sqrt(ridge) * np.eye(n_experts)])
    augmented_y = np.concatenate(
        [targets.astype(np.float64), math.sqrt(ridge) * uniform]
    )
    weights, _ = nnls(augmented_x, augmented_y)
    total = float(weights.sum())
    if total <= 1e-12:
        return uniform.astype(np.float32)
    return (weights / total).astype(np.float32)


def fit_moe_weights(
    targets: np.ndarray,
    experts: list[np.ndarray],
    familiarity: np.ndarray,
    *,
    ridge: float,
    minimum_regime_rows: int,
    regime_shrinkage: float,
) -> np.ndarray:
    """Fit target/regime weights on inner-calibration predictions only."""

    cube = np.stack(experts, axis=2)
    n_targets, n_experts = targets.shape[1], len(experts)
    weights = np.zeros((n_targets, 3, n_experts), dtype=np.float32)
    regimes = familiarity[:, 0].astype(np.int8)
    for target in range(n_targets):
        positive = targets[:, target] > 0
        global_weights = regularized_simplex_weights(
            cube[positive, target], targets[positive, target], ridge=ridge
        )
        for regime in range(3):
            mask = positive & (regimes == regime)
            if mask.sum() < minimum_regime_rows:
                weights[target, regime] = global_weights
                continue
            local = regularized_simplex_weights(
                cube[mask, target], targets[mask, target], ridge=ridge
            )
            strength = float(mask.sum()) / (float(mask.sum()) + regime_shrinkage)
            weights[target, regime] = strength * local + (1.0 - strength) * global_weights
    return weights


def apply_moe_weights(
    weights: np.ndarray, experts: list[np.ndarray], familiarity: np.ndarray
) -> np.ndarray:
    cube = np.stack(experts, axis=2)
    regimes = np.clip(familiarity[:, 0].astype(np.int8), 0, 2)
    prediction = np.empty(cube.shape[:2], dtype=np.float32)
    for row in range(len(prediction)):
        prediction[row] = np.sum(cube[row] * weights[:, regimes[row], :], axis=1)
    return np.maximum(prediction, 0.0)


def _prior_matrices(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    fit_targets: np.ndarray,
    fit_folds: np.ndarray,
    *,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fit_raw = build_cross_fitted_molecule_priors(
        fit, fit_targets, fit_folds, smoothing=smoothing
    )
    tables = fit_molecule_prior_tables(fit, fit_targets, smoothing=smoothing)
    validation_raw = transform_molecule_priors(validation, tables)
    scaler = StandardScaler().fit(fit_raw)
    fit_scaled = scaler.transform(fit_raw).astype(np.float32)
    validation_scaled = scaler.transform(validation_raw).astype(np.float32)
    n_targets = fit_targets.shape[1]
    return fit_raw[:, :n_targets], validation_raw[:, :n_targets], fit_scaled, validation_scaled


def _prediction_frame(
    train: pd.DataFrame, predictions: np.ndarray, prr_columns: list[str]
) -> pd.DataFrame:
    return pd.concat(
        [
            train[[ID_COLUMN]].reset_index(drop=True),
            pd.DataFrame(predictions, columns=prr_columns),
        ],
        axis=1,
    )


def run_prr_rupture_v2(config_path: Path) -> dict[str, object]:
    """Run six PRR improvements OOF. This workflow never reads test.csv."""

    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_everything(seed)
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(Path(paths["audit"]))
    audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    if not audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Le manifeste OOF ne correspond plus a l'audit anti-fuite.")

    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    targets = train[prr_columns].to_numpy(np.float32)
    folds = manifest["fold"].to_numpy()
    n_rows, n_targets = targets.shape
    device = _device(str(config["training"]["device"]))
    feature_config = yaml.safe_load(Path(paths["feature_cache_config"]).read_text("utf-8"))
    smoothing = float(feature_config["molecule_priors"]["smoothing"])
    expert_names = ["prior", "residual", "targeted_gbdt", "knn", "factorization"]
    predictions = {
        name: np.full((n_rows, n_targets), np.nan, dtype=np.float32)
        for name in [*expert_names, "adaptive_moe"]
    }
    fold_metadata: list[dict[str, object]] = []

    for outer_fold in sorted(int(value) for value in np.unique(folds)):
        validation_mask = folds == outer_fold
        fit_mask = ~validation_mask
        fit_frame = train.loc[fit_mask].reset_index(drop=True)
        validation_frame = train.loc[validation_mask].reset_index(drop=True)
        fit_targets = targets[fit_mask]
        fit_folds = folds[fit_mask]
        inner_values = sorted(int(value) for value in np.unique(fit_folds))
        calibration_fold = inner_values[outer_fold % len(inner_values)]
        calibration_mask = fit_folds == calibration_fold
        core_mask = ~calibration_mask
        print(
            f"PRR v2 fold {outer_fold}: core={core_mask.sum()} "
            f"calibration={calibration_mask.sum()} validation={validation_mask.sum()}",
            flush=True,
        )

        fit_base, validation_base, _, _ = _load_or_build_features(
            train, targets, folds, outer_fold, fingerprint, config
        )
        core_frame = fit_frame.loc[core_mask].reset_index(drop=True)
        calibration_frame = fit_frame.loc[calibration_mask].reset_index(drop=True)
        core_targets = fit_targets[core_mask]
        core_folds = fit_folds[core_mask]

        core_prior, calibration_prior, core_prior_scaled, calibration_prior_scaled = (
            _prior_matrices(
                core_frame,
                calibration_frame,
                core_targets,
                core_folds,
                smoothing=smoothing,
            )
        )
        core_features = np.hstack([fit_base[core_mask], core_prior_scaled]).astype(np.float32)
        calibration_features = np.hstack(
            [fit_base[calibration_mask], calibration_prior_scaled]
        ).astype(np.float32)
        calibration_residual = _fit_residual_network(
            core_features,
            core_prior,
            core_targets,
            calibration_features,
            calibration_prior,
            config["residual"],
            device,
            seed + outer_fold,
        )
        hard_targets = select_high_sse_targets(core_targets, int(config["gbdt"]["top_k"]))
        calibration_gbdt = _targeted_gbdt(
            core_features,
            calibration_features,
            core_targets,
            calibration_residual,
            hard_targets,
            config["gbdt"],
            seed + 1000 + outer_fold,
        )
        calibration_knn = knn_positive_prr(
            fit_base[core_mask],
            fit_base[calibration_mask],
            core_targets,
            calibration_prior,
            **config["knn"],
        )
        calibration_factorization, calibration_familiarity = _fit_factorization(
            fit_base[core_mask],
            core_prior,
            core_targets,
            core_frame,
            fit_base[calibration_mask],
            calibration_prior,
            calibration_frame,
            config["factorization"],
            device,
            seed + 2000 + outer_fold,
        )
        calibration_experts = [
            calibration_prior,
            calibration_residual,
            calibration_gbdt,
            calibration_knn,
            calibration_factorization,
        ]
        moe_weights = fit_moe_weights(
            fit_targets[calibration_mask],
            calibration_experts,
            calibration_familiarity,
            **config["mixture"],
        )

        fit_prior, validation_prior, fit_prior_scaled, validation_prior_scaled = (
            _prior_matrices(
                fit_frame,
                validation_frame,
                fit_targets,
                fit_folds,
                smoothing=smoothing,
            )
        )
        fit_features = np.hstack([fit_base, fit_prior_scaled]).astype(np.float32)
        validation_features = np.hstack(
            [validation_base, validation_prior_scaled]
        ).astype(np.float32)
        validation_residual = _fit_residual_network(
            fit_features,
            fit_prior,
            fit_targets,
            validation_features,
            validation_prior,
            config["residual"],
            device,
            seed + 100 + outer_fold,
        )
        validation_gbdt = _targeted_gbdt(
            fit_features,
            validation_features,
            fit_targets,
            validation_residual,
            hard_targets,
            config["gbdt"],
            seed + 1100 + outer_fold,
        )
        validation_knn = knn_positive_prr(
            fit_base,
            validation_base,
            fit_targets,
            validation_prior,
            **config["knn"],
        )
        validation_factorization, validation_familiarity = _fit_factorization(
            fit_base,
            fit_prior,
            fit_targets,
            fit_frame,
            validation_base,
            validation_prior,
            validation_frame,
            config["factorization"],
            device,
            seed + 2100 + outer_fold,
        )
        validation_experts = [
            validation_prior,
            validation_residual,
            validation_gbdt,
            validation_knn,
            validation_factorization,
        ]
        validation_moe = apply_moe_weights(
            moe_weights, validation_experts, validation_familiarity
        )
        for name, values in zip(expert_names, validation_experts, strict=True):
            predictions[name][validation_mask] = values
        predictions["adaptive_moe"][validation_mask] = validation_moe
        fold_metadata.append(
            {
                "fold": outer_fold,
                "calibration_fold": calibration_fold,
                "hard_target_indices": hard_targets.tolist(),
                "hard_target_names": [prr_columns[index] for index in hard_targets],
                "mean_moe_weights": moe_weights.mean(axis=(0, 1)).tolist(),
                "mean_validation_regime": validation_familiarity[:, 0].mean().item(),
            }
        )

    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("La couverture OOF PRR v2 est incomplete.")

    incumbent = pd.read_csv(paths["incumbent_oof"], low_memory=False)
    if not incumbent[ID_COLUMN].astype(str).equals(train[ID_COLUMN].astype(str)):
        raise ValueError("Les predictions OOF incumbent sont desalignees.")
    candidates = {
        "incumbent_blend": incumbent[prr_columns].to_numpy(np.float32),
        **predictions,
    }
    rows = []
    output_dir = Path(paths["output_dir"])
    run_dir = Path(paths["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, values in candidates.items():
        metrics = masked_prr_metrics(targets, values)
        rows.append({"candidate": name, **asdict(metrics)})
        _prediction_frame(train, values, prr_columns).to_csv(
            output_dir / f"oof_predictions_{name}.csv", index=False
        )
    comparison = pd.DataFrame(rows).sort_values("sprr", ascending=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    summary = {
        "run_id": config["run_id"],
        "train_sha256": _sha256(train_path),
        "validation_fingerprint": fingerprint,
        "comparison": comparison.to_dict(orient="records"),
        "fold_metadata": fold_metadata,
        "leakage_audit": {
            "passed": True,
            "pair_group_overlap_max": 0,
            "outer_validation_labels_used_for_fit": False,
            "stack_weights_fit_on_inner_calibration_only": True,
            "training_priors_cross_fitted": True,
            "hard_targets_selected_on_inner_core_only": True,
            "identity_masking_enabled": float(config["factorization"]["identity_mask_rate"])
            > 0,
            "test_loaded": False,
        },
        "automatic_submission": False,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(summary, run_dir / "summary.json")
    return summary
