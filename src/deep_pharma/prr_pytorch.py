from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .constants import ID_COLUMN, PRR_PREFIX
from .multimodal_features import MultimodalFeatureBuilder
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint

MethodName = Literal["progressive", "huber", "hurdle"]
PredictionMode = Literal["direct", "conditional", "expected"]
METHODS: tuple[MethodName, ...] = ("progressive", "huber", "hurdle")


@dataclass(frozen=True)
class PRRMetrics:
    masked_rmse: float
    sprr: float
    positive_count: int


class PRRNetwork(nn.Module):
    """One architecture for all losses; non-hurdle runs leave the binary head inactive."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.extend([nn.SiLU(), nn.Dropout(dropout)])
            previous_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.regression_head = nn.Linear(previous_dim, output_dim)
        self.binary_head = nn.Linear(previous_dim, output_dim)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(features)
        # Softplus makes the conditional/positive PRR prediction strictly positive.
        conditional_normalized = F.softplus(self.regression_head(hidden))
        binary_logits = self.binary_head(hidden)
        return conditional_normalized, binary_logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def masked_mse(prediction: Tensor, target: Tensor, positive_mask: Tensor) -> Tensor:
    if not positive_mask.any():
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[positive_mask], target[positive_mask])


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    positive_mask: Tensor,
    *,
    beta: float,
) -> Tensor:
    if not positive_mask.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(
        prediction[positive_mask],
        target[positive_mask],
        beta=beta,
    )


def progressive_raw_weight(epoch: int, log_only_epochs: int, transition_epochs: int) -> float:
    """Return 0 for log-only, linearly transition, then stay at 1 for raw masked MSE."""

    if epoch <= log_only_epochs:
        return 0.0
    if transition_epochs <= 0:
        return 1.0
    transition_position = epoch - log_only_epochs
    return float(min(1.0, transition_position / transition_epochs))


def masked_prr_metrics(target_raw: np.ndarray, prediction_raw: np.ndarray) -> PRRMetrics:
    if target_raw.shape != prediction_raw.shape:
        raise ValueError("Les formes PRR vraies et prédites doivent être identiques.")
    mask = target_raw > 0.0
    positive_count = int(mask.sum())
    if positive_count == 0:
        raise ValueError("Aucune valeur PRR positive pour calculer la métrique masquée.")
    error = target_raw[mask] - prediction_raw[mask]
    rmse = float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))
    return PRRMetrics(masked_rmse=rmse, sprr=1.0 / (1.0 + rmse), positive_count=positive_count)


def _target_scales(target_raw: np.ndarray, quantile: float) -> np.ndarray:
    scales = np.ones(target_raw.shape[1], dtype=np.float32)
    for column_index in range(target_raw.shape[1]):
        positives = target_raw[:, column_index][target_raw[:, column_index] > 0]
        if len(positives):
            scales[column_index] = max(float(np.quantile(positives, quantile)), 1e-6)
    return scales


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


def _builder_matches_config(
    builder: MultimodalFeatureBuilder, config: dict[str, object], seed: int
) -> bool:
    feature_config = config["features"]
    expected = {
        "text_components": int(feature_config["text_components"]),
        "morgan_components": int(feature_config["morgan_components"]),
        "word_max_features": int(feature_config["word_max_features"]),
        "char_max_features": int(feature_config["char_max_features"]),
        "min_df": int(feature_config["min_df"]),
        "include_warning_toxicity": bool(feature_config["include_warning_toxicity"]),
        "random_state": seed,
    }
    return all(getattr(builder, name) == value for name, value in expected.items())


def _device(training_config: dict[str, object]) -> torch.device:
    requested = str(training_config["device"])
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA a été demandé mais n'est pas disponible dans PyTorch.")
    return torch.device(requested)


def _loader(
    features: np.ndarray,
    targets_raw: np.ndarray,
    target_scales: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    normalized = targets_raw / target_scales
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32, copy=False)),
        torch.from_numpy(targets_raw.astype(np.float32, copy=False)),
        torch.from_numpy(normalized.astype(np.float32, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )


def _loss(
    method: MethodName,
    conditional_normalized: Tensor,
    binary_logits: Tensor,
    target_raw: Tensor,
    target_normalized: Tensor,
    target_scales: Tensor,
    *,
    epoch: int,
    config: dict[str, object],
) -> Tensor:
    training = config["training"]
    mask = target_raw > 0.0
    if method == "progressive":
        schedule = config["loss_schedules"]["progressive"]
        raw_weight = progressive_raw_weight(
            epoch,
            int(schedule["log_only_epochs"]),
            int(schedule["transition_epochs"]),
        )
        prediction_raw = conditional_normalized * target_scales
        log_loss = F.mse_loss(torch.log1p(prediction_raw), torch.log1p(target_raw))
        raw_loss = masked_mse(conditional_normalized, target_normalized, mask)
        return (1.0 - raw_weight) * log_loss + raw_weight * raw_loss
    if method == "huber":
        fine_tune_epochs = int(config["loss_schedules"]["huber"]["mse_finetune_epochs"])
        fine_tune_start = int(training["max_epochs"]) - fine_tune_epochs + 1
        prediction_raw = conditional_normalized * target_scales
        if epoch >= fine_tune_start:
            return masked_mse(prediction_raw, target_raw, mask)
        return masked_huber(
            prediction_raw,
            target_raw,
            mask,
            beta=float(training["huber_beta"]),
        )
    binary_target = mask.to(dtype=conditional_normalized.dtype)
    classification_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_target)
    regression_loss = masked_huber(
        conditional_normalized,
        target_normalized,
        mask,
        beta=float(training["huber_beta"]),
    )
    return (
        float(training["hurdle_bce_weight"]) * classification_loss
        + float(training["hurdle_regression_weight"]) * regression_loss
    )


def _run_epoch(
    model: PRRNetwork,
    loader: DataLoader,
    method: MethodName,
    target_scales: Tensor,
    config: dict[str, object],
    epoch: int,
    device: torch.device,
    optimizer: AdamW | None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0
    targets: list[np.ndarray] = []
    conditional_predictions: list[np.ndarray] = []
    expected_predictions: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for features, target_raw, target_normalized in loader:
            features = features.to(device)
            target_raw = target_raw.to(device)
            target_normalized = target_normalized.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            conditional_normalized, binary_logits = model(features)
            loss = _loss(
                method,
                conditional_normalized,
                binary_logits,
                target_raw,
                target_normalized,
                target_scales,
                epoch=epoch,
                config=config,
            )
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip_norm"])
                )
                optimizer.step()
            batch_rows = len(features)
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows
            conditional_raw = conditional_normalized * target_scales
            expected_raw = torch.sigmoid(binary_logits) * conditional_raw
            targets.append(target_raw.detach().cpu().numpy())
            conditional_predictions.append(conditional_raw.detach().cpu().numpy())
            expected_predictions.append(expected_raw.detach().cpu().numpy())
    return (
        total_loss / total_rows,
        np.vstack(targets),
        np.vstack(conditional_predictions),
        np.vstack(expected_predictions),
    )


def _architecture_signature(model: PRRNetwork) -> str:
    signature = [(name, list(parameter.shape)) for name, parameter in model.named_parameters()]
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def _method_metrics(
    method: MethodName,
    target: np.ndarray,
    conditional: np.ndarray,
    expected: np.ndarray,
    *,
    hurdle_selection_mode: str = "best",
) -> tuple[PredictionMode, PRRMetrics, dict[str, PRRMetrics]]:
    conditional_metrics = masked_prr_metrics(target, conditional)
    if method != "hurdle":
        return "direct", conditional_metrics, {"direct": conditional_metrics}
    expected_metrics = masked_prr_metrics(target, expected)
    candidates = {"conditional": conditional_metrics, "expected": expected_metrics}
    if hurdle_selection_mode in candidates:
        selected_mode = hurdle_selection_mode
        return selected_mode, candidates[selected_mode], candidates
    if hurdle_selection_mode != "best":
        raise ValueError("hurdle_selection_mode doit valoir 'best', 'conditional' ou 'expected'.")
    selected = max(candidates, key=lambda key: candidates[key].sprr)
    return selected, candidates[selected], candidates


def train_prr_method(
    method: MethodName,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    target_scales: np.ndarray,
    config: dict[str, object],
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    seed = int(config["seed"])
    seed_everything(seed)
    architecture = config["architecture"]
    training = config["training"]
    model = PRRNetwork(
        input_dim=train_features.shape[1],
        output_dim=train_targets.shape[1],
        hidden_dims=[int(value) for value in architecture["hidden_dims"]],
        dropout=float(architecture["dropout"]),
        use_batch_norm=bool(architecture["use_batch_norm"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
        min_lr=float(training["scheduler_min_lr"]),
    )
    loader_args = {
        "batch_size": int(training["batch_size"]),
        "seed": seed,
        "num_workers": int(training["num_workers"]),
    }
    train_loader = _loader(
        train_features, train_targets, target_scales, shuffle=True, **loader_args
    )
    train_evaluation_loader = _loader(
        train_features, train_targets, target_scales, shuffle=False, **loader_args
    )
    validation_loader = _loader(
        validation_features, validation_targets, target_scales, shuffle=False, **loader_args
    )
    target_scales_tensor = torch.from_numpy(target_scales).to(device)
    method_dir = run_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = method_dir / "best_checkpoint.pt"
    history: list[dict[str, object]] = []
    best_sprr = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    hurdle_selection_mode = str(training.get("hurdle_selection_mode", "best"))
    huber_schedule = config["loss_schedules"]["huber"]
    fine_tune_start = int(training["max_epochs"]) - int(huber_schedule["mse_finetune_epochs"]) + 1
    effective_min_epochs = int(training["min_epochs"])
    if method == "huber" and int(huber_schedule["mse_finetune_epochs"]) > 0:
        effective_min_epochs = max(effective_min_epochs, fine_tune_start)

    for epoch in range(1, int(training["max_epochs"]) + 1):
        training_phase = "main"
        if method == "huber" and epoch == fine_tune_start:
            training_phase = "raw_mse_finetune"
            restore_best = bool(huber_schedule.get("restore_best_before_finetune", False))
            if restore_best and checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"])
            if bool(huber_schedule.get("regression_head_only", False)):
                for parameter in model.parameters():
                    parameter.requires_grad = False
                for parameter in model.regression_head.parameters():
                    parameter.requires_grad = True
            fine_tune_lr = float(
                huber_schedule.get("finetune_learning_rate", training["learning_rate"])
            )
            optimizer = AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=fine_tune_lr,
                weight_decay=float(training["weight_decay"]),
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(training["scheduler_factor"]),
                patience=int(training["scheduler_patience"]),
                min_lr=float(training["scheduler_min_lr"]),
            )
            epochs_without_improvement = 0
        elif method == "huber" and epoch > fine_tune_start:
            training_phase = "raw_mse_finetune"
        _run_epoch(
            model,
            train_loader,
            method,
            target_scales_tensor,
            config,
            epoch,
            device,
            optimizer,
        )
        train_loss, train_target, train_conditional, train_expected = _run_epoch(
            model,
            train_evaluation_loader,
            method,
            target_scales_tensor,
            config,
            epoch,
            device,
            None,
        )
        validation_loss, validation_target, validation_conditional, validation_expected = (
            _run_epoch(
                model,
                validation_loader,
                method,
                target_scales_tensor,
                config,
                epoch,
                device,
                None,
            )
        )
        train_mode, train_metrics, train_candidates = _method_metrics(
            method,
            train_target,
            train_conditional,
            train_expected,
            hurdle_selection_mode=hurdle_selection_mode,
        )
        validation_mode, validation_metrics, validation_candidates = _method_metrics(
            method,
            validation_target,
            validation_conditional,
            validation_expected,
            hurdle_selection_mode=hurdle_selection_mode,
        )
        scheduler.step(validation_metrics.sprr)
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "training_phase": training_phase,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_prediction_mode": train_mode,
            "validation_prediction_mode": validation_mode,
            "train_masked_rmse": train_metrics.masked_rmse,
            "validation_masked_rmse": validation_metrics.masked_rmse,
            "train_sprr": train_metrics.sprr,
            "validation_sprr": validation_metrics.sprr,
        }
        if method == "hurdle":
            for split_name, candidates in (
                ("train", train_candidates),
                ("validation", validation_candidates),
            ):
                for mode, metrics in candidates.items():
                    epoch_record[f"{split_name}_{mode}_masked_rmse"] = metrics.masked_rmse
                    epoch_record[f"{split_name}_{mode}_sprr"] = metrics.sprr
        history.append(epoch_record)
        improved = validation_metrics.sprr > best_sprr + 1e-12
        if improved:
            best_sprr = validation_metrics.sprr
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "method": method,
                    "epoch": epoch,
                    "prediction_mode": validation_mode,
                    "validation_loss": validation_loss,
                    "validation_metrics": asdict(validation_metrics),
                    "architecture_signature": _architecture_signature(model),
                },
                checkpoint_path,
            )
        elif epoch >= effective_min_epochs:
            epochs_without_improvement += 1
        print(
            f"{method:11s} epoch={epoch:02d} val_loss={validation_loss:.5f} "
            f"val_RMSE={validation_metrics.masked_rmse:.5f} "
            f"val_SPRR={validation_metrics.sprr:.6f} mode={validation_mode}",
            flush=True,
        )
        if epoch >= effective_min_epochs and epochs_without_improvement >= int(
            training["patience"]
        ):
            break

    history_path = method_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    result = {
        "method": method,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_prediction_mode": checkpoint["prediction_mode"],
        "validation_loss": float(checkpoint["validation_loss"]),
        **checkpoint["validation_metrics"],
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "architecture_signature": checkpoint["architecture_signature"],
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(result, method_dir / "result.json")
    return result


def fit_prr_fixed_epochs(
    method: MethodName,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    target_scales: np.ndarray,
    config: dict[str, object],
    run_dir: Path,
    device: torch.device,
    *,
    epochs: int,
) -> dict[str, object]:
    """Refit a PRR network without consulting a validation or test set."""

    if epochs < 1:
        raise ValueError("epochs doit etre strictement positif.")
    seed = int(config["seed"])
    seed_everything(seed)
    architecture = config["architecture"]
    training = config["training"]
    model = PRRNetwork(
        input_dim=train_features.shape[1],
        output_dim=train_targets.shape[1],
        hidden_dims=[int(value) for value in architecture["hidden_dims"]],
        dropout=float(architecture["dropout"]),
        use_batch_norm=bool(architecture["use_batch_norm"]),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = _loader(
        train_features,
        train_targets,
        target_scales,
        batch_size=int(training["batch_size"]),
        seed=seed,
        num_workers=int(training["num_workers"]),
        shuffle=True,
    )
    scale_tensor = torch.from_numpy(target_scales).to(device)
    huber_schedule = config["loss_schedules"]["huber"]
    fine_tune_start = int(training["max_epochs"]) - int(
        huber_schedule["mse_finetune_epochs"]
    ) + 1
    for epoch in range(1, epochs + 1):
        if method == "huber" and epoch == fine_tune_start:
            if bool(huber_schedule.get("regression_head_only", False)):
                for parameter in model.parameters():
                    parameter.requires_grad = False
                for parameter in model.regression_head.parameters():
                    parameter.requires_grad = True
            optimizer = AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=float(huber_schedule.get("finetune_learning_rate", training["learning_rate"])),
                weight_decay=float(training["weight_decay"]),
            )
        _run_epoch(
            model,
            loader,
            method,
            scale_tensor,
            config,
            epoch,
            device,
            optimizer,
        )

    method_dir = run_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = method_dir / "fixed_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "method": method,
            "epoch": epochs,
            "prediction_mode": "direct",
            "architecture_signature": _architecture_signature(model),
        },
        checkpoint_path,
    )
    result = {
        "method": method,
        "best_epoch": epochs,
        "best_prediction_mode": "direct",
        "checkpoint": str(checkpoint_path),
        "architecture_signature": _architecture_signature(model),
        "selection_source": "fixed_before_refit",
    }
    write_json(result, method_dir / "fixed_result.json")
    return result


def predict_prr_arrays(
    result: dict[str, object],
    features: np.ndarray,
    target_scales: np.ndarray,
    config: dict[str, object],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Predict from one selected checkpoint without requiring target columns."""

    model, prediction_mode = _load_best_model(
        result,
        features.shape[1],
        len(target_scales),
        config,
        device,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features.astype(np.float32, copy=False))),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    conditional_batches: list[np.ndarray] = []
    expected_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    scale_tensor = torch.from_numpy(target_scales).to(device)
    with torch.inference_mode():
        for (batch_features,) in loader:
            conditional_normalized, binary_logits = model(batch_features.to(device))
            conditional = conditional_normalized * scale_tensor
            probability = torch.sigmoid(binary_logits)
            conditional_batches.append(conditional.cpu().numpy())
            probability_batches.append(probability.cpu().numpy())
            expected_batches.append((probability * conditional).cpu().numpy())
    conditional_array = np.vstack(conditional_batches)
    expected_array = np.vstack(expected_batches)
    selected_array = expected_array if prediction_mode == "expected" else conditional_array
    return {
        "selected": selected_array,
        "conditional": conditional_array,
        "expected": expected_array,
        "probability": np.vstack(probability_batches),
    }


def plot_training_history(
    history_path: Path,
    result: dict[str, object],
    output_path: Path,
) -> None:
    history = pd.read_csv(history_path)
    method = str(result["method"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"train": "#2563EB", "validation": "#D97706"}
    for split in ("train", "validation"):
        axes[0].plot(
            history["epoch"],
            history[f"{split}_loss"],
            label=split,
            color=colors[split],
        )
        axes[1].plot(
            history["epoch"],
            history[f"{split}_masked_rmse"],
            label=split,
            color=colors[split],
        )
        axes[2].plot(
            history["epoch"],
            history[f"{split}_sprr"],
            label=split,
            color=colors[split],
        )
    if method == "hurdle":
        axes[1].plot(
            history["epoch"],
            history["validation_expected_masked_rmse"],
            label="validation expected",
            color="#DB2777",
            linestyle="--",
        )
        axes[2].plot(
            history["epoch"],
            history["validation_expected_sprr"],
            label="validation expected",
            color="#DB2777",
            linestyle="--",
        )
    titles = ("Loss train / validation", "RMSE masqué sur PRR brut", "S_PRR = 1/(1+RMSE)")
    ylabels = ("Loss", "RMSE", "S_PRR")
    for axis, title, ylabel in zip(axes, titles, ylabels, strict=True):
        axis.axvline(
            int(result["best_epoch"]), color="#334155", linestyle=":", label="meilleure époque"
        )
        axis.set(title=title, ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.supxlabel("Époque", y=0.02)
    fig.suptitle(f"PRR PyTorch — méthode {method}", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_best_model(
    result: dict[str, object],
    input_dim: int,
    output_dim: int,
    config: dict[str, object],
    device: torch.device,
) -> tuple[PRRNetwork, PredictionMode]:
    architecture = config["architecture"]
    model = PRRNetwork(
        input_dim,
        output_dim,
        [int(value) for value in architecture["hidden_dims"]],
        float(architecture["dropout"]),
        bool(architecture["use_batch_norm"]),
    ).to(device)
    checkpoint = torch.load(result["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["prediction_mode"]


def _predict_test_once(
    test_path: Path,
    feature_builder: MultimodalFeatureBuilder,
    feature_scaler: StandardScaler,
    target_scales: np.ndarray,
    selected_result: dict[str, object],
    prr_columns: list[str],
    config: dict[str, object],
    output_path: Path,
    device: torch.device,
) -> dict[str, object]:
    # This is the first and only test read: all validation-based selection is complete.
    test = pd.read_csv(test_path, low_memory=False)
    forbidden = {column for column in test if column.startswith(PRR_PREFIX)}
    if forbidden:
        raise ValueError("Le test contient des cibles PRR inattendues.")
    test_features = feature_scaler.transform(feature_builder.transform(test)).astype(np.float32)
    model, prediction_mode = _load_best_model(
        selected_result,
        test_features.shape[1],
        len(prr_columns),
        config,
        device,
    )
    dummy_targets = np.zeros((len(test), len(prr_columns)), dtype=np.float32)
    test_loader = _loader(
        test_features,
        dummy_targets,
        target_scales,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        seed=int(config["seed"]),
        num_workers=int(config["training"]["num_workers"]),
    )
    target_scales_tensor = torch.from_numpy(target_scales).to(device)
    _, _, conditional, expected = _run_epoch(
        model,
        test_loader,
        selected_result["method"],
        target_scales_tensor,
        config,
        int(selected_result["best_epoch"]),
        device,
        None,
    )
    prediction = expected if prediction_mode == "expected" else conditional
    prediction_frame = pd.concat(
        [
            test[[ID_COLUMN]].reset_index(drop=True),
            pd.DataFrame(prediction, columns=prr_columns),
        ],
        axis=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_path, index=False)
    return {
        "rows": len(prediction_frame),
        "columns": len(prediction_frame.columns),
        "prediction_mode": prediction_mode,
        "minimum": float(prediction.min()),
        "maximum": float(prediction.max()),
        "finite": bool(np.isfinite(prediction).all()),
        "test_used_for_selection": False,
        "test_reads_after_selection": 1,
        "path": str(output_path),
    }


def _hurdle_output_comparison(results: list[dict[str, object]]) -> pd.DataFrame:
    hurdle_result = next(result for result in results if result["method"] == "hurdle")
    history = pd.read_csv(hurdle_result["history"])
    best = history.loc[history["epoch"] == int(hurdle_result["best_epoch"])].iloc[0]
    return pd.DataFrame(
        [
            {
                "output": "PRR_conditionnel",
                "masked_rmse": best["validation_conditional_masked_rmse"],
                "sprr": best["validation_conditional_sprr"],
            },
            {
                "output": "sigmoid(logit) × PRR_conditionnel",
                "masked_rmse": best["validation_expected_masked_rmse"],
                "sprr": best["validation_expected_sprr"],
            },
        ]
    )


def _write_comparison_report(
    comparison: pd.DataFrame,
    selected: dict[str, object],
    comparability: dict[str, object],
    test_inference: dict[str, object] | None,
    hurdle_outputs: pd.DataFrame,
    output_path: Path,
) -> None:
    table_lines = [
        "| Méthode | Meilleure époque | Sortie | Loss validation | RMSE masqué | S_PRR |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        table_lines.append(
            f"| {row.method} | {int(row.best_epoch)} | {row.best_prediction_mode} | "
            f"{row.validation_loss:.6f} | {row.masked_rmse:.6f} | {row.sprr:.6f} |"
        )
    table = "\n".join(table_lines)
    hurdle_lines = [
        "| Sortie hurdle | RMSE masqué validation | S_PRR validation |",
        "|---|---:|---:|",
    ]
    for row in hurdle_outputs.itertuples(index=False):
        hurdle_lines.append(f"| {row.output} | {row.masked_rmse:.6f} | {row.sprr:.6f} |")
    hurdle_table = "\n".join(hurdle_lines)
    test_note = (
        f"Le test a été lu une seule fois après sélection ({test_inference['rows']} lignes)."
        if test_inference
        else "L'inférence test a été désactivée."
    )
    output_path.write_text(
        f"""# Comparaison PyTorch des stratégies PRR

## Résultat

{table}

Méthode sélectionnée sur le seul `S_PRR` validation : **{selected["method"]}**, sortie
**{selected["best_prediction_mode"]}**, époque **{selected["best_epoch"]}**.

### Deux sorties du hurdle à sa meilleure époque

{hurdle_table}

## Comparabilité et absence de fuite

- Split : fold {comparability["validation_fold"]} du manifeste groupé par paire.
- Fingerprint : `{comparability["validation_fingerprint"]}`.
- Seed, architecture et hyperparamètres partagés : **oui**.
- Signatures d'architecture identiques : **{comparability["architecture_signatures_identical"]}**.
- Le preprocessing est ajusté uniquement sur les lignes fit du fold.
- {test_note}

Les losses de validation ne sont pas directement comparables en niveau, car chaque méthode
optimise un objectif différent. `RMSE_masked` et `S_PRR`, calculés sur les PRR bruts et le même
masque `PRR vrai > 0`, sont les critères comparables.
""",
        encoding="utf-8",
    )


def finalize_existing_prr_comparison(config_path: Path) -> dict[str, object]:
    """Rebuild summaries from completed artifacts without reading train or test."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["paths"]["run_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    results = [load_json(run_dir / method / "result.json") for method in METHODS]
    for result in results:
        plot_training_history(
            Path(result["history"]),
            result,
            output_dir / f"curves_{result['method']}.png",
        )
    comparison = pd.DataFrame(results).sort_values("sprr", ascending=False).reset_index(drop=True)
    selected = comparison.iloc[0].to_dict()
    comparability = load_json(run_dir / "comparability.json")
    test_audit_path = run_dir / "test_inference_audit.json"
    test_inference = load_json(test_audit_path) if test_audit_path.exists() else None
    hurdle_outputs = _hurdle_output_comparison(results)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    hurdle_outputs.to_csv(output_dir / "hurdle_output_comparison.csv", index=False)
    _write_comparison_report(
        comparison,
        selected,
        comparability,
        test_inference,
        hurdle_outputs,
        output_dir / "COMPARISON.md",
    )
    payload = {
        "run_id": config["run_id"],
        "selected": selected,
        "comparability": comparability,
        "hurdle_validation_outputs": hurdle_outputs.to_dict(orient="records"),
        "test_inference": test_inference,
    }
    write_json(payload, run_dir / "summary.json")
    return payload


def compare_prr_methods(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_everything(seed)
    paths = config["paths"]
    train_path = Path(paths["train"])
    test_path = Path(paths["test"])
    run_dir = Path(paths["run_dir"])
    output_dir = Path(paths["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately load train only. Test is not touched before the winning method is fixed.
    train = pd.read_csv(train_path, low_memory=False)
    folds_path = Path(config["validation"]["manifest"])
    audit_path = Path(config["validation"]["audit"])
    folds = pd.read_csv(folds_path).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(audit_path)
    current_audit = audit_pair_grouped_folds(train, folds)
    fingerprint = split_fingerprint(folds)
    if not current_audit["passed"] or fingerprint != stored_audit["fingerprint"]:
        raise ValueError("Le split partagé n'est plus conforme à son audit; expérience bloquée.")
    validation_fold = int(config["validation"]["fold"])
    validation_mask = folds["fold"].to_numpy() == validation_fold
    fit_mask = ~validation_mask
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    if not prr_columns:
        raise ValueError("Aucune cible PRR trouvée dans le train.")

    fit_frame = train.loc[fit_mask].reset_index(drop=True)
    validation_frame = train.loc[validation_mask].reset_index(drop=True)
    fit_targets = fit_frame[prr_columns].to_numpy(dtype=np.float32)
    validation_targets = validation_frame[prr_columns].to_numpy(dtype=np.float32)
    expected_target_scales = _target_scales(
        fit_targets, float(config["training"]["target_scale_quantile"])
    )
    bundle_path = run_dir / "fold_preprocessing.joblib"
    feature_cache_path = run_dir / "fold_features.npz"
    cache_signature = hashlib.sha256(
        json.dumps(
            {
                "seed": seed,
                "validation_fold": validation_fold,
                "validation_fingerprint": fingerprint,
                "features": config["features"],
                "prr_columns": prr_columns,
                "target_scale_quantile": config["training"]["target_scale_quantile"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache_metadata_path = run_dir / "fold_features_metadata.json"
    cache_metadata = load_json(cache_metadata_path) if cache_metadata_path.exists() else {}
    can_reuse_arrays = (
        bundle_path.exists()
        and feature_cache_path.exists()
        and cache_metadata.get("signature") == cache_signature
    )
    if can_reuse_arrays:
        print("Réutilisation du preprocessing et des features du fold partagé...", flush=True)
        bundle = joblib.load(bundle_path)
        cached = np.load(feature_cache_path)
        fit_features = cached["fit_features"]
        validation_features = cached["validation_features"]
        feature_builder = bundle["feature_builder"]
        feature_scaler = bundle["feature_scaler"]
        target_scales = bundle["target_scales"]
    else:
        can_reuse_builder = False
        if bundle_path.exists() and (run_dir / "comparability.json").exists():
            previous_comparability = load_json(run_dir / "comparability.json")
            bundle = joblib.load(bundle_path)
            can_reuse_builder = (
                previous_comparability.get("validation_fingerprint") == fingerprint
                and previous_comparability.get("validation_fold") == validation_fold
                and bundle.get("prr_columns") == prr_columns
                and _builder_matches_config(bundle["feature_builder"], config, seed)
                and np.allclose(bundle["target_scales"], expected_target_scales)
            )
        if can_reuse_builder:
            print("Réutilisation du preprocessing audité; recalcul des matrices...", flush=True)
            feature_builder = bundle["feature_builder"]
            feature_scaler = bundle["feature_scaler"]
            target_scales = bundle["target_scales"]
            fit_features = feature_scaler.transform(feature_builder.transform(fit_frame)).astype(
                np.float32
            )
            validation_features = feature_scaler.transform(
                feature_builder.transform(validation_frame)
            ).astype(np.float32)
        else:
            print("Ajustement unique du preprocessing sur le train du fold partagé...", flush=True)
            feature_builder = _feature_builder(config, seed)
            fit_features_unscaled = feature_builder.fit_transform(fit_frame)
            validation_features_unscaled = feature_builder.transform(validation_frame)
            feature_scaler = StandardScaler().fit(fit_features_unscaled)
            fit_features = feature_scaler.transform(fit_features_unscaled).astype(np.float32)
            validation_features = feature_scaler.transform(validation_features_unscaled).astype(
                np.float32
            )
            target_scales = expected_target_scales
        joblib.dump(
            {
                "feature_builder": feature_builder,
                "feature_scaler": feature_scaler,
                "target_scales": target_scales,
                "prr_columns": prr_columns,
            },
            bundle_path,
            compress=3,
        )
        np.savez_compressed(
            feature_cache_path,
            fit_features=fit_features,
            validation_features=validation_features,
        )
        write_json({"signature": cache_signature}, cache_metadata_path)
    device = _device(config["training"])
    print(f"Device PyTorch : {device}", flush=True)
    results = []
    for method in METHODS:
        result = train_prr_method(
            method,
            fit_features,
            fit_targets,
            validation_features,
            validation_targets,
            target_scales,
            config,
            run_dir,
            device,
        )
        results.append(result)
        plot_training_history(
            Path(result["history"]),
            result,
            output_dir / f"curves_{method}.png",
        )

    comparison = pd.DataFrame(results).sort_values("sprr", ascending=False).reset_index(drop=True)
    selected = comparison.iloc[0].to_dict()
    signatures = comparison["architecture_signature"].unique().tolist()
    comparability = {
        "same_split": True,
        "same_seed": True,
        "same_architecture": True,
        "same_hyperparameters_except_loss_definition": True,
        "validation_fold": validation_fold,
        "validation_fingerprint": fingerprint,
        "fit_rows": int(fit_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "pair_group_overlap": 0,
        "fit_row_indices_sha256": hashlib.sha256(
            np.flatnonzero(fit_mask).astype(np.int64).tobytes()
        ).hexdigest(),
        "validation_row_indices_sha256": hashlib.sha256(
            np.flatnonzero(validation_mask).astype(np.int64).tobytes()
        ).hexdigest(),
        "architecture_signatures": signatures,
        "architecture_signatures_identical": len(signatures) == 1,
        "test_loaded_before_selection": False,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    if len(signatures) != 1:
        raise RuntimeError("Les trois méthodes n'ont pas la même architecture PyTorch.")

    comparison.to_csv(output_dir / "comparison.csv", index=False)
    hurdle_outputs = _hurdle_output_comparison(results)
    hurdle_outputs.to_csv(output_dir / "hurdle_output_comparison.csv", index=False)
    write_json(comparability, run_dir / "comparability.json")
    write_json(selected, run_dir / "selected_method.json")
    test_inference = None
    if bool(config.get("predict_test_after_selection", False)):
        selected_result = next(
            result for result in results if result["method"] == selected["method"]
        )
        test_inference = _predict_test_once(
            test_path,
            feature_builder,
            feature_scaler,
            target_scales,
            selected_result,
            prr_columns,
            config,
            output_dir / "test_prr_predictions.csv",
            device,
        )
        write_json(test_inference, run_dir / "test_inference_audit.json")
    _write_comparison_report(
        comparison,
        selected,
        comparability,
        test_inference,
        hurdle_outputs,
        output_dir / "COMPARISON.md",
    )
    payload = {
        "run_id": config["run_id"],
        "selected": selected,
        "comparability": comparability,
        "hurdle_validation_outputs": hurdle_outputs.to_dict(orient="records"),
        "test_inference": test_inference,
    }
    write_json(payload, run_dir / "summary.json")
    return payload
