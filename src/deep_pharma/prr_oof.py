from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from .constants import ID_COLUMN, PRR_PREFIX
from .molecule_priors import (
    build_cross_fitted_molecule_priors,
    fit_molecule_prior_tables,
    transform_molecule_priors,
)
from .prr_pytorch import (
    MethodName,
    _device,
    _feature_builder,
    _target_scales,
    masked_prr_metrics,
    plot_training_history,
    predict_prr_arrays,
    seed_everything,
    train_prr_method,
)
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def per_target_prr_diagnostics(
    targets: np.ndarray,
    predictions: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_sse = 0.0
    for index, target_name in enumerate(target_names):
        mask = targets[:, index] > 0
        errors = predictions[mask, index] - targets[mask, index]
        squared_errors = np.square(errors, dtype=np.float64)
        sse = float(squared_errors.sum())
        total_sse += sse
        rows.append(
            {
                "target": target_name,
                "positive_count": int(mask.sum()),
                "sse": sse,
                "rmse": float(np.sqrt(squared_errors.mean())) if len(errors) else np.nan,
                "mean_true": float(targets[mask, index].mean()) if len(errors) else np.nan,
                "mean_prediction": float(predictions[mask, index].mean())
                if len(errors)
                else np.nan,
                "bias_prediction_minus_true": float(errors.mean()) if len(errors) else np.nan,
                "absolute_error_p90": float(np.quantile(np.abs(errors), 0.9))
                if len(errors)
                else np.nan,
            }
        )
    diagnostics = pd.DataFrame(rows)
    diagnostics["sse_contribution"] = diagnostics["sse"] / total_sse if total_sse > 0 else 0.0
    return diagnostics.sort_values("sse_contribution", ascending=False).reset_index(drop=True)


def _constant_positive_mean_oof(
    train: pd.DataFrame,
    targets: np.ndarray,
    fold_values: np.ndarray,
) -> np.ndarray:
    predictions = np.full_like(targets, np.nan, dtype=np.float32)
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
    if not np.isfinite(predictions).all() or len(predictions) != len(train):
        raise RuntimeError("Les prédictions OOF constantes sont incomplètes.")
    return predictions


def _positive_target_means(targets: np.ndarray) -> np.ndarray:
    positive = targets > 0
    return np.divide(
        np.where(positive, targets, 0.0).sum(axis=0),
        positive.sum(axis=0),
        out=np.zeros(targets.shape[1], dtype=np.float32),
        where=positive.sum(axis=0) > 0,
    ).astype(np.float32)


def _fit_convex_blend_weights(
    first: np.ndarray,
    second: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Fit per-target weights on positive fit targets: w*first + (1-w)*second."""

    weights = np.zeros(targets.shape[1], dtype=np.float32)
    for target_index in range(targets.shape[1]):
        mask = targets[:, target_index] > 0
        direction = first[mask, target_index] - second[mask, target_index]
        residual = targets[mask, target_index] - second[mask, target_index]
        denominator = float(np.dot(direction, direction))
        if denominator > 0:
            weights[target_index] = np.clip(
                float(np.dot(direction, residual)) / denominator, 0.0, 1.0
            )
    return weights


def _fit_affine_calibration(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    slope_min: float,
    slope_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.ones(targets.shape[1], dtype=np.float32)
    intercepts = np.zeros(targets.shape[1], dtype=np.float32)
    for target_index in range(targets.shape[1]):
        mask = targets[:, target_index] > 0
        x = predictions[mask, target_index].astype(np.float64)
        y = targets[mask, target_index].astype(np.float64)
        if len(x) < 2:
            continue
        centered_x = x - x.mean()
        centered_y = y - y.mean()
        slope = float(np.dot(centered_x, centered_y) / (np.dot(centered_x, centered_x) + ridge))
        slopes[target_index] = np.clip(slope, slope_min, slope_max)
        intercepts[target_index] = y.mean() - slopes[target_index] * x.mean()
    return slopes, intercepts


def _feature_cache_signature(config: dict[str, object], fingerprint: str, fold: int) -> str:
    payload = {
        "schema": 1,
        "seed": config["seed"],
        "fingerprint": fingerprint,
        "fold": fold,
        "features": config["features"],
        "molecule_priors": config["molecule_priors"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _experiment_signature(
    config: dict[str, object],
    fingerprint: str,
    fold: int,
    experiment: dict[str, object],
    input_dimension: int,
) -> str:
    payload = {
        "schema": 1,
        "seed": config["seed"],
        "fingerprint": fingerprint,
        "fold": fold,
        "experiment": experiment,
        "architecture": config["architecture"],
        "training": config["training"],
        "loss_schedules": config["loss_schedules"],
        "input_dimension": input_dimension,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _prepare_fold_features(
    train: pd.DataFrame,
    targets: np.ndarray,
    fold_values: np.ndarray,
    fold: int,
    config: dict[str, object],
    fingerprint: str,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "features.npz"
    metadata_path = cache_dir / "features_metadata.json"
    signature = _feature_cache_signature(config, fingerprint, fold)
    if cache_path.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata.get("signature") == signature:
            cached = np.load(cache_path)
            print(f"Fold {fold}: réutilisation des features auditées.", flush=True)
            return (
                cached["fit_base"],
                cached["validation_base"],
                cached["fit_with_priors"],
                cached["validation_with_priors"],
                cached["target_scales"],
            )

    validation_mask = fold_values == fold
    fit_frame = train.loc[~validation_mask].reset_index(drop=True)
    validation_frame = train.loc[validation_mask].reset_index(drop=True)
    fit_targets = targets[~validation_mask]
    inner_folds = fold_values[~validation_mask]
    print(f"Fold {fold}: ajustement des features multimodales sur le fit uniquement.", flush=True)
    builder = _feature_builder(config, int(config["seed"]))
    fit_base_raw = builder.fit_transform(fit_frame)
    validation_base_raw = builder.transform(validation_frame)
    base_scaler = StandardScaler().fit(fit_base_raw)
    fit_base = base_scaler.transform(fit_base_raw).astype(np.float32)
    validation_base = base_scaler.transform(validation_base_raw).astype(np.float32)

    smoothing = float(config["molecule_priors"]["smoothing"])
    print(f"Fold {fold}: construction cross-fittée des priors moléculaires.", flush=True)
    fit_priors_raw = build_cross_fitted_molecule_priors(
        fit_frame,
        fit_targets,
        inner_folds,
        smoothing=smoothing,
    )
    outer_tables = fit_molecule_prior_tables(fit_frame, fit_targets, smoothing=smoothing)
    validation_priors_raw = transform_molecule_priors(validation_frame, outer_tables)
    prior_scaler = StandardScaler().fit(fit_priors_raw)
    fit_priors = prior_scaler.transform(fit_priors_raw).astype(np.float32)
    validation_priors = prior_scaler.transform(validation_priors_raw).astype(np.float32)
    fit_with_priors = np.hstack([fit_base, fit_priors]).astype(np.float32)
    validation_with_priors = np.hstack([validation_base, validation_priors]).astype(np.float32)
    target_scales = _target_scales(fit_targets, float(config["training"]["target_scale_quantile"]))
    np.savez_compressed(
        cache_path,
        fit_base=fit_base,
        validation_base=validation_base,
        fit_with_priors=fit_with_priors,
        validation_with_priors=validation_with_priors,
        target_scales=target_scales,
    )
    write_json(
        {
            "signature": signature,
            "outer_fold": int(fold),
            "target_features_cross_fitted": True,
            "validation_statistics_fit_rows_only": True,
            "inner_group_values": sorted(int(value) for value in np.unique(inner_folds)),
            "base_dimensions": int(fit_base.shape[1]),
            "molecule_prior_dimensions": int(fit_priors.shape[1]),
        },
        metadata_path,
    )
    return fit_base, validation_base, fit_with_priors, validation_with_priors, target_scales


def _plot_target_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    top = diagnostics.head(15).sort_values("sse_contribution")
    labels = top["target"].str.removeprefix(PRR_PREFIX)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(labels, top["sse_contribution"] * 100, color="#2563EB")
    axes[0].set(xlabel="Contribution à la SSE globale (%)", title="15 cibles dominantes")
    axes[1].scatter(
        diagnostics["positive_count"],
        diagnostics["rmse"],
        c=diagnostics["sse_contribution"],
        cmap="magma",
        alpha=0.85,
    )
    axes[1].set(
        xlabel="Nombre de PRR positifs",
        ylabel="RMSE masqué par cible",
        title="Erreur et fréquence",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    comparison: pd.DataFrame,
    best_diagnostics: pd.DataFrame,
    audit: dict[str, object],
    output_path: Path,
) -> None:
    table_lines = [
        "| Expérience | Features | Méthode | RMSE OOF | S_PRR OOF |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        table_lines.append(
            f"| {row.experiment} | {row.feature_variant} | {row.method} | "
            f"{row.masked_rmse:.6f} | {row.sprr:.6f} |"
        )
    target_lines = [
        "| Cible | Positifs | RMSE | Contribution SSE | Biais |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best_diagnostics.head(10).itertuples(index=False):
        target_lines.append(
            f"| {row.target} | {row.positive_count} | {row.rmse:.4f} | "
            f"{100 * row.sse_contribution:.2f}% | {row.bias_prediction_minus_true:.4f} |"
        )
    output_path.write_text(
        "# PRR — validation OOF 5 folds sans fuite\n\n"
        + "\n".join(table_lines)
        + "\n\n## Cibles dominant l'erreur du meilleur modèle\n\n"
        + "\n".join(target_lines)
        + "\n\n## Garanties de validation\n\n"
        + f"- Empreinte du manifeste : `{audit['fingerprint']}`.\n"
        + "- Les paires identiques ou inversées restent dans le même fold.\n"
        + "- Le preprocessing multimodal est ajusté dans chaque outer-fit seulement.\n"
        + "- Les priors des lignes d'entraînement sont cross-fittés par groupes de paires.\n"
        + "- Les priors de validation utilisent exclusivement l'outer-fit.\n"
        + "- Les calibrations et mélanges par cible utilisent exclusivement l'outer-fit.\n"
        + "- `test.csv` n'est ni chargé ni évalué par cette commande.\n",
        encoding="utf-8",
    )


def run_prr_oof(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    run_dir = Path(config["paths"]["run_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intentionally no test path in this workflow.
    train_path = Path(config["paths"]["train"])
    train_sha256 = _sha256_file(train_path)
    train = pd.read_csv(train_path, low_memory=False)
    manifest = (
        pd.read_csv(Path(config["validation"]["manifest"]))
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    stored_audit = load_json(Path(config["validation"]["audit"]))
    current_audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    if not current_audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Le manifeste OOF ne correspond plus à l'audit anti-fuite.")

    fold_values = manifest["fold"].to_numpy()
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    targets = train[prr_columns].to_numpy(dtype=np.float32)
    experiments = config["experiments"]
    experiment_oof = {
        item["name"]: np.full_like(targets, np.nan, dtype=np.float32) for item in experiments
    }
    experiment_fold_rows: dict[str, list[dict[str, object]]] = {
        item["name"]: [] for item in experiments
    }
    postprocessing_config = config.get("postprocessing", {})
    postprocessing_enabled = bool(postprocessing_config.get("enabled", False))
    postprocess_names = (
        "blend_prior_base",
        "shrink_prior_to_mean",
        "affine_prior",
    )
    postprocess_oof = {
        name: np.full_like(targets, np.nan, dtype=np.float32) for name in postprocess_names
    }
    postprocess_fold_rows: dict[str, list[dict[str, object]]] = {
        name: [] for name in postprocess_names
    }
    device = _device(config["training"])
    print(f"Device PyTorch OOF : {device}", flush=True)

    for fold in sorted(int(value) for value in np.unique(fold_values)):
        validation_mask = fold_values == fold
        fit_targets = targets[~validation_mask]
        validation_targets = targets[validation_mask]
        prepared = _prepare_fold_features(
            train,
            targets,
            fold_values,
            fold,
            config,
            fingerprint,
            run_dir / "feature_cache" / f"fold_{fold}",
        )
        fit_base, validation_base, fit_priors, validation_priors, target_scales = prepared
        fold_fit_predictions: dict[str, np.ndarray] = {}
        fold_validation_predictions: dict[str, np.ndarray] = {}
        for experiment in experiments:
            name = str(experiment["name"])
            method: MethodName = str(experiment["method"])  # type: ignore[assignment]
            use_priors = bool(experiment["use_molecule_priors"])
            fit_features = fit_priors if use_priors else fit_base
            validation_features = validation_priors if use_priors else validation_base
            fold_dir = run_dir / name / f"fold_{fold}"
            result_path = fold_dir / method / "result.json"
            prediction_path = fold_dir / "validation_predictions.npy"
            fit_prediction_path = fold_dir / "fit_predictions.npy"
            completion_path = fold_dir / "completion.json"
            experiment_signature = _experiment_signature(
                config, fingerprint, fold, experiment, fit_features.shape[1]
            )
            completion = load_json(completion_path) if completion_path.exists() else {}
            can_reuse = (
                result_path.exists()
                and prediction_path.exists()
                and completion.get("signature") == experiment_signature
            )
            if can_reuse:
                print(f"{name} fold {fold}: réutilisation du checkpoint terminé.", flush=True)
                result = load_json(result_path)
                predictions = np.load(prediction_path)
            else:
                print(f"{name} fold {fold}: entraînement {method}.", flush=True)
                result = train_prr_method(
                    method,
                    fit_features,
                    fit_targets,
                    validation_features,
                    validation_targets,
                    target_scales,
                    config,
                    fold_dir,
                    device,
                )
                arrays = predict_prr_arrays(
                    result, validation_features, target_scales, config, device
                )
                predictions = arrays["conditional"] if method == "hurdle" else arrays["selected"]
                np.save(prediction_path, predictions)
                write_json(
                    {
                        "signature": experiment_signature,
                        "validation_rows": int(validation_mask.sum()),
                        "prediction_mode": "conditional"
                        if method == "hurdle"
                        else result["best_prediction_mode"],
                    },
                    completion_path,
                )
                plot_training_history(
                    Path(result["history"]),
                    result,
                    output_dir / "curves" / f"{name}_fold_{fold}.png",
                )
            if fit_prediction_path.exists() and can_reuse:
                fit_predictions = np.load(fit_prediction_path)
            else:
                fit_arrays = predict_prr_arrays(result, fit_features, target_scales, config, device)
                fit_predictions = (
                    fit_arrays["conditional"] if method == "hurdle" else fit_arrays["selected"]
                )
                np.save(fit_prediction_path, fit_predictions)
            metrics = masked_prr_metrics(validation_targets, predictions)
            experiment_oof[name][validation_mask] = predictions
            fold_fit_predictions[name] = fit_predictions
            fold_validation_predictions[name] = predictions
            experiment_fold_rows[name].append(
                {
                    "fold": fold,
                    "best_epoch": int(result["best_epoch"]),
                    "masked_rmse": metrics.masked_rmse,
                    "sprr": metrics.sprr,
                    "positive_count": metrics.positive_count,
                }
            )

        if postprocessing_enabled:
            base_name = str(postprocessing_config["base_experiment"])
            prior_name = str(postprocessing_config["prior_experiment"])
            base_fit = fold_fit_predictions[base_name]
            prior_fit = fold_fit_predictions[prior_name]
            base_validation = fold_validation_predictions[base_name]
            prior_validation = fold_validation_predictions[prior_name]
            fit_means = _positive_target_means(fit_targets)
            mean_fit = np.broadcast_to(fit_means, fit_targets.shape)
            mean_validation = np.broadcast_to(
                fit_means, (int(validation_mask.sum()), targets.shape[1])
            )

            blend_weights = _fit_convex_blend_weights(prior_fit, base_fit, fit_targets)
            shrink_weights = _fit_convex_blend_weights(prior_fit, mean_fit, fit_targets)
            slopes, intercepts = _fit_affine_calibration(
                prior_fit,
                fit_targets,
                ridge=float(postprocessing_config["affine_ridge"]),
                slope_min=float(postprocessing_config["affine_slope_min"]),
                slope_max=float(postprocessing_config["affine_slope_max"]),
            )
            fold_postprocessed = {
                "blend_prior_base": blend_weights * prior_validation
                + (1.0 - blend_weights) * base_validation,
                "shrink_prior_to_mean": shrink_weights * prior_validation
                + (1.0 - shrink_weights) * mean_validation,
                "affine_prior": np.maximum(0.0, slopes * prior_validation + intercepts),
            }
            parameter_rows = []
            for target_index, target_name in enumerate(prr_columns):
                parameter_rows.append(
                    {
                        "target": target_name,
                        "blend_prior_weight": float(blend_weights[target_index]),
                        "shrink_prior_weight": float(shrink_weights[target_index]),
                        "affine_slope": float(slopes[target_index]),
                        "affine_intercept": float(intercepts[target_index]),
                    }
                )
            pd.DataFrame(parameter_rows).to_csv(
                output_dir / f"postprocessing_parameters_fold_{fold}.csv", index=False
            )
            for name, predictions in fold_postprocessed.items():
                predictions = predictions.astype(np.float32)
                postprocess_oof[name][validation_mask] = predictions
                metrics = masked_prr_metrics(validation_targets, predictions)
                postprocess_fold_rows[name].append(
                    {
                        "fold": fold,
                        "best_epoch": np.nan,
                        "masked_rmse": metrics.masked_rmse,
                        "sprr": metrics.sprr,
                        "positive_count": metrics.positive_count,
                    }
                )

    comparison_rows: list[dict[str, object]] = []
    for experiment in experiments:
        name = str(experiment["name"])
        predictions = experiment_oof[name]
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"La couverture OOF de {name} est incomplète.")
        metrics = masked_prr_metrics(targets, predictions)
        prediction_frame = pd.concat(
            [
                train[[ID_COLUMN]].reset_index(drop=True),
                pd.DataFrame(predictions, columns=prr_columns),
            ],
            axis=1,
        )
        prediction_frame.to_csv(output_dir / f"oof_predictions_{name}.csv", index=False)
        fold_frame = pd.DataFrame(experiment_fold_rows[name])
        fold_frame.to_csv(output_dir / f"fold_metrics_{name}.csv", index=False)
        diagnostics = per_target_prr_diagnostics(targets, predictions, prr_columns)
        diagnostics.to_csv(output_dir / f"target_diagnostics_{name}.csv", index=False)
        comparison_rows.append(
            {
                "experiment": name,
                "feature_variant": "base+cross_fitted_molecule_priors"
                if bool(experiment["use_molecule_priors"])
                else "base",
                "method": experiment["method"],
                "masked_rmse": metrics.masked_rmse,
                "sprr": metrics.sprr,
                "positive_count": metrics.positive_count,
                "fold_sprr_mean": float(fold_frame["sprr"].mean()),
                "fold_sprr_std": float(fold_frame["sprr"].std(ddof=1)),
            }
        )

    for name in postprocess_names if postprocessing_enabled else ():
        predictions = postprocess_oof[name]
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"La couverture OOF post-traitée de {name} est incomplète.")
        metrics = masked_prr_metrics(targets, predictions)
        prediction_frame = pd.concat(
            [
                train[[ID_COLUMN]].reset_index(drop=True),
                pd.DataFrame(predictions, columns=prr_columns),
            ],
            axis=1,
        )
        prediction_frame.to_csv(output_dir / f"oof_predictions_{name}.csv", index=False)
        fold_frame = pd.DataFrame(postprocess_fold_rows[name])
        fold_frame.to_csv(output_dir / f"fold_metrics_{name}.csv", index=False)
        diagnostics = per_target_prr_diagnostics(targets, predictions, prr_columns)
        diagnostics.to_csv(output_dir / f"target_diagnostics_{name}.csv", index=False)
        comparison_rows.append(
            {
                "experiment": name,
                "feature_variant": "fit_only_targetwise_postprocessing",
                "method": "postprocessing",
                "masked_rmse": metrics.masked_rmse,
                "sprr": metrics.sprr,
                "positive_count": metrics.positive_count,
                "fold_sprr_mean": float(fold_frame["sprr"].mean()),
                "fold_sprr_std": float(fold_frame["sprr"].std(ddof=1)),
            }
        )

    constant_predictions = _constant_positive_mean_oof(train, targets, fold_values)
    constant_metrics = masked_prr_metrics(targets, constant_predictions)
    comparison_rows.append(
        {
            "experiment": "constant_positive_mean",
            "feature_variant": "fold_local_constant",
            "method": "constant",
            "masked_rmse": constant_metrics.masked_rmse,
            "sprr": constant_metrics.sprr,
            "positive_count": constant_metrics.positive_count,
            "fold_sprr_mean": None,
            "fold_sprr_std": None,
        }
    )
    comparison = (
        pd.DataFrame(comparison_rows).sort_values("sprr", ascending=False).reset_index(drop=True)
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    selected_name = str(comparison.iloc[0]["experiment"])
    all_learned_oof = {**experiment_oof, **postprocess_oof}
    if selected_name == "constant_positive_mean":
        best_diagnostics = per_target_prr_diagnostics(targets, constant_predictions, prr_columns)
    else:
        best_diagnostics = per_target_prr_diagnostics(
            targets, all_learned_oof[selected_name], prr_columns
        )
    _plot_target_diagnostics(best_diagnostics, output_dir / "target_error_analysis.png")
    report_audit = {
        "passed": True,
        "fingerprint": fingerprint,
        "train_sha256": train_sha256,
        "pair_group_overlap_max": max(
            int(item["pair_group_overlap"]) for item in current_audit["folds"]
        ),
        "outer_folds": sorted(int(value) for value in np.unique(fold_values)),
        "training_priors_cross_fitted": True,
        "validation_priors_fit_only": True,
        "postprocessing_parameters_fit_only": postprocessing_enabled,
        "test_loaded": False,
    }
    write_json(report_audit, run_dir / "leakage_audit.json")
    _write_report(comparison, best_diagnostics, report_audit, output_dir / "OOF_REPORT.md")
    comparison_records = json.loads(comparison.to_json(orient="records"))
    summary = {
        "run_id": config["run_id"],
        "selected_oof_experiment": selected_name,
        "comparison": comparison_records,
        "leakage_audit": report_audit,
        "test_inference": None,
    }
    write_json(summary, run_dir / "summary.json")
    return summary
