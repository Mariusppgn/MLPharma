"""Bounded CPU PRR search; outer labels are used only after frozen refits."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from .constants import ID_COLUMN, PRR_PREFIX
from .feature_transform_cache import (
    cache_fitted_text_transforms,
    clear_fitted_text_transform_cache,
)
from .molecule_priors import (
    build_cross_fitted_molecule_priors,
    fit_molecule_prior_tables,
    transform_molecule_priors,
)
from .nested_validation import build_nested_plan
from .prr_oof import _fit_convex_blend_weights, _sha256_file
from .prr_pytorch import (
    PRRNetwork,
    _architecture_signature,
    _feature_builder,
    _loader,
    _run_epoch,
    _target_scales,
    masked_prr_metrics,
    plot_training_history,
    predict_prr_arrays,
    seed_everything,
    train_prr_method,
)
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint


def candidate_config(base: dict, candidate: dict, seed: int) -> dict:
    config = copy.deepcopy(base)
    config["seed"] = seed
    for section in ("architecture", "training"):
        config[section].update(candidate.get(section, {}))
    config["loss_schedules"]["huber"].update(candidate.get("huber", {}))
    config["training"]["device"] = "cpu"
    return config


def replay_schedule(history: pd.DataFrame, config: dict, best_epoch: int) -> list[dict]:
    """Freeze learning rates and checkpoint restoration using inner history only."""
    training = config["training"]
    huber = config["loss_schedules"]["huber"]
    fine_start = training["max_epochs"] - huber["mse_finetune_epochs"] + 1
    schedule = []
    for epoch in range(1, best_epoch + 1):
        fine = huber["mse_finetune_epochs"] > 0 and epoch >= fine_start
        restore_epoch = None
        if epoch == fine_start and huber.get("restore_best_before_finetune", False):
            before = history.loc[history.epoch < epoch]
            if len(before):
                restore_epoch = int(before.loc[before.validation_sprr.idxmax(), "epoch"])
        if epoch == 1:
            rate = training["learning_rate"]
        elif epoch == fine_start:
            rate = huber.get("finetune_learning_rate", training["learning_rate"])
        else:
            rate = float(history.loc[history.epoch == epoch - 1, "learning_rate"].iloc[0])
        schedule.append(
            {
                "epoch": epoch,
                "learning_rate": rate,
                "fine_tune": fine,
                "restore_epoch": restore_epoch,
            }
        )
    return schedule


def refit_replaying_inner_schedule(features, targets, scales, config, schedule, run_dir):
    """No validation features/labels accepted: replay decisions fixed internally."""
    seed_everything(int(config["seed"]))
    device = torch.device("cpu")
    arch = config["architecture"]
    model = PRRNetwork(
        features.shape[1],
        targets.shape[1],
        arch["hidden_dims"],
        arch["dropout"],
        arch["use_batch_norm"],
    ).to(device)
    loader = _loader(
        features,
        targets,
        scales,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        seed=config["seed"],
        num_workers=0,
    )
    scale_tensor = torch.from_numpy(scales).to(device)
    optimizer = None
    states = {}
    restore_points = {step["restore_epoch"] for step in schedule if step["restore_epoch"]}
    previous_fine = False
    history = []
    for step in schedule:
        if step["restore_epoch"] is not None:
            model.load_state_dict(states[step["restore_epoch"]])
        if optimizer is None or (step["fine_tune"] and not previous_fine):
            if step["fine_tune"] and config["loss_schedules"]["huber"].get("regression_head_only"):
                for parameter in model.parameters():
                    parameter.requires_grad = False
                for parameter in model.regression_head.parameters():
                    parameter.requires_grad = True
            optimizer = AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=step["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )
        for group in optimizer.param_groups:
            group["lr"] = step["learning_rate"]
        loss, _, _, _ = _run_epoch(
            model, loader, "huber", scale_tensor, config, step["epoch"], device, optimizer
        )
        if step["epoch"] in restore_points:
            states[step["epoch"]] = copy.deepcopy(model.state_dict())
        history.append({**step, "train_loss": loss})
        previous_fine = step["fine_tune"]
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "refit_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prediction_mode": "direct",
            "epoch": len(schedule),
            "architecture_signature": _architecture_signature(model),
        },
        checkpoint,
    )
    pd.DataFrame(history).to_csv(run_dir / "refit_history.csv", index=False)
    write_json(schedule, run_dir / "frozen_inner_schedule.json")
    return {
        "checkpoint": str(checkpoint),
        "best_epoch": len(schedule),
        "best_prediction_mode": "direct",
    }


def fit_features(fit_frame, prediction_frame, fit_targets, inner_folds, config, priors):
    builder = _feature_builder(config, int(config["seed"]))
    cache_fitted_text_transforms(builder)
    print(f"PRR features: fit {len(fit_frame)} rows (TF-IDF/SVD/RDKit)", flush=True)
    fit = builder.fit_transform(fit_frame)
    print(f"PRR features: transform {len(prediction_frame)} held-out rows", flush=True)
    prediction = builder.transform(prediction_frame)
    # Remove local closure before serializing the fitted, fold-local builder.
    clear_fitted_text_transform_cache(builder)
    base_scaler = StandardScaler().fit(fit)
    base_arrays = (
        base_scaler.transform(fit).astype(np.float32),
        base_scaler.transform(prediction).astype(np.float32),
    )
    tables = None
    if priors:
        print("PRR features: supervised molecule priors, grouped cross-fitting", flush=True)
        fit_prior = build_cross_fitted_molecule_priors(
            fit_frame, fit_targets, inner_folds, smoothing=20.0
        )
        tables = fit_molecule_prior_tables(fit_frame, fit_targets, smoothing=20.0)
        prediction_prior = transform_molecule_priors(prediction_frame, tables)
        fit = np.column_stack([fit, fit_prior])
        prediction = np.column_stack([prediction, prediction_prior])
    scaler = StandardScaler().fit(fit)
    scales = _target_scales(fit_targets, config["training"]["target_scale_quantile"])
    return (
        scaler.transform(fit).astype(np.float32),
        scaler.transform(prediction).astype(np.float32),
        scales,
        {
            "builder": builder,
            "scaler": scaler,
            "base_scaler": base_scaler,
            "priors": tables,
            "target_scales": scales,
            "base_arrays": base_arrays,
        },
    )


def run_prr_nested_hpo(config_path: Path) -> dict:
    started = time.perf_counter()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(settings["base_config"]).read_text(encoding="utf-8"))
    for section, overrides in settings.get("base_overrides", {}).items():
        base[section].update(overrides)
    torch.set_num_threads(settings.get("cpu_threads", 4))
    train = pd.read_csv(settings["train"], low_memory=False)
    manifest = pd.read_csv(settings["manifest"]).sort_values("row_index").reset_index(drop=True)
    if not audit_pair_grouped_folds(train, manifest)["passed"]:
        raise ValueError("Outer manifest failed pair-group audit")
    columns = [c for c in train if c.startswith(PRR_PREFIX)]
    targets = train[columns].to_numpy(np.float32)
    candidates = settings["candidates"]
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names) or settings["baseline"] not in names:
        raise ValueError("Unique candidates and an explicit baseline are required")
    run_dir = Path(settings["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    train_sha = _sha256_file(Path(settings["train"]))
    config_sha = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    base_sha = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    implementation_sha = hashlib.sha256(
        b"".join(
            (Path(__file__).parent / filename).read_bytes()
            for filename in (
                "prr_nested_hpo.py",
                "prr_pytorch.py",
                "molecule_priors.py",
                "multimodal_features.py",
                "text_features.py",
                "nested_validation.py",
                "feature_transform_cache.py",
            )
        )
    ).hexdigest()
    signature = hashlib.sha256(
        (
            train_sha + config_sha + base_sha + implementation_sha + split_fingerprint(manifest)
        ).encode()
    ).hexdigest()
    write_json(settings, run_dir / "configuration.json")
    oof = {name: np.full_like(targets, np.nan) for name in ("baseline", "tuned")}
    records = []
    for fold in settings["outer_folds"]:
        fold_dir = run_dir / f"fold_{fold}"
        completion_path = fold_dir / "completion.json"
        if completion_path.exists():
            completion = load_json(completion_path)
            if completion["signature"] != signature:
                raise ValueError(
                    "Existing fold belongs to different data/config; use a new run_dir"
                )
            cached = np.load(fold_dir / "outer_predictions.npz")
            for name in oof:
                oof[name][cached["row_indices"]] = cached[name]
            records.extend(completion["records"])
            print(f"PRR nested fold {fold}: reuse completed audited fold", flush=True)
            continue
        plan = build_nested_plan(
            train, manifest, fold, inner_splits=settings["inner_splits"], seed=settings["seed"]
        )
        outer_fit = train.iloc[plan.outer_fit_indices].reset_index(drop=True)
        outer_val = train.iloc[plan.outer_validation_indices].reset_index(drop=True)
        outer_targets = targets[plan.outer_fit_indices]
        core, inner_val = list(plan.iter_inner_splits())[settings.get("inner_holdout_fold", 0)]
        config0 = candidate_config(base, candidates[0], settings["seed"] + fold)
        print(f"PRR nested fold {fold}: fit inner preprocessing", flush=True)
        inner_x, inner_v, inner_scales, inner_bundle = fit_features(
            outer_fit.iloc[core].reset_index(drop=True),
            outer_fit.iloc[inner_val].reset_index(drop=True),
            outer_targets[core],
            plan.inner_fold_values[core],
            config0,
            settings.get("use_molecule_priors", True),
        )
        results = {}
        inner_predictions = {}
        calibration = {}
        selection_scores = {}
        write_json(plan.audit, fold_dir / "nested_split_audit.json")
        inner_variants = {"base": inner_bundle.pop("base_arrays"), "prior": (inner_x, inner_v)}
        for candidate in candidates:
            cfg = candidate_config(base, candidate, settings["seed"] + fold)
            name = candidate["name"]
            results[name] = {}
            inner_predictions[name] = {}
            for component, (fit_array, val_array) in inner_variants.items():
                component_dir = fold_dir / "inner" / name / component
                result = train_prr_method(
                    "huber",
                    fit_array,
                    outer_targets[core],
                    val_array,
                    outer_targets[inner_val],
                    inner_scales,
                    cfg,
                    component_dir,
                    torch.device("cpu"),
                )
                results[name][component] = result
                plot_training_history(Path(result["history"]), result, component_dir / "curves.png")
                inner_predictions[name][component] = predict_prr_arrays(
                    result, val_array, inner_scales, cfg, torch.device("cpu")
                )["selected"]
            pred = inner_predictions[name]
            weights = _fit_convex_blend_weights(
                pred["prior"], pred["base"], outer_targets[inner_val]
            )
            calibration[name] = weights
            blended = weights * pred["prior"] + (1 - weights) * pred["base"]
            selection_scores[name] = masked_prr_metrics(outer_targets[inner_val], blended).sprr
        selected = max(names, key=lambda name: selection_scores[name])
        write_json(
            {
                "selected": selected,
                "results": results,
                "inner_blend_s_prr": selection_scores,
                "calibration_weights": {n: w.tolist() for n, w in calibration.items()},
                "outer_validation_consulted": False,
            },
            fold_dir / "inner_selection.json",
        )
        print(f"PRR nested fold {fold}: frozen selection {selected}; refit", flush=True)
        fit_x, val_x, scales, bundle = fit_features(
            outer_fit,
            outer_val,
            outer_targets,
            plan.inner_fold_values,
            config0,
            settings.get("use_molecule_priors", True),
        )
        outer_variants = {"base": bundle.pop("base_arrays"), "prior": (fit_x, val_x)}
        joblib.dump(bundle, fold_dir / "preprocessing.joblib", compress=3)
        predictions = {}
        for label, candidate_name in (("baseline", settings["baseline"]), ("tuned", selected)):
            if candidate_name not in predictions:
                candidate = candidates[names.index(candidate_name)]
                cfg = candidate_config(base, candidate, settings["seed"] + fold)
                component_predictions = {}
                for component, (fit_array, val_array) in outer_variants.items():
                    result = results[candidate_name][component]
                    schedule = replay_schedule(
                        pd.read_csv(result["history"]), cfg, result["best_epoch"]
                    )
                    refit = refit_replaying_inner_schedule(
                        fit_array,
                        outer_targets,
                        scales,
                        cfg,
                        schedule,
                        fold_dir / "refit" / candidate_name / component,
                    )
                    component_predictions[component] = predict_prr_arrays(
                        refit, val_array, scales, cfg, torch.device("cpu")
                    )["selected"]
                weights = calibration[candidate_name]
                predictions[candidate_name] = (
                    weights * component_predictions["prior"]
                    + (1 - weights) * component_predictions["base"]
                )
            pred = predictions[candidate_name]
            oof[label][plan.outer_validation_indices] = pred
            metrics = masked_prr_metrics(targets[plan.outer_validation_indices], pred)
            records.append(
                {
                    "fold": fold,
                    "branch": label,
                    "candidate": candidate_name,
                    "base_epoch": results[candidate_name]["base"]["best_epoch"],
                    "prior_epoch": results[candidate_name]["prior"]["best_epoch"],
                    "rmse_masked": metrics.masked_rmse,
                    "s_prr": metrics.sprr,
                }
            )
        pd.DataFrame(records).to_csv(run_dir / "fold_metrics.csv", index=False)
        np.savez_compressed(
            fold_dir / "outer_predictions.npz",
            row_indices=plan.outer_validation_indices,
            baseline=oof["baseline"][plan.outer_validation_indices],
            tuned=oof["tuned"][plan.outer_validation_indices],
        )
        write_json(
            {"signature": signature, "records": [row for row in records if row["fold"] == fold]},
            completion_path,
        )
    covered = np.isfinite(oof["baseline"]).all(axis=1)
    summary = {
        "folds": settings["outer_folds"],
        "rows": int(covered.sum()),
        "fingerprint": split_fingerprint(manifest),
        "test_accessed": False,
        "train_sha256": train_sha,
        "configuration_sha256": config_sha,
        "base_configuration_sha256": base_sha,
        "implementation_sha256": implementation_sha,
        "validation_fingerprint": split_fingerprint(manifest),
        "nested_audit_passed": True,
        "complete_oof": bool(covered.all()),
        "baseline_scope": "base+cross-fitted molecule-prior Huber/MSE convex blend",
        "selection": "single grouped inner holdout; outer scoring only",
        "duration_seconds": time.perf_counter() - started,
    }
    for name, pred in oof.items():
        metrics = masked_prr_metrics(targets[covered], pred[covered])
        summary[name] = {"rmse_masked": metrics.masked_rmse, "s_prr": metrics.sprr}
        pd.concat(
            [
                train.loc[covered, [ID_COLUMN]].reset_index(drop=True),
                pd.DataFrame(pred[covered], columns=columns),
            ],
            axis=1,
        ).to_csv(run_dir / f"oof_{'optimized' if name == 'tuned' else name}.csv", index=False)
    summary["delta_s_prr"] = summary["tuned"]["s_prr"] - summary["baseline"]["s_prr"]
    write_json(summary, run_dir / "summary.json")
    return summary
