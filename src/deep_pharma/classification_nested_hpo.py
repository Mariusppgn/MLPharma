"""Bounded nested CPU tuning of severity and relational binary + activity gate."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .feature_transform_cache import cache_fitted_text_transforms
from .four_branch_benchmark import (
    RelationalFactorization,
    _molecule_indices,
    _predict_relational,
    _relational_loader,
    _seed_everything,
    _target_scales,
    _train_relational_epochs,
    apply_platt_parameters,
    fit_platt_parameters,
)
from .nested_validation import build_nested_plan
from .prr_pytorch import _feature_builder
from .row_activity_benchmark import _activity_features
from .severity_priors_benchmark import _severity_features
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint


def select_gate_thresholds(y, probability, activity, thresholds, gates):
    """Selection accepts inner predictions only; no outer labels are available."""
    best = (-1.0, float(thresholds[0]), float(gates[0]))
    truth = np.asarray(y, dtype=bool)
    positives = int(truth.sum())
    for gate in gates:
        active = activity >= gate
        for threshold in thresholds:
            pred = (probability >= threshold) & active[:, None]
            tp = int((pred & truth).sum())
            score = 2 * tp / max(positives + int(pred.sum()), 1)
            if score > best[0]:
                best = (score, float(threshold), float(gate))
    return best


def relational_fit_predict(
    fit, held, y_binary, y_prr, settings, seed, *, epochs=None, selection_binary=None
):
    """Epoch selection is permitted only when explicit inner labels are supplied."""
    if epochs is None and selection_binary is None:
        raise ValueError("Inner labels required for epoch selection")
    if epochs is not None and selection_binary is not None:
        raise ValueError("Fixed refit must not receive validation labels")
    if epochs is not None and epochs < 1:
        raise ValueError("Refit epochs must be positive")
    _seed_everything(seed)
    count, encoded = _molecule_indices(fit, [fit, held])
    (left, right), (held_left, held_right) = encoded
    scales = _target_scales(y_prr, float(settings["target_scale_quantile"]))
    prevalence = y_binary.mean(axis=0)
    weights = np.clip(
        (1 - prevalence) / np.maximum(prevalence, 1e-4), 1, float(settings["positive_weight_max"])
    ).astype(np.float32)
    model = RelationalFactorization(
        count,
        y_binary.shape[1],
        int(settings["embedding_dim"]),
        int(settings["hidden_dim"]),
        float(settings["dropout"]),
    )
    loader = _relational_loader(left, right, y_binary, y_prr, int(settings["batch_size"]), True)
    optimizer = None
    best_score, best_epoch, stale = -1.0, 1, 0
    best_probability = None
    history = []
    maximum = int(settings["max_epochs"]) if epochs is None else int(epochs)
    for epoch in range(1, maximum + 1):
        optimizer = _train_relational_epochs(
            model,
            loader,
            epochs=1,
            config=settings,
            device=torch.device("cpu"),
            scales=scales,
            positive_weights=weights,
            optimizer=optimizer,
        )
        if epochs is not None and epoch != maximum:
            continue
        probability, _ = _predict_relational(
            model,
            held_left,
            held_right,
            batch_size=int(settings["batch_size"]),
            scales=scales,
            device=torch.device("cpu"),
        )
        if epochs is not None:
            return probability, maximum, history
        score, _, _ = select_gate_thresholds(
            selection_binary, probability, np.ones(len(held)), np.linspace(0.1, 0.8, 29), [0.0]
        )
        history.append({"epoch": epoch, "inner_binary_micro_f1": score})
        if score > best_score + 1e-7:
            best_score, best_epoch, stale = score, epoch, 0
            best_probability = probability.copy()
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    return best_probability, best_epoch, history


def _classifier(settings, seed, objective="multiclass"):
    return lgb.LGBMClassifier(
        objective=objective,
        n_estimators=int(settings["n_estimators"]),
        learning_rate=float(settings["learning_rate"]),
        num_leaves=int(settings["num_leaves"]),
        min_child_samples=int(settings["min_child_samples"]),
        reg_lambda=float(settings["reg_lambda"]),
        class_weight=settings.get("class_weight", "balanced")
        if objective == "multiclass"
        else None,
        n_jobs=int(settings.get("n_jobs", 4)),
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _representations(fit, held, labels, binary, folds, classes, feature_config, smoothing, seed):
    """Rebuild vocabulary/SVD and all supervised priors exclusively on current fit."""
    builder = _feature_builder(feature_config, seed)
    # Same fitted vocabulary; avoid tokenizing the same fit-side documents twice.
    cache_fitted_text_transforms(builder)
    x_fit = np.asarray(builder.fit_transform(fit), dtype=np.float32)
    x_held = np.asarray(builder.transform(held), dtype=np.float32)
    severity_fit, severity_held = _severity_features(fit, held, labels, folds, classes, smoothing)
    activity_fit, activity_held = _activity_features(
        fit, held, binary.any(axis=1).astype(np.int8), folds, smoothing
    )
    return (
        np.hstack([x_fit, severity_fit]),
        np.hstack([x_held, severity_held]),
        np.hstack([x_fit, activity_fit]),
        np.hstack([x_held, activity_held]),
    )


def run_classification_nested_hpo(config_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config["inner_splits"]) < 3:
        raise ValueError(
            "At least three inner folds are required for core-only cross-fitted priors"
        )
    if config["severity_candidates"][0] or config["binary_candidates"][0]:
        raise ValueError("Candidate zero must preserve the unmodified baseline settings")
    paths = config["paths"]
    if any("test" in str(key).lower() for key in paths):
        raise ValueError("Test paths are forbidden in tuning configurations")
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    train = pd.read_csv(paths["train"], low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    if not audit["passed"]:
        raise ValueError("Invalid outer fold audit")
    feature_config = yaml.safe_load(Path(paths["feature_config"]).read_text(encoding="utf-8"))
    y_severity = train[SEVERITY_COLUMN].astype(str).to_numpy()
    binary_columns = [c for c in train if c.startswith(BINARY_PREFIX)]
    prr_columns = [c for c in train if c.startswith(PRR_PREFIX)]
    y_binary = train[binary_columns].to_numpy(np.int8)
    y_prr = train[prr_columns].to_numpy(np.float32)
    seed = int(config["seed"])
    run_dir = Path(paths["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "feature_config_sha256": hashlib.sha256(
            Path(paths["feature_config"]).read_bytes()
        ).hexdigest(),
        "train_sha256": hashlib.sha256(Path(paths["train"]).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(Path(paths["manifest"]).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transform_cache_sha256": hashlib.sha256(
            Path(__file__).with_name("feature_transform_cache.py").read_bytes()
        ).hexdigest(),
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.exists() and load_json(contract_path) != contract:
        raise ValueError("Run contract changed; use a new run_dir instead of mixing experiments")
    write_json(contract, contract_path)
    outer_folds = config.get("outer_folds", sorted(manifest.fold.unique().tolist()))
    folds_summary = []
    for outer in outer_folds:
        fold_dir = run_dir / f"fold_{outer}"
        completed_path = fold_dir / "summary.json"
        required_paths = [
            fold_dir / f"{kind}_{variant}.{suffix}"
            for variant in ("baseline", "optimized")
            for kind, suffix in (("predictions", "csv"), ("probabilities", "npz"))
        ]
        if completed_path.exists() and all(path.exists() for path in required_paths):
            cached = load_json(completed_path)
            if cached.get("run_contract") != contract:
                raise ValueError("Completed fold has incompatible run contract")
            folds_summary.append(cached)
            print(f"Classification outer fold {outer}: resuming completed artifacts", flush=True)
            continue
        plan = build_nested_plan(
            train, manifest, int(outer), inner_splits=int(config["inner_splits"]), seed=seed
        )
        fit_ids, held_ids = plan.outer_fit_indices, plan.outer_validation_indices
        frame_fit = train.iloc[fit_ids].reset_index(drop=True)
        frame_held = train.iloc[held_ids].reset_index(drop=True)
        classes = sorted(np.unique(y_severity[fit_ids]))
        ys, yb, yp = y_severity[fit_ids], y_binary[fit_ids], y_prr[fit_ids]
        settings_severity = [
            dict(config["severity_baseline"], **variant)
            for variant in config["severity_candidates"]
        ]
        settings_binary = [
            dict(config["binary_baseline"], **variant) for variant in config["binary_candidates"]
        ]
        sev_inner = [np.empty(len(fit_ids), dtype=object) for _ in settings_severity]
        binary_inner = [np.full(yb.shape, np.nan, np.float32) for _ in settings_binary]
        epochs = [[] for _ in settings_binary]
        activity_inner = np.full(len(fit_ids), np.nan, np.float32)
        inner_history = []
        for inner, (core, calibration) in enumerate(plan.iter_inner_splits()):
            print(
                f"Classification fold {outer}, inner {inner}: fit-only feature preparation",
                flush=True,
            )
            core_frame = frame_fit.iloc[core].reset_index(drop=True)
            cal_frame = frame_fit.iloc[calibration].reset_index(drop=True)
            xs, xsv, xa, xav = _representations(
                core_frame,
                cal_frame,
                ys[core],
                yb[core],
                plan.inner_fold_values[core],
                classes,
                feature_config,
                float(config["prior_smoothing"]),
                seed + int(outer),
            )
            for candidate, setting in enumerate(settings_severity):
                model = _classifier(setting, seed + int(outer))
                model.fit(xs, ys[core])
                sev_inner[candidate][calibration] = model.predict(xsv)
            gate = _classifier(config["activity_model"], seed + int(outer), "binary")
            active = yb[core].any(axis=1).astype(np.int8)
            gate.fit(xa, active)
            positive = np.flatnonzero(np.asarray(gate.classes_) == 1)
            activity_inner[calibration] = (
                gate.predict_proba(xav)[:, positive[0]] if len(positive) else 0
            )
            for candidate, setting in enumerate(settings_binary):
                probability, best_epoch, history = relational_fit_predict(
                    core_frame,
                    cal_frame,
                    yb[core],
                    yp[core],
                    setting,
                    seed + int(outer) * 100 + inner,
                    selection_binary=yb[calibration],
                )
                binary_inner[candidate][calibration] = probability
                epochs[candidate].append(best_epoch)
                inner_history.append(
                    {"inner_fold": inner, "candidate": candidate, "history": history}
                )
            print(f"Classification fold {outer}, inner {inner}: completed", flush=True)
        if not np.isfinite(activity_inner).all() or any(
            not np.isfinite(p).all() for p in binary_inner
        ):
            raise ValueError("Incomplete inner OOF predictions")
        severity_scores = [
            float(f1_score(ys, p, average="macro", zero_division=0)) for p in sev_inner
        ]
        binary_scores, calibrators, threshold_pairs = [], [], []
        for probability in binary_inner:
            calibration = fit_platt_parameters(yb, probability, regularization=1.0)
            calibrated = apply_platt_parameters(probability, calibration)
            score, threshold, gate_threshold = select_gate_thresholds(
                yb,
                calibrated,
                activity_inner,
                config["binary_thresholds"],
                config["gate_thresholds"],
            )
            binary_scores.append(score)
            calibrators.append(calibration)
            threshold_pairs.append((threshold, gate_threshold))
        best_severity = int(np.argmax(severity_scores))
        best_binary = int(np.argmax(binary_scores))
        _, baseline_threshold, baseline_gate = select_gate_thresholds(
            yb,
            apply_platt_parameters(binary_inner[0], calibrators[0]),
            activity_inner,
            config["binary_thresholds"],
            [float(config.get("baseline_gate_threshold", 0.5))],
        )
        # Lock all selections before creating any outer predictions.
        selections = {
            "severity_index": best_severity,
            "binary_index": best_binary,
            "severity_inner_scores": severity_scores,
            "binary_inner_scores": binary_scores,
            "binary_epochs": epochs,
            "binary_thresholds": threshold_pairs,
            "baseline_binary_threshold": baseline_threshold,
            "baseline_gate_threshold": baseline_gate,
            "outer_labels_used_for_selection": False,
        }
        fold_dir.mkdir(exist_ok=True)
        write_json(selections, fold_dir / "locked_selection.json")
        write_json({"curves": inner_history}, fold_dir / "inner_training_history.json")
        xs, xsv, xa, xav = _representations(
            frame_fit,
            frame_held,
            ys,
            yb,
            plan.inner_fold_values,
            classes,
            feature_config,
            float(config["prior_smoothing"]),
            seed + int(outer),
        )
        gate = _classifier(config["activity_model"], seed + int(outer), "binary")
        gate.fit(xa, yb.any(axis=1).astype(np.int8))
        positive = np.flatnonzero(np.asarray(gate.classes_) == 1)
        activity = (
            gate.predict_proba(xav)[:, positive[0]] if len(positive) else np.zeros(len(held_ids))
        )
        variants, metrics = {}, {}
        severity_cache, binary_cache = {}, {}
        for variant, severity_id, binary_id in [
            ("baseline", 0, 0),
            ("optimized", best_severity, best_binary),
        ]:
            if severity_id not in severity_cache:
                model = _classifier(settings_severity[severity_id], seed + int(outer))
                model.fit(xs, ys)
                severity_cache[severity_id] = model.predict(xsv)
            if binary_id not in binary_cache:
                fixed_epochs = max(1, int(np.median(epochs[binary_id])))
                probability, _, _ = relational_fit_predict(
                    frame_fit,
                    frame_held,
                    yb,
                    yp,
                    settings_binary[binary_id],
                    seed + int(outer) * 100 + 50,
                    epochs=fixed_epochs,
                )
                binary_cache[binary_id] = apply_platt_parameters(
                    probability, calibrators[binary_id]
                )
            threshold, gate_threshold = threshold_pairs[binary_id]
            if variant == "baseline":
                threshold, gate_threshold = baseline_threshold, baseline_gate
            decision = (
                (binary_cache[binary_id] >= threshold) & (activity >= gate_threshold)[:, None]
            ).astype(np.int8)
            variants[variant] = {"severity": severity_cache[severity_id], "binary": decision}
            metrics[variant] = {
                "severity_macro_f1": float(
                    f1_score(
                        y_severity[held_ids],
                        variants[variant]["severity"],
                        average="macro",
                        zero_division=0,
                    )
                ),
                "binary_micro_f1": float(
                    f1_score(y_binary[held_ids], decision, average="micro", zero_division=0)
                ),
            }
            predictions = pd.concat(
                [
                    train.iloc[held_ids][[ID_COLUMN]].reset_index(drop=True),
                    pd.Series(variants[variant]["severity"], name=SEVERITY_COLUMN),
                    pd.DataFrame(decision, columns=binary_columns),
                ],
                axis=1,
            )
            predictions.to_csv(fold_dir / f"predictions_{variant}.csv", index=False)
            np.savez_compressed(
                fold_dir / f"probabilities_{variant}.npz",
                binary=binary_cache[binary_id],
                activity=activity,
                row_indices=held_ids,
            )
        entry = {
            "fold": int(outer),
            "metrics": metrics,
            "selection": selections,
            "run_contract": contract,
            "inner_fingerprint": plan.inner_fingerprint,
        }
        write_json(entry, fold_dir / "summary.json")
        folds_summary.append(entry)
        print(f"Classification outer fold {outer}: {metrics}", flush=True)
    complete = set(int(fold) for fold in outer_folds) == set(int(fold) for fold in manifest.fold)
    for variant in ("baseline", "optimized"):
        pieces = [
            pd.read_csv(run_dir / f"fold_{outer}" / f"predictions_{variant}.csv")
            for outer in outer_folds
        ]
        combined = pd.concat(pieces, ignore_index=True)
        if combined[ID_COLUMN].duplicated().any():
            raise ValueError("Duplicate IDs in classification outer predictions")
        combined[ID_COLUMN] = combined[ID_COLUMN].astype(str)
        expected = train[[ID_COLUMN]].copy()
        expected[ID_COLUMN] = expected[ID_COLUMN].astype(str)
        combined = expected.merge(combined, on=ID_COLUMN, how="left", validate="one_to_one")
        if complete and combined.isna().any().any():
            raise ValueError("Incomplete classification OOF coverage")
        combined.to_csv(run_dir / f"oof_{variant}.csv", index=False)
    summary = {
        "run_id": config["run_id"],
        "folds": folds_summary,
        "validation_fingerprint": split_fingerprint(manifest),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(Path(paths["train"]).read_bytes()).hexdigest(),
        "run_contract": contract,
        "duration_seconds": time.perf_counter() - started,
        "test_loaded": False,
        "complete_oof": complete,
        "audit": {
            "passed": True,
            "inner_feature_builders_fit_only": True,
            "supervised_priors_cross_fitted_on_current_fit_only": True,
            "outer_labels_used_for_selection": False,
            "same_outer_folds_baseline_and_optimized": True,
        },
        "automatic_submission": False,
        "scope": "severity priors and relational binary with independently learned activity gate",
    }
    write_json(summary, run_dir / "summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run_classification_nested_hpo(parser.parse_args().config)
