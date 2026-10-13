"""Leakage-safe PRR tail expert with fold-local side-effect dependencies."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .constants import ID_COLUMN, PRR_PREFIX
from .four_branch_benchmark import _load_or_build_features
from .molecule_priors import (
    build_cross_fitted_molecule_priors,
    fit_molecule_prior_tables,
    transform_molecule_priors,
)
from .prr_pytorch import _feature_builder
from .prr_rupture_v2 import select_high_sse_targets
from .submission import validate_submission
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_aligned(path: str, train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != len(train) or not frame[ID_COLUMN].astype(str).equals(
        train[ID_COLUMN].astype(str)
    ):
        raise ValueError(f"Prédictions OOF désalignées : {path}")
    return frame


def fit_coeffect_graph(
    targets: np.ndarray,
    *,
    neighbors: int,
    minimum_joint: int,
    joint_shrinkage: float,
) -> np.ndarray:
    """Estimate positive co-occurrence within active fit rows only."""

    if targets.ndim != 2 or neighbors < 0:
        raise ValueError("Cibles 2D et nombre de voisins positif ou nul requis.")
    active = (targets > 0).any(axis=1)
    flags = (targets[active] > 0).astype(np.float64)
    n_targets = targets.shape[1]
    graph = np.zeros((n_targets, n_targets), dtype=np.float32)
    if len(flags) < 3 or neighbors == 0:
        return graph
    centered = flags - flags.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.square(centered).sum(axis=0))
    denominator = np.outer(norms, norms)
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((n_targets, n_targets), dtype=np.float64),
        where=denominator > 0,
    )
    joint = flags.T @ flags
    strength = np.maximum(correlation, 0.0) * joint / (joint + joint_shrinkage)
    np.fill_diagonal(strength, 0.0)
    strength[joint < minimum_joint] = 0.0
    for target in range(n_targets):
        selected = np.argsort(strength[target])[::-1][:neighbors]
        weights = strength[target, selected]
        if weights.sum() > 0:
            graph[target, selected] = (weights / weights.sum()).astype(np.float32)
    return graph


def target_dependency_features(
    base: np.ndarray,
    with_priors: np.ndarray,
    graph: np.ndarray,
    target: int,
    coeffect_predictions: np.ndarray | None = None,
) -> np.ndarray:
    """Own-target and predicted-coeffect priors; no observed validation labels."""

    n_targets = graph.shape[0]
    if graph.shape != (n_targets, n_targets) or not 0 <= target < n_targets:
        raise ValueError("Graphe ou cible invalide.")
    if len(base) != len(with_priors) or with_priors.shape[1] != base.shape[1] + 6 * n_targets + 3:
        raise ValueError("Dimensions des priors moléculaires invalides.")
    prior = with_priors[:, base.shape[1] :]
    own = np.column_stack([prior[:, block * n_targets + target] for block in range(6)])
    related = np.column_stack(
        [
            prior[:, block * n_targets : (block + 1) * n_targets] @ graph[target]
            for block in range(6)
        ]
    )
    blocks = [base, own, related, prior[:, -3:]]
    if coeffect_predictions is not None:
        if coeffect_predictions.shape != (len(base), n_targets):
            raise ValueError("Prédictions de coeffets désalignées.")
        related_prediction = coeffect_predictions @ graph[target]
        blocks.extend(
            [
                coeffect_predictions[:, target : target + 1],
                related_prediction[:, None],
                coeffect_predictions.mean(axis=1, keepdims=True),
                coeffect_predictions.max(axis=1, keepdims=True),
            ]
        )
    return np.hstack(blocks).astype(np.float32)


def fit_reduced_rank_coeffects(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    validation_features: np.ndarray,
    *,
    rank: int,
    ridge: float,
) -> np.ndarray:
    """Fit a shared low-rank map from inputs to all PRR labels on fit rows only."""

    if rank < 1 or ridge <= 0 or len(fit_features) != len(fit_targets):
        raise ValueError("Paramètres du modèle multieffets invalides.")
    model = Ridge(alpha=ridge, solver="cholesky")
    model.fit(fit_features, np.log1p(np.maximum(fit_targets, 0.0)))
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    left, singular, right = np.linalg.svd(coefficients, full_matrices=False)
    effective_rank = min(rank, len(singular))
    compressed = (left[:, :effective_rank] * singular[:effective_rank]) @ right[:effective_rank]
    log_prediction = validation_features @ compressed.T + model.intercept_
    return np.expm1(np.clip(log_prediction, 0.0, 5.0)).astype(np.float32)


def crossfit_reduced_rank_coeffects(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    fit_folds: np.ndarray,
    validation_features: np.ndarray,
    *,
    rank: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Training coeffect features are predicted by models excluding their rows."""

    if len(fit_features) != len(fit_folds):
        raise ValueError("Folds et features désalignés.")
    oof = np.full(fit_targets.shape, np.nan, dtype=np.float32)
    for fold in np.unique(fit_folds):
        held_out = fit_folds == fold
        oof[held_out] = fit_reduced_rank_coeffects(
            fit_features[~held_out],
            fit_targets[~held_out],
            fit_features[held_out],
            rank=rank,
            ridge=ridge,
        )
    if not np.isfinite(oof).all():
        raise ValueError("Cross-fitting multieffets incomplet.")
    validation = fit_reduced_rank_coeffects(
        fit_features,
        fit_targets,
        validation_features,
        rank=rank,
        ridge=ridge,
    )
    return oof, validation


def _regressor(config: dict[str, object], seed: int, *, tail: bool = False) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=int(config["tail_estimators"] if tail else config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["tail_num_leaves"] if tail else config["num_leaves"]),
        min_child_samples=int(config["min_child_samples"]),
        reg_lambda=float(config["tail_reg_lambda"] if tail else config["reg_lambda"]),
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=1,
    )


def fit_predict_conditional_tail(
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    fit_values: np.ndarray,
    *,
    config: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """E[Y|Y>0,X] = E[min(Y,t)|Y>0,X] + P(Y>t|Y>0,X) E[Y-t|Y>t,X]."""

    positive = fit_values > 0
    count = int(positive.sum())
    if count < int(config["minimum_positive"]):
        value = float(fit_values[positive].mean()) if count else 0.0
        return np.full(len(validation_features), value, dtype=np.float32), {
            "positive_count": count,
            "tail_count": 0,
            "threshold": 0.0,
            "used_tail_head": False,
        }
    values = fit_values[positive].astype(np.float64)
    threshold = float(np.quantile(values, float(config["threshold_quantile"])))
    clipped = np.minimum(values, threshold)
    base_model = _regressor(config, seed)
    base_model.fit(fit_features[positive], clipped)
    base_prediction = np.clip(base_model.predict(validation_features), 0.0, threshold)
    exceed = values > threshold
    tail_count = int(exceed.sum())
    if tail_count < int(config["minimum_tail"]) or tail_count == count:
        return base_prediction.astype(np.float32), {
            "positive_count": count,
            "tail_count": tail_count,
            "threshold": threshold,
            "used_tail_head": False,
        }
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        min_child_samples=int(config["min_child_samples"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=seed + 10_000,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=1,
    )
    classifier.fit(fit_features[positive], exceed.astype(np.int8))
    probability = classifier.predict_proba(validation_features)[:, 1]
    tail_model = _regressor(config, seed + 20_000, tail=True)
    excess = values[exceed] - threshold
    tail_model.fit(fit_features[positive][exceed], excess)
    shrinkage = tail_count / (tail_count + float(config["tail_shrinkage"]))
    predicted_excess = shrinkage * np.maximum(tail_model.predict(validation_features), 0.0)
    predicted_excess += (1.0 - shrinkage) * float(excess.mean())
    prediction = base_prediction + probability * predicted_excess
    return np.clip(prediction, 0.0, float(values.max())).astype(np.float32), {
        "positive_count": count,
        "tail_count": tail_count,
        "threshold": threshold,
        "used_tail_head": True,
    }


def _fit_selected_targets(
    fit_base: np.ndarray,
    validation_base: np.ndarray,
    fit_with_priors: np.ndarray,
    validation_with_priors: np.ndarray,
    fit_targets: np.ndarray,
    graph: np.ndarray,
    selected: np.ndarray,
    config: dict[str, object],
    seed: int,
    fit_coeffects: np.ndarray | None = None,
    validation_coeffects: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    def fit_one(target: int) -> tuple[int, np.ndarray, dict[str, object]]:
        fit_features = target_dependency_features(
            fit_base, fit_with_priors, graph, target, fit_coeffects
        )
        validation_features = target_dependency_features(
            validation_base, validation_with_priors, graph, target, validation_coeffects
        )
        values, metadata = fit_predict_conditional_tail(
            fit_features,
            validation_features,
            fit_targets[:, target],
            config=config,
            seed=seed + target,
        )
        return target, values, {"target_index": target, **metadata}

    fitted = Parallel(n_jobs=int(config["target_n_jobs"]), prefer="threads")(
        delayed(fit_one)(int(target)) for target in selected
    )
    predictions = np.zeros((len(validation_base), fit_targets.shape[1]), dtype=np.float32)
    metadata = []
    for target, values, detail in fitted:
        predictions[:, target] = values
        metadata.append(detail)
    return predictions, metadata


def _score_prr(true: np.ndarray, predicted: np.ndarray) -> tuple[float, float, int]:
    positive = true > 0
    count = int(positive.sum())
    rmse = float(np.sqrt(np.square(true[positive] - predicted[positive]).mean()))
    return rmse, 1.0 / (1.0 + rmse), count


def _score_tail(
    true: np.ndarray, predicted: np.ndarray, *, threshold: float
) -> tuple[float, int]:
    mask = true >= threshold
    count = int(mask.sum())
    if count == 0:
        raise ValueError("Aucune observation dans la tranche PRR extrême.")
    return float(np.sqrt(np.square(true[mask] - predicted[mask]).mean())), count


def run_prr_tail_dependency(config_path: Path) -> dict[str, object]:
    """Evaluate a fixed tail/dependency blend on existing pair-grouped OOF folds."""

    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    stored_audit = load_json(Path(paths["audit"]))
    audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    if not audit["passed"] or fingerprint != stored_audit["fingerprint"]:
        raise ValueError("Folds groupés invalides.")
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    targets = train[prr_columns].to_numpy(np.float32)
    incumbent_frame = _read_aligned(paths["incumbent_oof"], train)
    incumbent = incumbent_frame[prr_columns].to_numpy(np.float32)
    folds = manifest["fold"].to_numpy()
    predictions = incumbent.copy()
    fold_details: list[dict[str, object]] = []
    alpha = float(config["selection"]["fixed_blend_weight"])
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Poids fixe hors de (0,1].")
    for fold in sorted(int(value) for value in np.unique(folds)):
        held_out = folds == fold
        fit_targets = targets[~held_out]
        selected = select_high_sse_targets(fit_targets, int(config["tail"]["top_k_targets"]))
        graph = fit_coeffect_graph(fit_targets, **config["graph"])
        fit_base, validation_base, fit_with_priors, validation_with_priors = (
            _load_or_build_features(train, targets, folds, fold, fingerprint, config)
        )
        fit_coeffects = None
        validation_coeffects = None
        if config.get("coeffect", {}).get("enabled", False):
            fit_coeffects, validation_coeffects = crossfit_reduced_rank_coeffects(
                fit_base,
                fit_targets,
                folds[~held_out],
                validation_base,
                rank=int(config["coeffect"]["rank"]),
                ridge=float(config["coeffect"]["ridge"]),
            )
        expert, target_details = _fit_selected_targets(
            fit_base,
            validation_base,
            fit_with_priors,
            validation_with_priors,
            fit_targets,
            graph,
            selected,
            config["tail"],
            int(config["seed"]) + 1000 * fold,
            fit_coeffects,
            validation_coeffects,
        )
        selected_values = np.maximum(
            (1.0 - alpha) * incumbent[held_out][:, selected] + alpha * expert[:, selected],
            0.0,
        )
        fold_predictions = predictions[held_out].copy()
        fold_predictions[:, selected] = selected_values
        predictions[held_out] = fold_predictions
        before_rmse, before_sprr, _ = _score_prr(targets[held_out], incumbent[held_out])
        after_rmse, after_sprr, _ = _score_prr(targets[held_out], fold_predictions)
        fold_details.append(
            {
                "fold": fold,
                "selected_targets": [prr_columns[int(index)] for index in selected],
                "before_rmse": before_rmse,
                "after_rmse": after_rmse,
                "before_sprr": before_sprr,
                "after_sprr": after_sprr,
                "sprr_gain": after_sprr - before_sprr,
                "tail_models": target_details,
                "graph_nonzero_edges": int(np.count_nonzero(graph)),
            }
        )
        print(
            f"Fold {fold}: S_PRR {before_sprr:.6f} -> {after_sprr:.6f} "
            f"({after_sprr - before_sprr:+.6f})",
            flush=True,
        )
    before_rmse, before_sprr, positive_count = _score_prr(targets, incumbent)
    after_rmse, after_sprr, _ = _score_prr(targets, predictions)
    tail_threshold = float(config["selection"]["tail_threshold"])
    before_tail_rmse, tail_count = _score_tail(targets, incumbent, threshold=tail_threshold)
    after_tail_rmse, _ = _score_tail(targets, predictions, threshold=tail_threshold)
    gain = after_sprr - before_sprr
    eligible = gain >= float(config["selection"]["minimum_sprr_gain"])
    eligible = eligible and after_tail_rmse < before_tail_rmse
    if bool(config["selection"]["require_every_fold_improved"]):
        eligible = eligible and all(detail["sprr_gain"] > 0 for detail in fold_details)
    # The historical incumbent selects PyTorch checkpoints on its reported outer fold.
    # This comparison is diagnostic, not an independent nested promotion test.
    strict_nested_comparison = False
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [train[[ID_COLUMN]].reset_index(drop=True), pd.DataFrame(predictions, columns=prr_columns)],
        axis=1,
    ).to_csv(output_dir / "oof_predictions_tail_dependency.csv", index=False)
    diagnostics = []
    for target, column in enumerate(prr_columns):
        positive = targets[:, target] > 0
        old_sse = float(np.square(targets[positive, target] - incumbent[positive, target]).sum())
        new_sse = float(np.square(targets[positive, target] - predictions[positive, target]).sum())
        diagnostics.append(
            {
                "target": column,
                "positive_count": int(positive.sum()),
                "old_sse": old_sse,
                "new_sse": new_sse,
                "sse_gain": old_sse - new_sse,
            }
        )
    pd.DataFrame(diagnostics).sort_values("old_sse", ascending=False).to_csv(
        output_dir / "target_diagnostics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in fold.items()
                if key not in {"tail_models", "selected_targets"}
            }
            for fold in fold_details
        ]
    ).to_csv(output_dir / "fold_comparison.csv", index=False)
    summary = {
        "run_id": config["run_id"],
        "config_sha256": _sha256(config_path),
        "train_sha256": _sha256(train_path),
        "validation_fingerprint": fingerprint,
        "incumbent_oof_sha256": _sha256(Path(paths["incumbent_oof"])),
        "before": {"rmse": before_rmse, "sprr": before_sprr},
        "after": {"rmse": after_rmse, "sprr": after_sprr},
        "sprr_gain": gain,
        "tail_threshold": tail_threshold,
        "tail_count": tail_count,
        "tail_rmse_before": before_tail_rmse,
        "tail_rmse_after": after_tail_rmse,
        "positive_count": positive_count,
        "folds": fold_details,
        "score_gate_eligible": bool(eligible),
        "promotion_eligible": bool(eligible and strict_nested_comparison),
        "strict_nested_comparison": strict_nested_comparison,
        "coeffect_model": config.get("coeffect", {"enabled": False}),
        "leakage_audit": {
            "passed": strict_nested_comparison,
            "new_expert_passed": True,
            "pair_group_overlap_max": 0,
            "target_selection_fit_only": True,
            "graph_fit_only": True,
            "molecule_priors_cross_fitted": True,
            "test_loaded": False,
            "incumbent_oof_used_only_for_validation_blend_and_scoring": True,
            "incumbent_checkpoint_selected_on_outer_validation": True,
        },
        "duration_seconds": time.perf_counter() - started,
    }
    run_dir = Path(paths["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, run_dir / "summary.json")
    return summary


def finalize_prr_tail_dependency(config_path: Path) -> dict[str, object]:
    """Refit on the official train only after a strict OOF win, then edit PRR columns."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    summary = load_json(Path(paths["run_dir"]) / "summary.json")
    if not summary["promotion_eligible"] or not summary.get("strict_nested_comparison", False):
        raise ValueError(
            "Le benchmark ne satisfait pas tous les garde-fous PRR globaux, "
            "par fold et extrêmes; aucune finalisation."
        )
    if summary.get("config_sha256") != _sha256(config_path):
        raise ValueError("La configuration a changé depuis la validation OOF.")
    if summary.get("incumbent_oof_sha256") != _sha256(Path(paths["incumbent_oof"])):
        raise ValueError("La référence OOF PRR a changé depuis la validation.")
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    if summary["train_sha256"] != _sha256(train_path):
        raise ValueError("Le train a changé depuis la sélection OOF.")
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    if not audit["passed"] or split_fingerprint(manifest) != summary["validation_fingerprint"]:
        raise ValueError("Les folds ont changé depuis la sélection OOF.")
    # This is intentionally the first read of the test set.
    test = pd.read_csv(paths["test"], low_memory=False)
    sample_path = Path(paths["sample_submission"])
    incumbent_path = Path(paths["incumbent_submission"])
    incumbent = pd.read_csv(incumbent_path, low_memory=False)
    if validate_submission(incumbent_path, sample_path):
        raise ValueError("La soumission de référence est invalide.")
    if not incumbent[ID_COLUMN].astype(str).equals(test[ID_COLUMN].astype(str)):
        raise ValueError("Test et soumission de référence désalignés.")
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    targets = train[prr_columns].to_numpy(np.float32)
    folds = manifest["fold"].to_numpy()
    feature_config = yaml.safe_load(Path(paths["feature_cache_config"]).read_text("utf-8"))
    builder = _feature_builder(feature_config, int(config["seed"]))
    fit_base_raw = builder.fit_transform(train)
    test_base_raw = builder.transform(test)
    base_scaler = StandardScaler().fit(fit_base_raw)
    fit_base = base_scaler.transform(fit_base_raw).astype(np.float32)
    test_base = base_scaler.transform(test_base_raw).astype(np.float32)
    smoothing = float(feature_config["molecule_priors"]["smoothing"])
    fit_prior_raw = build_cross_fitted_molecule_priors(train, targets, folds, smoothing=smoothing)
    tables = fit_molecule_prior_tables(train, targets, smoothing=smoothing)
    test_prior_raw = transform_molecule_priors(test, tables)
    prior_scaler = StandardScaler().fit(fit_prior_raw)
    fit_with_priors = np.hstack(
        [fit_base, prior_scaler.transform(fit_prior_raw).astype(np.float32)]
    )
    test_with_priors = np.hstack(
        [test_base, prior_scaler.transform(test_prior_raw).astype(np.float32)]
    )
    selected = select_high_sse_targets(targets, int(config["tail"]["top_k_targets"]))
    graph = fit_coeffect_graph(targets, **config["graph"])
    fit_coeffects = None
    test_coeffects = None
    if config.get("coeffect", {}).get("enabled", False):
        fit_coeffects, test_coeffects = crossfit_reduced_rank_coeffects(
            fit_base,
            targets,
            folds,
            test_base,
            rank=int(config["coeffect"]["rank"]),
            ridge=float(config["coeffect"]["ridge"]),
        )
    expert, detail = _fit_selected_targets(
        fit_base,
        test_base,
        fit_with_priors,
        test_with_priors,
        targets,
        graph,
        selected,
        config["tail"],
        int(config["seed"]) + 9000,
        fit_coeffects,
        test_coeffects,
    )
    alpha = float(config["selection"]["fixed_blend_weight"])
    candidate = incumbent.copy()
    current = candidate[prr_columns].to_numpy(np.float32)
    current[:, selected] = np.maximum(
        (1.0 - alpha) * current[:, selected] + alpha * expert[:, selected], 0.0
    )
    candidate[prr_columns] = current
    candidate_path = Path(paths["candidate_submission"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_path, index=False)
    errors = validate_submission(candidate_path, sample_path)
    if errors:
        candidate_path.unlink(missing_ok=True)
        raise ValueError("Soumission candidate invalide : " + "; ".join(errors))
    if not candidate.drop(columns=prr_columns).equals(incumbent.drop(columns=prr_columns)):
        raise AssertionError("Une colonne hors PRR a changé.")
    result = {
        "run_id": config["run_id"],
        "candidate_submission": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "reference_sha256": _sha256(incumbent_path),
        "changed_targets": [prr_columns[int(index)] for index in selected],
        "tail_models": detail,
        "test_used_for_fit": False,
        "test_used_for_scoring": False,
        "automatic_upload": False,
    }
    write_json(result, Path(paths["run_dir"]) / "candidate_summary.json")
    return result
