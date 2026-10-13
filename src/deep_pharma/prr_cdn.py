"""Leakage-safe conditional-mean dependency network for positive PRR.

This is deterministic fixed-point inference, not a joint probabilistic Gibbs sampler.
All settings are fixed before outer validation; convergence uses predictions only.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupKFold

from .chemistry_features import build_chemical_pair_features
from .constants import PRR_PREFIX
from .submission_gate import write_json
from .validation import audit_pair_grouped_folds, split_fingerprint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def stateless_features(frame: pd.DataFrame, bins: int) -> np.ndarray:
    """No fitted preprocessing and no target columns enter this representation."""
    fingerprints, descriptors = build_chemical_pair_features(frame)
    dense = fingerprints.toarray().astype(np.float32)
    folded = np.zeros((len(frame), bins * 3), dtype=np.float32)
    for block in range(3):
        for column in range(2048):
            folded[:, block * bins + column % bins] += dense[:, block * 2048 + column]
    hasher = HashingVectorizer(n_features=bins, alternate_sign=False, ngram_range=(1, 2))
    sides = []
    for side in ("A", "B"):
        columns = [c for c in frame if c.endswith("_" + side) and c != "SMILES_" + side]
        text = frame[columns].fillna("").astype(str).agg(" ".join, axis=1)
        sides.append(hasher.transform(text).toarray().astype(np.float32))
    left, right = sides
    return np.nan_to_num(np.concatenate(
        [descriptors.to_numpy(np.float32), folded, left + right,
         np.abs(left - right), left * right], axis=1
    ), nan=0, posinf=0, neginf=0)


def conditional_features(x: np.ndarray, predictions: np.ndarray, target: int) -> np.ndarray:
    """Explicitly exclude the current target from conditional inputs."""
    others = np.delete(predictions, target, axis=1)
    return np.concatenate([x, np.log1p(np.maximum(others, 0))], axis=1)


def fit_heads(x, y, config, coeffects=None):
    def fit_one(target):
        positive = y[:, target] > 0
        features = x if coeffects is None else conditional_features(x, coeffects, target)
        if positive.sum() < 10:
            return float(y[positive, target].mean()) if positive.any() else 0.0
        model = lgb.LGBMRegressor(
            objective="regression", n_estimators=config["trees"],
            num_leaves=config["leaves"], learning_rate=config["learning_rate"],
            min_child_samples=config["min_child_samples"], reg_lambda=config["reg_lambda"],
            random_state=config["seed"] + target, n_jobs=1, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
        model.fit(features[positive], y[positive, target])
        return model
    return Parallel(n_jobs=config["jobs"], prefer="threads")(
        delayed(fit_one)(target) for target in range(y.shape[1])
    )


def predict_heads(models, x, coeffects=None):
    values = []
    for target, model in enumerate(models):
        features = x if coeffects is None else conditional_features(x, coeffects, target)
        prediction = np.full(len(x), model) if isinstance(model, float) else model.predict(features)
        values.append(np.maximum(prediction, 0))
    return np.column_stack(values)


def crossfit_base(x, y, groups, config):
    """Each training row's conditioning vector excludes its complete pair group."""
    result = np.full_like(y, np.nan, dtype=float)
    audits = []
    splitter = GroupKFold(n_splits=config["inner_folds"])
    for inner, (fit, held) in enumerate(splitter.split(x, groups=groups)):
        overlap = len(set(groups[fit]) & set(groups[held]))
        if overlap:
            raise ValueError("Inner pair-group leakage")
        models = fit_heads(x[fit], y[fit], config)
        result[held] = predict_heads(models, x[held])
        audits.append({"inner_fold": inner, "fit_rows": len(fit),
                       "held_rows": len(held), "pair_group_overlap": overlap})
    if not np.isfinite(result).all():
        raise ValueError("Incomplete/nonfinite inner OOF predictions")
    return result, audits


def iterate_cdn(models, x, initial, *, damping, iterations, tolerance):
    """Synchronous updates; never accept labels or stop using validation scores."""
    if not 0 < damping <= 1 or iterations < 1 or tolerance < 0:
        raise ValueError("Invalid iteration settings")
    current = initial.copy()
    states = [current.copy()]
    diagnostics = []
    converged = False
    for sweep in range(1, iterations + 1):
        proposal = predict_heads(models, x, current)
        updated = (1 - damping) * current + damping * proposal
        if not np.isfinite(updated).all():
            raise ValueError("CDN diverged to nonfinite predictions")
        delta = updated - current
        relative = float(np.linalg.norm(delta) / max(np.linalg.norm(current), 1e-12))
        diagnostics.append({"sweep": sweep, "relative_change": relative,
                            "rms_change": float(np.sqrt(np.mean(delta**2))),
                            "max_change": float(np.max(np.abs(delta)))})
        current = updated
        states.append(current.copy())
        if relative <= tolerance:
            converged = True
            break
    return current, states, diagnostics, converged


def score(y, prediction):
    mask = y > 0
    rmse = float(np.sqrt(np.mean((y[mask] - prediction[mask]) ** 2)))
    return {"rmse_masked": rmse, "s_prr": 1 / (1 + rmse),
            "positive_count": int(mask.sum())}


def plots(curves, output):
    plt.style.use("ggplot")
    for metric, label in [("relative_change", "Variation relative des prédictions"),
                          ("s_prr", "S_PRR (validation externe, diagnostic)"),
                          ("rmse_masked", "RMSE masqué (PRR bruts)")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for fold, group in curves.groupby("fold"):
            ax.plot(group.sweep, group[metric], marker="o", ms=3, label=f"Fold {fold}")
        if metric == "relative_change":
            ax.set_yscale("log")
        ax.set(xlabel="Mise à jour (0 = modèle indépendant)", ylabel=label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"convergence_{metric}.png", dpi=180)
        plt.close(fig)


def run(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Completed run exists: choose a new output_dir")
    train_path = Path(config["train_path"])
    train = pd.read_csv(train_path)
    manifest = pd.read_csv(config["fold_manifest"]).sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    if not audit["passed"]:
        raise ValueError(audit["errors"])
    columns = [c for c in train if c.startswith(PRR_PREFIX)]
    y = train[columns].to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid PRR targets")
    x = stateless_features(train, config["feature_bins"])
    folds = manifest.fold.to_numpy()
    groups = manifest.pair_group.to_numpy()
    baseline = np.full_like(y, np.nan)
    cdn = np.full_like(y, np.nan)
    curves, results, inner_audits = [], [], []
    selected = config.get("folds", sorted(np.unique(folds).tolist()))
    for fold in selected:
        print(f"Fold {fold}: inner cross-fitting", flush=True)
        fit, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        coeffects, inner_audit = crossfit_base(x[fit], y[fit], groups[fit], config)
        inner_audits.append({"outer_fold": fold, "inner_folds": inner_audit})
        base_models = fit_heads(x[fit], y[fit], config)
        initial = predict_heads(base_models, x[held])
        conditional = fit_heads(x[fit], y[fit], config, coeffects)
        final, states, changes, converged = iterate_cdn(
            conditional, x[held], initial, damping=config["damping"],
            iterations=config["iterations"], tolerance=config["tolerance"],
        )
        baseline[held], cdn[held] = initial, final
        for sweep, state in enumerate(states):
            change = {"relative_change": np.nan, "rms_change": np.nan, "max_change": np.nan}
            if sweep:
                change = changes[sweep - 1]
            curves.append({"fold": fold, "sweep": sweep, **change, **score(y[held], state)})
        result = {"fold": fold, "baseline": score(y[held], initial),
                  "cdn": score(y[held], final), "converged": converged,
                  "sweeps": len(changes), "last_relative_change": changes[-1]["relative_change"]}
        results.append(result)
        pd.DataFrame(curves).to_csv(output / "iteration_metrics.csv", index=False)
        np.savez_compressed(output / f"fold_{fold}.npz", row_indices=held,
                            baseline=initial, cdn=final)
        print(result, flush=True)
    return finish_run(config, train, manifest, baseline, cdn, curves, results, inner_audits)


def finish_run(config, train, manifest, baseline, cdn, curves, results, inner_audits):
    output = Path(config["output_dir"])
    train_path = Path(config["train_path"])
    audit = audit_pair_grouped_folds(train, manifest)
    columns = [c for c in train if c.startswith(PRR_PREFIX)]
    y = train[columns].to_numpy(float)
    covered = np.isfinite(cdn).all(axis=1)
    summary = {"config": config, "fold_fingerprint": split_fingerprint(manifest),
               "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
               "outer_audit": audit, "inner_audits": inner_audits,
               "baseline": score(y[covered], baseline[covered]),
               "cdn": score(y[covered], cdn[covered]), "fold_results": results,
               "complete_oof": bool(covered.all()), "test_used": False,
               "selection": "fixed settings; prediction-only convergence; no promotion",
               "all_folds_improved": all(r["cdn"]["s_prr"] > r["baseline"]["s_prr"]
                                         for r in results)}
    write_json(summary, output / "summary.json")
    curves_frame = pd.DataFrame(curves)
    plots(curves_frame, output)
    effects = []
    for target, column in enumerate(columns):
        effects.append({"effect": column.removeprefix(PRR_PREFIX),
                        **{f"baseline_{k}": v for k, v in
                           score(y[covered, target], baseline[covered, target]).items()},
                        **{f"cdn_{k}": v for k, v in
                           score(y[covered, target], cdn[covered, target]).items()}})
    pd.DataFrame(effects).to_csv(output / "per_effect.csv", index=False)
    print(summary["baseline"], summary["cdn"], flush=True)
    return summary


def finalize_saved_run(config_path):
    """Recover final artifacts from saved fold predictions without retraining."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    train = pd.read_csv(config["train_path"])
    manifest = pd.read_csv(config["fold_manifest"]).sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    if not audit["passed"]:
        raise ValueError(audit["errors"])
    y = train[[c for c in train if c.startswith(PRR_PREFIX)]].to_numpy(float)
    baseline, cdn = np.full_like(y, np.nan), np.full_like(y, np.nan)
    curves = pd.read_csv(output / "iteration_metrics.csv")
    results, inner_audits = [], []
    groups = manifest.pair_group.to_numpy()
    for fold in sorted(curves.fold.unique()):
        data = np.load(output / f"fold_{fold}.npz")
        held = data["row_indices"]
        expected = np.flatnonzero(manifest.fold.to_numpy() == fold)
        if not np.array_equal(held, expected):
            raise ValueError("Saved fold row alignment mismatch")
        baseline[held], cdn[held] = data["baseline"], data["cdn"]
        last = curves[curves.fold == fold].iloc[-1]
        results.append({"fold": int(fold), "baseline": score(y[held], baseline[held]),
                        "cdn": score(y[held], cdn[held]),
                        "converged": bool(last.relative_change <= config["tolerance"]),
                        "sweeps": int(last.sweep),
                        "last_relative_change": float(last.relative_change)})
        fit = np.flatnonzero(manifest.fold.to_numpy() != fold)
        splits = GroupKFold(n_splits=config["inner_folds"])
        inner_audit = []
        for inner, (learn, held_inner) in enumerate(splits.split(fit, groups=groups[fit])):
            inner_audit.append({"inner_fold": inner, "fit_rows": len(learn),
                                "held_rows": len(held_inner), "pair_group_overlap":
                                len(set(groups[fit][learn]) & set(groups[fit][held_inner]))})
        inner_audits.append({"outer_fold": int(fold), "inner_folds": inner_audit})
    return finish_run(config, train, manifest, baseline, cdn,
                      curves.to_dict("records"), results, inner_audits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prr_cdn.yaml")
    run(parser.parse_args().config)
