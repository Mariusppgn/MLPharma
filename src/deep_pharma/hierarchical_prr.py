"""CPU friendly, fold safe hierarchical PRR expert.

Only positive PRR cells are observed by the competition's masked RMSE. Fit this
expert on the training portion of a split and call ``predict`` on held out rows.
For training predictions used by another model, use ``cross_fit_hierarchical_prr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .validation import canonicalize_smiles, frame_pair_keys


@dataclass(frozen=True)
class HierarchicalPRRConfig:
    ridge: float = 20.0
    max_iterations: int = 100
    svd_rank: int = 10
    svd_mix: float = 0.25
    min_prediction: float = 0.01

    def __post_init__(self) -> None:
        if self.ridge <= 0:
            raise ValueError("ridge must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        if self.svd_rank < 0:
            raise ValueError("svd_rank must be nonnegative")
        if not 0 <= self.svd_mix <= 1:
            raise ValueError("svd_mix must be in [0, 1]")
        if self.min_prediction <= 0:
            raise ValueError("min_prediction must be positive")


def _canonical_molecules(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    if not {"SMILES_A", "SMILES_B"}.issubset(frame.columns):
        raise ValueError("SMILES_A and SMILES_B are required")
    if frame[["SMILES_A", "SMILES_B"]].isna().any().any():
        raise ValueError("Missing SMILES")
    left = [canonicalize_smiles(str(value)) for value in frame["SMILES_A"]]
    right = [canonicalize_smiles(str(value)) for value in frame["SMILES_B"]]
    return left, right


class HierarchicalPRR:
    """Ridge molecular effects, shared low rank denoising, positive output.

    The target-specific model is ``global_mean + effect[A] + effect[B]``.
    Ridge regularization shrinks low-count molecules towards zero. Truncated
    SVD denoises the molecule-by-event effect matrix across targets.
    """

    def __init__(self, config: HierarchicalPRRConfig | None = None) -> None:
        self.config = config or HierarchicalPRRConfig()

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> HierarchicalPRR:
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(frame) or len(frame) == 0:
            raise ValueError("targets must have shape (len(frame), n_targets) with nonempty rows")
        if values.shape[1] == 0 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("targets must be finite, nonnegative, and have at least one column")

        left, right = _canonical_molecules(frame)
        molecules = tuple(sorted(set(left) | set(right)))
        lookup = {molecule: index for index, molecule in enumerate(molecules)}
        left_ids = np.fromiter((lookup[value] for value in left), dtype=np.int64)
        right_ids = np.fromiter((lookup[value] for value in right), dtype=np.int64)
        rows = np.tile(np.arange(len(frame), dtype=np.int64), 2)
        columns = np.concatenate((left_ids, right_ids))
        # A self-pair contributes two copies of the same molecule effect.
        design = sparse.coo_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, columns)),
            shape=(len(frame), len(molecules)),
        ).tocsr()

        positive = values > 0
        count = positive.sum(axis=0)
        positive_values = np.where(positive, values, 0.0)
        observed_mean = np.divide(
            positive_values.sum(axis=0), count,
            out=np.zeros(values.shape[1], dtype=np.float64), where=count > 0,
        )
        any_positive = positive_values[positive]
        fallback_mean = float(any_positive.mean()) if len(any_positive) else 1.0
        global_mean = np.where(count > 0, observed_mean, fallback_mean)
        effects = np.zeros((len(molecules), values.shape[1]), dtype=np.float64)
        for target_index in range(values.shape[1]):
            observed = positive[:, target_index]
            if not observed.any():
                continue
            effects[:, target_index] = lsqr(
                design[observed],
                values[observed, target_index] - global_mean[target_index],
                damp=np.sqrt(self.config.ridge),
                atol=1e-5,
                btol=1e-5,
                iter_lim=self.config.max_iterations,
            )[0]

        if self.config.svd_rank and self.config.svd_mix and min(effects.shape) > 1:
            # Scaling prevents high-variance PRR targets from monopolizing the SVD.
            target_scale = np.maximum(global_mean, self.config.min_prediction)
            scaled = effects / target_scale[None, :]
            u, singular, vt = np.linalg.svd(scaled, full_matrices=False)
            rank = min(self.config.svd_rank, len(singular))
            denoised = ((u[:, :rank] * singular[:rank]) @ vt[:rank]) * target_scale[None, :]
            effects = (1 - self.config.svd_mix) * effects + self.config.svd_mix * denoised

        self.molecules_ = molecules
        self.lookup_ = lookup
        self.global_mean_ = global_mean
        self.effects_ = effects
        self.positive_count_ = count
        self.n_targets_ = values.shape[1]
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "effects_"):
            raise RuntimeError("The expert must be fitted before predict")
        left, right = _canonical_molecules(frame)
        left_ids = np.fromiter((self.lookup_.get(value, -1) for value in left), dtype=np.int64)
        right_ids = np.fromiter((self.lookup_.get(value, -1) for value in right), dtype=np.int64)
        prediction = np.broadcast_to(self.global_mean_, (len(frame), self.n_targets_)).copy()
        seen_left = left_ids >= 0
        seen_right = right_ids >= 0
        prediction[seen_left] += self.effects_[left_ids[seen_left]]
        prediction[seen_right] += self.effects_[right_ids[seen_right]]
        return np.maximum(prediction, self.config.min_prediction).astype(np.float32)


def cross_fit_hierarchical_prr(
    frame: pd.DataFrame,
    targets: np.ndarray,
    fold_values: np.ndarray,
    *,
    config: HierarchicalPRRConfig | None = None,
) -> np.ndarray:
    """Predict each row with a model fitted on other pair-grouped folds only."""
    folds = np.asarray(fold_values)
    values = np.asarray(targets)
    if folds.ndim != 1 or len(folds) != len(frame) or len(values) != len(frame):
        raise ValueError("folds, targets, and frame must align")
    if len(np.unique(folds)) < 2:
        raise ValueError("at least two folds are required")
    pair_keys = frame_pair_keys(frame).to_numpy()
    prediction = np.empty(values.shape, dtype=np.float32)
    for fold in np.unique(folds):
        held_out = folds == fold
        if set(pair_keys[held_out]) & set(pair_keys[~held_out]):
            raise ValueError("a canonical molecule pair occurs in both fit and held-out rows")
        expert = HierarchicalPRR(config).fit(frame.loc[~held_out], values[~held_out])
        prediction[held_out] = expert.predict(frame.loc[held_out])
    return prediction
