from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import canonicalize_smiles


@dataclass(frozen=True)
class MoleculePriorTables:
    molecules: tuple[str, ...]
    positive_mean: np.ndarray
    prevalence: np.ndarray
    row_count: np.ndarray
    global_positive_mean: np.ndarray
    global_prevalence: np.ndarray


def _canonical_molecules(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    left = frame["SMILES_A"].astype(str).map(canonicalize_smiles).to_numpy()
    right = frame["SMILES_B"].astype(str).map(canonicalize_smiles).to_numpy()
    return left, right


def fit_molecule_prior_tables(
    frame: pd.DataFrame,
    targets: np.ndarray,
    *,
    smoothing: float,
) -> MoleculePriorTables:
    """Fit target statistics from the supplied rows only."""

    if len(frame) != len(targets):
        raise ValueError("Les lignes et les cibles des priors moléculaires sont désalignées.")
    if smoothing < 0:
        raise ValueError("Le lissage des priors moléculaires doit être positif ou nul.")

    left, right = _canonical_molecules(frame)
    molecules = np.concatenate([left, right])
    repeated_targets = np.concatenate([targets, targets], axis=0).astype(np.float64, copy=False)

    # A pair containing the same molecule twice contributes only once.
    row_ids = np.tile(np.arange(len(frame), dtype=np.int64), 2)
    unique_mask = ~pd.MultiIndex.from_arrays([row_ids, molecules]).duplicated()
    molecules = molecules[unique_mask]
    repeated_targets = repeated_targets[unique_mask]

    molecule_codes, unique_molecules = pd.factorize(molecules, sort=True)
    n_molecules = len(unique_molecules)
    n_targets = targets.shape[1]
    row_count = np.bincount(molecule_codes, minlength=n_molecules).astype(np.float64)
    positive = repeated_targets > 0
    positive_count = np.zeros((n_molecules, n_targets), dtype=np.float64)
    positive_sum = np.zeros_like(positive_count)
    for target_index in range(n_targets):
        np.add.at(
            positive_count[:, target_index],
            molecule_codes,
            positive[:, target_index].astype(np.float64),
        )
        np.add.at(
            positive_sum[:, target_index],
            molecule_codes,
            np.where(positive[:, target_index], repeated_targets[:, target_index], 0.0),
        )

    raw_targets = targets.astype(np.float64, copy=False)
    raw_positive = raw_targets > 0
    global_positive_count = raw_positive.sum(axis=0)
    global_positive_mean = np.divide(
        np.where(raw_positive, raw_targets, 0.0).sum(axis=0),
        global_positive_count,
        out=np.zeros(n_targets, dtype=np.float64),
        where=global_positive_count > 0,
    )
    global_prevalence = raw_positive.mean(axis=0)
    positive_mean = np.divide(
        positive_sum + smoothing * global_positive_mean,
        positive_count + smoothing,
        out=np.broadcast_to(global_positive_mean, positive_sum.shape).copy(),
        where=(positive_count + smoothing) > 0,
    )
    prevalence = (positive_count + smoothing * global_prevalence) / (row_count[:, None] + smoothing)
    return MoleculePriorTables(
        molecules=tuple(str(value) for value in unique_molecules),
        positive_mean=positive_mean.astype(np.float32),
        prevalence=prevalence.astype(np.float32),
        row_count=row_count.astype(np.float32),
        global_positive_mean=global_positive_mean.astype(np.float32),
        global_prevalence=global_prevalence.astype(np.float32),
    )


def transform_molecule_priors(
    frame: pd.DataFrame,
    tables: MoleculePriorTables,
) -> np.ndarray:
    """Build symmetric pair features with global fallbacks for unseen molecules."""

    left, right = _canonical_molecules(frame)
    lookup = {molecule: index for index, molecule in enumerate(tables.molecules)}
    left_indices = np.fromiter((lookup.get(value, -1) for value in left), dtype=np.int64)
    right_indices = np.fromiter((lookup.get(value, -1) for value in right), dtype=np.int64)

    def gather(values: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_values = np.broadcast_to(fallback, (len(frame), len(fallback))).copy()
        right_values = left_values.copy()
        left_seen = left_indices >= 0
        right_seen = right_indices >= 0
        left_values[left_seen] = values[left_indices[left_seen]]
        right_values[right_seen] = values[right_indices[right_seen]]
        return left_values, right_values

    mean_left, mean_right = gather(tables.positive_mean, tables.global_positive_mean)
    prev_left, prev_right = gather(tables.prevalence, tables.global_prevalence)

    # Every feature is invariant to swapping molecule A and molecule B.
    blocks = [
        (mean_left + mean_right) / 2.0,
        np.maximum(mean_left, mean_right),
        np.abs(mean_left - mean_right),
        (prev_left + prev_right) / 2.0,
        np.maximum(prev_left, prev_right),
        np.abs(prev_left - prev_right),
    ]
    counts = np.zeros((len(frame), 2), dtype=np.float32)
    left_seen = left_indices >= 0
    right_seen = right_indices >= 0
    counts[left_seen, 0] = tables.row_count[left_indices[left_seen]]
    counts[right_seen, 1] = tables.row_count[right_indices[right_seen]]
    log_counts = np.log1p(counts)
    blocks.extend(
        [
            log_counts.mean(axis=1, keepdims=True),
            log_counts.max(axis=1, keepdims=True),
            np.abs(log_counts[:, :1] - log_counts[:, 1:]),
        ]
    )
    features = np.hstack(blocks).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Les priors moléculaires contiennent des valeurs non finies.")
    return features


def build_cross_fitted_molecule_priors(
    frame: pd.DataFrame,
    targets: np.ndarray,
    inner_folds: np.ndarray,
    *,
    smoothing: float,
) -> np.ndarray:
    """Encode each training row using target statistics from other pair-group folds."""

    if len(frame) != len(inner_folds):
        raise ValueError("Les folds internes et les lignes sont désalignés.")
    encoded: np.ndarray | None = None
    covered = np.zeros(len(frame), dtype=bool)
    for fold in np.unique(inner_folds):
        held_out = inner_folds == fold
        tables = fit_molecule_prior_tables(
            frame.loc[~held_out].reset_index(drop=True),
            targets[~held_out],
            smoothing=smoothing,
        )
        fold_features = transform_molecule_priors(
            frame.loc[held_out].reset_index(drop=True), tables
        )
        if encoded is None:
            encoded = np.empty((len(frame), fold_features.shape[1]), dtype=np.float32)
        encoded[held_out] = fold_features
        covered[held_out] = True
    if encoded is None or not covered.all():
        raise RuntimeError("La couverture cross-fit des priors moléculaires est incomplète.")
    return encoded
