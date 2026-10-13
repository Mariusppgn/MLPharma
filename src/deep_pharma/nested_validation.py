"""Shared nested, canonical-pair grouped selection protocol.

Outer validation is evaluation-only. Build inner splits from outer-fit labels only;
fit preprocessing, priors, thresholds and checkpoints inside those inner splits.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .validation import (
    audit_pair_grouped_folds,
    build_pair_grouped_folds,
    frame_pair_keys,
    pair_group_hash,
    split_fingerprint,
)


def assert_group_disjoint(groups: np.ndarray, fit: np.ndarray, held: np.ndarray) -> None:
    """Reject own-row or own-pair leakage before fitting any estimator."""
    groups = np.asarray(groups)
    fit, held = np.asarray(fit), np.asarray(held)
    if np.intersect1d(fit, held).size:
        raise ValueError("Fit and held-out row indices overlap.")
    if set(groups[fit]) & set(groups[held]):
        raise ValueError("Canonical pair groups cross the split.")
    if not len(fit) or not len(held):
        raise ValueError("Fit and held-out partitions must both be nonempty.")


@dataclass(frozen=True)
class NestedFoldPlan:
    outer_fold: int
    outer_fit_indices: np.ndarray
    outer_validation_indices: np.ndarray
    inner_fold_values: np.ndarray
    outer_fingerprint: str
    inner_fingerprint: str
    audit: dict[str, object]

    @property
    def fingerprint(self) -> str:
        return self.outer_fingerprint

    def iter_inner_splits(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Return LOCAL indices into outer-fit, never into the full frame."""
        for fold in np.unique(self.inner_fold_values):
            yield (
                np.flatnonzero(self.inner_fold_values != fold),
                np.flatnonzero(self.inner_fold_values == fold),
            )


def build_nested_plan(
    frame: pd.DataFrame,
    outer_manifest: pd.DataFrame,
    outer_fold: int,
    *,
    inner_splits: int = 3,
    seed: int = 2026,
) -> NestedFoldPlan:
    """Validate outer grouping and stratify only the outer-fit subset.

    Shared deterministic inner folds make all candidates and branches comparable.
    The plan guarantees row/group exclusion, not estimator-level leakage safety.
    """
    frame = frame.reset_index(drop=True)
    manifest = outer_manifest.sort_values("row_index").reset_index(drop=True)
    outer_audit = audit_pair_grouped_folds(frame, manifest)
    if not outer_audit["passed"]:
        raise ValueError(str(outer_audit["errors"]))
    canonical_groups = frame_pair_keys(frame).map(pair_group_hash).to_numpy()
    if not np.array_equal(canonical_groups, manifest["pair_group"].to_numpy()):
        raise ValueError("Manifest pair-group identifiers do not match canonical chemistry.")
    folds = manifest["fold"].to_numpy()
    if pd.isna(folds).any() or not np.equal(folds, np.floor(folds)).all() or (folds < 0).any():
        raise ValueError("Outer fold labels must be nonnegative integers.")
    if outer_fold not in folds:
        raise ValueError(f"Unknown outer fold: {outer_fold}")
    fit = np.flatnonzero(folds != outer_fold)
    validation = np.flatnonzero(folds == outer_fold)
    assert_group_disjoint(canonical_groups, fit, validation)
    if inner_splits < 2 or len(np.unique(canonical_groups[fit])) < inner_splits:
        raise ValueError("At least two inner folds and enough distinct pairs are required.")
    inner_seed = seed + 1009 * (int(outer_fold) + 1)
    # No outer-validation target is accessed by inner stratification.
    inner, inner_audit = build_pair_grouped_folds(
        frame.iloc[fit].reset_index(drop=True), n_splits=inner_splits, seed=inner_seed
    )
    if not inner_audit["passed"]:
        raise ValueError(str(inner_audit["errors"]))
    inner_folds = inner["fold"].to_numpy(copy=True)
    for inner_fold in np.unique(inner_folds):
        assert_group_disjoint(
            canonical_groups[fit],
            np.flatnonzero(inner_folds != inner_fold),
            np.flatnonzero(inner_folds == inner_fold),
        )
    for array in (fit, validation, inner_folds):
        array.flags.writeable = False
    return NestedFoldPlan(
        outer_fold=int(outer_fold),
        outer_fit_indices=fit,
        outer_validation_indices=validation,
        inner_fold_values=inner_folds,
        outer_fingerprint=split_fingerprint(manifest),
        inner_fingerprint=split_fingerprint(inner),
        audit={
            "passed": True,
            "outer_fold": int(outer_fold),
            "outer_fit_rows": len(fit),
            "outer_validation_rows": len(validation),
            "outer_pair_group_overlap": 0,
            "inner": inner_audit,
            "selection_scope": "inner_only",
            "outer_validation_role": "evaluation_only",
            "test_accessed": False,
            "preprocessing_requirement": "fit inside each corresponding fit partition",
        },
    )


def validate_oof_coverage(predictions: np.ndarray, counts: np.ndarray) -> None:
    """Only publish OOF scores when every row was predicted exactly once."""
    predictions, counts = np.asarray(predictions), np.asarray(counts)
    if counts.shape != (len(predictions),) or not np.all(counts == 1):
        raise ValueError("OOF predictions must cover each row exactly once.")
    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions contain missing or nonfinite values.")
