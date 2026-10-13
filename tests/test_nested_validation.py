import numpy as np
import pandas as pd
import pytest

from deep_pharma.nested_validation import (
    assert_group_disjoint,
    build_nested_plan,
    validate_oof_coverage,
)
from deep_pharma.validation import build_pair_grouped_folds


def training_frame():
    rows = []
    for pair in range(18):
        for reverse in range(2):
            left, right = "C" * (pair + 1), "O"
            if reverse:
                left, right = right, left
            rows.append(
                {
                    "Pair_ID": f"pair_{pair}_{reverse}",
                    "SMILES_A": left,
                    "SMILES_B": right,
                    "Severity": ["Major", "Moderate", "Minor"][pair % 3],
                    "Target_Binary_effect": pair % 2,
                    "Target_PRR_effect": float(pair % 2) * (pair + 1),
                }
            )
    return pd.DataFrame(rows)


def test_nested_plan_has_group_exclusion_complete_local_coverage():
    frame = training_frame()
    manifest, _ = build_pair_grouped_folds(frame, n_splits=3, seed=2026)
    for outer in range(3):
        plan = build_nested_plan(frame, manifest, outer, inner_splits=3)
        assert plan.audit["passed"]
        assert plan.audit["outer_validation_role"] == "evaluation_only"
        assert plan.outer_fingerprint == plan.fingerprint
        counts = np.zeros(len(plan.outer_fit_indices), int)
        groups = manifest.pair_group.to_numpy()
        for fit, held in plan.iter_inner_splits():
            assert_group_disjoint(groups[plan.outer_fit_indices], fit, held)
            assert not set(plan.outer_fit_indices[held]) & set(plan.outer_validation_indices)
            counts[held] += 1
        assert (counts == 1).all()
        assert not plan.inner_fold_values.flags.writeable


def test_outer_label_mutation_cannot_change_inner_splits():
    frame = training_frame()
    manifest, _ = build_pair_grouped_folds(frame, n_splits=3, seed=2026)
    original = build_nested_plan(frame, manifest, 0)
    modified = frame.copy()
    held = original.outer_validation_indices
    modified.loc[held, "Severity"] = "DO_NOT_READ_OUTER_LABELS"
    modified.loc[held, "Target_Binary_effect"] = 100
    modified.loc[held, "Target_PRR_effect"] = 999999
    changed = build_nested_plan(modified, manifest, 0)
    np.testing.assert_array_equal(original.inner_fold_values, changed.inner_fold_values)
    assert original.inner_fingerprint == changed.inner_fingerprint


def test_manifest_cannot_forge_canonical_groups():
    frame = training_frame()
    manifest, _ = build_pair_grouped_folds(frame, n_splits=3, seed=2026)
    manifest["pair_group"] = "fake"
    with pytest.raises(ValueError, match="canonical chemistry"):
        build_nested_plan(frame, manifest, 0)


def test_plan_is_stable_when_manifest_row_order_changes():
    frame = training_frame()
    manifest, _ = build_pair_grouped_folds(frame, n_splits=3, seed=2026)
    a = build_nested_plan(frame, manifest, 0)
    b = build_nested_plan(frame, manifest.sample(frac=1, random_state=22), 0)
    assert a.inner_fingerprint == b.inner_fingerprint
    np.testing.assert_array_equal(a.outer_fit_indices, b.outer_fit_indices)


def test_group_overlap_rejected_even_without_row_overlap():
    with pytest.raises(ValueError, match="groups cross"):
        assert_group_disjoint(np.array(["same", "same", "other"]), [0, 2], [1])
    with pytest.raises(ValueError, match="row indices"):
        assert_group_disjoint(np.array(["a", "b"]), [0], [0, 1])


def test_oof_coverage_rejects_duplicate_missing_or_nonfinite_predictions():
    validate_oof_coverage(np.ones((3, 2)), np.ones(3))
    for counts in [np.array([1, 2, 1]), np.array([1, 0, 1]), np.ones(2)]:
        with pytest.raises(ValueError, match="exactly once"):
            validate_oof_coverage(np.ones((3, 2)), counts)
    with pytest.raises(ValueError, match="nonfinite"):
        validate_oof_coverage(np.array([1, np.nan]), np.ones(2))
