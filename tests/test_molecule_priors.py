import numpy as np
import pandas as pd

from deep_pharma.molecule_priors import (
    build_cross_fitted_molecule_priors,
    fit_molecule_prior_tables,
    transform_molecule_priors,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SMILES_A": ["CC", "CC", "O", "N"],
            "SMILES_B": ["O", "N", "CC", "CC"],
        }
    )


def test_cross_fitted_priors_do_not_use_held_out_targets() -> None:
    frame = _frame()
    folds = np.array([0, 0, 1, 1])
    targets = np.array([[2.0], [4.0], [8.0], [16.0]], dtype=np.float32)
    changed = targets.copy()
    changed[folds == 0] = 10_000.0

    encoded = build_cross_fitted_molecule_priors(frame, targets, folds, smoothing=2.0)
    encoded_changed = build_cross_fitted_molecule_priors(frame, changed, folds, smoothing=2.0)

    assert np.allclose(encoded[folds == 0], encoded_changed[folds == 0])
    assert not np.allclose(encoded[folds == 1], encoded_changed[folds == 1])


def test_pair_prior_features_are_symmetric() -> None:
    frame = _frame()
    targets = np.array([[2.0], [0.0], [8.0], [4.0]], dtype=np.float32)
    tables = fit_molecule_prior_tables(frame, targets, smoothing=2.0)
    original = transform_molecule_priors(frame, tables)
    swapped = transform_molecule_priors(
        frame.rename(columns={"SMILES_A": "SMILES_B", "SMILES_B": "SMILES_A"})[
            ["SMILES_A", "SMILES_B"]
        ],
        tables,
    )
    assert np.allclose(original, swapped)


def test_unseen_molecules_use_finite_global_fallbacks() -> None:
    frame = _frame()
    targets = np.array([[2.0], [0.0], [8.0], [4.0]], dtype=np.float32)
    tables = fit_molecule_prior_tables(frame, targets, smoothing=2.0)
    unseen = pd.DataFrame({"SMILES_A": ["Cl"], "SMILES_B": ["Br"]})
    encoded = transform_molecule_priors(unseen, tables)
    assert encoded.shape == (1, 9)
    assert np.isfinite(encoded).all()
