import numpy as np
import pandas as pd
import pytest

from deep_pharma.hierarchical_prr import (
    HierarchicalPRR,
    HierarchicalPRRConfig,
    cross_fit_hierarchical_prr,
)


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SMILES_A": ["CC", "CC", "CO", "CN", "CCl", "CO"],
            "SMILES_B": ["CO", "CN", "CCl", "CCl", "CC", "CN"],
        }
    )


def test_symmetric_prediction_and_unseen_fallback() -> None:
    frame = _pairs()
    target = np.array(
        [[2.0, 1.0], [4.0, 2.0], [6.0, 0.0], [3.0, 4.0], [5.0, 3.0], [7.0, 1.0]]
    )
    model = HierarchicalPRR(HierarchicalPRRConfig(ridge=1.0, svd_rank=1)).fit(frame, target)
    forward = model.predict(frame)
    reverse = model.predict(frame.rename(columns={"SMILES_A": "SMILES_B", "SMILES_B": "SMILES_A"}))
    assert np.allclose(forward, reverse)
    assert (forward > 0).all()

    unseen = pd.DataFrame({"SMILES_A": ["CF"], "SMILES_B": ["CBr"]})
    assert np.allclose(model.predict(unseen)[0], model.global_mean_)


def test_cross_fit_does_not_use_held_out_labels() -> None:
    frame = _pairs()
    target = np.array([[1.0], [3.0], [5.0], [7.0], [9.0], [11.0]])
    folds = np.array([0, 1, 0, 1, 0, 1])
    config = HierarchicalPRRConfig(ridge=1.0, svd_rank=0)
    reference = cross_fit_hierarchical_prr(frame, target, folds, config=config)
    modified = target.copy()
    modified[folds == 0] = 1000.0
    prediction = cross_fit_hierarchical_prr(frame, modified, folds, config=config)
    assert np.array_equal(prediction[folds == 0], reference[folds == 0])
    assert not np.array_equal(prediction[folds == 1], reference[folds == 1])


def test_cross_fit_rejects_pair_group_overlap_including_reversed_order() -> None:
    frame = pd.DataFrame({"SMILES_A": ["CC", "CO"], "SMILES_B": ["CO", "CC"]})
    with pytest.raises(ValueError, match="canonical molecule pair"):
        cross_fit_hierarchical_prr(frame, np.ones((2, 1)), np.array([0, 1]))


def test_only_positive_cells_train_each_target() -> None:
    frame = _pairs()
    target = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]])
    model = HierarchicalPRR(HierarchicalPRRConfig(svd_rank=0)).fit(frame, target)
    assert model.positive_count_.tolist() == [6, 0]
    assert np.allclose(model.effects_[:, 1], 0)
    assert np.allclose(model.predict(frame)[:, 1], target[:, 0].mean())
