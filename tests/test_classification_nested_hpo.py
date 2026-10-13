import numpy as np
import pandas as pd
import pytest

from deep_pharma.classification_nested_hpo import (
    _representations,
    relational_fit_predict,
    select_gate_thresholds,
)
from deep_pharma.prr_pytorch import _feature_builder
from deep_pharma.text_features import TEXT_FIELDS


def test_gate_selection_uses_probabilities_and_activity():
    y = np.array([[1, 0], [0, 0], [1, 1]])
    p = np.array([[0.8, 0.1], [0.9, 0.8], [0.7, 0.8]])
    score, threshold, gate = select_gate_thresholds(
        y, p, np.array([0.9, 0.1, 0.9]), [0.5], [0, 0.5]
    )
    assert score == 1
    assert threshold == 0.5
    assert gate == 0.5


def test_relational_requires_inner_labels_for_selection():
    with pytest.raises(ValueError, match="Inner labels"):
        relational_fit_predict(
            pd.DataFrame(), pd.DataFrame(), np.empty((0, 1)), np.empty((0, 1)), {}, 0
        )


def test_threshold_selection_deterministic_tie():
    score, threshold, gate = select_gate_thresholds(
        np.array([[1]]), np.array([[0.9]]), np.ones(1), [0.2, 0.5], [0, 0.5]
    )
    assert (score, threshold, gate) == (1, 0.2, 0)


def test_fixed_refit_rejects_validation_labels():
    with pytest.raises(ValueError, match="must not receive"):
        relational_fit_predict(
            pd.DataFrame(),
            pd.DataFrame(),
            np.empty((0, 1)),
            np.empty((0, 1)),
            {},
            0,
            epochs=1,
            selection_binary=np.ones((1, 1)),
        )


def test_relational_cpu_fit_predict_smoke():
    frame = pd.DataFrame({"SMILES_A": ["C", "CC", "O", "CC"], "SMILES_B": ["O", "O", "C", "C"]})
    binary = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int8)
    settings = {
        "embedding_dim": 4,
        "hidden_dim": 8,
        "dropout": 0.1,
        "target_scale_quantile": 0.95,
        "positive_weight_max": 5,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1,
        "binary_loss_weight": 1,
        "prr_loss_weight": 1,
    }
    pred, epochs, history = relational_fit_predict(
        frame, frame.iloc[:2], binary, binary.astype(np.float32) * 2, settings, 2026, epochs=1
    )
    assert pred.shape == (2, 2)
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()
    assert epochs == 1
    assert history == []


def test_text_memoization_keeps_fitted_feature_values_identical():
    smiles = ["C", "CC", "CCC", "O", "CO", "CCO"]
    fit = pd.DataFrame(
        {
            "SMILES_A": [smiles[i % 6] for i in range(30)],
            "SMILES_B": [smiles[(i + 1) % 6] for i in range(30)],
        }
    )
    for field in TEXT_FIELDS:
        for side in ("A", "B"):
            fit[f"{field}_{side}"] = [f"alpha beta gamma m{i % 6} delta" for i in range(30)]
    held = fit.iloc[:6].copy()
    settings = {
        "features": {
            "text_components": 2,
            "morgan_components": 2,
            "word_max_features": 100,
            "char_max_features": 100,
            "min_df": 1,
            "include_warning_toxicity": True,
        }
    }
    plain = _feature_builder(settings, 2026)
    expected_fit = plain.fit_transform(fit)
    expected_held = plain.transform(held)
    labels = np.array(["Major", "Moderate", "Minor"] * 10)
    binary = np.column_stack([np.arange(30) % 2, np.arange(30) % 3 == 0]).astype(np.int8)
    actual_fit, actual_held, _, _ = _representations(
        fit,
        held,
        labels,
        binary,
        np.arange(30) % 3,
        ["Major", "Moderate", "Minor"],
        settings,
        20.0,
        2026,
    )
    np.testing.assert_array_equal(actual_fit[:, : expected_fit.shape[1]], expected_fit)
    np.testing.assert_array_equal(actual_held[:, : expected_held.shape[1]], expected_held)
