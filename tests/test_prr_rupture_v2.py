import numpy as np
import torch

from deep_pharma.prr_rupture_v2 import (
    MaskedFactorizationPRR,
    apply_moe_weights,
    fit_moe_weights,
    knn_positive_prr,
    regularized_simplex_weights,
    select_high_sse_targets,
)


def test_factorization_is_symmetric_at_inference() -> None:
    torch.manual_seed(7)
    model = MaskedFactorizationPRR(
        input_dim=4,
        n_identities=8,
        n_targets=3,
        identity_dim=5,
        rank=6,
        dropout=0.0,
        identity_mask_rate=0.5,
    )
    model.eval()
    features = torch.randn(2, 4)
    prior = torch.full((2, 3), 2.0)
    left = torch.tensor([1, 2])
    right = torch.tensor([3, 4])
    forward = model(features, left, right, prior)
    reverse = model(features, right, left, prior)
    assert torch.allclose(forward, reverse, atol=1e-6)
    assert torch.all(forward > 0)


def test_regularized_simplex_weights_are_valid() -> None:
    experts = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    targets = np.array([1.1, 2.0, 2.9], dtype=np.float32)
    weights = regularized_simplex_weights(experts, targets, ridge=1.0)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_moe_uses_regimes_and_returns_finite_predictions() -> None:
    targets = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=np.float32)
    first = targets.copy()
    second = targets[::-1].copy()
    familiarity = np.array([[0], [0], [1], [1], [2], [2]], dtype=np.float32)
    weights = fit_moe_weights(
        targets,
        [first, second],
        familiarity,
        ridge=1.0,
        minimum_regime_rows=2,
        regime_shrinkage=1.0,
    )
    prediction = apply_moe_weights(weights, [first, second], familiarity)
    assert weights.shape == (1, 3, 2)
    assert np.allclose(weights.sum(axis=2), 1.0)
    assert np.isfinite(prediction).all()


def test_knn_positive_prr_falls_back_to_prior_without_positive_neighbor() -> None:
    fit_features = np.eye(3, dtype=np.float32)
    validation_features = np.eye(3, dtype=np.float32)
    targets = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]], dtype=np.float32)
    prior = np.array([[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    prediction = knn_positive_prr(
        fit_features,
        validation_features,
        targets,
        prior,
        neighbors=2,
        temperature=0.2,
        prior_strength=0.5,
        dimensions=3,
        n_jobs=1,
    )
    assert np.allclose(prediction[:, 0], prior[:, 0])
    assert np.isfinite(prediction).all()


def test_high_sse_selection_uses_positive_dispersion() -> None:
    targets = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.1, 8.0, 2.0],
            [0.9, 15.0, 2.1],
        ],
        dtype=np.float32,
    )
    selected = select_high_sse_targets(targets, top_k=1)
    assert selected.tolist() == [1]
