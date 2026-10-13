import numpy as np
import torch

from deep_pharma.four_branch_benchmark import (
    RelationalFactorization,
    _fit_micro_threshold,
    apply_platt_parameters,
    fit_platt_parameters,
    fixed_hard_vote,
    fixed_prr_blend,
)


def test_relational_factorization_is_pair_symmetric() -> None:
    torch.manual_seed(7)
    model = RelationalFactorization(
        n_molecules=8,
        n_targets=3,
        embedding_dim=4,
        hidden_dim=6,
        dropout=0.0,
    )
    model.eval()
    left = torch.tensor([1, 2, 3])
    right = torch.tensor([4, 5, 6])

    binary_lr, prr_lr = model(left, right)
    binary_rl, prr_rl = model(right, left)

    assert torch.allclose(binary_lr, binary_rl)
    assert torch.allclose(prr_lr, prr_rl)
    assert torch.all(prr_lr > 0)


def test_platt_calibration_and_micro_threshold_have_matching_shapes() -> None:
    targets = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.int8)
    probabilities = np.array(
        [[0.15, 0.75], [0.30, 0.85], [0.70, 0.35], [0.90, 0.15]],
        dtype=np.float32,
    )
    parameters = fit_platt_parameters(targets, probabilities, regularization=1.0)
    calibrated = apply_platt_parameters(probabilities, parameters)
    threshold, score = _fit_micro_threshold(targets, calibrated, minimum=0.1, maximum=0.9, steps=17)

    assert calibrated.shape == targets.shape
    assert np.isfinite(calibrated).all()
    assert 0.1 <= threshold <= 0.9
    assert score == 1.0


def test_fixed_blends_do_not_fit_on_targets() -> None:
    first_binary = np.array([[1, 0], [1, 1]], dtype=np.int8)
    second_binary = np.array([[0, 1], [1, 0]], dtype=np.int8)
    voted = fixed_hard_vote(
        {"first": first_binary, "second": second_binary},
        {"first": 0.75, "second": 0.25},
    )
    assert np.array_equal(voted, first_binary)

    first_prr = np.array([[2.0, 4.0]], dtype=np.float32)
    second_prr = np.array([[4.0, 8.0]], dtype=np.float32)
    blended = fixed_prr_blend(
        {"first": first_prr, "second": second_prr},
        {"first": 0.5, "second": 0.5},
    )
    assert np.allclose(blended, [[3.0, 6.0]])
