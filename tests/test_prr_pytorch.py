import numpy as np
import torch

from deep_pharma.prr_pytorch import (
    PRRNetwork,
    _loss,
    _method_metrics,
    masked_huber,
    masked_mse,
    masked_prr_metrics,
    progressive_raw_weight,
)


def test_progressive_loss_schedule() -> None:
    assert progressive_raw_weight(1, 3, 4) == 0.0
    assert progressive_raw_weight(3, 3, 4) == 0.0
    assert progressive_raw_weight(4, 3, 4) == 0.25
    assert progressive_raw_weight(5, 3, 4) == 0.5
    assert progressive_raw_weight(7, 3, 4) == 1.0
    assert progressive_raw_weight(20, 3, 4) == 1.0


def test_masked_losses_ignore_zero_targets() -> None:
    target = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
    prediction = torch.tensor([[100.0, 1.0], [200.0, 6.0]])
    mask = target > 0
    assert torch.isclose(masked_mse(prediction, target, mask), torch.tensor(2.5))
    expected_huber = torch.nn.functional.smooth_l1_loss(
        torch.tensor([1.0, 6.0]), torch.tensor([2.0, 4.0]), beta=0.5
    )
    assert torch.isclose(masked_huber(prediction, target, mask, beta=0.5), expected_huber)


def test_masked_prr_metric_is_on_raw_positive_targets() -> None:
    target = np.array([[0.0, 2.0], [4.0, 0.0]], dtype=np.float32)
    prediction = np.array([[99.0, 1.0], [6.0, 99.0]], dtype=np.float32)
    metrics = masked_prr_metrics(target, prediction)
    expected_rmse = np.sqrt((1.0**2 + 2.0**2) / 2.0)
    assert np.isclose(metrics.masked_rmse, expected_rmse)
    assert np.isclose(metrics.sprr, 1.0 / (1.0 + expected_rmse))
    assert metrics.positive_count == 2


def test_network_always_has_the_same_two_heads() -> None:
    model = PRRNetwork(10, 3, [8, 4], dropout=0.1, use_batch_norm=True)
    conditional, logits = model(torch.ones(2, 10))
    assert conditional.shape == logits.shape == (2, 3)
    assert torch.all(conditional > 0)
    assert hasattr(model, "regression_head")
    assert hasattr(model, "binary_head")


def test_huber_finetune_uses_raw_prr_mse() -> None:
    config = {
        "training": {
            "max_epochs": 10,
            "huber_beta": 0.5,
            "hurdle_bce_weight": 1.0,
            "hurdle_regression_weight": 1.0,
        },
        "loss_schedules": {
            "progressive": {"log_only_epochs": 2, "transition_epochs": 2},
            "huber": {"mse_finetune_epochs": 2},
        },
    }
    conditional_normalized = torch.tensor([[2.0, 2.0]])
    target_raw = torch.tensor([[10.0, 0.0]])
    target_normalized = torch.tensor([[1.0, 0.0]])
    scales = torch.tensor([10.0, 5.0])
    loss = _loss(
        "huber",
        conditional_normalized,
        torch.zeros_like(target_raw),
        target_raw,
        target_normalized,
        scales,
        epoch=9,
        config=config,
    )
    assert torch.isclose(loss, torch.tensor(100.0))


def test_hurdle_can_be_forced_to_use_conditional_output() -> None:
    target = np.array([[2.0, 4.0]], dtype=np.float32)
    conditional = target.copy()
    expected = np.zeros_like(target)
    mode, metrics, candidates = _method_metrics(
        "hurdle",
        target,
        conditional,
        expected,
        hurdle_selection_mode="conditional",
    )
    assert mode == "conditional"
    assert metrics.sprr == 1.0
    assert candidates["expected"].sprr < metrics.sprr
