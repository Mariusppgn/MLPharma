import numpy as np
import torch

from deep_pharma.rupture_benchmark import (
    HybridPairModel,
    molecular_graph,
    simplex_weights,
)


def test_molecular_graph_has_expected_padding_and_self_loops() -> None:
    features, adjacency, atom_types, original_size = molecular_graph("CCO", max_atoms=8)
    assert features.shape == (8, 12)
    assert adjacency.shape == (8, 8)
    assert atom_types.shape == (8,)
    assert original_size == 3
    assert np.all(np.diag(adjacency)[:3] == 1)
    assert np.all(atom_types[3:] == -100)


def test_hybrid_pair_model_is_pair_symmetric() -> None:
    torch.manual_seed(9)
    node_bank = torch.randn(5, 4, 8)
    pool_bank = node_bank.mean(dim=1)
    mask_bank = torch.ones(5, 4, dtype=torch.bool)
    config = {
        "identity_dim": 6,
        "pair_dim": 10,
        "target_dim": 7,
        "dropout": 0.0,
    }
    model = HybridPairModel(node_bank, pool_bank, mask_bank, 6, 3, 3, config)
    model.eval()
    left_graph = torch.tensor([1, 2])
    right_graph = torch.tensor([3, 4])
    left_identity = torch.tensor([1, 2])
    right_identity = torch.tensor([3, 4])
    familiarity = torch.tensor([[1.0, 1.0, 0.7, 0.2], [1.0, 1.0, 0.5, 0.1]])

    forward = model(left_graph, right_graph, left_identity, right_identity, familiarity)
    reverse = model(right_graph, left_graph, right_identity, left_identity, familiarity)
    for forward_value, reverse_value in zip(forward, reverse, strict=True):
        assert torch.allclose(forward_value, reverse_value, atol=1e-6)


def test_simplex_weights_are_nonnegative_and_normalized() -> None:
    experts = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 4.0], [3.0, 5.0, 3.0]])
    target = np.array([1.2, 2.1, 3.2])
    weights = simplex_weights(experts, target)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
