from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from deep_pharma.global_gnn import GlobalGNN, _indices, build_graph_bank, text_pair_features


def test_pair_invariance_and_unknown_identity() -> None:
    frame = pd.DataFrame(
        {
            "SMILES_A": ["CCO", "O", "N"],
            "SMILES_B": ["O", "CCO", "CCO"],
            "Mechanism_A": ["alpha", "beta", "gamma"],
            "Mechanism_B": ["beta", "alpha", "alpha"],
        }
    )
    bank = build_graph_bank(["CCO", "O", "N"], 32, torch.device("cpu"))
    identity = {"CCO": 1, "O": 2}
    indices = tuple(torch.from_numpy(array) for array in _indices(frame, bank.lookup, identity))
    text = text_pair_features(frame, 16)
    np.testing.assert_array_equal(text[0], text[1])
    assert indices[2][2] == 0
    model = GlobalGNN(
        2,
        3,
        2,
        text.shape[1],
        {
            "hidden_dim": 16,
            "gnn_layers": 2,
            "dropout": 0.0,
            "identity_dropout": 0.0,
            "identity_limit": 0.5,
        },
    ).eval()
    with torch.inference_mode():
        severity, binary, prr, gate = model(bank, *indices, torch.from_numpy(text))
    torch.testing.assert_close(severity[0], severity[1])
    torch.testing.assert_close(binary[0], binary[1])
    torch.testing.assert_close(prr[0], prr[1])
    assert bool((prr > 0).all())
    assert bool((gate <= 0.5).all())
