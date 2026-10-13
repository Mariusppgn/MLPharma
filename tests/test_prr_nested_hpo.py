from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from deep_pharma.prr_nested_hpo import (
    candidate_config,
    fit_features,
    refit_replaying_inner_schedule,
    replay_schedule,
)
from deep_pharma.prr_pytorch import train_prr_method
from deep_pharma.text_features import TEXT_FIELDS


def test_replay_restores_internal_best_and_replays_scheduler():
    config = {
        "training": {"max_epochs": 5, "learning_rate": 0.01},
        "loss_schedules": {
            "huber": {
                "mse_finetune_epochs": 2,
                "restore_best_before_finetune": True,
                "finetune_learning_rate": 0.001,
            }
        },
    }
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4, 5],
            "learning_rate": [0.01, 0.005, 0.005, 0.0005, 0.0005],
            "validation_sprr": [0.2, 0.3, 0.25, 0.31, 0.32],
        }
    )
    schedule = replay_schedule(history, config, 5)
    assert schedule[3]["restore_epoch"] == 2
    assert [s["learning_rate"] for s in schedule] == [0.01, 0.01, 0.005, 0.001, 0.0005]
    assert [s["fine_tune"] for s in schedule] == [False, False, False, True, True]


def test_candidate_config_is_isolated():
    base = {
        "training": {"device": "auto"},
        "architecture": {"dropout": 0.1},
        "loss_schedules": {"huber": {}},
    }
    cfg = candidate_config(base, {"architecture": {"dropout": 0.3}}, 9)
    assert cfg["training"]["device"] == "cpu"
    assert base["architecture"]["dropout"] == 0.1


def test_replay_reproduces_inner_checkpoint_on_identical_fit_rows(tmp_path):
    torch.set_num_threads(1)
    config = yaml.safe_load(Path("configs/prr_oof.yaml").read_text(encoding="utf-8"))
    config["architecture"].update(hidden_dims=[12, 6], dropout=0.1)
    config["training"].update(max_epochs=6, min_epochs=6, batch_size=16, num_workers=0)
    config["loss_schedules"]["huber"]["mse_finetune_epochs"] = 2
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(64, 8)).astype(np.float32)
    y = np.abs(rng.normal(size=(64, 3))).astype(np.float32) + 1
    scales = np.ones(3, np.float32)
    result = train_prr_method(
        "huber",
        x[:48],
        y[:48],
        x[48:],
        y[48:],
        scales,
        config,
        tmp_path / "inner",
        torch.device("cpu"),
    )
    schedule = replay_schedule(pd.read_csv(result["history"]), config, result["best_epoch"])
    refit = refit_replaying_inner_schedule(
        x[:48], y[:48], scales, config, schedule, tmp_path / "refit"
    )
    expected = torch.load(result["checkpoint"], weights_only=True)["model_state_dict"]
    actual = torch.load(refit["checkpoint"], weights_only=True)["model_state_dict"]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_fold_feature_cache_is_cleared_before_pickle(tmp_path):
    frame = pd.DataFrame(
        {"SMILES_A": ["C" * (i % 8 + 1) for i in range(24)], "SMILES_B": ["O"] * 24}
    )
    for suffix in ("A", "B"):
        for field in TEXT_FIELDS:
            frame[f"{field}_{suffix}"] = [f"text effect compound {i % 8}" for i in range(24)]
    config = yaml.safe_load(Path("configs/prr_oof.yaml").read_text(encoding="utf-8"))
    config["features"].update(
        text_components=2,
        morgan_components=2,
        word_max_features=100,
        char_max_features=100,
        min_df=1,
    )
    target = np.ones((18, 3), dtype=np.float32)
    fit_x, held_x, scales, bundle = fit_features(
        frame.iloc[:18], frame.iloc[18:], target, np.arange(18) % 3, config, True
    )
    assert len(fit_x) == 18 and len(held_x) == 6 and len(scales) == 3
    assert "_vectorize_documents" not in bundle["builder"].text_encoder.__dict__
    bundle.pop("base_arrays")
    joblib.dump(bundle, tmp_path / "preprocessing.joblib")
    reloaded = joblib.load(tmp_path / "preprocessing.joblib")
    np.testing.assert_array_equal(reloaded["target_scales"], scales)
