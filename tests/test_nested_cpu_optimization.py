import hashlib
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from deep_pharma.metric import hardcore_score
from deep_pharma.nested_cpu_optimization import _read_predictions, compare_pipeline
from deep_pharma.submission_gate import write_json
from deep_pharma.validation import build_pair_grouped_folds, split_fingerprint


def test_classification_priors_do_not_read_held_labels():
    from deep_pharma.row_activity_benchmark import _activity_features
    from deep_pharma.severity_priors_benchmark import _severity_features

    fit = pd.DataFrame(
        {
            "SMILES_A": ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"],
            "SMILES_B": ["O"] * 6,
            "Severity": ["Major", "Minor"] * 3,
            "Target_Binary_a": [1, 0, 1, 0, 1, 0],
        }
    )
    held = fit.iloc[:2].copy()
    corrupted = held.copy()
    corrupted["Severity"] = "FORBIDDEN_OUTER_LABEL"
    corrupted["Target_Binary_a"] = 99999
    labels = fit.Severity.to_numpy()
    active = fit.Target_Binary_a.to_numpy()
    folds = np.array([0, 1, 2, 0, 1, 2])
    severity = _severity_features(fit, held, labels, folds, ["Major", "Minor"], 20)
    changed = _severity_features(fit, corrupted, labels, folds, ["Major", "Minor"], 20)
    activity = _activity_features(fit, held, active, folds, 20)
    activity_changed = _activity_features(fit, corrupted, active, folds, 20)
    for original, modified in zip(
        (*severity, *activity), (*changed, *activity_changed), strict=True
    ):
        np.testing.assert_array_equal(original, modified)


def test_prediction_reader_rejects_id_mismatch_and_missing_rows(tmp_path):
    train = pd.DataFrame({"Pair_ID": ["one", "two"]})
    path = tmp_path / "prediction.csv"
    pd.DataFrame({"Pair_ID": ["two", "one"], "Target_PRR_a": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="identity mismatch"):
        _read_predictions(path, train, ["Target_PRR_a"])
    pd.DataFrame({"Pair_ID": ["one"], "Target_PRR_a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="identity mismatch"):
        _read_predictions(path, train, ["Target_PRR_a"])


def test_prediction_reader_rejects_incomplete_oof(tmp_path):
    train = pd.DataFrame({"Pair_ID": ["one", "two"]})
    path = tmp_path / "prediction.csv"
    pd.DataFrame({"Pair_ID": ["one", "two"], "Target_PRR_a": [1, np.nan]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incomplete OOF"):
        _read_predictions(path, train, ["Target_PRR_a"])


def test_joint_scores_and_fold_scores_match_exact_competition_metric(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "Pair_ID": f"pair_{i}",
                "SMILES_A": "C" * (i + 1),
                "SMILES_B": "O",
                "Severity": ["Major", "Moderate", "Minor"][i % 3],
                "Target_Binary_a": i % 2,
                "Target_Binary_b": 1 - i % 2,
                "Target_PRR_a": (i % 2) * (i + 1),
                "Target_PRR_b": (1 - i % 2) * (i + 2),
            }
            for i in range(18)
        ]
    )
    manifest, _ = build_pair_grouped_folds(frame, n_splits=3, seed=2026)
    train_path, manifest_path = tmp_path / "train.csv", tmp_path / "folds.csv"
    frame.to_csv(train_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    output, prr_dir, cls_dir = tmp_path / "output", tmp_path / "prr", tmp_path / "cls"
    for directory in (output, prr_dir, cls_dir):
        directory.mkdir()
    metadata = {
        "validation_fingerprint": split_fingerprint(manifest),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "complete_oof": True,
        "test_loaded": False,
    }
    for directory in (prr_dir, cls_dir):
        write_json(metadata, directory / "summary.json")
    binary = ["Target_Binary_a", "Target_Binary_b"]
    prr = ["Target_PRR_a", "Target_PRR_b"]
    expected = {}
    for variant, error in [("baseline", 2), ("optimized", 1)]:
        classification = frame[["Pair_ID", "Severity", *binary]].copy()
        regression = frame[["Pair_ID", *prr]].copy()
        regression[prr] += error
        # Predictions in zero-target cells must not affect the masked RMSE.
        for target in prr:
            regression.loc[frame[target] == 0, target] = 999999
        classification.to_csv(cls_dir / f"oof_{variant}.csv", index=False)
        regression.to_csv(prr_dir / f"oof_{variant}.csv", index=False)
        expected[variant] = hardcore_score(
            frame.Severity,
            classification.Severity,
            frame[binary],
            classification[binary],
            frame[prr],
            regression[prr],
        )
    summary = compare_pipeline(
        {"train": str(train_path), "manifest": str(manifest_path), "run_dir": str(output)},
        prr_dir,
        cls_dir,
    )
    for row in summary["comparison"]:
        for key, value in asdict(expected[row["variant"]]).items():
            assert row[key] == pytest.approx(value)
    assert summary["comparison"][0]["prr_score"] == pytest.approx(1 / 3)
    assert summary["comparison"][1]["prr_score"] == pytest.approx(1 / 2)
    assert summary["gains"]["total"] == pytest.approx(0.3 * (1 / 2 - 1 / 3))
    assert len(summary["folds"]) == 6
    assert summary["test_used"] is False
    assert summary["automatic_promotion"] is False
    for row in summary["folds"]:
        assert row["prr_score"] == pytest.approx(1 / 3 if row["variant"] == "baseline" else 1 / 2)
    assert (output / "fold_scores.png").exists()
    assert (output / "report.md").exists()


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("validation_fingerprint", "wrong", "protocol mismatch"),
        ("train_sha256", "wrong", "protocol mismatch"),
        ("complete_oof", False, "Incomplete stage"),
        ("test_loaded", True, "Test access"),
    ],
)
def test_pipeline_rejects_invalid_stage_contract(tmp_path, key, value, message):
    frame = pd.DataFrame(
        {
            "Pair_ID": [f"pair_{i}" for i in range(18)],
            "SMILES_A": ["C" * (i + 1) for i in range(18)],
            "SMILES_B": ["O"] * 18,
            "Severity": ["Major", "Moderate", "Minor"] * 6,
            "Target_Binary_a": [0, 1] * 9,
            "Target_PRR_a": [0, 2] * 9,
        }
    )
    manifest, _ = build_pair_grouped_folds(frame, n_splits=3, seed=2026)
    train_path, manifest_path = tmp_path / "train.csv", tmp_path / "folds.csv"
    frame.to_csv(train_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "validation_fingerprint": split_fingerprint(manifest),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "complete_oof": True,
    }
    metadata[key] = value
    stage = tmp_path / "stage"
    stage.mkdir()
    write_json(metadata, stage / "summary.json")
    with pytest.raises(ValueError, match=message):
        compare_pipeline(
            {"train": str(train_path), "manifest": str(manifest_path), "run_dir": str(tmp_path)},
            stage,
            stage,
        )
