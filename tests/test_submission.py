from pathlib import Path

import pandas as pd

from deep_pharma.submission import validate_submission


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Pair_ID": ["pair-1", "pair-2"],
            "Severity": ["Major", "Moderate"],
            "Target_Binary_A": [0, 1],
            "Target_PRR_A": [0.0, 1.5],
        }
    )


def test_valid_submission(tmp_path: Path) -> None:
    sample = _sample_frame()
    candidate_path = tmp_path / "candidate.csv"
    sample_path = tmp_path / "sample.csv"
    sample.to_csv(candidate_path, index=False)
    sample.to_csv(sample_path, index=False)

    assert validate_submission(candidate_path, sample_path) == []


def test_rejects_reordered_ids_and_invalid_values(tmp_path: Path) -> None:
    sample = _sample_frame()
    candidate = sample.iloc[::-1].reset_index(drop=True)
    candidate.loc[0, "Target_Binary_A"] = 2
    candidate.loc[1, "Target_PRR_A"] = -1.0
    candidate_path = tmp_path / "candidate.csv"
    sample_path = tmp_path / "sample.csv"
    candidate.to_csv(candidate_path, index=False)
    sample.to_csv(sample_path, index=False)

    errors = validate_submission(candidate_path, sample_path)

    assert any("Pair_ID" in error for error in errors)
    assert any("0 ou 1" in error for error in errors)
    assert any("positives ou nulles" in error for error in errors)
