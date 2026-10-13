import pandas as pd

from deep_pharma.model_audit import build_model_audit
from deep_pharma.text_features import TEXT_FIELDS


def _row(pair_id: str, left: str, right: str, binary: int, prr: float) -> dict:
    row = {
        "Pair_ID": pair_id,
        "SMILES_A": left,
        "SMILES_B": right,
        "Severity": "Major" if binary else "Moderate",
        "Target_Binary_Event": binary,
        "Target_PRR_Event": prr,
    }
    for suffix in ("A", "B"):
        for field in TEXT_FIELDS:
            row[f"{field}_{suffix}"] = "text"
    return row


def test_model_audit_measures_binary_prr_and_inverted_pairs() -> None:
    train = pd.DataFrame(
        [
            _row("Train_1", "CCO", "CCN", 1, 2.0),
            _row("Train_2", "NCC", "OCC", 0, 0.0),
        ]
    )
    test = train.drop(columns=["Severity", "Target_Binary_Event", "Target_PRR_Event"]).copy()
    test["Pair_ID"] = ["Test_1", "Test_2"]

    audit = build_model_audit(train, test)

    assert audit["binary_prr_relationship"]["all_events_exact"] is True
    assert audit["chemistry"]["repeated_unordered_pair_groups"] == 1
    assert audit["chemistry"]["inverted_pair_groups"] == 1
    assert audit["chemistry"]["conflicting_target_pair_groups"] == 1
