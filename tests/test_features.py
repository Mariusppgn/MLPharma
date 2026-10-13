import numpy as np
import pandas as pd

from deep_pharma.chemistry_features import MORGAN_SIZE, build_chemical_pair_features
from deep_pharma.multimodal_features import MultimodalFeatureBuilder
from deep_pharma.pk_features import extract_cyp_features, parse_half_life, parse_protein_binding
from deep_pharma.text_features import TEXT_FIELDS


def _feature_frame() -> pd.DataFrame:
    rows = []
    pairs = [("CCO", "c1ccccc1"), ("CCN", "O"), ("CCC", "N"), ("CCCl", "CO")]
    for index, (left, right) in enumerate(pairs):
        row = {"Pair_ID": f"Pair_{index}", "SMILES_A": left, "SMILES_B": right}
        for suffix in ("A", "B"):
            for field in TEXT_FIELDS:
                row[f"{field}_{suffix}"] = f"{field} text drug {suffix} {index}"
            row[f"Half_Life_{suffix}"] = "8-12 hours"
            row[f"Protein_Binding_{suffix}"] = "95% protein bound"
            row[f"CYP450_Enzymes_{suffix}"] = "substrate of CYP3A4"
            row[f"Metabolism_{suffix}"] = "metabolized by CYP3A4"
        rows.append(row)
    return pd.DataFrame(rows)


def _swap_drugs(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    for column in list(frame.columns):
        if column.endswith("_A"):
            base = column[:-2]
            swapped[f"{base}_A"] = frame[f"{base}_B"]
            swapped[f"{base}_B"] = frame[f"{base}_A"]
    return swapped


def test_chemical_pair_features_are_symmetric() -> None:
    frame = _feature_frame()
    swapped = _swap_drugs(frame)

    morgan, descriptors = build_chemical_pair_features(frame)
    swapped_morgan, swapped_descriptors = build_chemical_pair_features(swapped)

    assert morgan.shape == (len(frame), 3 * MORGAN_SIZE)
    assert np.allclose(morgan.toarray(), swapped_morgan.toarray())
    assert np.allclose(descriptors, swapped_descriptors, equal_nan=True)


def test_pk_parsers_handle_units_percentages_and_negation() -> None:
    assert parse_half_life("8-12 hours")["mean_hours"] == 10.0
    assert parse_half_life("2 days")["mean_hours"] == 48.0
    assert parse_half_life("30 minutes")["mean_hours"] == 0.5
    assert parse_protein_binding("approximately 95%")["mean_pct"] == 95.0
    assert extract_cyp_features("substrate of CYP3A4")["cyp3a4_substrate"] == 1.0
    assert extract_cyp_features("not an inhibitor of CYP3A4")["cyp3a4_inhibitor"] == 0.0


def test_complete_fold_preprocessor_is_finite_and_symmetric() -> None:
    frame = _feature_frame()
    builder = MultimodalFeatureBuilder(
        text_components=2,
        morgan_components=2,
        word_max_features=100,
        char_max_features=100,
        min_df=1,
        random_state=7,
    )

    features = builder.fit_transform(frame)
    swapped_features = builder.transform(_swap_drugs(frame))

    assert np.isfinite(features).all()
    assert np.allclose(features, swapped_features, atol=1e-5)
