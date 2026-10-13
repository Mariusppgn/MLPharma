from pathlib import Path

import pandas as pd

from deep_pharma.oof import run_smoke_oof
from deep_pharma.validation import (
    audit_pair_grouped_folds,
    build_drug_disjoint_holdout,
    build_pair_grouped_folds,
    canonical_pair_key,
    canonicalize_smiles,
)


def _training_frame() -> pd.DataFrame:
    pairs = [
        ("C", "CC", "Major", 1, 2.0),
        ("CC", "C", "Major", 1, 3.0),
        ("C", "CCC", "Moderate", 0, 0.0),
        ("CCC", "C", "Moderate", 0, 0.0),
        ("O", "CO", "Major", 1, 4.0),
        ("CO", "O", "Major", 1, 5.0),
        ("O", "N", "Moderate", 0, 0.0),
        ("N", "O", "Moderate", 0, 0.0),
        ("CN", "Cl", "Minor", 1, 1.0),
        ("Cl", "CN", "Minor", 1, 1.5),
        ("CN", "CCl", "Moderate", 0, 0.0),
        ("CCl", "CN", "Moderate", 0, 0.0),
    ]
    return pd.DataFrame(
        [
            {
                "Pair_ID": f"Train_{index:03d}",
                "SMILES_A": left,
                "SMILES_B": right,
                "Severity": severity,
                "Target_Binary_A": binary,
                "Target_PRR_A": prr,
            }
            for index, (left, right, severity, binary, prr) in enumerate(pairs)
        ]
    )


def test_pair_grouped_folds_have_complete_coverage_and_no_pair_overlap() -> None:
    frame = _training_frame()

    manifest, audit = build_pair_grouped_folds(frame, n_splits=2, seed=7)

    assert audit["passed"] is True
    assert audit["coverage_complete"] is True
    assert set(manifest["fold"]) == {0, 1}
    assert all(fold["pair_group_overlap"] == 0 for fold in audit["folds"])
    assert audit_pair_grouped_folds(frame, manifest)["passed"] is True


def test_canonicalization_groups_equivalent_and_reversed_smiles() -> None:
    aromatic = "c1ccncc1"
    kekulized = "C1=CC=NC=C1"

    assert canonicalize_smiles(aromatic) == canonicalize_smiles(kekulized)
    assert canonical_pair_key(aromatic, "CCO") == canonical_pair_key("OCC", kekulized)


def test_invalid_smiles_blocks_validation() -> None:
    try:
        canonicalize_smiles("this-is-not-smiles")
    except ValueError as error:
        assert "SMILES invalide" in str(error)
    else:
        raise AssertionError("Un SMILES invalide doit bloquer la validation.")


def test_drug_disjoint_holdout_purges_boundary_pairs() -> None:
    compounds = ["C" * length for length in range(1, 31)]
    frame = pd.DataFrame(
        [
            {"Pair_ID": f"Pair_{left}_{right}", "SMILES_A": left, "SMILES_B": right}
            for index, left in enumerate(compounds)
            for right in compounds[index + 1 :]
        ]
    )

    manifest, audit = build_drug_disjoint_holdout(frame, seed=2026, holdout_fraction=0.30)

    assert audit["passed"] is True
    assert audit["molecule_overlap"] == 0
    assert audit["train_rows"] > 0
    assert audit["validation_rows"] > 0
    assert audit["purged_rows"] > 0
    assert set(manifest["role"]) == {"train", "validation", "purged"}


def test_smoke_oof_is_complete_and_fold_local(tmp_path: Path) -> None:
    frame = _training_frame()
    manifest, audit = build_pair_grouped_folds(frame, n_splits=2, seed=7)

    metrics = run_smoke_oof(
        frame,
        manifest,
        audit,
        output_dir=tmp_path / "run",
        binary_prevalence_threshold=0.20,
    )

    assert metrics["leakage_audit"]["passed"] is True
    assert metrics["leakage_audit"]["pair_group_overlap_max"] == 0
    assert metrics["oof"]["coverage_complete"] is True
    assert metrics["oof"]["predicted_rows"] == len(frame)
    assert metrics["data_scope"]["test_used_for_fit"] is False
