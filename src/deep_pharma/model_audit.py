from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN


def _safe_canonical_smiles(value: object) -> str | None:
    if pd.isna(value):
        return None
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(str(value), parameters)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_series(series: pd.Series, mapping: dict[str, str | None]) -> pd.Series:
    def resolve(value: object) -> str:
        raw = str(value)
        canonical = mapping.get(raw)
        return canonical if canonical is not None else f"<INVALID:{raw}>"

    return series.map(resolve)


def _target_summaries(train: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    binary_summary = []
    prr_summary = []
    agreement_rows = []
    for binary_column, prr_column in zip(binary_columns, prr_columns, strict=True):
        binary = pd.to_numeric(train[binary_column], errors="coerce")
        prr = pd.to_numeric(train[prr_column], errors="coerce")
        positives = int((binary == 1).sum())
        binary_summary.append(
            {
                "target": binary_column,
                "positives": positives,
                "negatives": int((binary == 0).sum()),
                "missing": int(binary.isna().sum()),
                "positive_rate": _json_value(binary.mean()),
                "constant": int(binary.nunique(dropna=True)) <= 1,
                "rare_below_1pct": float(binary.mean()) < 0.01,
            }
        )
        finite = np.isfinite(prr.to_numpy(dtype=float, na_value=np.nan))
        finite_values = prr[finite]
        quantiles = finite_values.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        prr_summary.append(
            {
                "target": prr_column,
                "missing": int(prr.isna().sum()),
                "infinite": int((~finite & prr.notna().to_numpy()).sum()),
                "negative": int((finite_values < 0).sum()),
                "zero": int((finite_values == 0).sum()),
                "minimum": _json_value(finite_values.min()),
                "mean": _json_value(finite_values.mean()),
                "maximum": _json_value(finite_values.max()),
                "skew": _json_value(finite_values.skew()),
                "quantiles": {str(key): _json_value(value) for key, value in quantiles.items()},
            }
        )
        valid = binary.notna() & prr.notna() & finite
        derived = (prr[valid] > 0).astype(int)
        disagreements = int((binary[valid].astype(int) != derived).sum())
        agreement_rows.append(
            {
                "event": binary_column.removeprefix(BINARY_PREFIX),
                "threshold": 0.0,
                "observations": int(valid.sum()),
                "disagreements": disagreements,
                "agreement_rate": 1.0 - disagreements / int(valid.sum()),
            }
        )
    relationship = {
        "rule": "Binary_event == 1(PRR_event > 0)",
        "all_events_exact": all(row["disagreements"] == 0 for row in agreement_rows),
        "total_disagreements": sum(row["disagreements"] for row in agreement_rows),
        "events": agreement_rows,
    }
    return binary_summary, prr_summary, relationship


def build_model_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    expected_pairs = {column.removeprefix(BINARY_PREFIX) for column in binary_columns} == {
        column.removeprefix(PRR_PREFIX) for column in prr_columns
    }
    if not expected_pairs:
        raise ValueError("Les événements Binary et PRR ne correspondent pas.")

    all_smiles = pd.concat(
        [train["SMILES_A"], train["SMILES_B"], test["SMILES_A"], test["SMILES_B"]],
        ignore_index=True,
    )
    canonical_by_raw = {
        str(smiles): _safe_canonical_smiles(smiles)
        for smiles in all_smiles.dropna().astype(str).unique()
    }
    train_left = _canonical_series(train["SMILES_A"], canonical_by_raw)
    train_right = _canonical_series(train["SMILES_B"], canonical_by_raw)
    test_left = _canonical_series(test["SMILES_A"], canonical_by_raw)
    test_right = _canonical_series(test["SMILES_B"], canonical_by_raw)
    train_molecules = set(train_left.dropna()) | set(train_right.dropna())
    test_molecules = set(test_left.dropna()) | set(test_right.dropna())

    warm_a = test_left.isin(train_molecules)
    warm_b = test_right.isin(train_molecules)
    test_regime = np.select(
        [warm_a & warm_b, warm_a ^ warm_b], ["warm/warm", "warm/cold"], default="cold/cold"
    )

    unordered_pairs = pd.Series(
        [
            "\x1f".join(sorted((left, right)))
            for left, right in zip(train_left, train_right, strict=True)
        ],
        index=train.index,
    )
    directional_pairs = pd.Series(
        list(zip(train_left, train_right, strict=True)), index=train.index
    )
    group_sizes = unordered_pairs.value_counts()
    repeated_groups = set(group_sizes[group_sizes > 1].index)
    inverted_group_count = 0
    conflicting_group_count = 0
    target_columns = [SEVERITY_COLUMN, *binary_columns, *prr_columns]
    for pair_key in repeated_groups:
        indices = unordered_pairs.index[unordered_pairs == pair_key]
        directions = set(directional_pairs.loc[indices])
        if len(directions) > 1:
            inverted_group_count += 1
        if train.loc[indices, target_columns].drop_duplicates().shape[0] > 1:
            conflicting_group_count += 1

    text_columns = [
        column for column in test.columns if column not in {ID_COLUMN, "SMILES_A", "SMILES_B"}
    ]
    text_lengths = []
    for column in text_columns:
        values = train[column].dropna().astype(str).str.len()
        text_lengths.append(
            {
                "column": column,
                "missing": int(train[column].isna().sum()),
                "missing_rate": float(train[column].isna().mean()),
                "length_min": int(values.min()) if len(values) else None,
                "length_median": float(values.median()) if len(values) else None,
                "length_mean": float(values.mean()) if len(values) else None,
                "length_max": int(values.max()) if len(values) else None,
            }
        )

    binary_summary, prr_summary, relationship = _target_summaries(train)
    invalid_raw = [raw for raw, canonical in canonical_by_raw.items() if canonical is None]
    audit: dict[str, object] = {
        "schema_version": 1,
        "grain": "one drug pair per row, keyed by Pair_ID",
        "data": {
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "train_columns": list(train.columns),
            "test_columns": list(test.columns),
            "train_only_columns": [column for column in train if column not in test],
            "test_only_columns": [column for column in test if column not in train],
            "train_dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
            "test_dtypes": {column: str(dtype) for column, dtype in test.dtypes.items()},
            "train_missing": {column: int(value) for column, value in train.isna().sum().items()},
            "test_missing": {column: int(value) for column, value in test.isna().sum().items()},
        },
        "identifiers": {
            "train_duplicate_pair_ids": int(train[ID_COLUMN].duplicated().sum()),
            "test_duplicate_pair_ids": int(test[ID_COLUMN].duplicated().sum()),
        },
        "chemistry": {
            "unique_raw_smiles": len(canonical_by_raw),
            "invalid_raw_smiles": len(invalid_raw),
            "invalid_examples": invalid_raw[:10],
            "train_unique_molecules": len(train_molecules),
            "test_unique_molecules": len(test_molecules),
            "train_test_molecule_overlap": len(train_molecules & test_molecules),
            "test_regime_counts": pd.Series(test_regime).value_counts().to_dict(),
            "repeated_unordered_pair_groups": len(repeated_groups),
            "rows_in_repeated_pair_groups": int(unordered_pairs.isin(repeated_groups).sum()),
            "inverted_pair_groups": inverted_group_count,
            "conflicting_target_pair_groups": conflicting_group_count,
        },
        "severity": {
            "missing": int(train[SEVERITY_COLUMN].isna().sum()),
            "counts": train[SEVERITY_COLUMN].value_counts(dropna=False).to_dict(),
            "rates": train[SEVERITY_COLUMN].value_counts(normalize=True, dropna=False).to_dict(),
            "ordinal_assumption": False,
        },
        "binary_targets": binary_summary,
        "prr_targets": prr_summary,
        "binary_prr_relationship": relationship,
        "text_columns": text_lengths,
    }
    return audit


def audit_markdown(audit: dict[str, object]) -> str:
    data = audit["data"]
    chemistry = audit["chemistry"]
    severity = audit["severity"]
    relationship = audit["binary_prr_relationship"]
    regimes = chemistry["test_regime_counts"]
    return f"""# Audit de modélisation Deep-Pharma

## Contrat réel

- Train : {data["train_shape"][0]:,} lignes × {data["train_shape"][1]} colonnes.
- Test : {data["test_shape"][0]:,} lignes × {data["test_shape"][1]} colonnes.
- Clé : `Pair_ID`.
- Doublons train : {audit["identifiers"]["train_duplicate_pair_ids"]}.
- Doublons test : {audit["identifiers"]["test_duplicate_pair_ids"]}.

## Chimie et grain

- Molécules canoniques train : {chemistry["train_unique_molecules"]}.
- Molécules canoniques test : {chemistry["test_unique_molecules"]}.
- Recouvrement moléculaire train-test : {chemistry["train_test_molecule_overlap"]}.
- SMILES invalides : {chemistry["invalid_raw_smiles"]}.
- Groupes de paires répétés : {chemistry["repeated_unordered_pair_groups"]}.
- Lignes dans ces groupes : {chemistry["rows_in_repeated_pair_groups"]}.
- Groupes inversés A/B : {chemistry["inverted_pair_groups"]}.
- Groupes répétés avec cibles conflictuelles : {chemistry["conflicting_target_pair_groups"]}.

## Régimes du test

- warm/warm : {regimes.get("warm/warm", 0)}
- warm/cold : {regimes.get("warm/cold", 0)}
- cold/cold : {regimes.get("cold/cold", 0)}

## Cibles

- Severity : {severity["counts"]}.
- Relation Binary–PRR exacte au seuil zéro : {relationship["all_events_exact"]}.
- Contradictions Binary–PRR : {relationship["total_disagreements"]}.

## Décisions

- Conserver la validation groupée par paire canonique et le stress-test cold-drug.
- Construire des features invariantes à A/B et documenter les doublons conflictuels.
- Apprendre Binary et PRR séparément dans la première baseline.
- Comparer une contrainte de cohérence uniquement hors échantillon.
- Ajuster vocabulaire, SVD, imputation et seuil uniquement dans le train du fold.
"""


def write_model_audit(audit: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "data_audit.md").write_text(audit_markdown(audit), encoding="utf-8")
