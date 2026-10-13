from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .constants import (
    ALLOWED_SEVERITY_CLASSES,
    BINARY_PREFIX,
    EXPECTED_BINARY_TARGETS,
    EXPECTED_PRR_TARGETS,
    EXPECTED_SUBMISSION_COLUMNS,
    ID_COLUMN,
    PRR_PREFIX,
    SEVERITY_COLUMN,
)


@dataclass
class DataContractReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def inspect_data_contract(raw_dir: Path) -> DataContractReport:
    paths = {name: raw_dir / name for name in ("train.csv", "test.csv", "sample_submission.csv")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return DataContractReport(
            ok=False,
            errors=["Fichiers officiels manquants : " + ", ".join(missing)],
        )

    train = pd.read_csv(paths["train.csv"], low_memory=False)
    test = pd.read_csv(paths["test.csv"], low_memory=False)
    sample = pd.read_csv(paths["sample_submission.csv"], low_memory=False)

    errors: list[str] = []
    warnings: list[str] = []
    binary_columns = [column for column in train.columns if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train.columns if column.startswith(PRR_PREFIX)]
    sample_binary = [column for column in sample.columns if column.startswith(BINARY_PREFIX)]
    sample_prr = [column for column in sample.columns if column.startswith(PRR_PREFIX)]

    if len(binary_columns) != EXPECTED_BINARY_TARGETS:
        errors.append(f"Train contient {len(binary_columns)} cibles binaires au lieu de 50.")
    if len(prr_columns) != EXPECTED_PRR_TARGETS:
        errors.append(f"Train contient {len(prr_columns)} cibles PRR au lieu de 50.")
    if len(sample.columns) != EXPECTED_SUBMISSION_COLUMNS:
        errors.append(f"Le sample contient {len(sample.columns)} colonnes au lieu de 102.")
    if len(sample_binary) != EXPECTED_BINARY_TARGETS or len(sample_prr) != EXPECTED_PRR_TARGETS:
        errors.append("Les 50 colonnes binaires et/ou PRR manquent dans le sample.")
    if len(sample) != len(test):
        errors.append(f"Test ({len(test)} lignes) et sample ({len(sample)}) diffèrent.")
    if SEVERITY_COLUMN not in train.columns:
        errors.append("La cible Severity manque dans train.csv.")
        observed_severity: list[str] = []
    else:
        observed_severity = sorted(train[SEVERITY_COLUMN].dropna().astype(str).unique().tolist())
        unexpected = set(observed_severity) - ALLOWED_SEVERITY_CLASSES
        if unexpected:
            errors.append(f"Classes Severity inattendues : {sorted(unexpected)}")
        missing_documented = ALLOWED_SEVERITY_CLASSES - set(observed_severity)
        if missing_documented:
            warnings.append(
                "Classes autorisées par les règles mais absentes du train : "
                + ", ".join(sorted(missing_documented))
            )

    train_feature_columns = [
        column
        for column in train.columns
        if column != SEVERITY_COLUMN
        and not column.startswith(BINARY_PREFIX)
        and not column.startswith(PRR_PREFIX)
    ]
    if set(train_feature_columns) != set(test.columns):
        only_train = sorted(set(train_feature_columns) - set(test.columns))
        only_test = sorted(set(test.columns) - set(train_feature_columns))
        errors.append(
            f"Features train/test différentes. train-only={only_train}, test-only={only_test}"
        )

    if ID_COLUMN not in sample.columns:
        errors.append(f"La colonne d'identifiant {ID_COLUMN!r} manque dans le sample.")
    elif sample[ID_COLUMN].duplicated().any():
        errors.append("Le sample contient des Pair_ID dupliqués.")

    pair_id_match: bool | None = None
    if ID_COLUMN in sample.columns and "Pair_ID" in test.columns:
        pair_id_match = sample[ID_COLUMN].astype(str).equals(test["Pair_ID"].astype(str))
        if not pair_id_match:
            warnings.append("L'ordre/les valeurs de sample.Pair_ID et test.Pair_ID diffèrent.")

    target_missingness = {
        "severity": int(train[SEVERITY_COLUMN].isna().sum()) if SEVERITY_COLUMN in train else None,
        "binary": int(train[binary_columns].isna().sum().sum()) if binary_columns else None,
        "prr": int(train[prr_columns].isna().sum().sum()) if prr_columns else None,
    }
    facts: dict[str, object] = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_submission_shape": list(sample.shape),
        "feature_column_count": len(train_feature_columns),
        "binary_target_count": len(binary_columns),
        "prr_target_count": len(prr_columns),
        "observed_severity_classes": observed_severity,
        "target_missing_values": target_missingness,
        "duplicate_train_rows": int(train.duplicated().sum()),
        "duplicate_test_rows": int(test.duplicated().sum()),
        "sample_pair_id_matches_test_pair_id_in_order": pair_id_match,
    }
    return DataContractReport(ok=not errors, errors=errors, warnings=warnings, facts=facts)


def write_report(report: DataContractReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_json() + "\n", encoding="utf-8")
