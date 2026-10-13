from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ALLOWED_SEVERITY_CLASSES,
    BINARY_PREFIX,
    ID_COLUMN,
    PRR_PREFIX,
    SEVERITY_COLUMN,
)


def validate_submission(candidate_path: Path, sample_path: Path) -> list[str]:
    """Return validation errors; an empty list means the file matches the contract."""

    candidate = pd.read_csv(candidate_path, low_memory=False)
    sample = pd.read_csv(sample_path, low_memory=False)
    errors: list[str] = []

    if list(candidate.columns) != list(sample.columns):
        errors.append("Les colonnes ou leur ordre ne correspondent pas exactement au sample.")
        return errors
    if len(candidate) != len(sample):
        errors.append(f"Nombre de lignes incorrect : {len(candidate)} au lieu de {len(sample)}.")
        return errors
    if not candidate[ID_COLUMN].astype(str).equals(sample[ID_COLUMN].astype(str)):
        errors.append("Les Pair_ID ou leur ordre ne correspondent pas exactement au sample.")

    invalid_severity = sorted(
        set(candidate[SEVERITY_COLUMN].dropna().astype(str)) - ALLOWED_SEVERITY_CLASSES
    )
    if candidate[SEVERITY_COLUMN].isna().any() or invalid_severity:
        errors.append(f"Severity contient des valeurs manquantes/inconnues : {invalid_severity}")

    binary_columns = [column for column in candidate if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in candidate if column.startswith(PRR_PREFIX)]
    binary_numeric = candidate[binary_columns].apply(pd.to_numeric, errors="coerce")
    prr_numeric = candidate[prr_columns].apply(pd.to_numeric, errors="coerce")

    if binary_numeric.isna().any().any() or not binary_numeric.isin([0, 1]).all().all():
        errors.append("Les prédictions binaires doivent toutes valoir exactement 0 ou 1.")
    prr_array = prr_numeric.to_numpy(dtype=float)
    if not np.isfinite(prr_array).all():
        errors.append("Les prédictions PRR doivent toutes être numériques et finies.")
    elif (prr_array < 0).any():
        errors.append("Les prédictions PRR doivent être positives ou nulles.")

    return errors
