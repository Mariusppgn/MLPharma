from __future__ import annotations

import re
from functools import cache

import numpy as np
import pandas as pd

CYP_ENZYMES = (
    "CYP1A2",
    "CYP2A6",
    "CYP2B6",
    "CYP2C8",
    "CYP2C9",
    "CYP2C19",
    "CYP2D6",
    "CYP2E1",
    "CYP3A4",
    "CYP3A5",
)

_HALF_LIFE_PATTERN = re.compile(
    r"(?P<low>\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)?\s*"
    r"(?P<high>\d+(?:[.,]\d+)?)?\s*"
    r"(?P<unit>minutes?|mins?|hours?|hrs?|hr|h|days?|d|weeks?|w)\b",
    flags=re.IGNORECASE,
)
_PERCENT_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


def parse_half_life(text: object) -> dict[str, float]:
    if pd.isna(text) or not str(text).strip():
        return {
            "min_hours": np.nan,
            "max_hours": np.nan,
            "mean_hours": np.nan,
            "count": 0.0,
            "failed": 1.0,
        }
    values: list[float] = []
    for match in _HALF_LIFE_PATTERN.finditer(str(text)):
        low = float(match.group("low").replace(",", "."))
        high = match.group("high")
        numbers = [low] if high is None else [low, float(high.replace(",", "."))]
        unit = match.group("unit").lower()
        factor = 1.0
        if unit.startswith(("min",)):
            factor = 1.0 / 60.0
        elif unit.startswith("d"):
            factor = 24.0
        elif unit.startswith("w"):
            factor = 168.0
        values.extend(number * factor for number in numbers)
    if not values:
        return {
            "min_hours": np.nan,
            "max_hours": np.nan,
            "mean_hours": np.nan,
            "count": 0.0,
            "failed": 1.0,
        }
    return {
        "min_hours": min(values),
        "max_hours": max(values),
        "mean_hours": float(np.mean(values)),
        "count": float(len(values)),
        "failed": 0.0,
    }


def parse_protein_binding(text: object) -> dict[str, float]:
    if pd.isna(text) or not str(text).strip():
        return {
            "min_pct": np.nan,
            "max_pct": np.nan,
            "mean_pct": np.nan,
            "count": 0.0,
            "over_90": 0.0,
            "failed": 1.0,
        }
    values = [float(value.replace(",", ".")) for value in _PERCENT_PATTERN.findall(str(text))]
    if not values:
        return {
            "min_pct": np.nan,
            "max_pct": np.nan,
            "mean_pct": np.nan,
            "count": 0.0,
            "over_90": 0.0,
            "failed": 1.0,
        }
    return {
        "min_pct": min(values),
        "max_pct": max(values),
        "mean_pct": float(np.mean(values)),
        "count": float(len(values)),
        "over_90": float(max(values) > 90.0),
        "failed": 0.0,
    }


def _role_present(text: str, enzyme: str, role: str) -> bool:
    escaped = re.escape(enzyme.lower())
    role_patterns = {
        "inhibitor": rf"(?:inhibit\w*\s+(?:of\s+)?{escaped}|{escaped}.{{0,30}}inhibit\w*)",
        "inducer": rf"(?:induc\w*\s+(?:of\s+)?{escaped}|{escaped}.{{0,30}}induc\w*)",
        "substrate": (
            rf"(?:substrate\s+(?:of\s+)?{escaped}|"
            rf"metaboli[sz]\w*\s+by\s+{escaped}|{escaped}.{{0,30}}substrate)"
        ),
    }
    negative = rf"(?:not|neither|no)\s+(?:an?\s+)?{role}\w*\s+(?:of\s+)?{escaped}"
    return bool(re.search(role_patterns[role], text)) and not bool(re.search(negative, text))


def extract_cyp_features(text: object) -> dict[str, float]:
    normalized = "" if pd.isna(text) else re.sub(r"\s+", " ", str(text).lower())
    features: dict[str, float] = {}
    for enzyme in CYP_ENZYMES:
        key = enzyme.lower()
        features[f"{key}_mentioned"] = float(key in normalized)
        features[f"{key}_substrate"] = float(_role_present(normalized, enzyme, "substrate"))
        features[f"{key}_inhibitor"] = float(_role_present(normalized, enzyme, "inhibitor"))
        features[f"{key}_inducer"] = float(_role_present(normalized, enzyme, "inducer"))
    return features


@cache
def _cached_side_pk_features(
    half_life: str,
    protein_binding: str,
    cyp: str,
    metabolism: str,
    absorption: str,
    elimination: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(
        {f"half_life_{key}": value for key, value in parse_half_life(half_life).items()}
    )
    features.update(
        {
            f"protein_binding_{key}": value
            for key, value in parse_protein_binding(protein_binding).items()
        }
    )
    features.update(extract_cyp_features(f"{cyp} {metabolism}"))
    route_text = f"{absorption} {elimination}".lower()
    route_keywords = (
        "oral",
        "intravenous",
        "intramuscular",
        "subcutaneous",
        "renal",
        "hepatic",
        "fecal",
        "faecal",
        "urinary",
        "biliary",
        "gastrointestinal",
        "rapid absorption",
        "slow absorption",
        "delayed absorption",
    )
    for keyword in route_keywords:
        features[f"route_{keyword.replace(' ', '_')}"] = float(keyword in route_text)
    bioavailability = _PERCENT_PATTERN.findall(absorption)
    features["bioavailability_pct"] = (
        float(bioavailability[0].replace(",", ".")) if bioavailability else np.nan
    )
    return features


def _side_pk_features(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        values = [
            "" if pd.isna(row[f"{field}_{suffix}"]) else str(row[f"{field}_{suffix}"])
            for field in (
                "Half_Life",
                "Protein_Binding",
                "CYP450_Enzymes",
                "Metabolism",
                "Absorption",
                "Elimination_Route",
            )
        ]
        features = _cached_side_pk_features(
            values[0], values[1], values[2], values[3], values[4], values[5]
        )
        rows.append(features)
    return pd.DataFrame(rows, index=frame.index, dtype=np.float32)


def build_pk_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    side_a = _side_pk_features(frame, "A")
    side_b = _side_pk_features(frame, "B")
    matrix_a = side_a.to_numpy(dtype=np.float32)
    matrix_b = side_b.to_numpy(dtype=np.float32)
    pair_matrix = np.hstack([matrix_a + matrix_b, np.abs(matrix_a - matrix_b), matrix_a * matrix_b])
    names = [
        f"pk_{operation}_{column}"
        for operation in ("sum", "absdiff", "product")
        for column in side_a.columns
    ]
    return pd.DataFrame(pair_matrix, columns=names, index=frame.index)
