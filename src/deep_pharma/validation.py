from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from functools import cache

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedGroupKFold

from .constants import BINARY_PREFIX, ID_COLUMN, SEVERITY_COLUMN

PAIR_GROUPED_STRATEGY = "stratified_canonical_pair_grouped_v2"
DRUG_DISJOINT_STRATEGY = "hash_canonical_drug_disjoint_holdout_v2"


@dataclass(frozen=True)
class FoldAudit:
    fold: int
    train_rows: int
    validation_rows: int
    pair_group_overlap: int
    molecule_overlap: int


@cache
def canonicalize_smiles(smiles: str) -> str:
    parser_parameters = Chem.SmilesParserParams()
    # Preserve explicit standalone hydrogens/counterions in salts. RDKit otherwise
    # warns while parsing entries such as ``.[H+].[H+]`` even though they are valid.
    parser_parameters.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parser_parameters)
    if molecule is None:
        raise ValueError(f"SMILES invalide, validation bloquée : {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def canonical_pair_key(smiles_a: object, smiles_b: object) -> str:
    """Make chemically equivalent A+B and B+A share one pair key.

    RDKit preserves stereochemistry while canonicalizing the representation.
    """

    if pd.isna(smiles_a) or pd.isna(smiles_b):
        raise ValueError("SMILES manquant, validation bloquée.")
    left = canonicalize_smiles(str(smiles_a))
    right = canonicalize_smiles(str(smiles_b))
    return "\x1f".join(sorted((left, right)))


def pair_group_hash(pair_key: str) -> str:
    return hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:20]


def frame_pair_keys(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            canonical_pair_key(left, right)
            for left, right in zip(frame["SMILES_A"], frame["SMILES_B"], strict=True)
        ],
        index=frame.index,
        name="pair_key",
    )


def _stratification_labels(frame: pd.DataFrame) -> pd.Series:
    binary_columns = [column for column in frame if column.startswith(BINARY_PREFIX)]
    if not binary_columns:
        raise ValueError("Aucune cible binaire trouvée pour construire la stratification.")
    has_side_effect = frame[binary_columns].any(axis=1).astype(int).astype(str)
    return frame[SEVERITY_COLUMN].astype(str) + "|has_effect=" + has_side_effect


def split_fingerprint(manifest: pd.DataFrame) -> str:
    required = ["row_index", ID_COLUMN, "pair_group", "fold"]
    payload = manifest[required].sort_values("row_index").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_pair_grouped_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        frame = frame.reset_index(drop=True)
    if not frame[ID_COLUMN].is_unique:
        raise ValueError(f"{ID_COLUMN} doit être unique avant de construire les folds.")

    pair_keys = frame_pair_keys(frame)
    groups = pair_keys.map(pair_group_hash)
    stratification = _stratification_labels(frame)
    fold_values = np.full(len(frame), -1, dtype=np.int16)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (_, validation_indices) in enumerate(
        splitter.split(frame, y=stratification, groups=groups)
    ):
        fold_values[validation_indices] = fold

    if (fold_values < 0).any():
        raise RuntimeError("Certaines lignes n'ont reçu aucun fold de validation.")

    manifest = pd.DataFrame(
        {
            "row_index": np.arange(len(frame), dtype=np.int64),
            ID_COLUMN: frame[ID_COLUMN].astype(str),
            "pair_group": groups,
            "stratification": stratification,
            "fold": fold_values,
        }
    )
    audit = audit_pair_grouped_folds(frame, manifest)
    audit.update(
        {
            "strategy": PAIR_GROUPED_STRATEGY,
            "n_splits": n_splits,
            "seed": seed,
            "fingerprint": split_fingerprint(manifest),
        }
    )
    audit["protocol_id"] = (
        f"{PAIR_GROUPED_STRATEGY}:folds={n_splits}:seed={seed}:sha256={audit['fingerprint']}"
    )
    return manifest, audit


def audit_pair_grouped_folds(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    expected_rows = np.arange(len(frame), dtype=np.int64)
    sorted_manifest = manifest.sort_values("row_index").reset_index(drop=True)
    errors: list[str] = []
    if not np.array_equal(sorted_manifest["row_index"].to_numpy(), expected_rows):
        errors.append("Le manifeste ne couvre pas exactement chaque ligne une fois.")
    if not sorted_manifest[ID_COLUMN].astype(str).equals(frame[ID_COLUMN].astype(str)):
        errors.append(f"L'alignement des {ID_COLUMN} entre train et manifeste est invalide.")

    pair_keys = frame_pair_keys(frame)
    folds: list[FoldAudit] = []
    for fold in sorted(manifest["fold"].unique()):
        validation_mask = sorted_manifest["fold"].to_numpy() == fold
        train_mask = ~validation_mask
        train_pairs = set(pair_keys[train_mask])
        validation_pairs = set(pair_keys[validation_mask])
        train_molecules = set(
            frame.loc[train_mask, "SMILES_A"].astype(str).map(canonicalize_smiles)
        ) | set(frame.loc[train_mask, "SMILES_B"].astype(str).map(canonicalize_smiles))
        validation_molecules = set(
            frame.loc[validation_mask, "SMILES_A"].astype(str).map(canonicalize_smiles)
        ) | set(frame.loc[validation_mask, "SMILES_B"].astype(str).map(canonicalize_smiles))
        pair_overlap = len(train_pairs & validation_pairs)
        if pair_overlap:
            errors.append(f"Fold {fold}: {pair_overlap} groupes de paires traversent le split.")
        folds.append(
            FoldAudit(
                fold=int(fold),
                train_rows=int(train_mask.sum()),
                validation_rows=int(validation_mask.sum()),
                pair_group_overlap=pair_overlap,
                molecule_overlap=len(train_molecules & validation_molecules),
            )
        )
    return {
        "passed": not errors,
        "errors": errors,
        "coverage_complete": len(sorted_manifest) == len(frame),
        "folds": [asdict(fold) for fold in folds],
    }


def _is_held_out(compound: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}\x00{compound}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def build_drug_disjoint_holdout(
    frame: pd.DataFrame,
    *,
    seed: int,
    holdout_fraction: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create a strict drug-disjoint holdout and purge all boundary pairs."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction doit être strictement compris entre 0 et 1.")
    canonical_left = frame["SMILES_A"].astype(str).map(canonicalize_smiles)
    canonical_right = frame["SMILES_B"].astype(str).map(canonicalize_smiles)
    left_held = canonical_left.map(lambda value: _is_held_out(value, seed, holdout_fraction))
    right_held = canonical_right.map(lambda value: _is_held_out(value, seed, holdout_fraction))
    role = np.full(len(frame), "purged", dtype=object)
    role[(~left_held & ~right_held).to_numpy()] = "train"
    role[(left_held & right_held).to_numpy()] = "validation"

    train_mask = role == "train"
    validation_mask = role == "validation"
    train_molecules = set(canonical_left[train_mask]) | set(canonical_right[train_mask])
    validation_molecules = set(canonical_left[validation_mask]) | set(
        canonical_right[validation_mask]
    )
    overlap = train_molecules & validation_molecules
    errors = []
    if overlap:
        errors.append(f"{len(overlap)} molécules traversent le holdout strict.")
    if not validation_mask.any() or not train_mask.any():
        errors.append("Le holdout strict a produit un train ou une validation vide.")

    manifest = pd.DataFrame(
        {
            "row_index": np.arange(len(frame), dtype=np.int64),
            ID_COLUMN: frame[ID_COLUMN].astype(str),
            "role": role,
        }
    )
    audit = {
        "passed": not errors,
        "errors": errors,
        "strategy": DRUG_DISJOINT_STRATEGY,
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "purged_rows": int((role == "purged").sum()),
        "molecule_overlap": len(overlap),
    }
    return manifest, audit
