from __future__ import annotations

import hashlib
from functools import cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler

from ..chemistry_features import build_chemical_pair_features, molecule_features
from ..constants import BINARY_PREFIX, PRR_PREFIX
from ..pk_features import build_pk_pair_features
from ..text_features import TEXT_FIELDS, molecule_documents
from ..validation import canonicalize_smiles


def molecule_id(smiles):
    return "mol_" + hashlib.sha256(canonicalize_smiles(str(smiles)).encode()).hexdigest()[:12]


def pair_features(frame, config):
    """Reuse RDKit/PK/text features, without fitted transforms or target columns."""
    fingerprints, descriptors = build_chemical_pair_features(frame)
    bins = int(config["fingerprint_bins"])
    folded = np.zeros((len(frame), bins * 3), dtype=np.float32)
    for block in range(3):
        part = fingerprints[:, block * 2048 : (block + 1) * 2048]
        for column in range(bins):
            folded[:, block * bins + column] = np.asarray(part[:, column::bins].sum(axis=1)).ravel()
    sides = [molecule_documents(frame, side, include_warning_toxicity=True) for side in ("A", "B")]
    corpus = list(dict.fromkeys(sides[0] + sides[1]))
    hasher = HashingVectorizer(
        n_features=int(config["text_hash_bins"]), alternate_sign=False, ngram_range=(1, 2)
    )
    encoded = hasher.transform(corpus).toarray().astype(np.float32)
    lookup = {document: i for i, document in enumerate(corpus)}
    left, right = [encoded[[lookup[doc] for doc in docs]] for docs in sides]
    pk = build_pk_pair_features(frame)
    numeric = np.nan_to_num(
        np.hstack([descriptors.to_numpy(float), pk.to_numpy(float)]), nan=0, posinf=0, neginf=0
    )
    return np.hstack([numeric, folded, left + right, np.abs(left - right), left * right]).astype(
        np.float32
    )


@cache
def chemical_properties(smiles):
    canonical, descriptors, _ = molecule_features(str(smiles))
    molecule = Chem.MolFromSmiles(canonical)
    from rdkit.Chem import Descriptors, rdMolDescriptors

    return {
        "molecular_weight": float(descriptors[0]),
        "logp": float(descriptors[1]),
        "tpsa": float(descriptors[2]),
        "charge": int(Chem.GetFormalCharge(molecule)),
        "rings": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "fraction_csp3": float(Descriptors.FractionCSP3(molecule)),
    }


@cache
def _similarity_fingerprint(smiles):
    return Chem.RDKFingerprint(Chem.MolFromSmiles(canonicalize_smiles(str(smiles))))


def tanimoto(left, right):
    return float(
        DataStructs.TanimotoSimilarity(
            _similarity_fingerprint(str(left)), _similarity_fingerprint(str(right))
        )
    )


def pharmacology_text(frame, row, side, limit):
    """Unverified text remains text; never invent ATC/target/action annotations."""
    return {
        field: str(frame.at[row, field + "_" + side])[:limit]
        if pd.notna(frame.at[row, field + "_" + side])
        else None
        for field in TEXT_FIELDS
        if field + "_" + side in frame
    }


def profile_data(frame, groups, config):
    binary_columns = [c for c in frame if c.startswith(BINARY_PREFIX)]
    prr_columns = [c for c in frame if c.startswith(PRR_PREFIX)]
    raw_y = frame[binary_columns].to_numpy(float)
    if not np.isin(raw_y, [0, 1]).all():
        raise ValueError("Binary effect targets must be exactly zero or one")
    y = raw_y.astype(np.int8)
    prr = frame[prr_columns].to_numpy(float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(prr).all() or (prr < 0).any():
        raise ValueError("Invalid effect targets")
    molecules = {
        canonicalize_smiles(str(s)) for side in ("A", "B") for s in frame["SMILES_" + side]
    }
    nodes = []
    for j, column in enumerate(binary_columns):
        positives = int(y[:, j].sum())
        values = prr[:, j][prr[:, j] > 0]
        nodes.append(
            {
                "id": j,
                "effect": column.removeprefix(BINARY_PREFIX),
                "positive_pairs": positives,
                "observed_pairs": len(frame),
                "frequency": positives / len(frame),
                "rare": positives / len(frame) < config["dashboard"]["rare_prevalence"],
                "mean_positive_prr": float(values.mean()) if len(values) else None,
                "median_positive_prr": float(np.median(values)) if len(values) else None,
                "clinical_cases": None,
                "meddra": None,
                "soc": None,
            }
        )
    report = {
        "pairs": len(frame),
        "canonical_pair_groups": int(len(np.unique(groups))),
        "unique_molecules": len(molecules),
        "effect_count": len(binary_columns),
        "effect_sparsity": float(1 - y.mean()),
        "active_pairs": int(y.any(axis=1).sum()),
        "binary_prr_disagreements": int(((prr > 0) != y.astype(bool)).sum()),
        "missing_values": {c: int(frame[c].isna().sum()) for c in frame},
        "empty_or_missing_properties": {
            c: int((frame[c].isna() | frame[c].astype(str).str.strip().eq("")).sum())
            for c in frame
            if c.endswith(("_A", "_B")) and not c.startswith("SMILES")
        },
        "availability": {
            "time": False,
            "clinical_case_counts": False,
            "atc": False,
            "structured_targets": False,
            "meddra_soc": False,
            "cyp_text": True,
            "mechanism_text": True,
            "structures": True,
        },
        "interpretation": "Counts are annotated molecular pairs, not clinical cases.",
        "least_frequent": sorted(nodes, key=lambda n: n["frequency"])[:10],
        "most_frequent": sorted(nodes, key=lambda n: -n["frequency"])[:10],
    }
    return y, prr, nodes, report


def cluster_molecules(frame, clusters, seed):
    smiles = sorted(
        {canonicalize_smiles(str(s)) for side in ("A", "B") for s in frame["SMILES_" + side]}
    )
    properties = np.array([list(chemical_properties(s).values()) for s in smiles])
    scaled = StandardScaler().fit_transform(properties)
    model = KMeans(n_clusters=min(clusters, len(smiles)), random_state=seed, n_init=10)
    assignments = model.fit_predict(scaled)
    return {molecule_id(s): int(c) for s, c in zip(smiles, assignments, strict=True)}
