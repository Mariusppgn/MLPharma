from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from scipy import sparse

MORGAN_RADIUS = 2
MORGAN_SIZE = 2048

DESCRIPTOR_NAMES = [
    "mol_weight",
    "logp",
    "tpsa",
    "h_donors",
    "h_acceptors",
    "rotatable_bonds",
    "rings",
    "fraction_csp3",
    "formal_charge",
    "heavy_atoms",
    "hetero_atoms",
    "molar_refractivity",
    "fragment_count",
    "multiple_fragments",
    "has_formal_charge",
    "smiles_invalid",
]

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_SIZE,
    includeChirality=True,
)


def _parse_smiles(smiles: str) -> Chem.Mol | None:
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    return Chem.MolFromSmiles(smiles, parameters)


@cache
def molecule_features(smiles: str) -> tuple[str | None, tuple[float, ...], bytes]:
    molecule = _parse_smiles(smiles)
    if molecule is None:
        descriptors = [np.nan] * (len(DESCRIPTOR_NAMES) - 1) + [1.0]
        return None, tuple(descriptors), bytes(MORGAN_SIZE)

    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    fragments = Chem.GetMolFrags(molecule)
    formal_charge = float(Chem.GetFormalCharge(molecule))
    descriptors = (
        float(Descriptors.MolWt(molecule)),
        float(Crippen.MolLogP(molecule)),
        float(rdMolDescriptors.CalcTPSA(molecule)),
        float(Lipinski.NumHDonors(molecule)),
        float(Lipinski.NumHAcceptors(molecule)),
        float(Lipinski.NumRotatableBonds(molecule)),
        float(Lipinski.RingCount(molecule)),
        float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        formal_charge,
        float(molecule.GetNumHeavyAtoms()),
        float(Lipinski.NumHeteroatoms(molecule)),
        float(Crippen.MolMR(molecule)),
        float(len(fragments)),
        float(len(fragments) > 1),
        float(formal_charge != 0),
        0.0,
    )
    fingerprint = _MORGAN_GENERATOR.GetFingerprint(molecule)
    fingerprint_array = np.zeros(MORGAN_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    return canonical, descriptors, fingerprint_array.tobytes()


def _side_features(smiles: pd.Series) -> tuple[np.ndarray, sparse.csr_matrix]:
    descriptor_rows = []
    fingerprint_rows = []
    for value in smiles.astype(str):
        _, descriptors, fingerprint_bytes = molecule_features(value)
        descriptor_rows.append(descriptors)
        fingerprint_rows.append(np.frombuffer(fingerprint_bytes, dtype=np.uint8))
    descriptor_matrix = np.asarray(descriptor_rows, dtype=np.float32)
    fingerprint_matrix = sparse.csr_matrix(np.asarray(fingerprint_rows, dtype=np.float32))
    return descriptor_matrix, fingerprint_matrix


def build_chemical_pair_features(
    frame: pd.DataFrame,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    descriptors_a, morgan_a = _side_features(frame["SMILES_A"])
    descriptors_b, morgan_b = _side_features(frame["SMILES_B"])

    descriptor_matrix = np.hstack(
        [
            descriptors_a + descriptors_b,
            np.abs(descriptors_a - descriptors_b),
            descriptors_a * descriptors_b,
        ]
    )
    descriptor_names = [
        f"rdkit_{operation}_{name}"
        for operation in ("sum", "absdiff", "product")
        for name in DESCRIPTOR_NAMES
    ]
    descriptor_frame = pd.DataFrame(descriptor_matrix, columns=descriptor_names, index=frame.index)

    difference = (morgan_a - morgan_b).tocsr()
    difference.data = np.abs(difference.data)
    morgan_pair = sparse.hstack(
        [morgan_a + morgan_b, difference, morgan_a.multiply(morgan_b)], format="csr"
    )
    return morgan_pair, descriptor_frame
