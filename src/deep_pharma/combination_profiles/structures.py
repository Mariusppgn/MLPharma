"""Faithful 2D skeletal drawings from the supplied canonical SMILES, not 3D conformers."""

from functools import lru_cache
from xml.etree import ElementTree

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from ..validation import canonicalize_smiles


@lru_cache(maxsize=1024)
def molecular_structure(smiles: str) -> dict:
    canonical = canonicalize_smiles(smiles)
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False  # Keep standalone hydrogen ions/isotopes and all salt fragments.
    molecule = Chem.MolFromSmiles(canonical, parameters)
    if molecule is None:
        raise ValueError("Structure indisponible : SMILES invalide.")
    rdDepictor.Compute2DCoords(molecule, canonOrient=True, sampleSeed=2026)
    drawer = rdMolDraw2D.MolDraw2DSVG(640, 400)
    options = drawer.drawOptions()
    options.padding = 0.08
    options.minFontSize = 12
    options.maxFontSize = 25
    options.bondLineWidth = 1.8
    options.addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    root = ElementTree.fromstring(drawer.GetDrawingText())
    root.set("viewBox", "0 0 640 400")
    # Namespace registration produces plain <svg>, usable in image downloads and HTML previews.
    ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
    return {
        "svg": ElementTree.tostring(root, encoding="unicode"),
        "smiles": canonical,
        "atoms": molecule.GetNumAtoms(),
        "bonds": molecule.GetNumBonds(),
        "fragments": len(Chem.GetMolFrags(molecule)),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "method": "RDKit 2D skeletal depiction",
        "rdkit_version": rdBase.rdkitVersion,
    }
