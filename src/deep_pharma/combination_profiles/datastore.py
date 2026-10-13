"""Local train-only sources. Context cohorts exclude every row of the selected pair."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..graph.preprocessing import chemical_properties, molecule_id
from ..text_features import TEXT_FIELDS, normalize_biomedical_text
from ..validation import canonicalize_smiles
from .labels import LABELS
from .statistics import cell, compare_effect, filter_rows, number, summarize
from .structures import molecular_structure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProfileStore:
    def __init__(
        self,
        train: Path,
        *,
        oof_dir: Path | None = None,
        manifest: Path | None = None,
        profiles: Path | None = None,
    ):
        self.train = train
        self.frame = pd.read_csv(train)
        if not self.frame.Pair_ID.is_unique:
            raise ValueError("Pair_ID dupliqué : provenance ambiguë.")
        self.effects = [
            c.removeprefix("Target_Binary_") for c in self.frame if c.startswith("Target_Binary_")
        ]
        self.train_hash = sha256(train)
        self.pair_rows = defaultdict(list)
        self.molecule_rows = defaultdict(set)
        self.molecules = {}
        self.row_ids = []
        for i, row in self.frame.iterrows():
            ids = []
            for side in ("A", "B"):
                smiles = canonicalize_smiles(row[f"SMILES_{side}"])
                mid = molecule_id(smiles)
                ids.append(mid)
                self.molecule_rows[mid].add(i)
                if mid not in self.molecules:
                    self.molecules[mid] = {
                        "id": mid,
                        "smiles": smiles,
                        "example_pair": str(row.Pair_ID),
                        "side": side,
                    }
            self.row_ids.append(ids)
            self.pair_rows[tuple(sorted(ids))].append(i)
        self.binary = self.frame[["Target_Binary_" + e for e in self.effects]].to_numpy(float)
        self.prr = self.frame[["Target_PRR_" + e for e in self.effects]].to_numpy(float)
        if not np.isin(self.binary, [0, 1]).all():
            raise ValueError("Annotations invalides.")
        self.oof = None
        self.oof_status = "Non disponible"
        self.oof_sources = []
        if oof_dir is not None and manifest is not None:
            self._load_oof(oof_dir, manifest)
        self.imported = {}
        self.import_hash = None
        if profiles is not None:
            self._load_profiles(profiles)
        # Per-instance bounded caches avoid retaining discarded stores across reloads/tests.
        self.pharmacology = lru_cache(maxsize=256)(self.pharmacology)
        self.context = lru_cache(maxsize=256)(self.context)
        self.audit = {
            "train": str(train),
            "train_sha256": self.train_hash,
            "rows": len(self.frame),
            "molecules": len(self.molecules),
            "effects": len(self.effects),
            "canonical_pairs": len(self.pair_rows),
            "pairs_with_multiple_rows": sum(len(v) > 1 for v in self.pair_rows.values()),
            "individual_profiles": len({key[0] for key in self.imported if key[1] == ""}),
            "import_sha256": self.import_hash,
            "oof_status": self.oof_status,
            "oof_sources": self.oof_sources,
            "clinical_counts": any(c.get("n") is not None for c in self.imported.values()),
            "clinical_ci": any(c.get("ci") is not None for c in self.imported.values()),
            "meddra": any(c.get("soc") or c.get("pt") for c in self.imported.values()),
            "test_read": False,
            "training": False,
        }

    def _load_oof(self, directory: Path, manifest_path: Path):
        # Fail closed on missing/stale/misaligned predictions, never retrain.
        try:
            signature = json.loads((directory / "feature_signature.json").read_text())
            if signature["train_sha256"] != self.train_hash:
                raise ValueError("SHA train différent")
            manifest = pd.read_csv(manifest_path).sort_values("row_index")
            if not np.array_equal(manifest.row_index, np.arange(len(self.frame))):
                raise ValueError("Couverture manifeste invalide")
            if manifest.Pair_ID.tolist() != self.frame.Pair_ID.tolist():
                raise ValueError("Pair_ID désalignés")
            if manifest.groupby("pair_group").fold.nunique().max() != 1:
                raise ValueError("Un couple apparaît dans plusieurs folds")
            # Independently verify canonical pair grouping of the actual current train.
            for indices in self.pair_rows.values():
                if manifest.iloc[indices].fold.nunique() != 1:
                    raise ValueError("Fuite de couple dans le manifeste")
            values = np.full_like(self.binary, np.nan)
            coverage = np.zeros(len(self.frame), int)
            sources = []
            for fold in sorted(manifest.fold.unique()):
                path = directory / f"evaluation_pair_grouped_{fold}.npz"
                with np.load(path, allow_pickle=False) as arrays:
                    indices, probabilities = arrays["row_indices"], arrays["graph"]
                    if not np.array_equal(
                        np.sort(indices), manifest.loc[manifest.fold == fold, "row_index"]
                    ):
                        raise ValueError("Fold OOF désaligné")
                    if probabilities.shape != (len(indices), len(self.effects)):
                        raise ValueError("Forme OOF invalide")
                    if (
                        not np.isfinite(probabilities).all()
                        or not ((probabilities >= 0) & (probabilities <= 1)).all()
                    ):
                        raise ValueError("Probabilités OOF invalides")
                    values[indices] = probabilities
                    np.add.at(coverage, indices, 1)
                sources.append({"path": str(path), "sha256": sha256(path)})
            if not np.all(coverage == 1):
                raise ValueError("Couverture OOF incomplète/dupliquée")
            self.oof = values
            self.oof_status = (
                "OOF 5 folds pair-grouped vérifiés ; modèle de recherche graph binaire"
            )
            self.oof_sources = sources
        except (OSError, ValueError, KeyError) as exc:
            self.oof_status = f"Non disponible : {exc}"

    def _load_profiles(self, path: Path):
        data = pd.read_csv(path, keep_default_na=False)
        required = {
            "molecule_a",
            "molecule_b",
            "effect",
            "kind",
            "value",
            "source",
            "comparability_id",
            "meaning",
            "population",
            "period",
        }
        if not required.issubset(data.columns):
            raise ValueError(f"Colonnes profils requises : {sorted(required)}")
        for row in data.to_dict("records"):
            a, b = row.pop("molecule_a"), row.pop("molecule_b")
            effect = row.pop("effect")
            if a not in self.molecules or (b and b not in self.molecules):
                raise ValueError("Identifiant molécule inconnu")
            valid_kind = row["kind"] in ("prr", "probability", "annotation")
            if effect not in self.effects or not valid_kind:
                raise ValueError("Effet/type inconnu")
            if not row["source"]:
                raise ValueError("Source obligatoire")
            for key in ("n", "ci_low", "ci_high"):
                row[key] = number(row.get(key))
            low, high = row.pop("ci_low"), row.pop("ci_high")
            if (low is None) != (high is None) or (low is not None and low > high):
                raise ValueError("Intervalle incomplet/inversé")
            row["ci"] = [low, high] if low is not None else None
            key = (*sorted((a, b)), effect, row["kind"]) if b else (a, "", effect, row["kind"])
            if key in self.imported:
                raise ValueError("Profils dupliqués : sélectionner une source unique")
            self.imported[key] = cell(**row)
        self.import_hash = sha256(path)

    def catalog(self) -> dict:
        items = []
        partners = {mid: set() for mid in self.molecules}
        for a, b in self.pair_rows:
            partners[a].add(b)
            partners[b].add(a)
        for mid, meta in self.molecules.items():
            i = next(iter(sorted(self.molecule_rows[mid])))
            side = "A" if self.row_ids[i][0] == mid else "B"
            mechanism = self.frame.iloc[i].get(f"Mechanism_{side}", "")
            items.append(
                {
                    "id": mid,
                    "smiles": meta["smiles"],
                    "search": normalize_biomedical_text(mechanism)[:350],
                    "rows": len(self.molecule_rows[mid]),
                }
            )
        # Start with the first real row with an annotation, not an invented demo profile.
        initial = int(np.flatnonzero(self.binary.any(axis=1))[0]) if self.binary.any() else 0
        first = tuple(sorted(self.row_ids[initial]))
        return {
            "molecules": sorted(items, key=lambda m: m["id"]),
            "default": first,
            "partners": {mid: sorted(values) for mid, values in partners.items()},
            "audit": self.audit,
            "oof_available": self.oof is not None,
            "sources": ["binary", "prr", "oof"] if self.oof is not None else ["binary", "prr"],
            "soc": sorted({c["soc"] for c in self.imported.values() if c.get("soc")}),
            "pt": sorted({c["pt"] for c in self.imported.values() if c.get("pt")}),
        }

    def pharmacology(self, mid: str, preferred: int | None) -> dict:
        indices = sorted(self.molecule_rows[mid])
        i = preferred if preferred in indices else indices[0]
        side = "A" if self.row_ids[i][0] == mid else "B"
        row = self.frame.iloc[i]
        fields = {
            field: normalize_biomedical_text(row.get(f"{field}_{side}", ""))
            for field in TEXT_FIELDS
        }
        # Surface variation instead of selecting a false canonical monograph silently.
        variants = {
            tuple(
                normalize_biomedical_text(
                    self.frame.iloc[j].get(
                        f"{field}_{'A' if self.row_ids[j][0] == mid else 'B'}", ""
                    )
                )
                for field in TEXT_FIELDS
            )
            for j in indices
        }
        return {
            **self.molecules[mid],
            "fields": fields,
            "properties": chemical_properties(self.molecules[mid]["smiles"]),
            "text_source": str(row.Pair_ID),
            "text_variants": len(variants),
            "name": None,
            "pharmacological_class": None,
            "structure": molecular_structure(self.molecules[mid]["smiles"]),
        }

    def context(self, mid: str, excluded: tuple[str, str]) -> tuple:
        indices = sorted(self.molecule_rows[mid] - set(self.pair_rows.get(excluded, [])))
        groups = defaultdict(list)
        for i in indices:
            groups[tuple(sorted(self.row_ids[i]))].append(i)
        # One pair, one vote. Conflicting duplicate annotations => missing for that effect.
        unique = []
        for rows in groups.values():
            block = self.binary[rows]
            agrees = np.min(block, axis=0) == np.max(block, axis=0)
            unique.append(np.where(agrees, block[0], np.nan))
        if not unique:
            return tuple((None, 0, 0) for _ in self.effects)
        matrix = np.asarray(unique)
        counts = np.isfinite(matrix).sum(axis=0)
        positives = np.nansum(matrix, axis=0)
        return tuple(
            (float(k / n) if n else None, int(n), int(k))
            for n, k in zip(counts, positives, strict=True)
        )

    def compare(self, request: dict) -> dict:
        a, b = request["a"], request["b"]
        if a not in self.molecules or b not in self.molecules:
            raise ValueError("Molécule inconnue")
        source = request.get("source", "binary")
        if source not in ("binary", "prr", "oof", "imported_probability", "imported_prr"):
            raise ValueError("Source inconnue")
        key = tuple(sorted((a, b)))
        matches = self.pair_rows.get(key, [])
        pair_id = request.get("pair_id")
        i = next((j for j in matches if str(self.frame.iloc[j].Pair_ID) == pair_id), None)
        if pair_id and i is None:
            raise ValueError("Pair_ID incompatible avec le couple")
        if i is None and matches:
            i = matches[0]
        kind = (
            "prr"
            if source in ("prr", "imported_prr")
            else ("annotation" if source == "binary" else "probability")
        )
        threshold = number(request.get("threshold", 0 if kind == "prr" else 0.5))
        margin = number(request.get("margin", 0.05))
        if threshold is None or margin is None or (kind != "prr" and threshold > 1):
            raise ValueError("Seuil/marge invalides")
        minimum_n = number(request.get("min_support", 0))
        if minimum_n is None:
            raise ValueError("Effectif minimal invalide")
        context = bool(request.get("context", False)) and source == "binary"
        ca, cb = (self.context(a, key), self.context(b, key)) if context else (None, None)
        rows = []
        for e, effect in enumerate(self.effects):
            mono = []
            for mid, cohort in ((a, ca), (b, cb)):
                value = self.imported.get((mid, "", effect, kind), cell())
                if context:
                    freq, n, k = cohort[e]
                    value = cell(
                        freq,
                        kind="context_frequency",
                        n=n,
                        positives=k,
                        source="Autres couples canoniques du train ; couple étudié exclu",
                        comparability_id="train_context",
                        meaning="pair annotation frequency",
                        population="train pairs",
                        period="dataset unspecified",
                    )
                mono.append(value)
            ab = cell(kind=kind, source="Combinaison absente ou source indisponible")
            if source.startswith("imported_"):
                ab = self.imported.get((*key, effect, kind), ab)
            elif i is not None:
                raw = (
                    self.prr[i, e]
                    if source == "prr"
                    else self.binary[i, e]
                    if source == "binary"
                    else self.oof[i, e]
                    if self.oof is not None
                    else None
                )
                ab = cell(
                    raw,
                    kind=kind,
                    source=f"{source} : {self.frame.iloc[i].Pair_ID}",
                    prediction=source == "oof",
                    row_index=i,
                    n=None,
                    ci=None,
                )
            r = compare_effect(
                effect,
                *mono,
                ab,
                threshold=threshold,
                margin=margin,
                min_support=int(minimum_n),
                independence=bool(request.get("independence", False)),
            )
            r["label"] = LABELS.get(effect, effect.replace("_", " "))
            r["soc"], r["pt"] = ab.get("soc"), ab.get("pt")
            rows.append(r)
        filtered = filter_rows(rows, request.get("filters", {}))
        pharma_a, pharma_b = self.pharmacology(a, i), self.pharmacology(b, i)

        # Exact CYP mentions only, never infer inhibition, substrate or causal interaction.
        def cyp(fields):
            text = " ".join(fields[f] for f in ("CYP450_Enzymes", "Metabolism"))
            return set(re.findall(r"\bCYP\s*([1-4][A-Z]\d+)\b", text.upper()))

        return {
            "rows": filtered,
            "summary": summarize(filtered),
            "pharmacology": {
                "a": pharma_a,
                "b": pharma_b,
                "shared_cyp_mentions": sorted(cyp(pharma_a["fields"]) & cyp(pharma_b["fields"])),
            },
            "selection": {
                "a": a,
                "b": b,
                "source": source,
                "context": context,
                "pair_id": str(self.frame.iloc[i].Pair_ID) if i is not None else None,
                "available_pairs": [str(self.frame.iloc[j].Pair_ID) for j in matches],
                "threshold": threshold,
                "margin": margin,
                "min_support": minimum_n,
            },
            "provenance": self.audit,
            "limitations": [
                "Pas de profils monothérapie dans les données officielles.",
                "Fréquences contextuelles = couples annotés, pas incidence clinique.",
                "Doublons du couple étudié et annotations contextuelles conflictuelles exclus.",
                "Textes parfois variables/incohérents : aucun nom/classe inféré.",
                "PRR = disproportionalité ; annotations et probabilités du modèle ≠ incidence.",
                "Pas de dose, durée, exposition, effectif patient ou IC dans le train.",
                "Association non causale : sous-déclaration, sélection et confusion possibles.",
                "Recherche seulement : aucun entraînement, accès test ou soumission Kaggle.",
            ],
        }
