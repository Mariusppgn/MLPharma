"""Recompute the retained OOF score without loading test labels or uploading anything."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from deep_pharma.constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from deep_pharma.metric import hardcore_score
from deep_pharma.validation import audit_pair_grouped_folds, split_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def read_aligned(path: str, train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / path, low_memory=False)
    if len(frame) != len(train) or not frame[ID_COLUMN].astype(str).equals(
        train[ID_COLUMN].astype(str)
    ):
        raise ValueError(f"Prédictions OOF désalignées : {path}")
    return frame


def main() -> None:
    metrics = json.loads(
        (ROOT / "artifacts/runs/severity_activity_v1/metrics.json").read_text(encoding="utf-8")
    )
    train = pd.read_csv(ROOT / "data/raw/train.csv", low_memory=False)
    manifest = pd.read_csv(ROOT / "artifacts/validation/pair_grouped_folds.csv")
    audit = audit_pair_grouped_folds(train, manifest)
    if not audit["passed"] or split_fingerprint(manifest) != metrics["validation_fingerprint"]:
        raise ValueError("Audit des folds ou empreinte invalide.")

    sources = metrics["sources"]
    severity = read_aligned(sources["severity_oof"], train)
    binary = read_aligned(sources["binary_oof"], train)
    activity = read_aligned(sources["activity_oof"], train)
    prr = read_aligned(sources["prr_oof"], train)
    binary_columns = [c for c in train if c.startswith(BINARY_PREFIX)]
    prr_columns = [c for c in train if c.startswith(PRR_PREFIX)]
    gated_binary = binary[binary_columns].to_numpy(np.int8) * (
        activity["active_probability"].to_numpy() >= sources["activity_threshold"]
    )[:, None]
    score = hardcore_score(
        train[SEVERITY_COLUMN],
        severity[SEVERITY_COLUMN],
        train[binary_columns],
        gated_binary,
        train[prr_columns],
        prr[prr_columns],
    )
    for key, stored in metrics["score"].items():
        observed = getattr(score, key)
        if not math.isclose(observed, stored, abs_tol=1e-9):
            raise ValueError(f"Score {key} incohérent : {observed} != {stored}")
    candidate = ROOT / sources["candidate_csv"]
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != sources["candidate_sha256"]:
        raise ValueError("Le CSV soumis ne correspond plus à l'empreinte enregistrée.")
    registry = json.loads((ROOT / "artifacts/submission_registry.json").read_text("utf-8"))
    if registry["best_run_id"] != metrics["run_id"] or not math.isclose(
        registry["best_local_score"], score.total, abs_tol=1e-9
    ):
        raise ValueError("Le registre local ne correspond pas au candidat retenu.")
    print(
        json.dumps(
            {
                "score": score.__dict__,
                "severity_confusion_major_minor_moderate": confusion_matrix(
                    train[SEVERITY_COLUMN],
                    severity[SEVERITY_COLUMN],
                    labels=["Major", "Minor", "Moderate"],
                ).tolist(),
                "binary_confusion": confusion_matrix(
                    train[binary_columns].to_numpy().ravel(), gated_binary.ravel()
                ).tolist(),
                "validation_fingerprint": metrics["validation_fingerprint"],
                "candidate_sha256": digest,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
