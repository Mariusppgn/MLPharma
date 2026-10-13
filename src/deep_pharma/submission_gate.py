from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class GateDecision:
    eligible: bool
    candidate_run_id: str
    candidate_score: float | None
    reference_run_id: str | None
    reference_score: float | None
    improvement: float | None
    required_improvement: float
    reasons: list[str]
    automatic_submission: bool = False


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_reference_registry(
    metrics: dict[str, object],
    *,
    min_improvement: float = 0.0001,
) -> dict[str, object]:
    score = float(metrics["score"]["total"])
    return {
        "schema_version": 2,
        "policy": {
            "automatic_submission": False,
            "require_strict_local_improvement": True,
            "min_improvement": min_improvement,
            "require_same_protocol": True,
            "require_passed_leakage_audit": True,
        },
        "protocol_id": metrics["protocol_id"],
        "validation_fingerprint": metrics["validation_fingerprint"],
        "expected_oof_rows": int(metrics["oof"]["expected_rows"]),
        "best_local_score": score,
        "best_run_id": metrics["run_id"],
        "history": [
            {
                "run_id": metrics["run_id"],
                "score": score,
                "status": "reference_not_submitted",
                "recorded_at": datetime.now(UTC).isoformat(),
            }
        ],
    }


def evaluate_submission_gate(
    candidate: dict[str, object], registry: dict[str, object]
) -> GateDecision:
    reasons: list[str] = []
    run_id = str(candidate.get("run_id", "<missing>"))
    policy = registry.get("policy", {})
    min_improvement = float(policy.get("min_improvement", 0.0001))

    score_value = candidate.get("score", {}).get("total")
    candidate_score = float(score_value) if score_value is not None else None
    if candidate_score is None or not math.isfinite(candidate_score):
        reasons.append("Score composite candidat absent ou non fini.")
    elif not 0.0 <= candidate_score <= 1.0:
        reasons.append("Le score composite candidat doit être compris entre 0 et 1.")

    if candidate.get("protocol_id") != registry.get("protocol_id"):
        reasons.append("Le protocole de validation diffère de la référence.")
    if candidate.get("validation_fingerprint") != registry.get("validation_fingerprint"):
        reasons.append("L'empreinte des folds diffère de la référence.")

    leakage = candidate.get("leakage_audit", {})
    if not leakage.get("passed", False):
        reasons.append("L'audit de fuite n'est pas validé.")
    if int(leakage.get("pair_group_overlap_max", -1)) != 0:
        reasons.append("Au moins une paire moléculaire traverse train et validation.")
    if not leakage.get("fit_statistics_computed_within_each_fold", False):
        reasons.append("Le fit fold-local des transformations/modèles n'est pas attesté.")

    data_scope = candidate.get("data_scope", {})
    if not data_scope.get("train_only_for_fit", False):
        reasons.append("Le candidat n'atteste pas un entraînement limité au train.")
    if data_scope.get("test_used_for_fit", True):
        reasons.append("Le test a été utilisé pendant le fit.")
    if data_scope.get("test_used_for_local_scoring", True):
        reasons.append("Le test a été utilisé pour le score local.")

    oof = candidate.get("oof", {})
    if not oof.get("coverage_complete", False):
        reasons.append("La couverture OOF n'est pas complète.")
    if int(oof.get("predicted_rows", -1)) != int(oof.get("expected_rows", -2)):
        reasons.append("Chaque ligne train ne possède pas exactement une prédiction OOF.")
    if int(oof.get("expected_rows", -1)) != int(registry.get("expected_oof_rows", -2)):
        reasons.append("Le nombre de lignes OOF diffère de la référence complète.")

    reference_score = float(registry["best_local_score"])
    improvement = None if candidate_score is None else candidate_score - reference_score
    numerical_tolerance = 1e-12
    if improvement is None or improvement <= min_improvement + numerical_tolerance:
        reasons.append(
            f"Amélioration insuffisante : il faut dépasser {reference_score:.6f} "
            f"de plus de {min_improvement:.6f}."
        )

    return GateDecision(
        eligible=not reasons,
        candidate_run_id=run_id,
        candidate_score=candidate_score,
        reference_run_id=str(registry.get("best_run_id")),
        reference_score=reference_score,
        improvement=improvement,
        required_improvement=min_improvement,
        reasons=reasons,
    )


def promote_candidate(
    candidate: dict[str, object], registry: dict[str, object]
) -> tuple[GateDecision, dict[str, object]]:
    """Promote an eligible local run as the new comparison reference.

    Promotion is local bookkeeping only. It never calls Kaggle.
    """

    decision = evaluate_submission_gate(candidate, registry)
    if not decision.eligible:
        return decision, registry
    updated = json.loads(json.dumps(registry))
    updated["best_local_score"] = decision.candidate_score
    updated["best_run_id"] = decision.candidate_run_id
    updated["history"].append(
        {
            "run_id": decision.candidate_run_id,
            "score": decision.candidate_score,
            "status": "approved_not_submitted",
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    return decision, updated


def write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def decision_as_dict(decision: GateDecision) -> dict[str, object]:
    return asdict(decision)
