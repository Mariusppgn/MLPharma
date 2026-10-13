from deep_pharma.submission_gate import (
    create_reference_registry,
    evaluate_submission_gate,
    promote_candidate,
)


def _metrics(run_id: str, score: float) -> dict[str, object]:
    return {
        "run_id": run_id,
        "protocol_id": "pair-v1",
        "validation_fingerprint": "abc123",
        "score": {"total": score},
        "leakage_audit": {
            "passed": True,
            "pair_group_overlap_max": 0,
            "coverage_complete": True,
            "fit_statistics_computed_within_each_fold": True,
        },
        "data_scope": {
            "train_only_for_fit": True,
            "test_used_for_fit": False,
            "test_used_for_local_scoring": False,
        },
        "oof": {"coverage_complete": True, "expected_rows": 100, "predicted_rows": 100},
    }


def test_gate_accepts_only_strict_comparable_improvement() -> None:
    registry = create_reference_registry(_metrics("reference", 0.40), min_improvement=0.001)

    rejected = evaluate_submission_gate(_metrics("equal", 0.401), registry)
    accepted = evaluate_submission_gate(_metrics("better", 0.4011), registry)

    assert rejected.eligible is False
    assert accepted.eligible is True
    assert accepted.automatic_submission is False


def test_gate_rejects_any_leakage_or_protocol_change() -> None:
    registry = create_reference_registry(_metrics("reference", 0.40))
    candidate = _metrics("leaky", 0.90)
    candidate["protocol_id"] = "another-split"
    candidate["leakage_audit"]["pair_group_overlap_max"] = 1
    candidate["data_scope"]["test_used_for_fit"] = True

    decision = evaluate_submission_gate(candidate, registry)

    assert decision.eligible is False
    assert any("protocole" in reason for reason in decision.reasons)
    assert any("traverse" in reason for reason in decision.reasons)
    assert any("test" in reason.lower() for reason in decision.reasons)


def test_gate_rejects_a_partial_oof_score() -> None:
    registry = create_reference_registry(_metrics("reference", 0.40))
    candidate = _metrics("partial", 0.90)
    candidate["oof"]["expected_rows"] = 99
    candidate["oof"]["predicted_rows"] = 99

    decision = evaluate_submission_gate(candidate, registry)

    assert decision.eligible is False
    assert any("référence complète" in reason for reason in decision.reasons)


def test_gate_rejects_an_impossible_score() -> None:
    registry = create_reference_registry(_metrics("reference", 0.40))

    decision = evaluate_submission_gate(_metrics("impossible", 1.01), registry)

    assert decision.eligible is False
    assert any("compris entre 0 et 1" in reason for reason in decision.reasons)


def test_promotion_updates_only_an_eligible_local_reference() -> None:
    registry = create_reference_registry(_metrics("reference", 0.40), min_improvement=0.001)

    decision, updated = promote_candidate(_metrics("better", 0.42), registry)

    assert decision.eligible is True
    assert updated["best_local_score"] == 0.42
    assert updated["best_run_id"] == "better"
    assert updated["history"][-1]["status"] == "approved_not_submitted"
    assert registry["best_local_score"] == 0.40
