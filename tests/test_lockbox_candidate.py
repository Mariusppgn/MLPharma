from deep_pharma.lockbox_candidate import _is_improved
from deep_pharma.metric import HardcoreScore


def _score(total: float) -> HardcoreScore:
    return HardcoreScore(
        total=total,
        severity_macro_f1=0.6,
        side_effect_micro_f1=0.7,
        prr_score=0.2,
        prr_masked_rmse=4.0,
        prr_positive_count=10,
    )


def test_lockbox_gate_requires_strict_improvement() -> None:
    eligible, improvement = _is_improved(_score(0.5502), _score(0.55), 0.0001)
    assert eligible
    assert improvement > 0.0001

    tied, _ = _is_improved(_score(0.5501), _score(0.55), 0.0001)
    assert not tied
