import numpy as np
import pytest

from deep_pharma.metric import hardcore_score


def test_perfect_predictions_score_one() -> None:
    severity = np.array(["Major", "Moderate", "Minor", "Contraindication"])
    binary = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    prr = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 4.0], [0.0, 0.0]])

    score = hardcore_score(severity, severity, binary, binary, prr, prr)

    assert score.total == pytest.approx(1.0)
    assert score.prr_masked_rmse == pytest.approx(0.0)
    assert score.prr_positive_count == 4


def test_prr_rmse_ignores_true_zero_positions() -> None:
    score = hardcore_score(
        ["Major", "Minor"],
        ["Major", "Minor"],
        [[1], [0]],
        [[1], [0]],
        [[2.0, 0.0], [0.0, 4.0]],
        [[3.0, 999.0], [999.0, 3.0]],
    )

    assert score.prr_positive_count == 2
    assert score.prr_masked_rmse == pytest.approx(1.0)
    assert score.prr_score == pytest.approx(0.5)
    assert score.total == pytest.approx(0.85)


def test_no_positive_prr_is_explicit_error() -> None:
    with pytest.raises(ValueError, match="aucune vraie valeur"):
        hardcore_score(["Major"], ["Major"], [[1]], [[1]], [[0.0]], [[0.0]])
