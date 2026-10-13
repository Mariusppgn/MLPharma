from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class HardcoreScore:
    total: float
    severity_macro_f1: float
    side_effect_micro_f1: float
    prr_score: float
    prr_masked_rmse: float
    prr_positive_count: int


def _as_matching_arrays(
    y_true: ArrayLike, y_pred: ArrayLike, name: str
) -> tuple[np.ndarray, np.ndarray]:
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)
    if true_array.shape != pred_array.shape:
        raise ValueError(
            f"{name}: shapes différentes, vrai={true_array.shape}, prédit={pred_array.shape}."
        )
    return true_array, pred_array


def hardcore_score(
    y_true_severity: ArrayLike,
    y_pred_severity: ArrayLike,
    y_true_binary: ArrayLike,
    y_pred_binary: ArrayLike,
    y_true_prr: ArrayLike,
    y_pred_prr: ArrayLike,
) -> HardcoreScore:
    """Compute the competition composite score exactly as documented.

    PRR errors are included only where the true PRR is strictly positive.
    Severity macro-F1 follows scikit-learn's observed-label convention; when the
    organizer's scorer is available, this implementation should be compared against it.
    """

    severity_true, severity_pred = _as_matching_arrays(y_true_severity, y_pred_severity, "Severity")
    binary_true, binary_pred = _as_matching_arrays(y_true_binary, y_pred_binary, "Side effects")
    prr_true, prr_pred = _as_matching_arrays(y_true_prr, y_pred_prr, "PRR")

    if (
        not np.isfinite(prr_true.astype(float)).all()
        or not np.isfinite(prr_pred.astype(float)).all()
    ):
        raise ValueError("PRR: toutes les valeurs doivent être finies.")

    severity_macro_f1 = float(
        f1_score(severity_true, severity_pred, average="macro", zero_division=0)
    )
    side_effect_micro_f1 = float(
        f1_score(binary_true, binary_pred, average="micro", zero_division=0)
    )

    positive_mask = prr_true.astype(float) > 0.0
    positive_count = int(positive_mask.sum())
    if positive_count == 0:
        raise ValueError("PRR: aucune vraie valeur strictement positive dans le masque.")

    errors = prr_true.astype(float)[positive_mask] - prr_pred.astype(float)[positive_mask]
    masked_rmse = float(np.sqrt(np.mean(np.square(errors))))
    prr_score = 1.0 / (1.0 + masked_rmse)
    total = 0.4 * severity_macro_f1 + 0.3 * side_effect_micro_f1 + 0.3 * prr_score

    return HardcoreScore(
        total=float(total),
        severity_macro_f1=severity_macro_f1,
        side_effect_micro_f1=side_effect_micro_f1,
        prr_score=float(prr_score),
        prr_masked_rmse=masked_rmse,
        prr_positive_count=positive_count,
    )
