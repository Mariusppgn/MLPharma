import numpy as np

from deep_pharma.prr_oof import per_target_prr_diagnostics


def test_target_diagnostics_attribute_total_squared_error() -> None:
    targets = np.array([[1.0, 0.0], [3.0, 2.0]], dtype=np.float32)
    predictions = np.array([[2.0, 99.0], [5.0, 5.0]], dtype=np.float32)
    diagnostics = per_target_prr_diagnostics(targets, predictions, ["a", "b"])

    assert np.isclose(diagnostics["sse"].sum(), 14.0)
    assert np.isclose(diagnostics["sse_contribution"].sum(), 1.0)
    assert diagnostics.iloc[0]["target"] == "b"
