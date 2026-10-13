import numpy as np

from deep_pharma import prr_cdn
from deep_pharma.prr_cdn import conditional_features, iterate_cdn


class Constant:
    def predict(self, x):
        return np.full(len(x), 2.0)


def test_own_target_never_conditions_prediction():
    x = np.ones((3, 2))
    predictions = np.arange(9).reshape(3, 3).astype(float)
    expected = conditional_features(x, predictions, 1)
    predictions[:, 1] = 100000
    np.testing.assert_array_equal(expected, conditional_features(x, predictions, 1))


def test_convergence_is_prediction_only():
    final, states, changes, converged = iterate_cdn(
        [Constant(), Constant()], np.ones((4, 2)), np.zeros((4, 2)),
        damping=0.5, iterations=50, tolerance=1e-5,
    )
    assert converged
    assert len(changes) < 50
    np.testing.assert_allclose(final, 2, atol=5e-5)
    assert len(states) == len(changes) + 1


def test_nonconvergence_is_reported():
    _, _, changes, converged = iterate_cdn(
        [Constant()], np.ones((4, 2)), np.zeros((4, 1)),
        damping=0.1, iterations=2, tolerance=1e-10,
    )
    assert not converged
    assert len(changes) == 2


def test_internal_crossfit_excludes_entire_pair(monkeypatch):
    x = np.arange(12).reshape(6, 2)
    y = np.arange(6).reshape(6, 1).astype(float)
    groups = np.repeat(["pair_a", "pair_b", "pair_c"], 2)
    seen = []

    def fake_fit(features, targets, config):
        seen.append(features[:, 0].copy())
        return [float(targets.mean())]

    monkeypatch.setattr(prr_cdn, "fit_heads", fake_fit)
    prediction, audits = prr_cdn.crossfit_base(x, y, groups, {"inner_folds": 3})
    assert np.isfinite(prediction).all()
    assert all(audit["pair_group_overlap"] == 0 for audit in audits)
    assert len(seen) == 3
    for pair in range(3):
        indices = np.flatnonzero(groups == f"pair_{chr(97 + pair)}")
        expected = y[groups != groups[indices[0]]].mean()
        np.testing.assert_array_equal(prediction[indices], expected)
