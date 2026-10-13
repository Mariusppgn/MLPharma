import numpy as np
import pytest
import yaml

from deep_pharma.prr_tail_dependency import (
    crossfit_reduced_rank_coeffects,
    finalize_prr_tail_dependency,
    fit_coeffect_graph,
    fit_predict_conditional_tail,
    target_dependency_features,
)


def test_graph_uses_conditional_cooccurrence_and_excludes_self() -> None:
    targets = np.array(
        [[2, 3, 0], [1, 2, 0], [0, 0, 4], [3, 2, 0], [0, 0, 0]] * 12,
        dtype=np.float32,
    )
    graph = fit_coeffect_graph(targets, neighbors=1, minimum_joint=2, joint_shrinkage=1.0)
    assert graph.shape == (3, 3)
    assert np.allclose(np.diag(graph), 0.0)
    assert graph[0, 1] == 1.0
    assert graph[1, 0] == 1.0
    assert graph[2].sum() == 0.0


def test_dependency_features_use_only_supplied_priors() -> None:
    base = np.array([[10.0], [20.0]], dtype=np.float32)
    blocks = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)] * 6
    prior = np.hstack([*blocks, np.zeros((2, 3), dtype=np.float32)])
    graph = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    features = target_dependency_features(base, np.hstack([base, prior]), graph, 0)
    assert features.shape == (2, 16)
    assert np.allclose(features[:, 1:7], [[1] * 6, [3] * 6])
    assert np.allclose(features[:, 7:13], [[2] * 6, [4] * 6])


def test_zero_neighbor_graph_is_a_valid_ablation() -> None:
    targets = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    graph = fit_coeffect_graph(targets, neighbors=0, minimum_joint=1, joint_shrinkage=1.0)
    assert np.all(graph == 0)


def test_tail_prediction_is_finite_positive_and_uses_training_values_only() -> None:
    rng = np.random.default_rng(7)
    fit_features = rng.normal(size=(90, 4)).astype(np.float32)
    validation_features = rng.normal(size=(12, 4)).astype(np.float32)
    values = (4 + fit_features[:, 0] + rng.exponential(2, size=90)).astype(np.float32)
    values[:8] += 20.0
    config = {
        "minimum_positive": 20,
        "threshold_quantile": 0.80,
        "minimum_tail": 5,
        "n_estimators": 8,
        "tail_estimators": 8,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "tail_num_leaves": 5,
        "min_child_samples": 3,
        "reg_lambda": 1.0,
        "tail_reg_lambda": 2.0,
        "tail_shrinkage": 10.0,
    }
    first, metadata = fit_predict_conditional_tail(
        fit_features, validation_features, values, config=config, seed=11
    )
    second, _ = fit_predict_conditional_tail(
        fit_features, validation_features, values, config=config, seed=11
    )
    assert metadata["used_tail_head"]
    assert np.isfinite(first).all()
    assert (first >= 0).all()
    assert np.allclose(first, second)


def test_multieffect_features_are_cross_fitted() -> None:
    features = np.eye(8, dtype=np.float32)
    targets = np.column_stack([np.arange(8), np.arange(8)[::-1]]).astype(np.float32)
    folds = np.arange(8) % 2
    fit_oof, validation = crossfit_reduced_rank_coeffects(
        features,
        targets,
        folds,
        features[:2],
        rank=1,
        ridge=1.0,
    )
    assert fit_oof.shape == targets.shape
    assert validation.shape == (2, 2)
    assert np.isfinite(fit_oof).all()
    assert not np.allclose(fit_oof[:2], validation)


def test_finalization_refuses_failed_oof_without_loading_test(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text('{"promotion_eligible": false}', encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"run_dir": str(run_dir), "test": "missing_test.csv"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="aucune finalisation"):
        finalize_prr_tail_dependency(config_path)
