import json

import numpy as np

from deep_pharma.graph.bootstrap import grouped_resample
from deep_pharma.graph.conditional_graph import edge_gates
from deep_pharma.graph.evaluation import add_neighbors, crossfit_probabilities
from deep_pharma.graph.pipeline import edge_records
from deep_pharma.graph.screening import conditional_information, contingency
from deep_pharma.graph.structure_learning import bic_dag, pc_skeleton


def test_independence_and_association():
    x = np.tile([0, 0, 1, 1], 100)
    y = np.tile([0, 1, 0, 1], 100)
    assert abs(contingency(x, y)["phi"]) < 1e-10
    assert contingency(x, x)["phi"] > 0.98
    assert contingency(x, x)["odds_ratio"] > 100


def test_sparse_is_not_independence_evidence():
    y = np.zeros((100, 3), dtype=int)
    assert not conditional_information(y, 0, 1, [2], 3)["testable"]


def test_bootstrap_keeps_complete_groups():
    groups = np.repeat(np.arange(20), 2)
    sampled = grouped_resample(groups, np.random.default_rng(2))
    counts = np.bincount(sampled, minlength=len(groups))
    assert np.array_equal(counts[::2], counts[1::2])


def test_coeffect_features_use_probabilities_not_labels():
    x = np.ones((8, 4))
    p = np.ones((8, 3)) * 0.2
    first = add_neighbors(x, p, [1, 2])
    p[:, 0] = 0.9
    np.testing.assert_array_equal(first, add_neighbors(x, p, [1, 2]))


def test_crossfit_excludes_every_held_group(monkeypatch):
    import deep_pharma.graph.evaluation as module

    x = np.arange(12).reshape(-1, 1)
    groups = np.repeat(np.arange(6), 2)
    y = np.zeros((12, 2), dtype=int)

    def fit(core, labels, *args):
        return set(core[:, 0])

    def predict(heads, held):
        assert not heads.intersection(held[:, 0])
        return np.full((len(held), 2), 0.3)

    monkeypatch.setattr(module, "fit_binary_heads", fit)
    monkeypatch.setattr(module, "predict_heads", predict)
    p = crossfit_probabilities(x, y, groups, {"inner_splits": 3}, 1, 1)
    assert np.isfinite(p).all()


def test_bic_direction_is_not_overwritten_by_marginal():
    graphs = {
        "marginal": [{"source": 0, "target": 1, "phi": 0.2}],
        "candidates": {(0, 1)},
        "pc": set(),
        "neighborhood": set(),
        "bic": {(1, 0)},
        "weights": {},
        "conditional_tests": {},
    }
    edge = next(e for e in edge_records(graphs, {}, 2) if e["method"] == "bic")
    assert (edge["source"], edge["target"]) == (1, 0)


def test_pc_prunes_indirect_common_parent_association():
    rng = np.random.default_rng(23)
    parent = rng.binomial(1, 0.5, 20000)
    left = parent ^ rng.binomial(1, 0.15, len(parent))
    right = parent ^ rng.binomial(1, 0.15, len(parent))
    y = np.column_stack([left, right, parent])
    config = {
        "max_conditioning": 1,
        "max_conditioning_neighbors": 3,
        "min_expected": 3,
        "alpha": 0.001,
    }
    selected, diagnostics = pc_skeleton(y, {(0, 1), (0, 2), (1, 2)}, config)
    assert (0, 1) not in selected
    assert {(0, 2), (1, 2)} <= selected
    assert diagnostics[(0, 1)]["separation_set"] == [2]


def test_bic_is_acyclic_and_parent_bounded():
    import networkx as nx

    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.4, (500, 4))
    y[:, 1] = y[:, 0] ^ rng.binomial(1, 0.1, len(y))
    candidates = {(i, j) for i in range(4) for j in range(i + 1, 4)}
    edges, score = bic_dag(y, candidates, max_parents=1, max_steps=20)
    graph = nx.DiGraph(list(edges))
    assert nx.is_directed_acyclic_graph(graph)
    assert max(dict(graph.in_degree()).values()) <= 1
    assert np.isfinite(score)


def test_conditional_gate_uses_context_only():
    model = {
        "scaler_mean": [0.0],
        "scaler_scale": [1.0],
        "gate_dimensions": 1,
        "heads": {
            "0": {"neighbors": [1], "base_coefficients": [0.3], "gating_coefficients": [[0.2]]}
        },
    }
    values = edge_gates(np.array([[0.0], [2.0]]), model, [{"source": 0, "target": 1}])
    np.testing.assert_allclose(values[:, 0], [0.3, 0.7])


def test_html_is_offline_and_escapes_embedded_descriptions(tmp_path):
    from deep_pharma.graph.visualization import generate_dashboard

    payload = {
        "metadata": {"config": {"bootstrap": {"stability_default": 0.6}}},
        "nodes": [{"id": 0, "effect": "</script><script>alert(1)</script>"}],
        "edges": [],
        "documents": {},
        "pairs": [{"id": "p", "effects": [], "documents": []}],
    }
    (tmp_path / "dashboard_data.json").write_text(json.dumps(payload), encoding="utf-8")
    html = generate_dashboard(tmp_path).read_text(encoding="utf-8")
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script>" in html
    assert "<script src=" not in html
    assert "{{DATA}}" not in html
    assert "{{DESIGN_CSS}}" not in html
    assert 'id="labelSize"' in html
    assert 'id="readableNodeList"' in html
    assert (tmp_path / "effect_graph_report.html").exists()
