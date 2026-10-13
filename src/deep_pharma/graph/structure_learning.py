from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .screening import candidate_edges, conditional_information, marginal_graph


def pc_skeleton(y, candidates, config):
    """Bounded PC-stable-style skeleton; deliberately no causal orientation."""
    remaining = set(candidates)
    diagnostics = {
        edge: {"conditional_p": None, "cmi": None, "testable": False, "separation_set": None}
        for edge in candidates
    }
    for order in range(int(config["max_conditioning"]) + 1):
        frozen = {j: set() for j in range(y.shape[1])}
        for i, j in remaining:
            frozen[i].add(j)
            frozen[j].add(i)
        removed = set()
        for i, j in sorted(remaining):
            # Frozen adjacency makes this level independent of removal order.
            neighbors = sorted((frozen[i] | frozen[j]) - {i, j})
            neighbors = neighbors[: int(config["max_conditioning_neighbors"])]
            sets = combinations(neighbors, order)
            for conditioning in sets:
                result = conditional_information(y, i, j, conditioning, config["min_expected"])
                if not result["testable"]:
                    continue
                previous = diagnostics[(i, j)]["conditional_p"]
                if previous is None or result["p_value"] > previous:
                    diagnostics[(i, j)].update(
                        conditional_p=result["p_value"], cmi=result["cmi"], testable=True
                    )
                if result["p_value"] >= config["alpha"]:
                    diagnostics[(i, j)]["separation_set"] = list(conditioning)
                    removed.add((i, j))
                    break
        remaining -= removed
    # Unpowered pairs remain unresolved, rather than being called independent or robust.
    return {edge for edge in remaining if diagnostics[edge]["testable"]}, diagnostics


def neighborhood_features(x, y, target, neighbors, gates):
    values = y[:, neighbors] if neighbors else np.empty((len(y), 0))
    interactions = (values[:, :, None] * x[:, None, :gates]).reshape(len(y), -1)
    return np.hstack([x, values, interactions])


def conditional_neighborhoods(y, x, candidates, config):
    """Sparse conditional logistic associations with feature-dependent log-odds slopes."""
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    # All pair features adjust the association; only a small block modulates each edge.
    gates = min(int(config["gate_dimensions"]), z.shape[1])
    heads, selected, weights = {}, set(), {}
    for target in range(y.shape[1]):
        neighbors = sorted({j if i == target else i for i, j in candidates if target in (i, j)})
        if not neighbors or np.unique(y[:, target]).size < 2:
            continue
        features = neighborhood_features(z, y, target, neighbors, gates)
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=config["l1_c"], max_iter=250, random_state=2026
        )
        model.fit(features, y[:, target])
        coefficients = model.coef_[0]
        base = coefficients[z.shape[1] : z.shape[1] + len(neighbors)]
        modulation = coefficients[z.shape[1] + len(neighbors) :].reshape(len(neighbors), gates)
        heads[target] = {
            "neighbors": neighbors,
            "intercept": float(model.intercept_[0]),
            "x_coefficients": coefficients[: z.shape[1]].tolist(),
            "base_coefficients": base.tolist(),
            "gating_coefficients": modulation.tolist(),
        }
        for position, neighbor in enumerate(neighbors):
            key = tuple(sorted((target, neighbor)))
            magnitude = abs(base[position]) + np.linalg.norm(modulation[position])
            if magnitude >= config["coefficient_threshold"]:
                selected.add(key)
            weights.setdefault(key, []).append(float(base[position]))
    weights = {edge: float(np.mean(values)) for edge, values in weights.items()}
    return (
        selected,
        weights,
        {
            "heads": heads,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "context_dim": z.shape[1],
            "gate_dimensions": gates,
            "meaning": "Conditional predictive log-odds, not causal effects",
        },
    )


def bic_dag(y, candidates, max_parents=2, max_steps=60):
    """Greedy add/delete BIC DAG search; factorization direction is not biological causality."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(y.shape[1]))
    cache = {}

    def score(target, parents):
        key = target, tuple(sorted(parents))
        if key not in cache:
            codes = np.zeros(len(y), dtype=int)
            for bit, parent in enumerate(key[1]):
                codes += y[:, parent].astype(int) << bit
            table = np.bincount(codes * 2 + y[:, target], minlength=2 ** (len(parents) + 1))
            table = table.reshape(-1, 2).astype(float)
            totals = table.sum(axis=1)
            probability = np.divide(
                table, totals[:, None], out=np.ones_like(table), where=totals[:, None] > 0
            )
            nonzero = table > 0
            likelihood = float(np.sum(table[nonzero] * np.log(probability[nonzero])))
            cache[key] = likelihood - 0.5 * 2 ** len(parents) * np.log(max(len(y), 2))
        return cache[key]

    for _ in range(max_steps):
        best_gain, best = 0.0, None
        for a, b in sorted(candidates):
            for source, target in [(a, b), (b, a)]:
                parents = set(graph.predecessors(target))
                if graph.has_edge(source, target):
                    gain = score(target, parents - {source}) - score(target, parents)
                    action = "delete"
                elif len(parents) < max_parents and not nx.has_path(graph, target, source):
                    gain = score(target, parents | {source}) - score(target, parents)
                    action = "add"
                else:
                    continue
                if gain > best_gain + 1e-9:
                    best_gain, best = gain, (action, source, target)
        if best is None:
            break
        action, source, target = best
        if action == "add":
            graph.add_edge(source, target)
        else:
            graph.remove_edge(source, target)
    return set(graph.edges()), float(sum(score(j, set(graph.predecessors(j))) for j in graph))


def learn_graphs(y, x, config):
    marginal = marginal_graph(y)
    candidates = candidate_edges(marginal, config["screening_neighbors"], y.shape[1])
    pc, conditional_tests = pc_skeleton(y, candidates, config)
    sparse, weights, gate_model = conditional_neighborhoods(y, x, candidates, config)
    dag, bic = bic_dag(y, candidates, config["bic_max_parents"], config["bic_max_steps"])
    return {
        "marginal": marginal,
        "candidates": candidates,
        "pc": pc,
        "conditional_tests": conditional_tests,
        "neighborhood": sparse,
        "weights": weights,
        "gate_model": gate_model,
        "bic": dag,
        "bic_score": bic,
    }
