from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from scipy.special import logit


def analyze_graph(nodes, edges, method, stability):
    graph = nx.Graph()
    graph.add_nodes_from(n["id"] for n in nodes)
    for edge in edges:
        if edge["method"] == method and edge["stability"] >= stability:
            strength = abs(edge["weight"])
            graph.add_edge(
                edge["source"],
                edge["target"],
                weight=max(strength, 1e-8),
                distance=1 / max(strength, 1e-8),
            )
    communities = (
        list(nx.community.greedy_modularity_communities(graph, weight="weight"))
        if graph.number_of_edges()
        else [{node} for node in graph]
    )
    mapping = {node: i for i, community in enumerate(communities) for node in community}
    measures = {
        "degree_centrality": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph, weight="distance"),
        "closeness": nx.closeness_centrality(graph, distance="distance"),
        "pagerank": nx.pagerank(graph, weight="weight"),
    }
    for node in nodes:
        j = node["id"]
        node.update({name: float(values[j]) for name, values in measures.items()})
        node.update(
            degree=int(graph.degree(j)),
            weighted_degree=float(graph.degree(j, weight="weight")),
            community=mapping[j],
        )
    return {
        "nodes": len(graph),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "connected_components": nx.number_connected_components(graph),
        "communities": [sorted(c) for c in communities],
        "triangles": int(sum(nx.triangles(graph).values()) // 3),
        "top_hubs": sorted(nodes, key=lambda n: -n["weighted_degree"])[:10],
    }


def nonadditive_motifs(y, edges, maximum=30, minimum=20):
    """Smoothed four-cell logit interaction; exploratory, not synergistic causality."""
    neighbors = {j: set() for j in range(y.shape[1])}
    for edge in edges:
        if edge["method"] == "neighborhood" and edge["stability"] >= 0.5:
            neighbors[edge["source"]].add(edge["target"])
            neighbors[edge["target"]].add(edge["source"])
    results = []
    for target, adjacent in neighbors.items():
        for left, right in combinations(sorted(adjacent)[:6], 2):
            cells = y[:, left] * 2 + y[:, right]
            counts = np.bincount(cells, minlength=4)
            if counts.min() < minimum:
                continue
            positive = np.bincount(cells, weights=y[:, target], minlength=4)
            probability = (positive + 0.5) / (counts + 1)
            logits = logit(probability)
            contrast = float(logits[3] - logits[2] - logits[1] + logits[0])
            variance = float(np.sum(1 / (positive + 0.5) + 1 / (counts - positive + 0.5)))
            results.append(
                {
                    "left": left,
                    "right": right,
                    "target": target,
                    "logit_interaction": contrast,
                    "approx_low": contrast - 1.96 * np.sqrt(variance),
                    "approx_high": contrast + 1.96 * np.sqrt(variance),
                    "minimum_cell_pairs": int(counts.min()),
                    "meaning": "Exploratory unadjusted interaction, no causal claim",
                }
            )
    return sorted(results, key=lambda r: -abs(r["logit_interaction"]))[:maximum]
