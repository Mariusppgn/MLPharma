from __future__ import annotations

import numpy as np
import pandas as pd

from .structure_learning import learn_graphs


def grouped_resample(groups, rng):
    unique = np.unique(groups)
    lookup = {group: np.flatnonzero(groups == group) for group in unique}
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([lookup[group] for group in chosen])


def bootstrap_graphs(y, x, groups, structure, iterations, seed, output):
    """Pair-group bootstrap measures selection stability, not causal truth."""
    if iterations < 2:
        raise ValueError("At least two bootstrap replicates required")
    rng = np.random.default_rng(seed)
    rows, selected, weight_samples = [], {}, {}
    for repeat in range(iterations):
        indices = grouped_resample(groups, rng)
        graphs = learn_graphs(y[indices], x[indices], structure)
        marginal_weights = {
            tuple(sorted((int(row["source"]), int(row["target"])))): row["phi"]
            for row in graphs["marginal"]
        }
        for method in ("marginal", "pc", "neighborhood", "bic"):
            collection = graphs["candidates"] if method == "marginal" else graphs[method]
            for source, target in collection:
                edge = tuple(sorted((source, target)))
                key = method, *edge
                selected[key] = selected.get(key, 0) + 1
                weight = (
                    graphs["weights"].get(edge, 0)
                    if method == "neighborhood"
                    else marginal_weights[edge]
                )
                weight_samples.setdefault(key, []).append(weight)
                rows.append(
                    {
                        "bootstrap": repeat,
                        "method": method,
                        "source": source,
                        "target": target,
                        "weight": weight,
                    }
                )
        pd.DataFrame(rows).to_csv(output / "bootstrap_edges.csv", index=False)
        print(f"Effect graph bootstrap {repeat + 1}/{iterations}", flush=True)
    summaries = {}
    for key, count in selected.items():
        probability = count / iterations
        # Wilson interval is Monte Carlo uncertainty in selection frequency only.
        denominator = 1 + 1.96**2 / iterations
        center = (probability + 1.96**2 / (2 * iterations)) / denominator
        half = (
            1.96
            * np.sqrt(probability * (1 - probability) / iterations + 1.96**2 / (4 * iterations**2))
            / denominator
        )
        values = weight_samples[key] + [0.0] * (iterations - count)
        summaries[key] = {
            "stability": probability,
            "stability_mc_low": max(0, center - half),
            "stability_mc_high": min(1, center + half),
            "weight_low": float(np.quantile(values, 0.025)),
            "weight_high": float(np.quantile(values, 0.975)),
            "bootstrap_selected": count,
            "bootstrap_total": iterations,
        }
    return summaries
