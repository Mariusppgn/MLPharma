"""Descriptive sensitivity diagnostics, never used to select predictive models."""

from collections import Counter

import numpy as np

from .screening import contingency


def annotation_diagnostics(payload):
    pairs, nodes = payload["pairs"], payload["nodes"]
    y = np.zeros((len(pairs), len(nodes)), dtype=np.int8)
    for row, pair in enumerate(pairs):
        y[row, pair["effects"]] = 1
    counts = y.sum(axis=1)
    active = counts > 0
    comparisons = []
    documentation = []
    for pair in pairs:
        available = sum(
            value is not None and bool(str(value).strip())
            for doc in pair["documents"]
            for value in payload["documents"][doc]["text"].values()
        )
        documentation.append({"pair_id": pair["id"], "available_text_fields": available})
    for edge in payload["edges"]:
        if edge["method"] != "marginal":
            continue
        i, j = edge["source"], edge["target"]
        restricted = contingency(y[active, i], y[active, j]) if active.any() else None
        comparisons.append(
            {
                "source": i,
                "target": j,
                "phi_all": edge["phi"],
                "phi_active_only": restricted["phi"] if restricted else None,
            }
        )
    return {
        "pairs_without_annotation": int((~active).sum()),
        "pairs_with_annotation": int(active.sum()),
        "mean_effects_active_pair": float(counts[active].mean()) if active.any() else None,
        "annotation_count_distribution": [
            {"effect_count": int(k), "pairs": int(v)} for k, v in sorted(Counter(counts).items())
        ],
        "marginal_sensitivity_active_only": comparisons,
        "least_documented_pairs": sorted(
            documentation, key=lambda row: row["available_text_fields"]
        )[:20],
        "documentation_note": (
            "Availability of text fields, not assessed accuracy or clinical completeness"
        ),
        "warning": (
            "An all-zero block can induce strong marginal association. Active-only conditioning "
            "can itself induce selection/collider bias; this comparison is descriptive, not causal."
        ),
    }
