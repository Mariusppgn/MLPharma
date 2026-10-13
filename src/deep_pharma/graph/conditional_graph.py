from __future__ import annotations

import numpy as np


def edge_gates(x, gate_model, edges):
    """Conditional mean log-odds slopes; no observation's effect label is an input."""
    mean, scale = np.asarray(gate_model["scaler_mean"]), np.asarray(gate_model["scaler_scale"])
    z = (x - mean) / scale
    dimensions = gate_model["gate_dimensions"]
    output = np.zeros((len(x), len(edges)), dtype=np.float32)
    for position, edge in enumerate(edges):
        values = []
        for target, neighbor in [
            (edge["source"], edge["target"]),
            (edge["target"], edge["source"]),
        ]:
            head = gate_model["heads"].get(target, gate_model["heads"].get(str(target)))
            if head is None or neighbor not in head["neighbors"]:
                continue
            index = head["neighbors"].index(neighbor)
            values.append(
                head["base_coefficients"][index]
                + z[:, :dimensions] @ np.asarray(head["gating_coefficients"][index])
            )
        if values:
            output[:, position] = np.mean(values, axis=0)
    return output
