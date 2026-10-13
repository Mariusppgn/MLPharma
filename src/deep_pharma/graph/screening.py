from __future__ import annotations

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def benjamini_hochberg(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.minimum.accumulate(
        (values[order] * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def contingency(x, y, smoothing=0.5):
    raw = np.bincount(np.asarray(x, int) * 2 + np.asarray(y, int), minlength=4).reshape(2, 2)
    table = raw.astype(float) + smoothing
    probability = table / table.sum()
    expected = probability.sum(axis=1)[:, None] * probability.sum(axis=0)[None, :]
    mi = float(np.sum(xlogy(probability, probability / expected)))
    entropies = [
        -float(np.sum(xlogy(margin, margin)))
        for margin in [probability.sum(axis=1), probability.sum(axis=0)]
    ]
    nmi = mi / max(np.sqrt(entropies[0] * entropies[1]), 1e-15)
    a, b, c, d = table.ravel()
    phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    odds = a * d / (b * c)
    lift = probability[1, 1] / (probability[1].sum() * probability[:, 1].sum())
    raw_probability = raw / max(raw.sum(), 1)
    raw_expected = raw_probability.sum(axis=1)[:, None] * raw_probability.sum(axis=0)[None, :]
    valid = (raw_probability > 0) & (raw_expected > 0)
    raw_mi = float(
        np.sum(xlogy(raw_probability[valid], raw_probability[valid] / raw_expected[valid]))
    )
    standard_error = np.sqrt(np.sum(1 / table))
    return {
        "mi": mi,
        "nmi": nmi,
        "phi": float(phi),
        "odds_ratio": float(odds),
        "odds_ratio_low": float(np.exp(np.log(odds) - 1.96 * standard_error)),
        "odds_ratio_high": float(np.exp(np.log(odds) + 1.96 * standard_error)),
        "lift": float(lift),
        "cooccurrence": int(raw[1, 1]),
        "support": int(raw.sum()),
        "p_value": float(chi2.sf(2 * raw.sum() * raw_mi, 1)),
        "smoothing": smoothing,
    }


def marginal_graph(y):
    edges = [
        {"source": i, "target": j, **contingency(y[:, i], y[:, j])}
        for i in range(y.shape[1])
        for j in range(i + 1, y.shape[1])
    ]
    q = benjamini_hochberg([e["p_value"] for e in edges])
    for edge, value in zip(edges, q, strict=True):
        edge["q_value"] = float(value)
    return edges


def candidate_edges(edges, count, targets):
    chosen = set()
    for node in range(targets):
        neighbors = [e for e in edges if node in (e["source"], e["target"])]
        for edge in sorted(neighbors, key=lambda e: -e["mi"])[:count]:
            chosen.add((edge["source"], edge["target"]))
    return chosen


def conditional_information(y, i, j, conditioning, min_expected):
    """Discrete G² test. Sparse cells are untestable, not evidence of independence."""
    strata = np.zeros(len(y), dtype=int)
    for bit, node in enumerate(conditioning):
        strata += y[:, node].astype(int) << bit
    code = strata * 4 + y[:, i].astype(int) * 2 + y[:, j].astype(int)
    counts = np.bincount(code, minlength=4 * 2 ** len(conditioning)).reshape(-1, 2, 2)
    information, df, supported = 0.0, 0, True
    for table in counts:
        n = table.sum()
        if not n:
            continue
        expected = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :] / n
        if (expected < min_expected).any():
            supported = False
        if (expected > 0).all():
            valid = table > 0
            information += float(np.sum(xlogy(table[valid], table[valid] / expected[valid])))
            df += 1
    cmi = information / max(len(y), 1)
    return {
        "cmi": cmi,
        "p_value": float(chi2.sf(2 * information, df)) if df else None,
        "testable": supported and df > 0,
        "df": df,
    }
