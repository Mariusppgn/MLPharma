"""Nullable metrics: an annotation, a probability and a PRR are different objects."""

from __future__ import annotations

import math
from collections import Counter

from .classification import classify

CONTEXT_KEYS = ("comparability_id", "meaning", "population", "period")


def number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def cell(value=None, *, kind="missing", source="", **metadata) -> dict:
    result = number(value)
    if kind in ("probability", "annotation", "context_frequency"):
        if result is not None and (result > 1 or (kind == "annotation" and result not in (0, 1))):
            raise ValueError(f"Valeur invalide pour {kind}")
    return {"value": result, "kind": kind, "source": source, **metadata}


def comparable(cells: list[dict]) -> bool:
    return (
        all(
            cells[0].get(key) and all(c.get(key) == cells[0][key] for c in cells)
            for key in CONTEXT_KEYS
        )
        and len({c["kind"] for c in cells}) == 1
    )


def presence(c: dict, threshold: float, min_support: int = 0) -> bool | None:
    value = c["value"]
    if value is None or (min_support and (c.get("n") is None or c["n"] < min_support)):
        return None
    # PRR=0 is recorded non-detection, not a clinical absence or a significance test.
    return value > threshold if c["kind"] == "prr" else value >= threshold


def compare_effect(
    effect: str,
    a: dict,
    b: dict,
    ab: dict,
    *,
    threshold=0.5,
    margin=0.05,
    min_support=0,
    independence=False,
) -> dict:
    cells = [a, b, ab]
    available = all(c["value"] is not None for c in cells)
    valid = comparable(cells)
    reference = delta = None
    reason = "Profils individuels ou profil du couple manquants."
    if available:
        reason = "Échelles, populations, périodes ou significations non comparables."
        if valid and a["kind"] == "probability":
            reason = "Référence indépendante non autorisée : hypothèse à accepter explicitement."
            if independence:
                reference = 1 - (1 - a["value"]) * (1 - b["value"])
                delta = ab["value"] - reference
                reason = "Référence exploratoire, non causale ; delta en points de pourcentage."
        elif a["kind"] == "prr":
            reason = "Un PRR n'est pas une probabilité : aucune référence d'union."
        else:
            reason = "Annotations/fréquences de couples : pas de probabilités individuelles."
    detected = [presence(c, threshold, min_support) for c in cells]
    individual_valid = comparable([a, b]) and a["kind"] != "context_frequency"
    categories = classify(
        cells, detected, individual_valid=individual_valid, valid=valid, delta=delta, margin=margin
    )
    raw_deltas = log_ratios = None
    if valid and available and a["kind"] == "prr":
        raw_deltas = [ab["value"] - a["value"], ab["value"] - b["value"]]
        if all(c["value"] > 0 for c in cells):
            log_ratios = [math.log(ab["value"] / a["value"]), math.log(ab["value"] / b["value"])]
    return {
        "effect": effect,
        "a": a,
        "b": b,
        "ab": ab,
        "reference": reference,
        "delta": delta,
        "reason": reason,
        "categories": categories,
        "detected": detected,
        "prr_deltas": raw_deltas,
        "prr_log_ratios": log_ratios,
    }


def filter_rows(rows: list[dict], filters: dict) -> list[dict]:
    result = []
    for r in rows:
        haystack = f"{r['effect']} {r.get('label', '')}".casefold()
        if filters.get("search", "").casefold() not in haystack:
            continue
        if filters.get("category") and filters["category"] not in r["categories"]:
            continue
        value = r["ab"]["value"]
        if filters.get("minimum") is not None and (value is None or value < filters["minimum"]):
            continue
        if filters.get("maximum") is not None and (value is None or value > filters["maximum"]):
            continue
        sign = filters.get("delta_sign")
        if sign and (
            r["delta"] is None
            or (sign == "positive" and r["delta"] <= 0)
            or (sign == "negative" and r["delta"] >= 0)
        ):
            continue
        if filters.get("soc") and r.get("soc") != filters["soc"]:
            continue
        if filters.get("pt") and r.get("pt") != filters["pt"]:
            continue
        n = r["ab"].get("n")
        if filters.get("minimum_n") and (n is None or n < filters["minimum_n"]):
            continue
        ci = r["ab"].get("ci")
        if filters.get("max_ci_width") is not None:
            if ci is None or ci[1] - ci[0] > filters["max_ci_width"]:
                continue
        result.append(r)
    sort = filters.get("sort", "name")
    result.sort(
        key=(lambda r: -(r["ab"]["value"] or 0))
        if sort == "value"
        else (lambda r: -abs(r["delta"] or 0))
        if sort == "delta"
        else lambda r: r.get("label", r["effect"])
    )
    return result


def summarize(rows: list[dict]) -> dict:
    counts = Counter(c for r in rows for c in r["categories"])
    detected = []
    for i in range(3):
        values = [r["detected"][i] for r in rows]
        detected.append(sum(values) if values and all(v is not None for v in values) else None)
    known = bool(rows) and all(
        r["detected"][0] is not None
        and r["detected"][1] is not None
        and comparable([r["a"], r["b"]])
        for r in rows
    )
    union = sum(bool(r["detected"][0] or r["detected"][1]) for r in rows) if known else 0
    intersection = sum(bool(r["detected"][0] and r["detected"][1]) for r in rows) if known else 0
    deltas = [r for r in rows if r["delta"] is not None]
    return {
        "rows": len(rows),
        "detected": detected,
        "categories": dict(counts),
        "jaccard": intersection / union if union else None,
        "jaccard_reason": "Sur effets filtrés, au seuil affiché ; contexte ≠ monothérapie.",
        "top_positive": [
            r["effect"] for r in sorted(deltas, key=lambda r: -r["delta"]) if r["delta"] > 0
        ][:5],
        "top_negative": [
            r["effect"] for r in sorted(deltas, key=lambda r: r["delta"]) if r["delta"] < 0
        ][:5],
    }
