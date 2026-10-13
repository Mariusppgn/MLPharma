"""Overlapping descriptive categories; missing detection never counts as absence."""


def classify(cells, detected, *, individual_valid, valid, delta, margin):
    a, b, ab = cells
    categories = []
    if individual_valid and detected[0] is not None and detected[1] is not None:
        if detected[0] and detected[1]:
            categories.append("shared")
        elif detected[0]:
            categories.append("a_specific")
        elif detected[1]:
            categories.append("b_specific")
        if valid and not detected[0] and not detected[1] and detected[2]:
            categories.append("emerging")
    if all(d is not None for d in detected):
        if delta is not None:
            if delta > margin:
                categories.append("strengthened")
            elif delta < -margin:
                categories.append("attenuated")
        elif valid and a["kind"] == "prr":
            if ab["value"] > max(a["value"], b["value"]) + margin:
                categories.append("strengthened")
            elif ab["value"] < min(a["value"], b["value"]) - margin:
                categories.append("attenuated")
    if a["kind"] == b["kind"] == "context_frequency":
        if detected[0] is not None and detected[1] is not None:
            if detected[0] and detected[1]:
                categories.append("context_shared")
            elif detected[0]:
                categories.append("context_a")
            elif detected[1]:
                categories.append("context_b")
    return categories
