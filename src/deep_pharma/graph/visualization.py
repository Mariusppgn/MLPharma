from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..submission_gate import write_json
from .diagnostics import annotation_diagnostics
from .graph_analysis import analyze_graph


def generate_dashboard(run_dir):
    """One offline Cytoscape renderer, used for both explorer and report entry points."""
    run_dir = Path(run_dir)
    assets = Path(__file__).parent / "assets"
    vendor = assets / "cytoscape.min.js"
    if not vendor.exists():
        raise FileNotFoundError("Run scripts/vendor_effect_graph.py once")
    payload = json.loads((run_dir / "dashboard_data.json").read_text(encoding="utf-8"))
    # Centralities use the same global selection and RMS strength as the default view.
    analysis_edges = [
        {**edge, "weight": edge.get("weight_strength", abs(edge["weight"]))}
        for edge in payload["edges"]
        if edge["selected_full"]
    ]
    payload["metadata"]["analysis"] = analyze_graph(
        payload["nodes"],
        analysis_edges,
        "neighborhood",
        payload["metadata"]["config"]["bootstrap"]["stability_default"],
    )
    payload["metadata"]["centrality_weight"] = "RMS conditional slope over all training pairs"
    pd.DataFrame(payload["nodes"]).to_csv(run_dir / "nodes.csv", index=False)
    write_json(payload["metadata"], run_dir / "graph_metadata.json")
    diagnostics = annotation_diagnostics(payload)
    payload["annotation_diagnostics"] = diagnostics
    write_json(diagnostics, run_dir / "annotation_diagnostics.json")
    write_json(payload, run_dir / "dashboard_data.json")
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Untrusted descriptions cannot close the embedded JSON script.
    serialized = (
        serialized.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    )
    template = (assets / "dashboard.html").read_text(encoding="utf-8")
    template = template.replace(
        "{{DESIGN_CSS}}", (assets / "dashboard.css").read_text(encoding="utf-8")
    )
    template = template.replace("{{CYTOSCAPE}}", vendor.read_text(encoding="utf-8"))
    template = template.replace("{{DATA}}", serialized)
    template = template.replace(
        "{{APPLICATION}}", (assets / "dashboard.js").read_text(encoding="utf-8")
    )
    for name, view in [("effect_graph.html", "graph"), ("effect_graph_report.html", "evaluation")]:
        (run_dir / name).write_text(
            template.replace("{{INITIAL_VIEW}}", json.dumps(view)), encoding="utf-8"
        )
    return run_dir / "effect_graph.html"
