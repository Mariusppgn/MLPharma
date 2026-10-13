from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from ..constants import ID_COLUMN
from ..nested_cpu_optimization import _Tee
from ..pk_features import extract_cyp_features
from ..submission_gate import write_json
from ..validation import audit_pair_grouped_folds, split_fingerprint
from .bootstrap import bootstrap_graphs
from .conditional_graph import edge_gates
from .evaluation import evaluate_graph_prediction
from .graph_analysis import analyze_graph, nonadditive_motifs
from .preprocessing import (
    chemical_properties,
    cluster_molecules,
    molecule_id,
    pair_features,
    pharmacology_text,
    profile_data,
    tanimoto,
)
from .structure_learning import learn_graphs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _set_status(output, stage, state="running"):
    write_json(
        {"state": state, "stage": stage, "updated_utc": datetime.now(UTC).isoformat()},
        output / "status.json",
    )


def edge_records(graphs, bootstrap, iterations):
    marginal = {(edge["source"], edge["target"]): edge for edge in graphs["marginal"]}
    result = []
    for method in ("marginal", "pc", "neighborhood", "bic"):
        fitted = graphs["candidates"] if method == "marginal" else graphs[method]
        undirected = {tuple(sorted(edge)) for edge in fitted}
        union = undirected | {(i, j) for m, i, j in bootstrap if m == method}
        if method == "marginal":
            union = set(marginal)
        for i, j in sorted(union):
            source, target = i, j
            if method == "bic" and (j, i) in fitted:
                source, target = j, i
            statistics = bootstrap.get(
                (method, i, j),
                {
                    "stability": 0.0,
                    "bootstrap_selected": 0,
                    "bootstrap_total": iterations,
                    "weight_low": None,
                    "weight_high": None,
                    "stability_mc_low": None,
                    "stability_mc_high": None,
                },
            )
            weight = (
                graphs["weights"].get((i, j), 0.0)
                if method == "neighborhood"
                else marginal[(i, j)]["phi"]
            )
            conditional = graphs["conditional_tests"].get((i, j), {})
            result.append(
                {
                    "id": f"{method}_{i}_{j}",
                    "method": method,
                    **marginal[(i, j)],
                    "source": source,
                    "target": target,
                    "directed": method == "bic",
                    "weight": weight,
                    "selected_full": (i, j) in undirected,
                    **statistics,
                    "conditional_mi": conditional.get("cmi"),
                    "conditional_p": conditional.get("conditional_p"),
                    "conditional_testable": conditional.get("testable", False),
                    "clinical_cases": None,
                }
            )
    return result


def _dashboard_pairs(frame, y, prr, x, edges, gate_model, clusters, config, base_oof, graph_oof):
    molecules, documents, pairs = {}, {}, []
    conditional_edges = [e for e in edges if e["method"] == "neighborhood"]
    gated = edge_gates(x, gate_model, conditional_edges)
    for position, edge in enumerate(conditional_edges):
        edge["weight_strength"] = float(np.sqrt(np.mean(gated[:, position] ** 2)))
    for row in range(len(frame)):
        ids, document_ids = [], []
        for side in ("A", "B"):
            smiles = str(frame.at[row, "SMILES_" + side])
            identity = molecule_id(smiles)
            ids.append(identity)
            if identity not in molecules:
                molecules[identity] = {
                    "id": identity,
                    "smiles": smiles,
                    "label": identity,
                    "cluster": clusters[identity],
                    **chemical_properties(smiles),
                }
            text = pharmacology_text(frame, row, side, config["dashboard"]["metadata_text_limit"])
            full_cyp = " ".join(
                str(frame.at[row, field + "_" + side]) for field in ["CYP450_Enzymes", "Metabolism"]
            )
            cyp = extract_cyp_features(full_cyp)
            key = hashlib.sha256(json.dumps([text, cyp], sort_keys=True).encode()).hexdigest()[:14]
            documents.setdefault(key, {"text": text, "cyp_heuristic": cyp})
            document_ids.append(key)
        observed = np.flatnonzero(y[row]).tolist()
        positive_prr = {str(j): round(float(prr[row, j]), 5) for j in np.flatnonzero(prr[row] > 0)}
        pairs.append(
            {
                "id": str(frame.at[row, ID_COLUMN]),
                "a": ids[0],
                "b": ids[1],
                "documents": document_ids,
                "effects": observed,
                "prr": positive_prr,
                "similarity": round(
                    tanimoto(frame.at[row, "SMILES_A"], frame.at[row, "SMILES_B"]), 5
                ),
                "edge_weights": np.round(gated[row], 4).tolist(),
                "baseline_oof": np.round(base_oof[row], 4).tolist()
                if np.isfinite(base_oof[row]).all()
                else None,
                "graph_oof": np.round(graph_oof[row], 4).tolist()
                if np.isfinite(graph_oof[row]).all()
                else None,
            }
        )
    return molecules, documents, pairs, [edge["id"] for edge in conditional_edges]


def _execute(config, output):
    _set_status(output, "profile")
    train = pd.read_csv(config["train"], low_memory=False)
    manifest = pd.read_csv(config["manifest"]).sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    if not audit["passed"]:
        raise ValueError(audit["errors"])
    y, prr, nodes, profile = profile_data(train, manifest.pair_group.to_numpy(), config)
    write_json(profile, output / "data_profile.json")
    print(f"Graph research data: {len(train)} pairs, {len(nodes)} effects", flush=True)
    _set_status(output, "features")
    feature_signature = hashlib.sha256(
        json.dumps(config["features"], sort_keys=True).encode()
    ).hexdigest()
    signature = {"train_sha256": sha256(config["train"]), "features": feature_signature}
    cache = output / "features.npz"
    if cache.exists() and json.loads((output / "feature_signature.json").read_text()) == signature:
        x = np.load(cache)["x"]
    else:
        x = pair_features(train, config["features"])
        np.savez_compressed(cache, x=x)
        write_json(signature, output / "feature_signature.json")
    swapped = train.iloc[:5].copy()
    columns = {
        c: c[:-1] + ("B" if c.endswith("_A") else "A") for c in swapped if c.endswith(("_A", "_B"))
    }
    swapped = swapped.rename(columns=columns)[train.columns]
    if not np.allclose(x[:5], pair_features(swapped, config["features"]), atol=1e-6):
        raise ValueError("Pair features fail A/B invariance")
    _set_status(output, "structure")
    graphs = learn_graphs(y, x, config["structure"])
    _set_status(output, "bootstrap")
    bootstrap = bootstrap_graphs(
        y,
        x,
        manifest.pair_group.to_numpy(),
        config["structure"],
        config["bootstrap"]["iterations"],
        config["seed"],
        output,
    )
    edges = edge_records(graphs, bootstrap, config["bootstrap"]["iterations"])
    evaluation = {"status": "disabled", "pair_oof_complete": False}
    base_oof, graph_oof = np.full(y.shape, np.nan), np.full(y.shape, np.nan)
    if config["evaluation"]["enabled"]:
        _set_status(output, "predictive_evaluation")
        evaluation, base_oof, graph_oof = evaluate_graph_prediction(
            train, x, y, manifest, config, output
        )
    write_json(evaluation, output / "evaluation.json")
    _set_status(output, "dashboard")
    clusters = cluster_molecules(train, config["features"]["molecule_clusters"], config["seed"])
    cluster_graphs = []
    for cluster in sorted(set(clusters.values())):
        mask = np.array(
            [
                clusters[molecule_id(a)] == cluster or clusters[molecule_id(b)] == cluster
                for a, b in zip(train.SMILES_A, train.SMILES_B, strict=True)
            ]
        )
        if mask.sum() < 40:
            continue
        local = learn_graphs(y[mask], x[mask], config["structure"])
        reference = set(graphs["neighborhood"])
        selected = set(local["neighborhood"])
        cluster_graphs.append(
            {
                "cluster": cluster,
                "pairs": int(mask.sum()),
                "edges": [
                    {"source": i, "target": j, "weight": local["weights"][(i, j)]}
                    for i, j in sorted(selected)
                ],
                "jaccard_global": len(reference & selected) / max(len(reference | selected), 1),
                "scope": (
                    "Exploratory fit; overlapping populations; no local bootstrap or OOF gain claim"
                ),
            }
        )
    write_json(cluster_graphs, output / "cluster_graphs.json")
    molecules, documents, pairs, conditional_ids = _dashboard_pairs(
        train, y, prr, x, edges, graphs["gate_model"], clusters, config, base_oof, graph_oof
    )
    analysis = analyze_graph(nodes, edges, "neighborhood", config["bootstrap"]["stability_default"])
    motifs = nonadditive_motifs(y, edges)
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "train_sha256": signature["train_sha256"],
        "validation_fingerprint": split_fingerprint(manifest),
        "config": config,
        "symmetry_sample_passed": True,
        "time_present": False,
        "clinical_case_counts": False,
        "test_read": False,
        "causal_identification": False,
        "bootstrap_unit": "canonical pair group",
        "bootstrap_limit": (
            "Shared molecules remain dependent; stability is not causal confidence."
        ),
        "pc_limit": "Screened, bounded PC-stable-style skeleton; no full CPDAG claim.",
        "bic_limit": "Greedy BIC add/delete search, not an optimal DAG or causal orientation.",
        "pharmacology": (
            "CYP roles are heuristic text extractions, not curated validated annotations."
        ),
        "probability_scope": "Probability of dataset annotation, not clinical event incidence.",
        "unavailable": ["clinical_cases", "time", "ATC", "MedDRA", "SOC", "Ki", "IC50", "EC50"],
        "methods": {
            "marginal": "MI/phi/OR/lift; candidate-screen stability",
            "pc": "bounded conditional G2 pruning",
            "neighborhood": "L1 conditional logistic + feature-dependent edge gating",
            "bic": "binary Bayesian-network greedy BIC",
        },
        "analysis": analysis,
        "references": [
            "https://www.jmlr.org/papers/v15/colombo14a.html",
            "https://arxiv.org/abs/0809.2932",
            "https://js.cytoscape.org/",
        ],
    }
    for name, rows in [("nodes", nodes), ("edges", edges)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    write_json(graphs["gate_model"], output / "conditional_model.json")
    write_json(metadata, output / "graph_metadata.json")
    payload = {
        "metadata": metadata,
        "profile": profile,
        "nodes": nodes,
        "edges": edges,
        "molecules": molecules,
        "documents": documents,
        "pairs": pairs,
        "conditional_edge_ids": conditional_ids,
        "motifs": motifs,
        "evaluation": evaluation,
        "cluster_graphs": cluster_graphs,
    }
    write_json(payload, output / "dashboard_data.json")
    from .visualization import generate_dashboard

    generate_dashboard(output)
    _set_status(output, "complete", "complete")
    print(f"Effect graph ready: {output / 'effect_graph.html'}", flush=True)
    return metadata


def run_effect_graph(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config["bootstrap"].get("unit") != "pair_group":
        raise ValueError("Only canonical pair-group bootstrap is implemented")
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "configuration.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != config:
        raise ValueError("Configuration changed: choose a new run_dir")
    write_json(config, frozen)
    if (output / "graph_metadata.json").exists():
        raise FileExistsError("Completed graph run exists; regenerate HTML or choose a new run_dir")
    with (output / "run.log").open("a", encoding="utf-8") as log:
        with (
            contextlib.redirect_stdout(_Tee(sys.stdout, log)),
            threadpool_limits(config["cpu_threads"]),
        ):
            try:
                return _execute(config, output)
            except Exception:
                _set_status(output, "failed", "failed")
                traceback.print_exc(file=log)
                raise
