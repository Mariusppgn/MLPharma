from __future__ import annotations

import hashlib
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import GroupKFold

from ..submission_gate import write_json
from ..validation import canonicalize_smiles
from .metrics import probability_metrics
from .screening import candidate_edges, marginal_graph
from .structure_learning import conditional_neighborhoods


def fit_binary_heads(x, y, config, seed, jobs, coeffects=None, adjacency=None):
    def fit_one(target):
        features = x if coeffects is None else add_neighbors(x, coeffects, adjacency[target])
        if np.unique(y[:, target]).size < 2:
            return float((y[:, target].sum() + 0.5) / (len(y) + 1))
        model = lgb.LGBMClassifier(
            n_estimators=config["baseline_trees"],
            num_leaves=config["baseline_leaves"],
            learning_rate=0.05,
            reg_lambda=5.0,
            min_child_samples=30,
            random_state=seed + target,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(features, y[:, target])
        return model

    return Parallel(n_jobs=jobs, prefer="threads")(delayed(fit_one)(j) for j in range(y.shape[1]))


def add_neighbors(x, probabilities, neighbors):
    if not neighbors:
        return x
    neighbor = probabilities[:, neighbors]
    # Fixed bounded interaction transforms preserve A/B invariance.
    context = np.log1p(np.maximum(x[:, :3], 0))
    interaction = (neighbor[:, :, None] * context[:, None, :]).reshape(len(x), -1)
    return np.hstack([x, neighbor, interaction])


def predict_heads(heads, x, coeffects=None, adjacency=None):
    columns = []
    for target, model in enumerate(heads):
        features = x if coeffects is None else add_neighbors(x, coeffects, adjacency[target])
        columns.append(
            np.full(len(x), model)
            if isinstance(model, float)
            else model.predict_proba(features)[:, 1]
        )
    return np.clip(np.column_stack(columns), 1e-6, 1 - 1e-6)


def crossfit_probabilities(x, y, groups, config, seed, jobs):
    result = np.full(y.shape, np.nan, dtype=float)
    for inner, (core, held) in enumerate(
        GroupKFold(n_splits=config["inner_splits"]).split(x, groups=groups)
    ):
        if set(groups[core]) & set(groups[held]):
            raise ValueError("Pair-group leakage in inner predictions")
        heads = fit_binary_heads(x[core], y[core], config, seed + inner, jobs)
        result[held] = predict_heads(heads, x[held])
    if not np.isfinite(result).all():
        raise ValueError("Incomplete inner OOF conditioning")
    return result


def held_bucket(key, seed, fraction):
    number = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")
    return number / 2**64 < fraction


def evaluation_splits(frame, manifest, config, seed):
    a = frame.SMILES_A.astype(str).map(canonicalize_smiles).to_numpy()
    b = frame.SMILES_B.astype(str).map(canonicalize_smiles).to_numpy()
    result = []
    if "pair_grouped" in config["protocols"]:
        for fold in config["outer_folds"]:
            held = manifest.fold.to_numpy() == fold
            result.append(
                ("pair_grouped", str(fold), np.flatnonzero(~held), np.flatnonzero(held), None)
            )
    for protocol in ["molecule_disjoint", "scaffold_disjoint"]:
        if protocol not in config["protocols"]:
            continue

        def key(smiles, protocol=protocol):
            if protocol == "molecule_disjoint":
                return smiles
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))
            return scaffold or "acyclic:" + smiles

        role_a = np.array([held_bucket(key(s), seed, config["drug_holdout_fraction"]) for s in a])
        role_b = np.array([held_bucket(key(s), seed, config["drug_holdout_fraction"]) for s in b])
        fit, held = ~role_a & ~role_b, role_a & role_b
        result.append((protocol, "holdout", np.flatnonzero(fit), np.flatnonzero(held), None))
    if "leave_one_drug_out" in config["protocols"]:
        counts = Counter(np.concatenate([a, b]))
        for molecule, _ in counts.most_common(config["leave_one_drug_count"]):
            held = (a == molecule) | (b == molecule)
            result.append(
                (
                    "leave_one_drug_out",
                    hashlib.sha256(molecule.encode()).hexdigest()[:10],
                    np.flatnonzero(~held),
                    np.flatnonzero(held),
                    molecule,
                )
            )
    return result


def evaluate_graph_prediction(frame, x, y, manifest, config, output):
    settings = config["evaluation"]
    groups = manifest.pair_group.to_numpy()
    oof_base = np.full(y.shape, np.nan)
    oof_graph = np.full(y.shape, np.nan)
    records = []
    for protocol, fold, fit, held, focal in evaluation_splits(
        frame, manifest, settings, config["seed"]
    ):
        if len(fit) < settings["min_rows"] or len(held) < settings["min_rows"]:
            records.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "status": "skipped",
                    "reason": "Insufficient pairs",
                    "fit_rows": len(fit),
                    "held_rows": len(held),
                }
            )
            continue
        print(f"Graph evaluation {protocol}/{fold}: {len(fit)} fit / {len(held)} held", flush=True)
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Canonical pair group crossed an evaluation split")
        molecules_fit = {
            canonicalize_smiles(str(s))
            for side in ("A", "B")
            for s in frame.iloc[fit]["SMILES_" + side]
        }
        molecules_held = {
            canonicalize_smiles(str(s))
            for side in ("A", "B")
            for s in frame.iloc[held]["SMILES_" + side]
        }
        if (
            protocol in ["molecule_disjoint", "scaffold_disjoint"]
            and molecules_fit & molecules_held
        ):
            raise ValueError("Molecule leakage in disjoint evaluation")
        if focal is not None and focal in molecules_fit:
            raise ValueError("Held-out focal drug present in fit")
        baseline = fit_binary_heads(x[fit], y[fit], settings, config["seed"], config["cpu_threads"])
        base_prediction = predict_heads(baseline, x[held])
        # Never condition on held-out observed effects or global fitted graph structure.
        inner = crossfit_probabilities(
            x[fit], y[fit], groups[fit], settings, config["seed"], config["cpu_threads"]
        )
        marginal = marginal_graph(y[fit])
        candidate = candidate_edges(
            marginal, config["structure"]["screening_neighbors"], y.shape[1]
        )
        sparse, _, _ = conditional_neighborhoods(y[fit], x[fit], candidate, config["structure"])
        adjacency = {
            j: sorted({b if a == j else a for a, b in sparse if j in (a, b)})
            for j in range(y.shape[1])
        }
        graph_heads = fit_binary_heads(
            x[fit], y[fit], settings, config["seed"], config["cpu_threads"], inner, adjacency
        )
        graph_prediction = predict_heads(graph_heads, x[held], base_prediction, adjacency)
        rare = y[fit].mean(axis=0) < settings["rare_prevalence"]
        baseline_metrics = probability_metrics(
            y[held], base_prediction, settings["classification_threshold"], settings["top_k"], rare
        )
        graph_metrics = probability_metrics(
            y[held], graph_prediction, settings["classification_threshold"], settings["top_k"], rare
        )
        records.append(
            {
                "protocol": protocol,
                "fold": fold,
                "status": "complete",
                "baseline": baseline_metrics,
                "graph": graph_metrics,
                "fit_rows": len(fit),
                "held_rows": len(held),
                "molecule_overlap": len(molecules_fit & molecules_held),
                "pair_group_overlap": 0,
                "focal_drug_absent_from_fit": focal not in molecules_fit,
                "structure_edges_fit_only": len(sparse),
                "labels_held_used_for_selection": False,
            }
        )
        np.savez_compressed(
            output / f"evaluation_{protocol}_{fold}.npz",
            row_indices=held,
            baseline=base_prediction,
            graph=graph_prediction,
        )
        if protocol == "pair_grouped":
            oof_base[held], oof_graph[held] = base_prediction, graph_prediction
        write_json({"records": records, "test_read": False}, output / "evaluation_progress.json")
    complete = bool(np.isfinite(oof_base).all() and np.isfinite(oof_graph).all())
    aggregate = None
    if complete:
        rare = y.mean(axis=0) < settings["rare_prevalence"]
        aggregate = {
            "baseline": probability_metrics(
                y, oof_base, settings["classification_threshold"], settings["top_k"], rare
            ),
            "graph": probability_metrics(
                y, oof_graph, settings["classification_threshold"], settings["top_k"], rare
            ),
        }
        per_effect, curves = [], {}
        for name, probabilities in [("baseline", oof_base), ("graph", oof_graph)]:
            precision, recall, _ = precision_recall_curve(y.ravel(), probabilities.ravel())
            indices = np.unique(np.linspace(0, len(recall) - 1, 160).astype(int))
            curves[name] = {
                "recall": recall[indices].tolist(),
                "precision": precision[indices].tolist(),
            }
            for target in range(y.shape[1]):
                per_effect.append(
                    {
                        "model": name,
                        "effect_index": target,
                        "positive_pairs": int(y[:, target].sum()),
                        "auprc": float(
                            average_precision_score(y[:, target], probabilities[:, target])
                        )
                        if y[:, target].sum()
                        else None,
                        "brier": float(brier_score_loss(y[:, target], probabilities[:, target])),
                    }
                )
        pd.DataFrame(per_effect).to_csv(output / "per_effect_metrics.csv", index=False)
        write_json(curves, output / "precision_recall.json")
    else:
        curves = None
    return (
        {
            "records": records,
            "pair_oof_complete": complete,
            "pair_oof": aggregate,
            "precision_recall": curves,
            "test_read": False,
            "threshold_selection": "fixed before evaluation",
            "rare_metric_note": "Null when no rare positive labels exist",
            "scope": "Independent and graph-feature LightGBM; not the retained Kaggle ensemble",
        },
        oof_base,
        oof_graph,
    )
