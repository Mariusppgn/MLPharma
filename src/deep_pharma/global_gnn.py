"""End-to-end, fold-isolated molecular GNN for the three competition tasks."""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score
from torch import nn

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .metric import hardcore_score
from .rupture_benchmark import _atom_features
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, canonicalize_smiles, split_fingerprint


@dataclass
class GraphBank:
    atoms: torch.Tensor
    edges: torch.Tensor
    bonds: torch.Tensor
    graph: torch.Tensor
    counts: torch.Tensor
    lookup: dict[str, int]
    truncated: int


def build_graph_bank(smiles: list[str], max_atoms: int, device: torch.device) -> GraphBank:
    molecules = sorted(set(smiles))
    lookup = {smile: index for index, smile in enumerate(molecules)}
    atom_parts: list[np.ndarray] = []
    source: list[int] = []
    target: list[int] = []
    bonds: list[list[float]] = []
    graph: list[int] = []
    counts: list[int] = []
    truncated = 0
    offset = 0
    for graph_id, smile in enumerate(molecules):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"SMILES invalide : {smile!r}")
        size = min(mol.GetNumAtoms(), max_atoms)
        if size == 0:
            raise ValueError(f"Molécule sans atomes : {smile!r}")
        truncated += mol.GetNumAtoms() > max_atoms
        atom_parts.extend(_atom_features(mol.GetAtomWithIdx(i)) for i in range(size))
        graph.extend([graph_id] * size)
        counts.append(size)
        for bond in mol.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left >= size or right >= size:
                continue
            kind = bond.GetBondType()
            attribute = [
                float(kind == Chem.BondType.SINGLE),
                float(kind == Chem.BondType.DOUBLE),
                float(kind == Chem.BondType.TRIPLE),
                float(kind == Chem.BondType.AROMATIC),
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
            ]
            source.extend((offset + left, offset + right))
            target.extend((offset + right, offset + left))
            bonds.extend((attribute, attribute))
        offset += size
    edge_array = np.asarray([source, target], dtype=np.int64).reshape(2, -1)
    bond_array = np.asarray(bonds, dtype=np.float32).reshape(-1, 6)
    return GraphBank(
        atoms=torch.as_tensor(np.asarray(atom_parts), dtype=torch.float32, device=device),
        edges=torch.as_tensor(edge_array, dtype=torch.long, device=device),
        bonds=torch.as_tensor(bond_array, dtype=torch.float32, device=device),
        graph=torch.as_tensor(graph, dtype=torch.long, device=device),
        counts=torch.as_tensor(counts, dtype=torch.float32, device=device),
        lookup=lookup,
        truncated=truncated,
    )


def _text_side(frame: pd.DataFrame, suffix: str) -> list[str]:
    columns = sorted(
        column for column in frame if column.endswith(suffix) and column != f"SMILES{suffix}"
    )
    return [
        " ".join(f"{column[:-2]} {str(value)}" for column, value in zip(columns, row, strict=True))
        for row in frame[columns].fillna("").itertuples(index=False, name=None)
    ]


def text_pair_features(frame: pd.DataFrame, n_features: int) -> np.ndarray:
    """Stateless hashing: no vocabulary or target statistics can cross folds."""
    vectorizer = HashingVectorizer(
        n_features=n_features, alternate_sign=False, ngram_range=(1, 2), norm="l2"
    )
    left = vectorizer.transform(_text_side(frame, "_A")).toarray().astype(np.float32)
    right = vectorizer.transform(_text_side(frame, "_B")).toarray().astype(np.float32)
    return np.concatenate([left + right, np.abs(left - right), left * right], axis=1)


class BondMessageLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.edge = nn.Linear(6, hidden, bias=False)
        self.update = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, bank: GraphBank) -> torch.Tensor:
        src, dst = bank.edges
        messages = torch.nn.functional.silu(nodes[src] + self.edge(bank.bonds))
        aggregate = torch.zeros_like(nodes).index_add_(0, dst, messages)
        degrees = torch.zeros((len(nodes), 1), device=nodes.device).index_add_(
            0, dst, torch.ones((len(dst), 1), device=nodes.device)
        )
        return self.norm(nodes + self.update(aggregate / degrees.clamp_min(1)))


class GlobalGNN(nn.Module):
    def __init__(
        self, n_molecules: int, n_targets: int, n_severity: int, text_dim: int, cfg: dict
    ) -> None:
        super().__init__()
        hidden = int(cfg["hidden_dim"])
        dropout = float(cfg["dropout"])
        self.atom_input = nn.Linear(12, hidden)
        self.layers = nn.ModuleList(
            BondMessageLayer(hidden, dropout) for _ in range(int(cfg["gnn_layers"]))
        )
        self.graph_project = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())
        self.text_project = nn.Sequential(nn.Linear(text_dim, hidden), nn.SiLU())
        self.identity = nn.Embedding(n_molecules + 1, hidden, padding_idx=0)
        self.identity_dropout = float(cfg["identity_dropout"])
        self.identity_limit = float(cfg["identity_limit"])
        self.identity_gate = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.trunk = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden),
            nn.LayerNorm(2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        )
        self.severity = nn.Linear(hidden, n_severity)
        self.binary = nn.Linear(hidden, n_targets)
        self.prr = nn.Linear(hidden, n_targets)

    def encode(self, bank: GraphBank) -> torch.Tensor:
        nodes = torch.nn.functional.silu(self.atom_input(bank.atoms))
        for layer in self.layers:
            nodes = layer(nodes, bank)
        n_graphs = len(bank.counts)
        mean = (
            torch.zeros((n_graphs, nodes.shape[1]), device=nodes.device).index_add_(
                0, bank.graph, nodes
            )
            / bank.counts[:, None]
        )
        maximum = torch.full_like(mean, -torch.inf).scatter_reduce_(
            0, bank.graph[:, None].expand_as(nodes), nodes, reduce="amax", include_self=True
        )
        return self.graph_project(torch.cat([mean, maximum], dim=1))

    def forward(
        self,
        bank: GraphBank,
        left: torch.Tensor,
        right: torch.Tensor,
        left_identity: torch.Tensor,
        right_identity: torch.Tensor,
        text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        molecule = self.encode(bank)
        a, b = molecule[left], molecule[right]
        structure = torch.cat([a + b, torch.abs(a - b), a * b], dim=1)
        text_latent = self.text_project(text)
        ia, ib = self.identity(left_identity), self.identity(right_identity)
        identity = (ia + ib + ia * ib) / 3.0
        if self.training and self.identity_dropout:
            keep = torch.rand((len(left), 1), device=left.device) >= self.identity_dropout
            identity = identity * keep
        gate = self.identity_limit * torch.sigmoid(
            self.identity_gate(torch.cat([structure, text_latent], dim=1))
        )
        latent = self.trunk(torch.cat([structure, text_latent + gate * identity], dim=1))
        return (
            self.severity(latent),
            self.binary(latent),
            nn.functional.softplus(self.prr(latent)),
            gate,
        )


def _indices(frame: pd.DataFrame, graph_lookup: dict[str, int], identity: dict[str, int]):
    left = frame["SMILES_A"].astype(str).map(canonicalize_smiles)
    right = frame["SMILES_B"].astype(str).map(canonicalize_smiles)
    return (
        np.asarray([graph_lookup[value] for value in left], dtype=np.int64),
        np.asarray([graph_lookup[value] for value in right], dtype=np.int64),
        np.asarray([identity.get(value, 0) for value in left], dtype=np.int64),
        np.asarray([identity.get(value, 0) for value in right], dtype=np.int64),
    )


def _micro_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.25, 0.8, 56):
        score = f1_score(targets.ravel(), (probabilities >= threshold).ravel(), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def _score(
    severity: np.ndarray,
    binary: np.ndarray,
    prr: np.ndarray,
    predictions: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, float]:
    result = hardcore_score(
        severity,
        predictions["severity"],
        binary,
        predictions["probability"] >= threshold,
        prr,
        predictions["prr"],
    )
    return asdict(result)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _train_one(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    fit_y: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_y: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    config: dict,
    *,
    epochs: int | None = None,
    select_epoch: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model_cfg = config["model"]
    train_cfg = config["training"]
    device_name = str(model_cfg["device"])
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA demandé mais indisponible.")
    _seed(int(config["seed"]))
    fit_molecules = {
        canonicalize_smiles(str(value))
        for column in ("SMILES_A", "SMILES_B")
        for value in fit[column]
    }
    all_molecules = list(fit_molecules) + [
        canonicalize_smiles(str(value))
        for column in ("SMILES_A", "SMILES_B")
        for value in validation[column]
    ]
    bank = build_graph_bank(all_molecules, int(model_cfg["max_atoms"]), device)
    identity = {value: index + 1 for index, value in enumerate(sorted(fit_molecules))}
    fit_idx = tuple(torch.as_tensor(x, device=device) for x in _indices(fit, bank.lookup, identity))
    val_idx = tuple(
        torch.as_tensor(x, device=device) for x in _indices(validation, bank.lookup, identity)
    )
    text_fit = torch.as_tensor(
        text_pair_features(fit, int(model_cfg["text_hash_dim"])), device=device
    )
    text_val = torch.as_tensor(
        text_pair_features(validation, int(model_cfg["text_hash_dim"])), device=device
    )
    sev_fit, bin_fit, prr_fit = fit_y
    y_sev = torch.as_tensor(sev_fit, dtype=torch.long, device=device)
    y_bin = torch.as_tensor(bin_fit, dtype=torch.float32, device=device)
    y_prr = torch.as_tensor(prr_fit, dtype=torch.float32, device=device)
    positive = y_prr > 0
    scale = float(torch.sqrt((y_prr[positive] ** 2).mean()).clamp_min(1).item())
    counts = np.bincount(sev_fit, minlength=int(config["n_severity"]))
    severity_weights = torch.as_tensor(
        1 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32, device=device
    )
    severity_weights /= severity_weights.mean()
    model = GlobalGNN(
        len(identity),
        bin_fit.shape[1],
        int(config["n_severity"]),
        text_fit.shape[1],
        model_cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    maximum = int(epochs or train_cfg["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum)
    best_score, best_epoch, best_threshold = -math.inf, 0, 0.5
    history: list[dict[str, float]] = []
    for epoch in range(1, maximum + 1):
        model.train()
        sev_out, bin_out, prr_out, _ = model(bank, *fit_idx, text_fit)
        severity_loss = nn.functional.cross_entropy(sev_out, y_sev, weight=severity_weights)
        binary_loss = nn.functional.binary_cross_entropy_with_logits(bin_out, y_bin)
        prr_loss = nn.functional.mse_loss(prr_out[positive], y_prr[positive] / scale)
        loss = (
            float(train_cfg["severity_weight"]) * severity_loss
            + float(train_cfg["binary_weight"]) * binary_loss
            + float(train_cfg["prr_weight"]) * prr_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        record = {"epoch": epoch, "train_loss": float(loss.detach())}
        if select_epoch:
            if validation_y is None:
                raise ValueError("Sélection d'époque impossible sans validation interne.")
            predicted = _predict(model, bank, val_idx, text_val, scale)
            threshold, _ = _micro_threshold(validation_y[1], predicted["probability"])
            score = _score(*validation_y, predicted, threshold)
            record.update({"inner_score": score["total"], "inner_prr_score": score["prr_score"]})
            if score["total"] > best_score + 1e-6:
                best_score, best_epoch, best_threshold = score["total"], epoch, threshold
            if epoch - best_epoch >= int(train_cfg["patience"]) and epoch >= int(
                train_cfg["min_epochs"]
            ):
                history.append(record)
                break
        history.append(record)
    if not select_epoch:
        best_epoch = maximum
    predicted = _predict(model, bank, val_idx, text_val, scale)
    return predicted, {
        "selected_epoch": best_epoch,
        "threshold": best_threshold,
        "inner_best_score": None if not select_epoch else best_score,
        "history": history,
        "graph_molecules": len(bank.lookup),
        "graph_truncated": bank.truncated,
        "validation_unknown_identity": int(((val_idx[2] == 0) | (val_idx[3] == 0)).sum()),
        "device": str(device),
    }


@torch.inference_mode()
def _predict(
    model: GlobalGNN,
    bank: GraphBank,
    indices: tuple[torch.Tensor, ...],
    text: torch.Tensor,
    scale: float,
) -> dict[str, np.ndarray]:
    model.eval()
    severity, binary, prr, gate = model(bank, *indices, text)
    return {
        "severity": severity.argmax(dim=1).cpu().numpy(),
        "probability": torch.sigmoid(binary).cpu().numpy(),
        "prr": (prr * scale).cpu().numpy(),
        "mean_identity_gate": float(gate.mean()),
    }


def run_global_gnn(config_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    current_audit = audit_pair_grouped_folds(train, manifest)
    stored_audit = load_json(Path(paths["audit"]))
    fingerprint = split_fingerprint(manifest)
    if not current_audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Audit des folds invalide; benchmark arrêté.")
    binary_cols = [col for col in train if col.startswith(BINARY_PREFIX)]
    prr_cols = [col for col in train if col.startswith(PRR_PREFIX)]
    labels = sorted(train[SEVERITY_COLUMN].astype(str).unique())
    severity = train[SEVERITY_COLUMN].map({x: i for i, x in enumerate(labels)}).to_numpy(np.int64)
    binary = train[binary_cols].to_numpy(np.float32)
    prr = train[prr_cols].to_numpy(np.float32)
    config["n_severity"] = len(labels)
    folds = manifest["fold"].to_numpy()
    selected_folds = sorted(np.unique(folds))[: int(config["training"].get("max_folds", 5))]
    oof_severity = np.full(len(train), -1, dtype=np.int64)
    oof_binary = np.full((len(train), len(binary_cols)), np.nan, dtype=np.float32)
    oof_prr = np.full_like(oof_binary, np.nan)
    fold_results = []
    for fold in selected_folds:
        fit_mask = folds != fold
        val_mask = folds == fold
        fit_folds = folds[fit_mask]
        calibration_fold = sorted(np.unique(fit_folds))[int(fold) % (len(np.unique(fit_folds)))]
        core_mask = fit_folds != calibration_fold
        inner_mask = ~core_mask
        fit = train.loc[fit_mask].reset_index(drop=True)
        core = fit.loc[core_mask].reset_index(drop=True)
        inner = fit.loc[inner_mask].reset_index(drop=True)
        outer = train.loc[val_mask].reset_index(drop=True)
        y_fit = (severity[fit_mask], binary[fit_mask], prr[fit_mask])
        y_core = tuple(y[core_mask] for y in y_fit)
        y_inner = tuple(y[inner_mask] for y in y_fit)
        print(
            f"Global GNN fold {fold}: sélection interne, core={len(core)}, calib={len(inner)}",
            flush=True,
        )
        _, selection = _train_one(core, inner, y_core, y_inner, config, select_epoch=True)
        print(f"Global GNN fold {fold}: refit {selection['selected_epoch']} époques", flush=True)
        prediction, refit = _train_one(
            fit, outer, y_fit, None, config, epochs=int(selection["selected_epoch"])
        )
        outer_score = _score(
            severity[val_mask],
            binary[val_mask],
            prr[val_mask],
            prediction,
            float(selection["threshold"]),
        )
        oof_severity[val_mask] = prediction["severity"]
        oof_binary[val_mask] = (prediction["probability"] >= selection["threshold"]).astype(
            np.float32
        )
        oof_prr[val_mask] = prediction["prr"]
        fold_results.append(
            {
                "fold": int(fold),
                "calibration_fold": int(calibration_fold),
                "inner_selected_epoch": selection["selected_epoch"],
                "inner_threshold": selection["threshold"],
                "inner_score": selection["inner_best_score"],
                "outer_score": outer_score,
                "mean_identity_gate_outer": prediction["mean_identity_gate"],
                "graph_truncated": refit["graph_truncated"],
                "unknown_identity_outer": refit["validation_unknown_identity"],
                "history": selection["history"],
            }
        )
        print(f"Global GNN fold {fold}: score externe {outer_score['total']:.6f}", flush=True)
    complete = len(selected_folds) == len(np.unique(folds))
    if complete and (oof_severity < 0).any() or complete and not np.isfinite(oof_prr).all():
        raise RuntimeError("Couverture OOF incomplète.")
    mask = oof_severity >= 0
    overall = _score(
        severity[mask],
        binary[mask],
        prr[mask],
        {"severity": oof_severity[mask], "probability": oof_binary[mask], "prr": oof_prr[mask]},
        0.5,
    )
    run_dir = Path(paths["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": config["run_id"],
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_fingerprint": fingerprint,
        "folds_complete": complete,
        "score_evaluated_rows": int(mask.sum()),
        "score": overall,
        "fold_results": fold_results,
        "severity_labels": labels,
        "leakage_audit": {
            "outer_validation_never_used_for_fit_selection_or_threshold": True,
            "inner_selection_only_within_outer_fit": True,
            "text_hashing_stateless": True,
            "identity_indices_fit_only": True,
            "test_loaded": False,
            "historical_incumbent_comparison_strict": False,
            "reason": "Existing PRR OOF checkpoint selection used outer validation.",
        },
        "automatic_submission": False,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(summary, run_dir / "summary.json")
    if complete:
        output = pd.concat(
            [
                train[[ID_COLUMN]].reset_index(drop=True),
                pd.Series([labels[i] for i in oof_severity], name=SEVERITY_COLUMN),
                pd.DataFrame(oof_binary.astype(np.int8), columns=binary_cols),
                pd.DataFrame(oof_prr, columns=prr_cols),
            ],
            axis=1,
        )
        output_path = Path(paths["output_oof"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
    return summary
