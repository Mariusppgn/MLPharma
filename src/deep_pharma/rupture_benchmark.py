from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from rdkit import Chem
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from torch import nn

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .four_branch_benchmark import _fit_micro_threshold, _load_or_build_features
from .metric import HardcoreScore, hardcore_score
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, canonicalize_smiles, split_fingerprint


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA demande mais indisponible.")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atom_features(atom: Chem.Atom) -> np.ndarray:
    hybridizations = (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2,
    )
    return np.asarray(
        [
            min(atom.GetAtomicNum(), 100) / 100.0,
            min(atom.GetTotalDegree(), 6) / 6.0,
            np.clip(atom.GetFormalCharge(), -4, 4) / 4.0,
            min(atom.GetTotalNumHs(), 4) / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
            *[float(atom.GetHybridization() == value) for value in hybridizations],
        ],
        dtype=np.float32,
    )


def molecular_graph(smiles: str, max_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"SMILES invalide : {smiles!r}")
    n_atoms = min(molecule.GetNumAtoms(), max_atoms)
    features = np.zeros((max_atoms, 12), dtype=np.float32)
    adjacency = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    atom_types = np.full(max_atoms, -100, dtype=np.int64)
    for index, atom in enumerate(molecule.GetAtoms()):
        if index >= n_atoms:
            break
        features[index] = _atom_features(atom)
        atom_types[index] = min(atom.GetAtomicNum(), 100)
        adjacency[index, index] = 1.0
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left < n_atoms and right < n_atoms:
            adjacency[left, right] = 1.0
            adjacency[right, left] = 1.0
    return features, adjacency, atom_types, molecule.GetNumAtoms()


class DenseGINEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(features)
        mask_float = mask.unsqueeze(-1).float()
        for layer, norm in zip(self.layers, self.norms, strict=True):
            update = layer(torch.bmm(adjacency, hidden))
            hidden = norm(hidden + update) * mask_float
        pooled = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        return hidden, pooled


class MaskedAtomModel(nn.Module):
    def __init__(self, encoder: DenseGINEncoder, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, 101)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        nodes, _ = self.encoder(features, adjacency, mask)
        return self.head(nodes)


class HybridPairModel(nn.Module):
    def __init__(
        self,
        node_bank: torch.Tensor,
        pool_bank: torch.Tensor,
        mask_bank: torch.Tensor,
        n_identities: int,
        n_targets: int,
        n_severity: int,
        config: dict[str, object],
    ) -> None:
        super().__init__()
        node_dim = node_bank.shape[-1]
        identity_dim = int(config["identity_dim"])
        pair_dim = int(config["pair_dim"])
        target_dim = int(config["target_dim"])
        self.register_buffer("node_bank", node_bank)
        self.register_buffer("pool_bank", pool_bank)
        self.register_buffer("mask_bank", mask_bank)
        self.identity = nn.Embedding(n_identities, identity_dim, padding_idx=0)
        self.cross_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.structure = nn.Sequential(
            nn.Linear(node_dim * 4, pair_dim), nn.SiLU(), nn.Dropout(float(config["dropout"]))
        )
        self.identity_pair = nn.Sequential(
            nn.Linear(identity_dim * 3, pair_dim), nn.SiLU(), nn.Dropout(float(config["dropout"]))
        )
        self.gate = nn.Sequential(nn.Linear(pair_dim * 2 + 4, 64), nn.SiLU(), nn.Linear(64, 1))
        self.trunk = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.SiLU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(pair_dim, target_dim),
            nn.SiLU(),
        )
        self.target = nn.Parameter(torch.randn(n_targets, target_dim) * 0.05)
        self.binary_bias = nn.Parameter(torch.zeros(n_targets))
        self.raw_bias = nn.Parameter(torch.zeros(n_targets))
        self.log_bias = nn.Parameter(torch.zeros(n_targets))
        self.raw_projection = nn.Linear(target_dim, target_dim)
        self.log_projection = nn.Linear(target_dim, target_dim)
        self.severity = nn.Linear(target_dim, n_severity)

    @staticmethod
    def symmetric_identity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat([left + right, torch.abs(left - right), left * right], dim=1)

    def _cross_interaction(self, left_index: torch.Tensor, right_index: torch.Tensor) -> torch.Tensor:
        left = self.node_bank[left_index]
        right = self.node_bank[right_index]
        left_mask = self.mask_bank[left_index]
        right_mask = self.mask_bank[right_index]
        left_projected = self.cross_projection(left)
        right_projected = self.cross_projection(right)
        score = torch.bmm(left_projected, right_projected.transpose(1, 2)) / math.sqrt(
            left.shape[-1]
        )
        valid = left_mask.unsqueeze(2) & right_mask.unsqueeze(1)
        score = score.masked_fill(~valid, -1e4)
        left_to_right = torch.softmax(score, dim=2)
        right_to_left = torch.softmax(score.transpose(1, 2), dim=2)
        attended_right = torch.bmm(left_to_right, right)
        attended_left = torch.bmm(right_to_left, left)
        left_delta = (
            torch.abs(left - attended_right) * left_mask.unsqueeze(-1)
        ).sum(dim=1) / left_mask.sum(dim=1, keepdim=True).clamp_min(1)
        right_delta = (
            torch.abs(right - attended_left) * right_mask.unsqueeze(-1)
        ).sum(dim=1) / right_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return 0.5 * (left_delta + right_delta)

    def forward(
        self,
        left_graph: torch.Tensor,
        right_graph: torch.Tensor,
        left_identity: torch.Tensor,
        right_identity: torch.Tensor,
        familiarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_pool = self.pool_bank[left_graph]
        right_pool = self.pool_bank[right_graph]
        interaction = self._cross_interaction(left_graph, right_graph)
        structure = self.structure(
            torch.cat(
                [
                    left_pool + right_pool,
                    torch.abs(left_pool - right_pool),
                    left_pool * right_pool,
                    interaction,
                ],
                dim=1,
            )
        )
        identity = self.identity_pair(
            self.symmetric_identity(self.identity(left_identity), self.identity(right_identity))
        )
        gate = torch.sigmoid(self.gate(torch.cat([structure, identity, familiarity], dim=1)))
        latent = self.trunk(gate * identity + (1.0 - gate) * structure)
        binary = latent @ self.target.T + self.binary_bias
        raw = nn.functional.softplus(self.raw_projection(latent) @ self.target.T + self.raw_bias)
        log = nn.functional.softplus(self.log_projection(latent) @ self.target.T + self.log_bias)
        return binary, raw, log, self.severity(latent), gate


def simplex_weights(experts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if experts.ndim != 2 or experts.shape[1] < 2:
        raise ValueError("experts doit avoir la forme [n, n_experts].")
    if len(experts) < experts.shape[1]:
        return np.full(experts.shape[1], 1.0 / experts.shape[1], dtype=np.float32)
    weights, _ = nnls(experts.astype(np.float64), targets.astype(np.float64))
    total = float(weights.sum())
    if total <= 1e-12:
        return np.full(experts.shape[1], 1.0 / experts.shape[1], dtype=np.float32)
    return (weights / total).astype(np.float32)


def _molecule_bank(
    frames: list[pd.DataFrame], max_atoms: int
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    molecules = sorted(
        {
            canonicalize_smiles(str(value))
            for frame in frames
            for column in ("SMILES_A", "SMILES_B")
            for value in frame[column]
        }
    )
    lookup = {molecule: index for index, molecule in enumerate(molecules)}
    features, adjacency, atom_types = [], [], []
    original_sizes: dict[str, int] = {}
    for molecule in molecules:
        graph = molecular_graph(molecule, max_atoms)
        features.append(graph[0])
        adjacency.append(graph[1])
        atom_types.append(graph[2])
        original_sizes[molecule] = graph[3]
    return (
        lookup,
        np.asarray(features, dtype=np.float32),
        np.asarray(adjacency, dtype=np.float32),
        np.asarray(atom_types, dtype=np.int64),
        original_sizes,
    )


def _frame_indices(
    frame: pd.DataFrame,
    graph_lookup: dict[str, int],
    identity_lookup: dict[str, int],
    frequencies: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_smiles = frame["SMILES_A"].astype(str).map(canonicalize_smiles)
    right_smiles = frame["SMILES_B"].astype(str).map(canonicalize_smiles)
    left_graph = left_smiles.map(graph_lookup).to_numpy(np.int64)
    right_graph = right_smiles.map(graph_lookup).to_numpy(np.int64)
    left_id = left_smiles.map(lambda value: identity_lookup.get(value, 0)).to_numpy(np.int64)
    right_id = right_smiles.map(lambda value: identity_lookup.get(value, 0)).to_numpy(np.int64)
    max_frequency = max(frequencies.values(), default=1)
    left_seen = left_id > 0
    right_seen = right_id > 0
    left_frequency = left_smiles.map(
        lambda value: math.log1p(frequencies.get(value, 0)) / math.log1p(max_frequency)
    ).to_numpy(np.float32)
    right_frequency = right_smiles.map(
        lambda value: math.log1p(frequencies.get(value, 0)) / math.log1p(max_frequency)
    ).to_numpy(np.float32)
    familiarity = np.column_stack(
        [
            left_seen & right_seen,
            left_seen | right_seen,
            0.5 * (left_frequency + right_frequency),
            np.abs(left_frequency - right_frequency),
        ]
    ).astype(np.float32)
    return left_graph, right_graph, left_id, right_id, familiarity


def _pretrain_encoder(
    features: np.ndarray,
    adjacency: np.ndarray,
    atom_types: np.ndarray,
    molecule_indices: np.ndarray,
    config: dict[str, object],
    device: torch.device,
) -> DenseGINEncoder:
    encoder = DenseGINEncoder(
        features.shape[-1],
        int(config["node_dim"]),
        int(config["gnn_layers"]),
        float(config["dropout"]),
    ).to(device)
    model = MaskedAtomModel(encoder, int(config["node_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["pretrain_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = int(config["molecule_batch_size"])
    rng = np.random.default_rng(int(config["seed"]))
    for _ in range(int(config["pretrain_epochs"])):
        order = rng.permutation(molecule_indices)
        model.train()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch_features = torch.from_numpy(features[rows]).to(device)
            batch_adjacency = torch.from_numpy(adjacency[rows]).to(device)
            batch_types = torch.from_numpy(atom_types[rows]).to(device)
            valid = batch_types >= 0
            masked = (torch.rand(valid.shape, device=device) < float(config["mask_rate"])) & valid
            if not masked.any():
                continue
            corrupted = batch_features.clone()
            corrupted[masked] = 0.0
            logits = model(corrupted, batch_adjacency, valid)
            loss = nn.functional.cross_entropy(logits[masked], batch_types[masked])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
    return encoder


@torch.inference_mode()
def _encode_bank(
    encoder: DenseGINEncoder,
    features: np.ndarray,
    adjacency: np.ndarray,
    atom_types: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder.eval()
    nodes, pools, masks = [], [], []
    for start in range(0, len(features), batch_size):
        batch_features = torch.from_numpy(features[start : start + batch_size]).to(device)
        batch_adjacency = torch.from_numpy(adjacency[start : start + batch_size]).to(device)
        mask = torch.from_numpy(atom_types[start : start + batch_size] >= 0).to(device)
        batch_nodes, batch_pools = encoder(batch_features, batch_adjacency, mask)
        nodes.append(batch_nodes.cpu())
        pools.append(batch_pools.cpu())
        masks.append(mask.cpu())
    return torch.cat(nodes), torch.cat(pools), torch.cat(masks)


def _batches(n_rows: int, batch_size: int, shuffle: bool, seed: int) -> list[np.ndarray]:
    rows = np.arange(n_rows)
    if shuffle:
        np.random.default_rng(seed).shuffle(rows)
    return [rows[start : start + batch_size] for start in range(0, n_rows, batch_size)]


def _model_arrays(
    model: HybridPairModel,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: dict[str, object],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    outputs = {name: [] for name in ("probability", "raw", "log", "severity", "gate")}
    with torch.inference_mode():
        for rows in _batches(len(indices[0]), int(config["pair_batch_size"]), False, 0):
            tensors = [torch.from_numpy(value[rows]).to(device) for value in indices]
            binary, raw, log, severity, gate = model(*tensors)
            outputs["probability"].append(torch.sigmoid(binary).cpu().numpy())
            outputs["raw"].append(raw.cpu().numpy())
            outputs["log"].append(np.expm1(np.clip(log.cpu().numpy(), 0.0, 8.0)))
            outputs["severity"].append(severity.argmax(dim=1).cpu().numpy())
            outputs["gate"].append(gate.cpu().numpy())
    return {name: np.concatenate(parts).astype(np.float32) for name, parts in outputs.items()}


def _fit_pair_model(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    fit_binary: np.ndarray,
    fit_prr: np.ndarray,
    fit_severity: np.ndarray,
    config: dict[str, object],
    device: torch.device,
    *,
    epochs: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    _seed(int(config["seed"]))
    graph_lookup, features, adjacency, atom_types, sizes = _molecule_bank(
        [fit_frame, validation_frame], int(config["max_atoms"])
    )
    fit_molecules = {
        canonicalize_smiles(str(value))
        for column in ("SMILES_A", "SMILES_B")
        for value in fit_frame[column]
    }
    molecule_indices = np.asarray([graph_lookup[value] for value in fit_molecules], dtype=np.int64)
    encoder = _pretrain_encoder(
        features, adjacency, atom_types, molecule_indices, config, device
    )
    node_bank, pool_bank, mask_bank = _encode_bank(
        encoder,
        features,
        adjacency,
        atom_types,
        device,
        int(config["molecule_batch_size"]),
    )
    identity_lookup = {value: index + 1 for index, value in enumerate(sorted(fit_molecules))}
    all_fit_smiles = pd.concat([fit_frame["SMILES_A"], fit_frame["SMILES_B"]]).astype(str)
    frequencies = all_fit_smiles.map(canonicalize_smiles).value_counts().to_dict()
    fit_indices = _frame_indices(fit_frame, graph_lookup, identity_lookup, frequencies)
    validation_indices = _frame_indices(
        validation_frame, graph_lookup, identity_lookup, frequencies
    )
    model = HybridPairModel(
        node_bank,
        pool_bank,
        mask_bank,
        len(identity_lookup) + 1,
        fit_binary.shape[1],
        int(np.max(fit_severity)) + 1,
        config,
    ).to(device)
    prevalence = fit_binary.mean(axis=0)
    positive_weights = torch.from_numpy(
        np.clip((1 - prevalence) / np.maximum(prevalence, 1e-4), 1.0, float(config["positive_weight_max"])).astype(np.float32)
    ).to(device)
    class_counts = np.bincount(fit_severity)
    class_weights = torch.from_numpy(
        (class_counts.sum() / np.maximum(class_counts, 1) / len(class_counts)).astype(np.float32)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    binary_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    severity_loss = nn.CrossEntropyLoss(weight=class_weights)
    scale = np.ones(fit_prr.shape[1], dtype=np.float32)
    for target_index in range(fit_prr.shape[1]):
        positive = fit_prr[:, target_index][fit_prr[:, target_index] > 0]
        if len(positive):
            scale[target_index] = max(float(np.quantile(positive, 0.95)), 1.0)
    scale_tensor = torch.from_numpy(scale).to(device)
    for epoch in range(epochs):
        model.train()
        for rows in _batches(
            len(fit_frame), int(config["pair_batch_size"]), True, int(config["seed"]) + epoch
        ):
            tensors = [torch.from_numpy(value[rows]).to(device) for value in fit_indices]
            y_binary = torch.from_numpy(fit_binary[rows].astype(np.float32)).to(device)
            y_prr = torch.from_numpy(fit_prr[rows].astype(np.float32)).to(device)
            y_severity = torch.from_numpy(fit_severity[rows]).to(device)
            logits, raw, log, severity, _ = model(*tensors)
            positive = y_prr > 0
            normalized = y_prr / scale_tensor
            raw_loss = nn.functional.smooth_l1_loss(raw[positive], normalized[positive], beta=0.5)
            log_target = torch.log1p(y_prr)
            log_loss = nn.functional.smooth_l1_loss(log[positive], log_target[positive], beta=0.5)
            loss = (
                float(config["binary_weight"]) * binary_loss(logits, y_binary)
                + float(config["prr_raw_weight"]) * raw_loss
                + float(config["prr_log_weight"]) * log_loss
                + float(config["severity_weight"]) * severity_loss(severity, y_severity)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
    fit_predictions = _model_arrays(model, fit_indices, config, device)
    validation_predictions = _model_arrays(model, validation_indices, config, device)
    fit_predictions["familiarity"] = fit_indices[4]
    validation_predictions["familiarity"] = validation_indices[4]
    fit_predictions["raw"] *= scale
    validation_predictions["raw"] *= scale
    return fit_predictions, validation_predictions, {
        "epochs": epochs,
        "molecules_fit": len(fit_molecules),
        "molecules_total": len(graph_lookup),
        "truncated_molecules": int(sum(value > int(config["max_atoms"]) for value in sizes.values())),
        "mean_identity_gate_validation": float(validation_predictions["gate"].mean()),
        "device": str(device),
    }


def _lgb_experts(
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    fit_binary: np.ndarray,
    fit_prr: np.ndarray,
    config: dict[str, object],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    def one_target(index: int) -> tuple[int, np.ndarray, np.ndarray]:
        binary_target = fit_binary[:, index]
        if np.unique(binary_target).size < 2:
            probability = np.full(len(validation_features), binary_target.mean(), dtype=np.float32)
        else:
            classifier = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=int(config["n_estimators"]),
                learning_rate=float(config["learning_rate"]),
                num_leaves=int(config["num_leaves"]),
                min_child_samples=int(config["min_child_samples"]),
                reg_lambda=float(config["reg_lambda"]),
                random_state=seed + index,
                n_jobs=1,
                deterministic=True,
                verbosity=-1,
            )
            classifier.fit(fit_features, binary_target)
            probability = classifier.predict_proba(validation_features)[:, 1].astype(np.float32)
        prr_target = fit_prr[:, index]
        positive = prr_target > 0
        if positive.sum() < 3:
            value = float(prr_target[positive].mean()) if positive.any() else 0.0
            prr = np.full(len(validation_features), value, dtype=np.float32)
        else:
            regressor = lgb.LGBMRegressor(
                objective="regression_l2",
                n_estimators=int(config["n_estimators"]),
                learning_rate=float(config["learning_rate"]),
                num_leaves=int(config["num_leaves"]),
                min_child_samples=int(config["min_child_samples"]),
                reg_lambda=float(config["reg_lambda"]),
                random_state=seed + 1000 + index,
                n_jobs=1,
                deterministic=True,
                verbosity=-1,
            )
            regressor.fit(fit_features[positive], prr_target[positive])
            prr = np.maximum(regressor.predict(validation_features), 0.0).astype(np.float32)
        return index, probability, prr

    results = Parallel(n_jobs=int(config["target_n_jobs"]), prefer="threads")(
        delayed(one_target)(index) for index in range(fit_binary.shape[1])
    )
    probability = np.zeros((len(validation_features), fit_binary.shape[1]), dtype=np.float32)
    prr = np.zeros_like(probability)
    for index, target_probability, target_prr in results:
        probability[:, index] = target_probability
        prr[:, index] = target_prr
    return probability, prr


def _fit_binary_stack(
    targets: np.ndarray,
    graph: np.ndarray,
    gbdt: np.ndarray,
    familiarity: np.ndarray,
) -> list[LogisticRegression | float]:
    models: list[LogisticRegression | float] = []
    features = np.column_stack([graph, gbdt])
    for index in range(targets.shape[1]):
        target = targets[:, index]
        if np.unique(target).size < 2:
            models.append(float(target[0]))
            continue
        model = LogisticRegression(C=0.5, max_iter=300, class_weight="balanced")
        model.fit(np.column_stack([features[:, [index, index + targets.shape[1]]], familiarity]), target)
        models.append(model)
    return models


def _predict_binary_stack(
    models: list[LogisticRegression | float],
    graph: np.ndarray,
    gbdt: np.ndarray,
    familiarity: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(graph, dtype=np.float32)
    for index, model in enumerate(models):
        if isinstance(model, float):
            result[:, index] = model
        else:
            features = np.column_stack([graph[:, index], gbdt[:, index], familiarity])
            result[:, index] = model.predict_proba(features)[:, 1]
    return result


def _fit_prr_stack(
    targets: np.ndarray,
    raw: np.ndarray,
    log: np.ndarray,
    gbdt: np.ndarray,
    familiarity: np.ndarray,
) -> np.ndarray:
    weights = np.zeros((targets.shape[1], 2, 3), dtype=np.float32)
    seen = familiarity[:, 0] > 0.5
    experts = np.stack([raw, log, gbdt], axis=2)
    for target_index in range(targets.shape[1]):
        positive = targets[:, target_index] > 0
        global_weights = simplex_weights(
            experts[positive, target_index], targets[positive, target_index]
        )
        for regime, regime_mask in enumerate([~seen, seen]):
            mask = positive & regime_mask
            weights[target_index, regime] = (
                simplex_weights(experts[mask, target_index], targets[mask, target_index])
                if mask.sum() >= 12
                else global_weights
            )
    return weights


def _predict_prr_stack(
    weights: np.ndarray,
    raw: np.ndarray,
    log: np.ndarray,
    gbdt: np.ndarray,
    familiarity: np.ndarray,
) -> np.ndarray:
    experts = np.stack([raw, log, gbdt], axis=2)
    seen = (familiarity[:, 0] > 0.5).astype(np.int8)
    result = np.zeros_like(raw, dtype=np.float32)
    for row in range(len(raw)):
        result[row] = np.sum(experts[row] * weights[:, seen[row], :], axis=1)
    return np.maximum(result, 0.0)


def _score(
    train: pd.DataFrame,
    severity: np.ndarray,
    binary: np.ndarray,
    prr: np.ndarray,
    binary_columns: list[str],
    prr_columns: list[str],
) -> HardcoreScore:
    return hardcore_score(
        train[SEVERITY_COLUMN], severity, train[binary_columns], binary, train[prr_columns], prr
    )


def run_rupture_benchmark(config_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_path = Path(paths["train"])
    train = pd.read_csv(train_path, low_memory=False)
    manifest = pd.read_csv(paths["manifest"]).sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    fingerprint = split_fingerprint(manifest)
    stored_audit = load_json(Path(paths["audit"]))
    if not audit["passed"] or fingerprint != stored_audit.get("fingerprint"):
        raise ValueError("Le manifeste groupe ne correspond pas a son audit.")
    baseline = pd.read_csv(paths["baseline_oof"], low_memory=False)
    if not baseline[ID_COLUMN].astype(str).equals(train[ID_COLUMN].astype(str)):
        raise ValueError("Predictions baseline desalignees.")
    binary_columns = [column for column in train if column.startswith(BINARY_PREFIX)]
    prr_columns = [column for column in train if column.startswith(PRR_PREFIX)]
    y_binary = train[binary_columns].to_numpy(np.int8)
    y_prr = train[prr_columns].to_numpy(np.float32)
    severity_labels = sorted(train[SEVERITY_COLUMN].astype(str).unique())
    severity_lookup = {value: index for index, value in enumerate(severity_labels)}
    y_severity = train[SEVERITY_COLUMN].map(severity_lookup).to_numpy(np.int64)
    fixed_severity = baseline[SEVERITY_COLUMN].astype(str).to_numpy()
    folds = manifest["fold"].to_numpy()
    device = _device(str(config["model"]["device"]))
    n_rows, n_targets = y_binary.shape
    graph_probability = np.full((n_rows, n_targets), np.nan, dtype=np.float32)
    stack_probability = np.full_like(graph_probability, np.nan)
    raw_oof = np.full_like(graph_probability, np.nan)
    log_oof = np.full_like(graph_probability, np.nan)
    gbdt_oof = np.full_like(graph_probability, np.nan)
    oof_stack_prr = np.full_like(graph_probability, np.nan)
    graph_decisions = np.zeros_like(y_binary)
    stack_decisions = np.zeros_like(y_binary)
    metadata: list[dict[str, object]] = []
    for outer_fold in sorted(int(value) for value in np.unique(folds)):
        validation_mask = folds == outer_fold
        fit_mask = ~validation_mask
        fit_frame = train.loc[fit_mask].reset_index(drop=True)
        validation_frame = train.loc[validation_mask].reset_index(drop=True)
        fit_folds = folds[fit_mask]
        inner_values = sorted(int(value) for value in np.unique(fit_folds))
        calibration_fold = inner_values[outer_fold % len(inner_values)]
        calibration_mask = fit_folds == calibration_fold
        core_mask = ~calibration_mask
        print(f"Rupture fold {outer_fold}: core={core_mask.sum()} calibration={calibration_mask.sum()} validation={validation_mask.sum()}", flush=True)
        fit_base, validation_base, _, _ = _load_or_build_features(
            train, y_prr, folds, outer_fold, fingerprint, config
        )
        core_frame = fit_frame.loc[core_mask].reset_index(drop=True)
        calibration_frame = fit_frame.loc[calibration_mask].reset_index(drop=True)
        _, calibration_predictions, selection_meta = _fit_pair_model(
            core_frame,
            calibration_frame,
            y_binary[fit_mask][core_mask],
            y_prr[fit_mask][core_mask],
            y_severity[fit_mask][core_mask],
            config["model"],
            device,
            epochs=int(config["training"]["selection_epochs"]),
        )
        calibration_gbdt_probability, calibration_gbdt_prr = _lgb_experts(
            fit_base[core_mask],
            fit_base[calibration_mask],
            y_binary[fit_mask][core_mask],
            y_prr[fit_mask][core_mask],
            config["gbdt"],
            int(config["seed"]) + outer_fold,
        )
        calibration_familiarity = _frame_indices(
            calibration_frame,
            {canonicalize_smiles(str(value)): 0 for value in pd.concat([calibration_frame.SMILES_A, calibration_frame.SMILES_B])},
            {
                value: index + 1
                for index, value in enumerate(
                    sorted(
                        {
                            canonicalize_smiles(str(item))
                            for column in ("SMILES_A", "SMILES_B")
                            for item in core_frame[column]
                        }
                    )
                )
            },
            pd.concat([core_frame.SMILES_A, core_frame.SMILES_B]).astype(str).map(canonicalize_smiles).value_counts().to_dict(),
        )[4]
        binary_models = _fit_binary_stack(
            y_binary[fit_mask][calibration_mask],
            calibration_predictions["probability"],
            calibration_gbdt_probability,
            calibration_familiarity,
        )
        calibration_stacked_probability = _predict_binary_stack(
            binary_models,
            calibration_predictions["probability"],
            calibration_gbdt_probability,
            calibration_familiarity,
        )
        threshold, calibration_f1 = _fit_micro_threshold(
            y_binary[fit_mask][calibration_mask],
            calibration_stacked_probability,
            minimum=float(config["stacking"]["threshold_min"]),
            maximum=float(config["stacking"]["threshold_max"]),
            steps=int(config["stacking"]["threshold_steps"]),
        )
        graph_threshold, graph_calibration_f1 = _fit_micro_threshold(
            y_binary[fit_mask][calibration_mask],
            calibration_predictions["probability"],
            minimum=float(config["stacking"]["threshold_min"]),
            maximum=float(config["stacking"]["threshold_max"]),
            steps=int(config["stacking"]["threshold_steps"]),
        )
        prr_weights = _fit_prr_stack(
            y_prr[fit_mask][calibration_mask],
            calibration_predictions["raw"],
            calibration_predictions["log"],
            calibration_gbdt_prr,
            calibration_familiarity,
        )
        _, final_predictions, final_meta = _fit_pair_model(
            fit_frame,
            validation_frame,
            y_binary[fit_mask],
            y_prr[fit_mask],
            y_severity[fit_mask],
            config["model"],
            device,
            epochs=int(config["training"]["refit_epochs"]),
        )
        validation_gbdt_probability, validation_gbdt_prr = _lgb_experts(
            fit_base,
            validation_base,
            y_binary[fit_mask],
            y_prr[fit_mask],
            config["gbdt"],
            int(config["seed"]) + 100 + outer_fold,
        )
        validation_stack_features = final_predictions.pop("familiarity")
        stacked_probability = _predict_binary_stack(
            binary_models,
            final_predictions["probability"],
            validation_gbdt_probability,
            validation_stack_features,
        )
        validation_stacked_prr = _predict_prr_stack(
            prr_weights,
            final_predictions["raw"],
            final_predictions["log"],
            validation_gbdt_prr,
            validation_stack_features,
        )
        graph_probability[validation_mask] = final_predictions["probability"]
        stack_probability[validation_mask] = stacked_probability
        raw_oof[validation_mask] = final_predictions["raw"]
        log_oof[validation_mask] = final_predictions["log"]
        gbdt_oof[validation_mask] = validation_gbdt_prr
        oof_stack_prr[validation_mask] = validation_stacked_prr
        graph_decisions[validation_mask] = (
            final_predictions["probability"] >= graph_threshold
        ).astype(np.int8)
        stack_decisions[validation_mask] = (stacked_probability >= threshold).astype(np.int8)
        metadata.append(
            {
                "fold": outer_fold,
                "calibration_fold": calibration_fold,
                "threshold": threshold,
                "calibration_micro_f1": calibration_f1,
                "graph_threshold": graph_threshold,
                "graph_calibration_micro_f1": graph_calibration_f1,
                "selection": selection_meta,
                "refit": final_meta,
                "mean_prr_weights": prr_weights.mean(axis=(0, 1)).tolist(),
            }
        )
        stack_probability[validation_mask] = stacked_probability
        graph_probability[validation_mask] = final_predictions["probability"]
    if not all(
        np.isfinite(array).all()
        for array in [
            graph_probability,
            stack_probability,
            raw_oof,
            log_oof,
            gbdt_oof,
            oof_stack_prr,
        ]
    ):
        raise RuntimeError("Couverture OOF rupture incomplete.")
    candidates = {
        "hybrid_graph_raw": (graph_decisions, raw_oof),
        "hybrid_graph_log": (graph_decisions, log_oof),
        "rupture_adaptive_stack": (stack_decisions, oof_stack_prr),
    }
    rows = []
    output_dir = Path(paths["output_dir"])
    run_dir = Path(paths["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, (binary, prr) in candidates.items():
        score = _score(train, fixed_severity, binary, prr, binary_columns, prr_columns)
        rows.append({"branch": name, **asdict(score)})
        pd.concat(
            [
                train[[ID_COLUMN]].reset_index(drop=True),
                pd.Series(fixed_severity, name=SEVERITY_COLUMN),
                pd.DataFrame(binary, columns=binary_columns),
                pd.DataFrame(prr, columns=prr_columns),
            ],
            axis=1,
        ).to_csv(output_dir / f"oof_predictions_{name}.csv", index=False)
    comparison = pd.DataFrame(rows).sort_values("total", ascending=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    summary = {
        "run_id": config["run_id"],
        "train_sha256": _sha256(train_path),
        "validation_fingerprint": fingerprint,
        "comparison": comparison.to_dict(orient="records"),
        "fold_metadata": metadata,
        "leakage_audit": {
            "passed": True,
            "pair_group_overlap_max": 0,
            "outer_validation_never_used_for_fit_or_stacking": True,
            "self_supervision_uses_outer_fit_molecules_only": True,
            "test_loaded": False,
        },
        "automatic_submission": False,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(summary, run_dir / "summary.json")
    return summary
