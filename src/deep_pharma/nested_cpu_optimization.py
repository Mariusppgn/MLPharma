"""Sequential CPU optimization and comparable whole-pipeline OOF reporting."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .constants import BINARY_PREFIX, ID_COLUMN, PRR_PREFIX, SEVERITY_COLUMN
from .metric import hardcore_score
from .submission_gate import load_json, write_json
from .validation import audit_pair_grouped_folds, split_fingerprint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def _status(output, state, stage):
    write_json({"state": state, "stage": stage, "pid": os.getpid(),
                "updated_utc": datetime.now(UTC).isoformat()}, output / "status.json")


def _read_predictions(path, train, columns):
    frame = pd.read_csv(path)
    if not frame[ID_COLUMN].astype(str).equals(train[ID_COLUMN].astype(str)):
        raise ValueError(f"OOF row identity mismatch: {path}")
    if frame[columns].isna().any().any():
        raise ValueError(f"Incomplete OOF predictions: {path}")
    return frame[columns]


def compare_pipeline(config, prr_dir, classification_dir):
    output = Path(config["run_dir"])
    train = pd.read_csv(config["train"])
    manifest = pd.read_csv(config["manifest"]).sort_values("row_index").reset_index(drop=True)
    audit = audit_pair_grouped_folds(train, manifest)
    if not audit["passed"]:
        raise ValueError(audit["errors"])
    fingerprint = split_fingerprint(manifest)
    train_hash = _sha(config["train"])
    for directory in (prr_dir, classification_dir):
        stage = load_json(Path(directory) / "summary.json")
        stage_fingerprint = stage.get("validation_fingerprint", stage.get("fingerprint"))
        if stage_fingerprint != fingerprint or stage.get("train_sha256") != train_hash:
            raise ValueError("Stage train/fold protocol mismatch")
        if stage.get("complete_oof") is not True:
            raise ValueError("Incomplete stage OOF coverage")
        if any(stage.get(key, False) for key in ("test_loaded", "test_used", "test_accessed")):
            raise ValueError("Test access is forbidden in optimization")
    binary = [c for c in train if c.startswith(BINARY_PREFIX)]
    prr = [c for c in train if c.startswith(PRR_PREFIX)]
    records, fold_records = [], []
    for variant in ("baseline", "optimized"):
        classification = _read_predictions(
            Path(classification_dir) / f"oof_{variant}.csv", train, [SEVERITY_COLUMN, *binary]
        )
        regression = _read_predictions(Path(prr_dir) / f"oof_{variant}.csv", train, prr)
        if not np.isin(classification[binary].to_numpy(), [0, 1]).all():
            raise ValueError("Binary predictions must be hard 0/1 decisions")
        prediction = pd.concat([train[[ID_COLUMN]], classification, regression], axis=1)
        prediction.to_csv(output / f"oof_{variant}.csv", index=False)
        for fold in [None, *sorted(manifest.fold.unique())]:
            mask = np.ones(len(train), dtype=bool) if fold is None else manifest.fold == fold
            metrics = hardcore_score(
                train.loc[mask, SEVERITY_COLUMN], prediction.loc[mask, SEVERITY_COLUMN],
                train.loc[mask, binary], prediction.loc[mask, binary],
                train.loc[mask, prr], prediction.loc[mask, prr],
            )
            row = {"variant": variant, **asdict(metrics)}
            if fold is None:
                records.append(row)
            else:
                fold_records.append({"fold": int(fold), **row})
    comparison = pd.DataFrame(records)
    folds = pd.DataFrame(fold_records)
    comparison.to_csv(output / "comparison.csv", index=False)
    folds.to_csv(output / "fold_comparison.csv", index=False)
    baseline, optimized = records
    gains = {k: optimized[k] - baseline[k] for k in baseline if k != "variant"}
    fold_gains = folds.pivot(index="fold", columns="variant", values="total")
    fold_gains["delta"] = fold_gains.optimized - fold_gains.baseline
    summary = {"comparison": records, "gains": gains, "folds": fold_records,
               "validation_fingerprint": fingerprint,
               "train_sha256": train_hash, "complete_oof": True,
               "automatic_promotion": False, "test_used": False,
               "positive_gain_folds": int((fold_gains.delta > 0).sum()),
               "all_folds_improved": bool((fold_gains.delta > 0).all()),
               "scope": "bounded nested tuning, not an estimate of the absolute ceiling",
               "outer_reuse_caveat": "Historical method choices already examined these folds."}
    write_json(summary, output / "summary.json")
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, metric, title in zip(
        axes, ["total", "severity_macro_f1", "side_effect_micro_f1", "prr_score"],
        ["Score global", "Sévérité macro-F1", "Effets micro-F1", "S_PRR"], strict=True,
    ):
        for variant, group in folds.groupby("variant", sort=False):
            ax.plot(group.fold, group[metric], marker="o", label=variant)
        ax.set(title=title, xlabel="Fold externe", ylim=(0, 1))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "fold_scores.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fold_gains.index, fold_gains.delta)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Fold externe", ylabel="Gain du score global (optimisé − référence)")
    fig.tight_layout()
    fig.savefig(output / "fold_gains.png", dpi=180)
    plt.close(fig)
    lines = ["# Optimisation CPU : bilan comparable", "",
             "Sélection interne uniquement ; aucun test ni soumission.", "",
             "| Mesure | Référence imbriquée | Optimisé | Variation |",
             "|---|---:|---:|---:|"]
    for metric in ["total", "severity_macro_f1", "side_effect_micro_f1", "prr_score",
                   "prr_masked_rmse"]:
        lines.append(f"| {metric} | {baseline[metric]:.6f} | {optimized[metric]:.6f} "
                     f"| {gains[metric]:+.6f} |")
    lines.extend(["", "Les paramètres sont choisis dans l'apprentissage de chaque fold.",
                  "Ces essais bornés mesurent une marge d'optimisation, pas un plafond absolu.",
                  "Les folds ont déjà été consultés pour des choix de méthodes historiques :",
                  "ils ne constituent pas un holdout entièrement neuf.",
                  "Aucun mélange de branches n'est choisi a posteriori à partir de ce tableau.",
                  "La soumission actuelle n'est pas remplacée automatiquement."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_nested_cpu_optimization(config_path, *, finalize_only=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    paths = [config_path, config["prr_config"], config["classification_config"],
             config["train"], config["manifest"]]
    for stage_config in (config["prr_config"], config["classification_config"]):
        stage = yaml.safe_load(Path(stage_config).read_text(encoding="utf-8"))
        feature_config = stage.get("base_config", stage.get("paths", {}).get("feature_config"))
        if feature_config:
            paths.append(feature_config)
    fingerprints = {str(path): _sha(path) for path in paths}
    contract = output / "frozen_inputs.json"
    if contract.exists() and load_json(contract) != fingerprints:
        raise ValueError("Input/config changed; use a new run_dir")
    write_json(fingerprints, contract)
    if not finalize_only:
        from .classification_nested_hpo import run_classification_nested_hpo
        from .prr_nested_hpo import run_prr_nested_hpo

        with (output / "run.log").open("a", encoding="utf-8") as log:
            with (contextlib.redirect_stdout(_Tee(sys.stdout, log)),
                  threadpool_limits(limits=int(config.get("cpu_threads", 4)))):
                try:
                    _status(output, "running", "prr")
                    print("Stage 1/2: nested PRR tuning (CPU)", flush=True)
                    run_prr_nested_hpo(Path(config["prr_config"]))
                    _status(output, "running", "classification")
                    print("Stage 2/2: nested classification tuning (CPU)", flush=True)
                    run_classification_nested_hpo(Path(config["classification_config"]))
                except BaseException:
                    _status(output, "failed", "training")
                    traceback.print_exc(file=log)
                    raise
    prr_config = yaml.safe_load(Path(config["prr_config"]).read_text(encoding="utf-8"))
    cls_config = yaml.safe_load(Path(config["classification_config"]).read_text(encoding="utf-8"))
    prr_dir = prr_config.get("run_dir", prr_config.get("paths", {}).get("run_dir"))
    cls_dir = cls_config.get("run_dir", cls_config.get("paths", {}).get("run_dir"))
    try:
        summary = compare_pipeline(config, prr_dir, cls_dir)
    except BaseException:
        _status(output, "failed", "comparison")
        raise
    _status(output, "complete", "comparison")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nested_cpu_optimization.yaml"))
    parser.add_argument("--finalize-only", action="store_true")
    arguments = parser.parse_args()
    run_nested_cpu_optimization(arguments.config, finalize_only=arguments.finalize_only)
