from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def probability_metrics(y, probability, threshold, top_k, rare_mask):
    flat_y, flat_p = y.ravel(), probability.ravel()
    decision = probability >= threshold
    bins = np.minimum((flat_p * 10).astype(int), 9)
    ece = sum(
        float((bins == b).mean())
        * abs(float(flat_p[bins == b].mean()) - float(flat_y[bins == b].mean()))
        for b in range(10)
        if (bins == b).any()
    )
    top = np.argsort(-probability, axis=1)[:, : min(top_k, y.shape[1])]
    recovered = np.take_along_axis(y, top, axis=1).sum(axis=1)
    observed = y.sum(axis=1)
    present = observed > 0
    target_ap = [
        float(average_precision_score(y[:, j], probability[:, j]))
        for j in range(y.shape[1])
        if y[:, j].sum()
    ]
    rare_true = y[:, rare_mask]
    rare_pred = decision[:, rare_mask]
    return {
        "auprc_micro": float(average_precision_score(flat_y, flat_p)) if flat_y.sum() else None,
        "auprc_macro": float(np.mean(target_ap)) if target_ap else None,
        "auroc_micro": float(roc_auc_score(flat_y, flat_p))
        if np.unique(flat_y).size == 2
        else None,
        "precision_micro": float(precision_score(y, decision, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y, decision, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y, decision, average="micro", zero_division=0)),
        "brier": float(brier_score_loss(flat_y, flat_p)),
        "ece_10_bins": ece,
        "precision_at_k": float(np.mean(recovered / top.shape[1])),
        "recall_at_k": float(np.mean(recovered[present] / observed[present]))
        if present.any()
        else None,
        "rare_recall": float((rare_true & rare_pred).sum() / rare_true.sum())
        if rare_true.sum()
        else None,
        "rare_targets": int(rare_mask.sum()),
        "pairs": len(y),
        "positive_labels": int(y.sum()),
    }
