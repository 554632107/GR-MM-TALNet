# -*- coding: utf-8 -*-
"""
Compute AUROC, AUPRC, and classification metrics at the Youden and maximum-F1 thresholds.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    return float(thresholds[int(np.argmax(youden_j))])


def find_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Probability threshold that maximizes F1 on one fold's test set."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float | None = None) -> dict:
    if threshold is None:
        threshold = find_youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "threshold": threshold,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def format_metrics(m: dict) -> str:
    return (
        f"AUROC={m['auroc']:.4f} | AUPRC={m['auprc']:.4f} | "
        f"BA={m['balanced_acc']:.4f} | "
        f"Sens={m['sensitivity']:.4f} | Spec={m['specificity']:.4f} | "
        f"F1={m['f1']:.4f} | thr={m['threshold']:.3f}"
    )
