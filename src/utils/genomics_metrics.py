"""Metrics and plotting helpers for genomic classification."""

from __future__ import annotations

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask] == (y_prob[mask] >= 0.5)))
        ece += (np.sum(mask) / len(y_prob)) * abs(acc - conf)
    return float(ece)


def compute_binary_genomic_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    try:
        auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except ValueError:
        auroc = 0.0
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fnr": fnr,
        "ppv": ppv,
        "npv": npv,
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_binary_calibration(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    bins = np.linspace(0.0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    observed = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= left) & (y_prob < right)
        observed.append(float(np.mean(y_true[mask])) if np.any(mask) else np.nan)

    path = os.path.join(output_dir, f"{name}_calibration_curve.png")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.plot(centers, observed, marker="o", label="Observed")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title(f"{name} calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path

