"""Evaluation metrics for code review filtering.

Computes standard classification metrics plus domain-specific ones:
  - Action rate: fraction of comments surfaced (measures filter aggressiveness)
  - Calibration error: how well scores predict actual outcomes
  - Coverage: fraction of addressed comments that were surfaced
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


@dataclass
class MetricsResult:
    """Container for all evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    brier_score: float
    action_rate: float
    coverage: float
    true_positive_rate: float
    false_positive_rate: float
    n_samples: int
    n_positive: int
    n_negative: int
    confusion: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_str(self) -> str:
        return (
            f"Acc={self.accuracy:.3f} | F1={self.f1:.3f} | "
            f"AUROC={self.auroc:.3f} | ActionRate={self.action_rate:.3f} | "
            f"Coverage={self.coverage:.3f}"
        )


def compute_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    scores: list[float] | np.ndarray | None = None,
) -> MetricsResult:
    """Compute comprehensive evaluation metrics.

    Args:
        labels: Ground truth binary labels (1=addressed, 0=ignored)
        predictions: Binary predictions (1=surface, 0=filter)
        scores: Continuous scores [0, 1] for AUROC/calibration (optional)
    """
    labels = np.array(labels, dtype=int)
    predictions = np.array(predictions, dtype=int)

    if scores is None:
        scores = predictions.astype(float)
    else:
        scores = np.array(scores, dtype=float)
        scores = np.clip(scores, 0, 1)

    n = len(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = n - n_pos

    acc = float(accuracy_score(labels, predictions))
    prec = float(precision_score(labels, predictions, zero_division=0))
    rec = float(recall_score(labels, predictions, zero_division=0))
    f1_val = float(f1_score(labels, predictions, zero_division=0))

    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(labels, scores))
    except ValueError:
        auprc = 0.0

    brier = float(brier_score_loss(labels, scores))
    action_rate = float(np.mean(predictions))

    addressed_mask = labels == 1
    if np.sum(addressed_mask) > 0:
        coverage = float(np.mean(predictions[addressed_mask]))
    else:
        coverage = 0.0

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    return MetricsResult(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1_val,
        auroc=auroc,
        auprc=auprc,
        brier_score=brier,
        action_rate=action_rate,
        coverage=coverage,
        true_positive_rate=float(tpr),
        false_positive_rate=float(fpr),
        n_samples=n,
        n_positive=n_pos,
        n_negative=n_neg,
        confusion={"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    )


def compute_calibration_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute calibration curve data for reliability diagrams."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (scores == bins[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
            bin_accuracies.append(float(np.mean(labels[mask])))
            bin_counts.append(int(np.sum(mask)))

    ece = 0.0
    total = sum(bin_counts)
    for acc, cnt, center in zip(bin_accuracies, bin_counts, bin_centers):
        ece += (cnt / max(total, 1)) * abs(acc - center)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "ece": ece,
    }
