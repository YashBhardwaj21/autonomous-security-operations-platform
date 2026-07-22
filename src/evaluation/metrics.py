"""Evaluation metrics — macro-F1, top-k, ECE, Brier, confusion matrix, bootstrap CIs.

Net-new: neither source repo had ML metrics (asop's evaluation/metrics.py was
parser-funnel counters). Implements the calibration metrics the plan requires
(ECE, Brier) which are essential because a calibrated probability feeds the SOAR
blast-radius gate (REPORT.md H1).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def macro_precision(y_true, y_pred) -> float:
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


def macro_recall(y_true, y_pred) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def top_k_accuracy(y_true, proba: np.ndarray, classes: Sequence, k: int = 3) -> float:
    """Fraction of samples whose true label is within the top-k predicted probs."""
    proba = np.asarray(proba)
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    topk = np.argsort(-proba, axis=1)[:, :k]
    hits = 0
    for i, yt in enumerate(y_true):
        if yt in idx and idx[yt] in topk[i]:
            hits += 1
    return hits / len(y_true) if len(y_true) else 0.0


def expected_calibration_error(y_true, proba: np.ndarray, classes: Sequence,
                               n_bins: int = 10) -> float:
    """ECE over the predicted (max-prob) class, equal-width confidence bins."""
    proba = np.asarray(proba)
    classes = list(classes)
    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    pred = [classes[i] for i in pred_idx]
    correct = np.array([1.0 if pred[i] == y_true[i] else 0.0 for i in range(len(y_true))])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    if n == 0:
        return 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def brier_score(y_true, proba: np.ndarray, classes: Sequence) -> float:
    """Multiclass Brier score = mean squared error vs one-hot true labels."""
    proba = np.asarray(proba)
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba)
    for r, yt in enumerate(y_true):
        if yt in idx:
            onehot[r, idx[yt]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def confusion(y_true, y_pred, labels: Sequence) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 y_true: Sequence, y_pred: Sequence,
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Return (point, lo, hi) percentile bootstrap CI for a paired-array metric."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    n = len(yt)
    point = metric_fn(yt, yp)
    if n == 0:
        return point, point, point
    stats = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        stats[b] = metric_fn(yt[i], yp[i])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return float(point), lo, hi
