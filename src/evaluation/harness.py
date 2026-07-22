"""Nested LOSO cross-validation harness — fixes REPORT.md M16/M17.

The source repos had NO evaluation harness (the only eval ever run was one fold of
a diagnostic script). This provides the discipline the user's own review demanded:

  * OUTER loop: Leave-One-Scenario-Out (grouped by scenario) — reported metrics.
  * INNER loop: grouped k-fold within each outer-train set — for hyperparameter
    search and calibration fitting ONLY. Never touches the outer-test fold.
  * A loud leakage assertion: no scenario may appear in both train and test.
  * Baselines: majority-class and uniform-random.

This is MACHINERY. It is model-agnostic via an ``estimator_factory`` callable and
is unit-tested on dummy arrays under tests/harness_selftest. It is intentionally
NOT run against a real model here — that is the [YOU — TRAIN/INSPECT] boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from src.evaluation import metrics as M


class GroupLeakageError(AssertionError):
    pass


def assert_no_group_leakage(train_groups: Sequence, test_groups: Sequence) -> None:
    overlap = set(map(str, train_groups)) & set(map(str, test_groups))
    if overlap:
        raise GroupLeakageError(
            f"Scenario(s) span train AND test — leakage: {sorted(overlap)}"
        )


@dataclass
class FoldResult:
    held_out_group: str
    n_test: int
    macro_f1: float
    macro_precision: float
    macro_recall: float
    top3: float
    ece: float
    brier: float


@dataclass
class CVReport:
    per_fold: List[FoldResult] = field(default_factory=list)
    classes: List = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"LOSO folds: {len(self.per_fold)}  classes: {len(self.classes)}"]
        for k, v in self.aggregate.items():
            lines.append(f"  {k}: {v:.4f}")
        for name, b in self.baselines.items():
            lines.append(f"  baseline[{name}].macro_f1: {b.get('macro_f1', float('nan')):.4f}")
        return "\n".join(lines)


# ---- estimator protocol -----------------------------------------------------
# estimator_factory() -> object with:
#     .fit(X_train, y_train, groups_train=None)   (groups optional; used for inner CV)
#     .predict_proba(X_test) -> (n, n_classes)
#     .classes_  (sequence, order matching predict_proba columns)
# A thin sklearn wrapper satisfies this; see tests/harness_selftest.


def _predict_labels(proba: np.ndarray, classes: Sequence):
    idx = proba.argmax(axis=1)
    return [classes[i] for i in idx]


def nested_loso_cv(estimator_factory: Callable[[], object],
                   X: np.ndarray, y: Sequence, groups: Sequence,
                   top_k: int = 3, n_bins: int = 10) -> CVReport:
    X = np.asarray(X, dtype=np.float64)
    y = list(y)
    groups = list(map(str, groups))
    classes = sorted(set(y))
    logo = LeaveOneGroupOut()
    report = CVReport(classes=classes)

    all_true: List = []
    all_pred: List = []

    for train_idx, test_idx in logo.split(X, y, groups):
        g_train = [groups[i] for i in train_idx]
        g_test = [groups[i] for i in test_idx]
        assert_no_group_leakage(g_train, g_test)  # loud on any leakage

        est = estimator_factory()
        try:
            est.fit(X[train_idx], [y[i] for i in train_idx], groups_train=g_train)
        except TypeError:
            est.fit(X[train_idx], [y[i] for i in train_idx])

        proba = np.asarray(est.predict_proba(X[test_idx]))
        est_classes = list(getattr(est, "classes_", classes))
        y_test = [y[i] for i in test_idx]
        pred = _predict_labels(proba, est_classes)

        all_true.extend(y_test)
        all_pred.extend(pred)

        report.per_fold.append(FoldResult(
            held_out_group=g_test[0] if g_test else "?",
            n_test=len(test_idx),
            macro_f1=M.macro_f1(y_test, pred),
            macro_precision=M.macro_precision(y_test, pred),
            macro_recall=M.macro_recall(y_test, pred),
            top3=M.top_k_accuracy(y_test, proba, est_classes, k=top_k),
            ece=M.expected_calibration_error(y_test, proba, est_classes, n_bins=n_bins),
            brier=M.brier_score(y_test, proba, est_classes),
        ))

    # aggregate with bootstrap CIs over the pooled out-of-fold predictions
    f1_point, f1_lo, f1_hi = M.bootstrap_ci(M.macro_f1, all_true, all_pred)
    report.aggregate = {
        "macro_f1": f1_point, "macro_f1_ci_lo": f1_lo, "macro_f1_ci_hi": f1_hi,
        "macro_precision": M.macro_precision(all_true, all_pred),
        "macro_recall": M.macro_recall(all_true, all_pred),
        "mean_fold_macro_f1": float(np.mean([f.macro_f1 for f in report.per_fold])),
        "mean_fold_top3": float(np.mean([f.top3 for f in report.per_fold])),
        "mean_fold_ece": float(np.mean([f.ece for f in report.per_fold])),
        "mean_fold_brier": float(np.mean([f.brier for f in report.per_fold])),
    }
    report.baselines = _baselines(y, all_true, all_pred)
    return report


def _baselines(y_all: Sequence, all_true: Sequence, all_pred: Sequence) -> Dict[str, Dict[str, float]]:
    classes, counts = np.unique(np.asarray(y_all, dtype=object), return_counts=True)
    majority = classes[int(np.argmax(counts))]
    maj_pred = [majority] * len(all_true)
    rng = np.random.default_rng(0)
    rand_pred = list(rng.choice(classes, size=len(all_true)))
    return {
        "majority_class": {"macro_f1": M.macro_f1(all_true, maj_pred)},
        "uniform_random": {"macro_f1": M.macro_f1(all_true, rand_pred)},
    }


def inner_group_kfold(groups_train: Sequence, n_splits: int = 3):
    """Grouped inner splitter for HPO/calibration; caps n_splits at #groups."""
    n_groups = len(set(map(str, groups_train)))
    return GroupKFold(n_splits=max(2, min(n_splits, n_groups)))
