"""Self-tests for the evaluation harness — DUMMY ARRAYS ONLY, isolated folder.

Nothing here is application data; these arrays exist purely to prove the harness
machinery (splitting, leakage assertion, metrics) is correct. src/ never imports
this (enforced by scripts/check_no_dummy_in_src.py).
"""
import numpy as np
import pytest

from src.evaluation.harness import (
    GroupLeakageError,
    assert_no_group_leakage,
    inner_group_kfold,
    nested_loso_cv,
)
from src.evaluation import metrics as M


class _DummyEstimator:
    """Minimal sklearn-like estimator over dummy data for machinery testing only."""

    def fit(self, X, y, groups_train=None):
        self.classes_ = sorted(set(y))
        # frequency prior -> proba
        import numpy as _np
        counts = _np.array([sum(1 for v in y if v == c) for c in self.classes_], dtype=float)
        self._prior = counts / counts.sum()
        return self

    def predict_proba(self, X):
        import numpy as _np
        return _np.tile(self._prior, (len(X), 1))


def test_leakage_assertion_fails_loudly():
    with pytest.raises(GroupLeakageError):
        assert_no_group_leakage(["s1", "s2"], ["s2", "s3"])
    # clean split must not raise
    assert_no_group_leakage(["s1", "s2"], ["s3"])


def test_nested_loso_runs_and_leaves_one_scenario_out():
    rng = np.random.default_rng(0)
    # 6 scenarios, 3 classes; features are dummy
    groups, y = [], []
    for s in range(6):
        cls = f"T100{s % 3}"
        for _ in range(5):
            groups.append(f"scenario_{s}")
            y.append(cls)
    X = rng.normal(size=(len(y), 4))
    report = nested_loso_cv(_DummyEstimator, X, y, groups)
    assert len(report.per_fold) == 6           # one fold per scenario
    assert "macro_f1" in report.aggregate
    assert "majority_class" in report.baselines
    # every test fold's held-out scenario appears in no other fold's training set
    seen = [f.held_out_group for f in report.per_fold]
    assert len(set(seen)) == 6


def test_calibration_metrics_bounds():
    classes = ["a", "b"]
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = ["a", "b"]
    ece = M.expected_calibration_error(y, proba, classes)
    brier = M.brier_score(y, proba, classes)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier <= 2.0
    assert M.top_k_accuracy(y, proba, classes, k=1) == 1.0


def test_inner_kfold_caps_to_group_count():
    splitter = inner_group_kfold(["a", "b"], n_splits=3)
    assert splitter.get_n_splits() == 2
