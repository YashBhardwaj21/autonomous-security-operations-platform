"""Scaffold plumbing tests — DUMMY ARRAYS ONLY, isolated folder.

Verifies the attribution model satisfies the harness estimator protocol and that
an UNTRAINED model raises rather than fabricating a confidence (REPORT.md C3).
These fit on dummy arrays purely to exercise plumbing — no real metric is reported.
"""
import numpy as np
import pytest

from src.attribution.model import HierarchicalAttributionModel
from src.attribution.loader import AttributionService


def _dummy(n_per_class=12, n_classes=3, n_feat=8, seed=0):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    gid = 0
    for c in range(n_classes):
        for _ in range(n_per_class):
            X.append(rng.normal(loc=c, size=n_feat))
            y.append(f"T100{c}")
            groups.append(f"scenario_{gid % 6}")
            gid += 1
    return np.asarray(X), y, groups


def test_untrained_model_raises_not_fabricates():
    m = HierarchicalAttributionModel()
    with pytest.raises(RuntimeError):
        m.predict_proba(np.zeros((1, 8)))


def test_uncalibrated_plumbing_satisfies_protocol():
    X, y, groups = _dummy()
    m = HierarchicalAttributionModel(n_estimators=20, calibrate=False).fit(X, y)
    proba = m.predict_proba(X[:5])
    assert proba.shape == (5, len(m.classes_))
    assert set(m.classes_) == {"T1000", "T1001", "T1002"}


def test_grouped_calibration_plumbing():
    X, y, groups = _dummy(n_per_class=20)
    m = HierarchicalAttributionModel(n_estimators=20, calibrate=True).fit(X, y, groups_train=groups)
    proba = m.predict_proba(X[:3])
    assert proba.shape[1] == len(m.classes_)
    # calibrated probabilities are valid distributions
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_service_reports_model_unavailable_when_no_artifact(tmp_path):
    svc = AttributionService(artifact_path=str(tmp_path / "nope.joblib"))
    assert svc.available is False
    res = svc.predict({"proc_powershell": 2.0})
    assert res.status == "model_unavailable"
    assert res.confidence is None   # never a fabricated 0.65
