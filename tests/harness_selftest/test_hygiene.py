"""Self-tests for the fitted hygiene transform — DUMMY ARRAYS ONLY, isolated."""
import numpy as np

from src.features.hygiene import HygieneTransform


def test_drops_constant_and_correlated_and_artifact():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.column_stack([
        a,               # keep
        a * 2.0 + 1.0,   # perfectly correlated -> dropped
        np.ones(200),    # constant -> dropped
        rng.normal(size=200),  # keep
        rng.normal(size=200),  # will be named as a lab artifact -> dropped
    ])
    names = ["proc_x", "proc_x_scaled", "const_col", "net_y", "vmware_tools_flag"]
    h = HygieneTransform().fit(X, names)
    out = h.transform(X)
    assert "const_col" in h.dropped_constant_
    assert "proc_x_scaled" in h.dropped_correlated_
    assert "vmware_tools_flag" in h.dropped_artifact_
    assert h.feature_names_out_ == ["proc_x", "net_y"]
    assert out.shape[1] == 2


def test_transform_replays_frozen_selection(tmp_path):
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=50), np.ones(50)])
    h = HygieneTransform().fit(X, ["keep", "const"])
    p = tmp_path / "hygiene.joblib"
    h.save(str(p))
    h2 = HygieneTransform.load(str(p))
    Xnew = np.column_stack([rng.normal(size=10), np.ones(10)])
    assert h2.transform(Xnew).shape == (10, 1)   # same frozen columns on new data
