"""Real SHAP explanations — replaces etbackend's Gini-importance x value substitute
(REPORT.md M10). Lazy shap import so importing this module never requires shap.

Explanations are best-effort: if SHAP can't be computed (e.g. a calibrated wrapper
around a non-tree estimator), callers get an empty list — never a fabricated one.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _underlying_tree_model(model):
    """Reach the fitted tree estimator inside a CalibratedClassifierCV wrapper."""
    inner = getattr(model, "_model", model)
    calibs = getattr(inner, "calibrated_classifiers_", None)
    if calibs:
        est = getattr(calibs[0], "estimator", None) or getattr(calibs[0], "base_estimator", None)
        if est is not None:
            return est
    return inner


def shap_contributions(artifact, X_row: np.ndarray, predicted_class: str,
                       top_k: int = 5) -> List[Tuple[str, float]]:
    import shap  # lazy

    tree = _underlying_tree_model(artifact.model)
    names = artifact.hygiene.feature_names_out_
    explainer = shap.TreeExplainer(tree)
    vals = explainer.shap_values(X_row)

    # multiclass -> list per class; pick the predicted class column
    classes = list(getattr(tree, "classes_", []))
    if isinstance(vals, list):
        ci = classes.index(predicted_class) if predicted_class in classes else 0
        row = np.asarray(vals[ci])[0]
    else:
        arr = np.asarray(vals)
        row = arr[0] if arr.ndim == 2 else arr[0, :, 0]

    order = np.argsort(-np.abs(row))[:top_k]
    return [(names[i], float(row[i])) for i in order if i < len(names)]
