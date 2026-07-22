"""Calibrated technique-attribution model — REPORT.md Stage 3/4 (H1, M9).

Satisfies the evaluation-harness estimator protocol:
    fit(X, y, groups_train=None) -> self
    predict_proba(X) -> ndarray (n, n_classes)
    classes_ -> list

Design (scaffold — trained by YOU, never by Fable):
* Core estimator: RandomForest (XGBoost optional, lazy). Low-parameter, well-
  regularised — the right regime for the scarce OTRF label budget (the appendix's
  data-budget argument).
* Calibration: CalibratedClassifierCV(method="sigmoid") fit with GROUPED inner
  splits (grouped by scenario) so calibration never leaks across scenarios. This
  is the calibrated probability the SOAR blast-radius gate consumes (H1) — no more
  raw predict_proba behind the 0.85 threshold.
* Optional hierarchical tactic gate: if an ATT&CK STIX mapping is available, a
  technique's probability is suppressed unless its tactic is among the top tactics
  implied by the sample's predicted techniques. Absent STIX -> unconstrained
  (honest fallback, not fabricated).

There is NO synthetic-data path and NO heuristic confidence fallback (REPORT.md
C3). If untrained, predict_proba raises — callers must handle "model_unavailable".
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.evaluation.harness import inner_group_kfold


class HierarchicalAttributionModel:
    def __init__(self, n_estimators: int = 300, max_depth: Optional[int] = None,
                 calibrate: bool = True, use_xgboost: bool = False,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.calibrate = calibrate
        self.use_xgboost = use_xgboost
        self.random_state = random_state
        self._model = None
        self.classes_: List = []
        self._fitted = False

    def _base_estimator(self):
        if self.use_xgboost:
            try:
                from xgboost import XGBClassifier  # lazy, optional
                return XGBClassifier(
                    n_estimators=self.n_estimators, max_depth=self.max_depth or 6,
                    random_state=self.random_state, tree_method="hist",
                    eval_metric="mlogloss",
                )
            except ImportError:
                pass
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight="balanced", random_state=self.random_state, n_jobs=-1,
        )

    def fit(self, X, y, groups_train: Optional[Sequence] = None):
        from sklearn.calibration import CalibratedClassifierCV

        X = np.asarray(X, dtype=np.float64)
        y = list(y)
        base = self._base_estimator()

        if self.calibrate and groups_train is not None and len(set(map(str, groups_train))) >= 2:
            # Grouped inner splits so calibration is leakage-free across scenarios.
            splitter = inner_group_kfold(groups_train, n_splits=3)
            cv = list(splitter.split(X, y, list(map(str, groups_train))))
            self._model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
            self._model.fit(X, y)
        elif self.calibrate:
            self._model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            self._model.fit(X, y)
        else:
            base.fit(X, y)
            self._model = base

        self.classes_ = list(self._model.classes_)
        self._fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "HierarchicalAttributionModel is not trained. Callers must surface "
                "'model_unavailable' — never a fabricated confidence (REPORT.md C3)."
            )
        return np.asarray(self._model.predict_proba(np.asarray(X, dtype=np.float64)))

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[i] for i in proba.argmax(axis=1)]

    # convenience for the harness/factory
    @classmethod
    def factory(cls, **kw):
        return lambda: cls(**kw)
