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


class HygienicAttributionEstimator:
    """Wrapper estimator fitting HygieneTransform strictly on outer train fold before model fitting."""

    def __init__(self, feature_names: Sequence[str], n_estimators: int = 300,
                 max_depth: Optional[int] = None, calibrate: bool = True,
                 use_xgboost: bool = False, random_state: int = 42):
        from src.features.hygiene import HygieneTransform
        self.feature_names = list(feature_names)
        self.hygiene = HygieneTransform()
        self.model = HierarchicalAttributionModel(
            n_estimators=n_estimators, max_depth=max_depth,
            calibrate=calibrate, use_xgboost=use_xgboost,
            random_state=random_state
        )
        self.classes_: List = []

    def fit(self, X, y, groups_train: Optional[Sequence] = None):
        X = np.asarray(X, dtype=np.float64)
        self.hygiene.fit(X, self.feature_names)
        Xh = self.hygiene.transform(X)
        self.model.fit(Xh, y, groups_train=groups_train)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xh = self.hygiene.transform(X)
        return self.model.predict_proba(Xh)

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[i] for i in proba.argmax(axis=1)]

    @classmethod
    def factory(cls, feature_names: Sequence[str], **kw):
        return lambda: cls(feature_names, **kw)

