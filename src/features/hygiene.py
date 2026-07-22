"""Fitted, persisted feature-hygiene transform — fixes REPORT.md M14/M4.

The old code only *printed* diagnostics (constant-feature counts, correlated
pairs). This makes hygiene a real fit-on-train-only transform that:

  1. drops constant (zero-variance) features (VarianceThreshold),
  2. drops one of each highly-correlated pair (|rho| >= 0.98),
  3. scrubs lab-artifact feature columns by name (VM tool / capture-harness
     fingerprints) so the model can't learn the lab instead of the attack.

CRITICAL: fit() is called on TRAIN folds only; transform() replays the frozen
selection on val/test. The fitted selection is persisted (joblib) alongside any
model artifact so serving uses the identical columns (no train/serve skew).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import joblib
import numpy as np

# Feature-name substrings that indicate lab/environment artifacts rather than
# attack behaviour. Our current extractors are count/histogram based and do not
# emit hostname strings as columns, so this is a forward guard: if a future
# extractor adds such a column, it is scrubbed here rather than learned.
LAB_ARTIFACT_SUBSTRINGS = (
    "vmware", "virtualbox", "vbox", "theshire", "mordordc", "capture",
    "harness", "vagrant", "packer",
)


@dataclass
class HygieneTransform:
    correlation_threshold: float = 0.98
    feature_names_in_: List[str] = field(default_factory=list)
    keep_index_: List[int] = field(default_factory=list)
    feature_names_out_: List[str] = field(default_factory=list)
    dropped_constant_: List[str] = field(default_factory=list)
    dropped_correlated_: List[str] = field(default_factory=list)
    dropped_artifact_: List[str] = field(default_factory=list)
    _fitted: bool = False

    def fit(self, X: np.ndarray, feature_names: Sequence[str]) -> "HygieneTransform":
        X = np.asarray(X, dtype=np.float64)
        names = list(feature_names)
        n_features = X.shape[1]
        assert len(names) == n_features, "feature_names length must match X columns"
        self.feature_names_in_ = names

        drop = set()

        # 1. lab-artifact columns by name
        for i, nm in enumerate(names):
            low = nm.lower()
            if any(sub in low for sub in LAB_ARTIFACT_SUBSTRINGS):
                drop.add(i)
                self.dropped_artifact_.append(nm)

        # 2. constant (zero-variance) columns
        stds = X.std(axis=0)
        for i in range(n_features):
            if i in drop:
                continue
            if stds[i] == 0.0:
                drop.add(i)
                self.dropped_constant_.append(names[i])

        # 3. highly-correlated pairs — keep the earlier column, drop the later
        candidate = [i for i in range(n_features) if i not in drop]
        if len(candidate) > 1:
            sub = X[:, candidate]
            # guard against zero-variance sneaking in
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(sub, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            m = len(candidate)
            for a in range(m):
                ia = candidate[a]
                if ia in drop:
                    continue
                for b in range(a + 1, m):
                    ib = candidate[b]
                    if ib in drop:
                        continue
                    if abs(corr[a, b]) >= self.correlation_threshold:
                        drop.add(ib)
                        self.dropped_correlated_.append(names[ib])

        self.keep_index_ = [i for i in range(n_features) if i not in drop]
        self.feature_names_out_ = [names[i] for i in self.keep_index_]
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("HygieneTransform.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.keep_index_]

    def fit_transform(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def report(self) -> dict:
        return {
            "n_in": len(self.feature_names_in_),
            "n_out": len(self.feature_names_out_),
            "dropped_constant": self.dropped_constant_,
            "dropped_correlated": self.dropped_correlated_,
            "dropped_artifact": self.dropped_artifact_,
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HygieneTransform":
        return joblib.load(path)
