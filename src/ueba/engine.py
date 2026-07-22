"""UEBA anomaly engine — ported from etbackend's online detector.

Ensemble: per-entity Welford running z-score + a global online IsolationForest.
Kept because it was the most honest ML component in either repo. Improvements:

* REPORT.md M6 — this operates on its OWN feature space (raw per-entity behavioural
  counts), SEPARATE from attribution features. It does NOT emit an anomaly_score
  that then becomes an attribution input.
* REPORT.md M-6 (self-poisoning) — the IsolationForest training buffer excludes
  vectors that scored as anomalies, so sustained attack traffic does not normalise
  itself into the baseline.
* No pre-trained pkl and no benign-baseline claim until a real benign corpus (LANL/
  CICIDS/TON_IoT) is wired in (see src/ueba/baseline.py loaders in Phase 8 scripts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class _Welford:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def z(self, x: float) -> float:
        if self.n < 2:
            return 0.0
        std = math.sqrt(self.m2 / (self.n - 1))
        return 0.0 if std == 0 else (x - self.mean) / std


@dataclass
class AnomalyResult:
    is_anomalous: bool
    score: float                      # 0..1
    contributing_features: List[str] = field(default_factory=list)


class UEBAEngine:
    SPACE = "ueba"
    MAX_TRAIN = 5000
    RETRAIN_EVERY = 500

    def __init__(self, anomaly_threshold: float = 0.5, contamination: float = 0.05):
        self.anomaly_threshold = anomaly_threshold
        self.contamination = contamination
        self._stats: Dict[str, _Welford] = {}
        self._feature_order: Optional[List[str]] = None
        self._buffer: List[List[float]] = []
        self._if = None
        self._seen = 0

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        if self._feature_order is None:
            self._feature_order = sorted(features.keys())
        return np.array([float(features.get(k, 0.0)) for k in self._feature_order], dtype=np.float64)

    def _stat_score(self, entity_key: str, features: Dict[str, float]) -> (float, List[str]):
        contributing = []
        zsum, n = 0.0, 0
        for k, v in features.items():
            w = self._stats.setdefault(f"{entity_key}::{k}", _Welford())
            z = w.z(float(v))
            w.update(float(v))
            zabs = min(abs(z), 10.0)
            zsum += zabs
            n += 1
            if zabs > 1.0:
                contributing.append(k)
        mean_z = (zsum / n) if n else 0.0
        return min(1.0, mean_z / 6.0), contributing

    def _if_score(self, vec: np.ndarray) -> float:
        if self._if is None:
            return 0.5  # neutral until trained
        raw = float(self._if.score_samples(vec.reshape(1, -1))[0])
        return 1.0 / (1.0 + math.exp(5.0 * raw))

    def _maybe_retrain(self) -> None:
        if self._seen % self.RETRAIN_EVERY != 0 or len(self._buffer) < 50:
            return
        from sklearn.ensemble import IsolationForest
        X = np.asarray(self._buffer[-self.MAX_TRAIN:])
        self._if = IsolationForest(contamination=self.contamination, random_state=42).fit(X)

    def process(self, entity_key: str, features: Dict[str, float]) -> AnomalyResult:
        vec = self._vectorize(features)
        stat, contributing = self._stat_score(entity_key, features)
        ifs = self._if_score(vec)
        score = 0.6 * stat + 0.4 * ifs
        is_anom = score >= self.anomaly_threshold
        self._seen += 1
        # self-poisoning guard: only feed NON-anomalous vectors to the baseline buffer
        if not is_anom:
            self._buffer.append(vec.tolist())
            self._maybe_retrain()
        return AnomalyResult(is_anomalous=is_anom, score=round(score, 4),
                             contributing_features=contributing[:5])
