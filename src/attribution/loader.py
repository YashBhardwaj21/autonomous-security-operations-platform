"""Attribution artifact loader — replaces etbackend's heuristic-fallback classifier.

REPORT.md C3/H1: the old classifier silently fabricated a 0.65-confidence label
when no model was present. Here, if no trained artifact exists, we return an
explicit ``model_unavailable`` result — never a made-up label or confidence.

A prediction bundles the calibrated probability AND the SHAP contributing features
(populated once a model is trained; see src/attribution/explain.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.features.hygiene import HygieneTransform


@dataclass
class AttributionResult:
    status: str                      # "ok" | "model_unavailable"
    technique: Optional[str] = None
    confidence: Optional[float] = None       # CALIBRATED probability (H1)
    top_k: List[tuple] = field(default_factory=list)   # [(technique, prob), ...]
    contributing_features: List[tuple] = field(default_factory=list)  # SHAP (technique)
    model_version: Optional[str] = None


@dataclass
class AttributionArtifact:
    model: object                    # HierarchicalAttributionModel
    hygiene: HygieneTransform
    feature_names_in: List[str]
    version: str

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AttributionArtifact":
        return joblib.load(path)


class AttributionService:
    """Loads an artifact if present; otherwise every prediction is model_unavailable."""

    def __init__(self, artifact_path: str = os.path.join("models", "attribution.joblib")):
        self.artifact_path = artifact_path
        self._artifact: Optional[AttributionArtifact] = None
        if os.path.exists(artifact_path):
            self._artifact = AttributionArtifact.load(artifact_path)

    @property
    def available(self) -> bool:
        return self._artifact is not None

    def predict(self, feature_dict: Dict[str, float], top_k: int = 3,
                explain: bool = True) -> AttributionResult:
        if self._artifact is None:
            return AttributionResult(status="model_unavailable")

        art = self._artifact
        row = np.array([[feature_dict.get(n, 0.0) for n in art.feature_names_in]], dtype=np.float64)
        Xh = art.hygiene.transform(row)
        proba = art.model.predict_proba(Xh)[0]
        classes = art.model.classes_
        order = np.argsort(-proba)
        top = [(classes[i], float(proba[i])) for i in order[:top_k]]

        contributing: List[tuple] = []
        if explain:
            try:
                from src.attribution.explain import shap_contributions
                contributing = shap_contributions(art, Xh, predicted_class=classes[order[0]])
            except Exception:
                contributing = []  # explanation is best-effort, never fabricated

        return AttributionResult(
            status="ok",
            technique=classes[order[0]],
            confidence=float(proba[order[0]]),
            top_k=top,
            contributing_features=contributing,
            model_version=art.version,
        )
