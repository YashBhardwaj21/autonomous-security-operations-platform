"""Deterministic next-step predictor over the data-derived transition matrix.

REPORT.md Stage 6: no neural forecasting, no hardcoded probabilities. Loads the
matrix built by src/prediction/transition.py (real OTRF-derived counts) and returns
the most probable next techniques with their support. If the matrix is absent or a
technique is unseen, returns an empty list — honest, never fabricated.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from src.prediction.transition import TransitionMatrix

DEFAULT_MATRIX = os.path.join("models", "transition_matrix.json")


class NextStepPredictor:
    def __init__(self, matrix_path: str = DEFAULT_MATRIX):
        self.matrix_path = matrix_path
        self._tm: Optional[TransitionMatrix] = None
        if os.path.exists(matrix_path):
            self._tm = TransitionMatrix.load(matrix_path)

    @property
    def available(self) -> bool:
        return self._tm is not None and bool(self._tm.matrix)

    def predict_next(self, technique: str, top_k: int = 3) -> List[dict]:
        if self._tm is None:
            return []
        out = []
        for tech, prob in self._tm.next_techniques(technique, top_k):
            support = self._tm.support.get(technique.split(".")[0], {}).get(tech)
            out.append({"technique": tech, "probability": round(prob, 4),
                        "support": support, "provenance": self._tm.provenance})
        return out
