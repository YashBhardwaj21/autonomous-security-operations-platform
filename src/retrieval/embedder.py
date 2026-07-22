"""Pluggable text embedders for retrieval.

Default is TF-IDF (scikit-learn) — deterministic, no model download, works offline
over the REAL corpus on disk. SentenceTransformer is an optional lazy upgrade
(better semantics, requires the model). Both embed the SAME real corpus; neither
fabricates content.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class TfidfEmbedder:
    backend = "tfidf"

    def __init__(self, **kw):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(stop_words="english", max_features=4096, **kw)
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> "TfidfEmbedder":
        self._vec.fit(list(corpus))
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.encode before fit")
        return self._vec.transform(list(texts)).toarray().astype(np.float32)


class SentenceTransformerEmbedder:
    backend = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy
        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: Sequence[str]) -> "SentenceTransformerEmbedder":
        return self  # no fit needed

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True),
                          dtype=np.float32)


def default_embedder(prefer_transformer: bool = False):
    """Return a SentenceTransformer embedder if requested and importable, else TF-IDF."""
    if prefer_transformer:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return TfidfEmbedder()
