"""Embedding retrieval index over the real threat-intel corpus.

Uses FAISS when available (lazy) else a NumPy cosine-similarity fallback — both
give identical top-k semantics over normalized vectors. Non-gating by construction:
this class only reads text and returns documents; it holds no reference to the SOAR
gate or the classifier and cannot influence any decision (asserted in tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.retrieval.corpus import Document
from src.retrieval.embedder import TfidfEmbedder, default_embedder


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@dataclass
class RetrievalHit:
    document: Document
    score: float


class RetrievalIndex:
    def __init__(self, embedder=None):
        self.embedder = embedder or default_embedder(prefer_transformer=False)
        self._docs: List[Document] = []
        self._emb: Optional[np.ndarray] = None

    def build(self, documents: Sequence[Document]) -> "RetrievalIndex":
        self._docs = list(documents)
        if not self._docs:
            self._emb = None
            return self
        texts = [d.text for d in self._docs]
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(texts)
        self._emb = _normalize(self.embedder.encode(texts))
        return self

    @property
    def size(self) -> int:
        return len(self._docs)

    def query(self, text: str, top_k: int = 5,
              sources: Optional[Sequence[str]] = None) -> List[RetrievalHit]:
        if self._emb is None or not text.strip():
            return []
        q = _normalize(self.embedder.encode([text]))[0]
        sims = self._emb @ q
        order = np.argsort(-sims)
        hits: List[RetrievalHit] = []
        for i in order:
            doc = self._docs[i]
            if sources and doc.source not in sources:
                continue
            hits.append(RetrievalHit(document=doc, score=float(sims[i])))
            if len(hits) >= top_k:
                break
        return hits
