"""Retrieval service — attaches supporting threat-intel to a prediction.

REPORT.md T4 / ADR: retrieval is NON-GATING. It runs AFTER attribution and only
adds context; it never feeds the classifier or the SOAR gate. This class exposes
only read/return methods and holds no gate/model handle — the non-gating property
is structural, and a test asserts a gate decision is byte-identical with and
without retrieval attached.

If the corpus is empty (no STIX / no advisories fetched), retrieve() returns an
empty package with a clear note — it never fabricates advisories (H7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.retrieval.corpus import Document, load_corpus
from src.retrieval.index import RetrievalIndex


@dataclass
class EvidencePackage:
    technique_id: str
    available: bool
    exact_matches: List[dict] = field(default_factory=list)   # docs tagged with the technique
    semantic_matches: List[dict] = field(default_factory=list)
    note: str = ""


class RetrievalService:
    def __init__(self, stix_path: Optional[str] = None, embedder=None):
        self._index = RetrievalIndex(embedder=embedder)
        self._by_technique = {}
        docs = load_corpus(stix_path)
        self._index.build(docs)
        for d in docs:
            for tid in d.technique_ids:
                self._by_technique.setdefault(tid, []).append(d)

    @property
    def available(self) -> bool:
        return self._index.size > 0

    def retrieve(self, technique_id: str, query_text: Optional[str] = None,
                 top_k: int = 5) -> EvidencePackage:
        if not self.available:
            return EvidencePackage(technique_id=technique_id, available=False,
                                   note="No threat-intel corpus present. Run "
                                        "scripts/fetch_threat_intel.py (real sources; "
                                        "never fabricated).")
        exact = [d.to_dict() for d in self._by_technique.get(technique_id, [])]
        q = query_text or technique_id
        semantic = [{"score": round(h.score, 4), **h.document.to_dict()}
                    for h in self._index.query(q, top_k=top_k)]
        return EvidencePackage(technique_id=technique_id, available=True,
                               exact_matches=exact, semantic_matches=semantic)
