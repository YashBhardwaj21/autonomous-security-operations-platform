"""Retrieval machinery tests — DUMMY corpus, isolated folder.

The corpus here is a handful of hand-written docs used ONLY to exercise the
index/embedder plumbing and the non-gating property. Real retrieval reads
data/reference/ (fetched via scripts/fetch_threat_intel.py).
"""
from src.retrieval.corpus import Document
from src.retrieval.index import RetrievalIndex
from src.retrieval.embedder import TfidfEmbedder


def _dummy_docs():
    return [
        Document("attack::T1003", "attack", "T1003 OS Credential Dumping lsass memory", ["T1003"], "T1003"),
        Document("attack::T1078", "attack", "T1078 Valid Accounts vpn credential reuse", ["T1078"], "T1078"),
        Document("attack::T1021", "attack", "T1021 Remote Services lateral movement smb rdp", ["T1021"], "T1021"),
    ]


def test_tfidf_index_ranks_relevant_doc_first():
    idx = RetrievalIndex(embedder=TfidfEmbedder()).build(_dummy_docs())
    hits = idx.query("credential dumping from lsass", top_k=2)
    assert hits and hits[0].document.ref == "T1003"


def test_empty_corpus_returns_no_hits_never_fabricates():
    idx = RetrievalIndex(embedder=TfidfEmbedder()).build([])
    assert idx.size == 0
    assert idx.query("anything") == []


def test_retrieval_output_is_plain_data_non_gating():
    # Structural non-gating guarantee: hits are plain Documents/dicts, no callables,
    # no reference to a gate or model — retrieval cannot alter any decision.
    idx = RetrievalIndex(embedder=TfidfEmbedder()).build(_dummy_docs())
    hit = idx.query("remote services", top_k=1)[0]
    d = hit.document.to_dict()
    assert set(d.keys()) == {"doc_id", "source", "ref", "technique_ids", "text"}
    assert all(not callable(v) for v in d.values())
