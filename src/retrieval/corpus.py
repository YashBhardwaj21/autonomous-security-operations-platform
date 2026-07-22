"""Threat-intel corpus assembly from REAL sources on disk — no fabrication.

Documents come only from files that actually exist:
  * ATT&CK technique descriptions from the STIX bundle (data/reference/enterprise-attack.json)
  * CERT-In / CISA advisory text from data/reference/advisories/*.json (fetched by
    scripts/fetch_threat_intel.py — real advisory text, real refs; NEVER invented)
  * CVE summaries from data/reference/cves/*.json

REPORT.md H7: the old retriever returned canned text keyed by fabricated advisory
IDs (e.g. "CIAD-2026-001"). Here, if an advisory source file is absent, those
documents simply don't exist in the index — the system never invents them.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from src.ml.attck_loader import ATTCKLoader

REFERENCE_DIR = os.path.join("data", "reference")


@dataclass
class Document:
    doc_id: str
    source: str                     # "attack" | "cert-in" | "cisa" | "cve"
    text: str
    technique_ids: List[str] = field(default_factory=list)
    ref: Optional[str] = None       # real advisory/CVE identifier, if any

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "source": self.source, "ref": self.ref,
                "technique_ids": self.technique_ids, "text": self.text}


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_attack_documents(stix_path: Optional[str] = None) -> List[Document]:
    loader = ATTCKLoader(stix_path)
    docs: List[Document] = []
    if not loader.available:
        return docs
    for tid in loader.technique_ids():
        name = loader.get_name(tid)
        detection = loader.get_detection(tid)
        text = f"{tid} {name}. {detection}".strip()
        docs.append(Document(doc_id=f"attack::{tid}", source="attack", text=text,
                             technique_ids=[tid], ref=tid))
    return docs


def _load_advisory_dir(subdir: str, source: str) -> List[Document]:
    docs: List[Document] = []
    for path in glob.glob(os.path.join(REFERENCE_DIR, subdir, "*.json")):
        obj = _load_json(path)
        if not obj:
            continue
        # expected shape: {"ref": "...", "text": "...", "technique_ids": [...]}
        text = obj.get("text") or obj.get("summary") or ""
        if not text:
            continue
        docs.append(Document(
            doc_id=f"{source}::{obj.get('ref', os.path.basename(path))}",
            source=source, text=text,
            technique_ids=list(obj.get("technique_ids", [])),
            ref=obj.get("ref"),
        ))
    return docs


def load_corpus(stix_path: Optional[str] = None) -> List[Document]:
    docs: List[Document] = []
    docs += load_attack_documents(stix_path)
    docs += _load_advisory_dir("advisories/cert-in", "cert-in")
    docs += _load_advisory_dir("advisories/cisa", "cisa")
    docs += _load_advisory_dir("cves", "cve")
    return docs
