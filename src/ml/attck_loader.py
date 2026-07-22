"""MITRE ATT&CK STIX loader — ported from asop src/ml/attck_loader.py.

Provides technique -> tactic(s) mapping used to constrain the hierarchical
classifier (technique candidates limited to the predicted tactic — REPORT.md
Stage 3). Degrades gracefully: if the STIX bundle is absent, mapping is empty and
the classifier falls back to unconstrained technique classification (honest, not
fabricated). Fetch the bundle with scripts/fetch_attack_stix.py (write-only; YOU run).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

_DEFAULT_STIX = os.path.join("data", "reference", "enterprise-attack.json")


class ATTCKLoader:
    def __init__(self, stix_path: Optional[str] = None):
        self.stix_path = stix_path or _DEFAULT_STIX
        self._techniques: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.stix_path):
            self._techniques = {}
            return
        with open(self.stix_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            attck_id = next((r["external_id"] for r in obj.get("external_references", [])
                             if r.get("source_name") == "mitre-attack"), None)
            if attck_id:
                self._techniques[attck_id] = obj

    @property
    def available(self) -> bool:
        return bool(self._techniques)

    def get_name(self, attck_id: str) -> str:
        obj = self._techniques.get(attck_id)
        return obj.get("name", "Unknown") if obj else "Unknown"

    def get_tactics(self, attck_id: str) -> List[str]:
        obj = self._techniques.get(attck_id)
        if not obj:
            return []
        return [p.get("phase_name") for p in obj.get("kill_chain_phases", [])
                if p.get("kill_chain_name") == "mitre-attack"]

    def get_detection(self, attck_id: str) -> str:
        obj = self._techniques.get(attck_id)
        return obj.get("x_mitre_detection", "") if obj else ""

    def technique_ids(self) -> List[str]:
        return list(self._techniques)

    def tactic_to_techniques(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for tid in self._techniques:
            for tactic in self.get_tactics(tid):
                out.setdefault(tactic, []).append(tid)
        return out
