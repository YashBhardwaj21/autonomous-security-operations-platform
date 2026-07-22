"""Vulnerability prioritization scorer — ported from etbackend, with EPSS added.

REPORT.md B4: the old scorer had NO EPSS (the ML plan and SOC workflow both name
it), used a boolean exploit flag, and computed "attack-path exposure" as a static
label lookup. Here:

* EPSS (exploit probability, 0..1) is a first-class input.
* attack-path exposure can be supplied as a REAL reachability fraction from the
  digital twin (twin.blast_radius) rather than a hardcoded zone label.
* CVE data comes from a real feed the operator fetches (no hardcoded seed CVEs).

Score is a transparent weighted sum; weights are explicit and documented, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_WEIGHTS = {
    "epss": 0.30,              # probability of exploitation in the wild (real EPSS)
    "cvss": 0.25,             # normalized severity
    "asset_criticality": 0.20,
    "attack_path_exposure": 0.15,  # real twin reachability fraction when available
    "ttp_overlap": 0.10,      # predicted-technique relevance
}


@dataclass
class VulnScore:
    cve: str
    risk: float               # 0..1
    components: Dict[str, float]


def _tier_to_criticality(tier: int) -> float:
    return {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}.get(int(tier), 0.25)


def score_vulnerability(cve: str, cvss: float, epss: float,
                        asset_criticality_tier: int,
                        attack_path_exposure: float = 0.0,
                        ttp_overlap: float = 0.0,
                        weights: Optional[Dict[str, float]] = None) -> VulnScore:
    w = weights or DEFAULT_WEIGHTS
    comps = {
        "epss": max(0.0, min(1.0, epss)),
        "cvss": max(0.0, min(1.0, cvss / 10.0)),
        "asset_criticality": _tier_to_criticality(asset_criticality_tier),
        "attack_path_exposure": max(0.0, min(1.0, attack_path_exposure)),
        "ttp_overlap": max(0.0, min(1.0, ttp_overlap)),
    }
    risk = sum(w[k] * comps[k] for k in comps)
    return VulnScore(cve=cve, risk=round(risk, 4), components=comps)
