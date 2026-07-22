"""Data-derived ATT&CK transition matrix — fixes REPORT.md H8/M11.

The old etbackend matrix was 27 hand-written rows carrying a fabricated
``source="attck_procedures+campaign_data"`` label. This derives transitions from
REAL data: the ordered ``attack_mappings`` sequences of OTRF compound + APT29-evals
multi-stage scenarios. Provenance recorded here is TRUE and reproducible — the
matrix is a deterministic count over observed scenario technique orderings.

Method:
  * For each multi-stage scenario, read its ordered attack_mappings -> a sequence
    of parent techniques (consecutive duplicates collapsed).
  * Count directed adjacent transitions A->B across all scenarios.
  * Normalise per source technique to probabilities.
Every probability traces to a raw count (``support``); nothing is invented. If no
multi-stage scenarios are available, the matrix is empty (honest), not fabricated.
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

DEFAULT_ROOT = os.path.join("data", "raw", "Security-Datasets")
MULTISTAGE_TYPES = {"compound"}


def _parent(t: str) -> str:
    return str(t).split(".")[0]


def technique_sequence(metadata: dict) -> List[str]:
    seq: List[str] = []
    for m in (metadata.get("attack_mappings") or []):
        t = m.get("technique")
        if not t:
            continue
        p = _parent(t)
        if not seq or seq[-1] != p:  # collapse consecutive duplicates
            seq.append(p)
    return seq


def _is_multistage(metadata: dict, seq_len: int) -> bool:
    if str(metadata.get("type", "")).lower() in MULTISTAGE_TYPES:
        return True
    # APT29-evals day scenarios are multi-stage even if not tagged "compound"
    title = str(metadata.get("title", "")).lower()
    if "apt29" in title or "evals" in title:
        return True
    return seq_len >= 3


@dataclass
class TransitionMatrix:
    provenance: str
    matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)   # A -> {B: prob}
    support: Dict[str, Dict[str, int]] = field(default_factory=dict)    # A -> {B: count}
    n_scenarios: int = 0
    n_transitions: int = 0

    def next_techniques(self, technique: str, top_k: int = 3) -> List[Tuple[str, float]]:
        row = self.matrix.get(_parent(technique), {})
        return sorted(row.items(), key=lambda kv: -kv[1])[:top_k]

    def to_json(self) -> dict:
        return {
            "provenance": self.provenance,
            "n_scenarios": self.n_scenarios,
            "n_transitions": self.n_transitions,
            "matrix": self.matrix,
            "support": self.support,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    @staticmethod
    def load(path: str) -> "TransitionMatrix":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return TransitionMatrix(
            provenance=d["provenance"], matrix=d["matrix"], support=d.get("support", {}),
            n_scenarios=d.get("n_scenarios", 0), n_transitions=d.get("n_transitions", 0),
        )


def build_transition_matrix(root: str = DEFAULT_ROOT) -> TransitionMatrix:
    counts: Dict[str, Counter] = defaultdict(Counter)
    n_scen, n_trans = 0, 0

    meta_glob = os.path.join(root, "datasets", "**", "_metadata", "*.yaml")
    for meta_path in glob.glob(meta_glob, recursive=True):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
        except Exception:
            continue
        seq = technique_sequence(meta)
        if not _is_multistage(meta, len(seq)) or len(seq) < 2:
            continue
        n_scen += 1
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
            n_trans += 1

    matrix: Dict[str, Dict[str, float]] = {}
    support: Dict[str, Dict[str, int]] = {}
    for a, ctr in counts.items():
        total = sum(ctr.values())
        matrix[a] = {b: c / total for b, c in ctr.items()}
        support[a] = dict(ctr)

    return TransitionMatrix(
        provenance=("derived from ordered attack_mappings of OTRF compound + "
                    "APT29-evals multi-stage scenarios (counts in `support`)"),
        matrix=matrix, support=support, n_scenarios=n_scen, n_transitions=n_trans,
    )
