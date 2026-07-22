"""Multi-label ATT&CK labeling from OTRF metadata — fixes REPORT.md H6/M7/M15.

The old code took ``metadata["techniques"][0]`` — first-technique-wins — which
mislabels exactly the multi-stage scenarios that matter. The real OTRF metadata
yaml (verified) carries a full ``attack_mappings`` list, each entry:

    - technique: T1552
      sub-technique: "004"
      tactics: [TA0006]

We build the FULL multi-label target (all techniques, all tactics) per scenario.
Sub-techniques are folded to parent techniques for label stability, but the raw
sub-technique is retained for reporting.

``unsupported_class`` (M15): any technique appearing in fewer than
``MIN_SCENARIOS_PER_CLASS`` scenarios is excluded from the supervised label set
and reported, never silently trained on.

CRITICAL DATA BOUNDARY (REPORT.md C1): benign-only sources (LANL/CICIDS/TON_IoT)
must NEVER receive an attack label. ``label_from_metadata`` refuses a non-empty
label for a benign source and raises — the boundary is enforced in code, not docs.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import yaml

from src.canon.schema import BENIGN_ONLY_SOURCES, SourceType

MIN_SCENARIOS_PER_CLASS = 3
UNSUPPORTED_CLASS = "unsupported_class"


@dataclass
class ScenarioLabel:
    scenario_id: str
    techniques: Set[str] = field(default_factory=set)      # parent techniques, e.g. {"T1552","T1078"}
    sub_techniques: Set[str] = field(default_factory=set)   # e.g. {"T1552.004"}
    tactics: Set[str] = field(default_factory=set)          # e.g. {"TA0006","TA0001"}

    def is_empty(self) -> bool:
        return not self.techniques


def _parent(technique: str) -> str:
    return str(technique).split(".")[0]


def label_from_metadata(metadata: Dict[str, Any], scenario_id: str,
                        source: SourceType = SourceType.OTRF) -> ScenarioLabel:
    """Build a multi-label target from one scenario's metadata dict."""
    label = ScenarioLabel(scenario_id=scenario_id)
    mappings = metadata.get("attack_mappings") or []
    for m in mappings:
        tech = m.get("technique")
        if not tech:
            continue
        sub = m.get("sub-technique") or m.get("sub_technique")
        parent = _parent(tech)
        label.techniques.add(parent)
        if sub:
            label.sub_techniques.add(f"{parent}.{sub}")
        for ta in (m.get("tactics") or []):
            if ta:
                label.tactics.add(str(ta))

    if source in BENIGN_ONLY_SOURCES and not label.is_empty():
        raise ValueError(
            f"Benign-only source {source.value} received attack labels "
            f"{sorted(label.techniques)} for scenario {scenario_id}. "
            "Benign sources feed the UEBA baseline only (REPORT.md C1)."
        )
    return label


def label_from_yaml(path: str, scenario_id: str,
                    source: SourceType = SourceType.OTRF) -> ScenarioLabel:
    with open(path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f) or {}
    return label_from_metadata(metadata, scenario_id, source)


def build_label_space(labels: List[ScenarioLabel],
                      min_scenarios: int = MIN_SCENARIOS_PER_CLASS):
    """Return (supported_techniques, unsupported_techniques, technique_scenario_counts).

    A technique is *supported* iff it appears in >= min_scenarios distinct scenarios.
    Unsupported techniques are collapsed to UNSUPPORTED_CLASS downstream.
    """
    counts: Counter = Counter()
    for lab in labels:
        for tech in lab.techniques:
            counts[tech] += 1
    supported = {t for t, c in counts.items() if c >= min_scenarios}
    unsupported = {t for t, c in counts.items() if c < min_scenarios}
    return supported, unsupported, dict(counts)


def project_labels(labels: List[ScenarioLabel], supported: Set[str]) -> List[Set[str]]:
    """Map each scenario's technique set into the supported label space, routing
    anything unsupported to UNSUPPORTED_CLASS (never dropped silently)."""
    out: List[Set[str]] = []
    for lab in labels:
        projected = {t for t in lab.techniques if t in supported}
        if lab.techniques - supported:
            projected.add(UNSUPPORTED_CLASS)
        out.append(projected)
    return out
