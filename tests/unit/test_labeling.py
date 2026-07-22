import pytest

from src.canon.schema import SourceType
from src.features.labeling import (
    UNSUPPORTED_CLASS,
    build_label_space,
    label_from_metadata,
    project_labels,
)
from tests._fixtures import sysmon_events as fx


def test_multilabel_from_full_attack_mappings():
    lab = label_from_metadata(fx.GOLDEN_SAML_METADATA, "GoldenSAML", SourceType.OTRF)
    # full multi-technique / multi-tactic target, NOT first-technique-wins (H6)
    assert lab.techniques == {"T1552", "T1606", "T1078", "T1114"}
    assert {"TA0006", "TA0001", "TA0009"}.issubset(lab.tactics)
    assert "T1552.004" in lab.sub_techniques


def test_benign_source_rejects_labels():
    with pytest.raises(ValueError):
        label_from_metadata(fx.GOLDEN_SAML_METADATA, "lanl_x", SourceType.LANL)


def test_unsupported_class_excludes_rare_techniques():
    labels = [
        label_from_metadata({"attack_mappings": [{"technique": "T1003", "tactics": ["TA0006"]}]}, f"s{i}")
        for i in range(3)  # T1003 appears 3x -> supported
    ] + [
        label_from_metadata({"attack_mappings": [{"technique": "T1999", "tactics": ["TA0006"]}]}, "rare")
    ]
    supported, unsupported, counts = build_label_space(labels, min_scenarios=3)
    assert "T1003" in supported and "T1999" in unsupported
    projected = project_labels(labels, supported)
    assert projected[-1] == {UNSUPPORTED_CLASS}  # rare routed, never silently dropped
