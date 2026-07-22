import pytest

from src.twin.simulator import DigitalTwinSimulator, TopologyRequiredError

ASSETS = [
    {"asset_id": "web", "criticality_tier": 3, "network_position": "dmz"},
    {"asset_id": "app", "criticality_tier": 2, "network_position": "internal"},
    {"asset_id": "db", "criticality_tier": 1, "network_position": "internal"},
    {"asset_id": "dc", "criticality_tier": 0, "network_position": "internal"},
]
EDGES = [
    {"from_asset": "web", "to_asset": "app", "technique_id": "T1021"},
    {"from_asset": "app", "to_asset": "db", "technique_id": "T1078"},
    {"from_asset": "app", "to_asset": "dc", "technique_id": "T1021.002"},
]


def test_build_requires_real_edges_no_fabrication():
    t = DigitalTwinSimulator()
    with pytest.raises(TopologyRequiredError):
        t.build(ASSETS, edges=[])          # H10: no _auto_generate_edges fallback


def test_blast_radius_uses_real_reachability():
    t = DigitalTwinSimulator()
    t.build(ASSETS, EDGES)
    br = t.blast_radius("web")
    assert set(br["reachable_nodes"]) == {"app", "db", "dc"}
    assert br["reachable_count"] == 3
    assert br["critical_reachable"] == 2     # db(tier1) + dc(tier0)


def test_simulate_reports_critical_paths():
    t = DigitalTwinSimulator()
    t.build(ASSETS, EDGES)
    res = t.simulate_compromise("web")
    assert "db" in res["reachable_nodes"]
    assert res["critical_paths"]             # at least one path to a tier<=1 asset
