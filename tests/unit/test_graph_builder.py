from datetime import datetime
import torch
import pytest

from src.canon.schema import (
    Activity,
    FileEntity,
    NetworkConnectionEntity,
    ProcessEntity,
    RegistryEntity,
    ServiceEntity,
    SourceType,
    UserEntity,
)
from src.graph.graph_builder import GraphBuilder
from src.graph.types import FEATURE_DIM, RelationshipType


def _activity(**kwargs) -> Activity:
    defaults = {
        "activity_id": "test_act_001",
        "scenario_id": "SDWIN-1001",
        "host": "WORKSTATION1",
        "source": SourceType.OTRF,
        "start_time": datetime(2026, 1, 1, 12, 0, 0),
        "end_time": datetime(2026, 1, 1, 12, 5, 0),
    }
    defaults.update(kwargs)
    return Activity(**defaults)


def test_empty_activity():
    act = _activity()
    builder = GraphBuilder()
    data = builder.build(act)

    assert data.x.shape == (0, FEATURE_DIM)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0,)
    assert data.activity_id == "test_act_001"
    assert data.scenario_id == "SDWIN-1001"
    assert data.host == "WORKSTATION1"


def test_single_process():
    p1 = ProcessEntity(entity_id="proc_1", process_id=100, image="explorer.exe", command_line="explorer.exe")
    act = _activity(processes={"proc_1": p1})

    builder = GraphBuilder()
    data = builder.build(act)

    assert data.x.shape == (1, FEATURE_DIM)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0,)
    assert not torch.isnan(data.x).any()


def test_parent_child_process_spawned():
    p1 = ProcessEntity(entity_id="proc_1", process_id=100, image="explorer.exe")
    p2 = ProcessEntity(entity_id="proc_2", process_id=200, image="cmd.exe", parent_guid="proc_1")
    act = _activity(
        processes={"proc_1": p1, "proc_2": p2},
        relationships=[("proc_1", RelationshipType.SPAWNED.value, "proc_2")],
    )

    data = GraphBuilder().build(act)

    assert data.x.shape == (2, FEATURE_DIM)
    assert data.edge_index.shape == (2, 1)
    assert data.edge_attr.shape == (1,)
    # Check edge source/target match node indices
    assert data.edge_index[0, 0].item() == 0
    assert data.edge_index[1, 0].item() == 1


def test_process_loaded_file():
    p1 = ProcessEntity(entity_id="proc_1", process_id=100, image="whoami.exe")
    f1 = FileEntity(entity_id="file_1", file_path="C:\\kernel32.dll", extension="dll")
    act = _activity(
        processes={"proc_1": p1},
        files={"file_1": f1},
        relationships=[("proc_1", RelationshipType.LOADED.value, "file_1")],
    )

    data = GraphBuilder().build(act)

    assert data.x.shape == (2, FEATURE_DIM)
    assert data.edge_index.shape == (2, 1)
    assert data.edge_attr.shape == (1,)


def test_process_modified_registry():
    p1 = ProcessEntity(entity_id="proc_1", process_id=100, image="reg.exe")
    r1 = RegistryEntity(entity_id="reg_1", registry_hive="HKLM", key_path="Software\\Run", operation="Set")
    act = _activity(
        processes={"proc_1": p1},
        registry={"reg_1": r1},
        relationships=[("proc_1", RelationshipType.MODIFIED_REGISTRY.value, "reg_1")],
    )

    data = GraphBuilder().build(act)

    assert data.x.shape == (2, FEATURE_DIM)
    assert data.edge_index.shape == (2, 1)


def test_process_connected_network():
    p1 = ProcessEntity(entity_id="proc_1", process_id=100, image="curl.exe")
    net1 = NetworkConnectionEntity(
        entity_id="net_1", source_ip="10.0.0.1", dest_ip="8.8.8.8", source_port=1234, dest_port=80, protocol="TCP"
    )
    act = _activity(
        processes={"proc_1": p1},
        network={"net_1": net1},
        relationships=[("proc_1", RelationshipType.CONNECTED_TO.value, "net_1")],
    )

    data = GraphBuilder().build(act)

    assert data.x.shape == (2, FEATURE_DIM)
    assert data.edge_index.shape == (2, 1)


def test_missing_destination_node_ignored():
    p1 = ProcessEntity(entity_id="proc_1", process_id=100, image="cmd.exe")
    # Relationship points to non-existent proc_99
    act = _activity(
        processes={"proc_1": p1},
        relationships=[("proc_1", RelationshipType.SPAWNED.value, "proc_99")],
    )

    data = GraphBuilder().build(act)

    assert data.x.shape == (1, FEATURE_DIM)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0,)


def test_unknown_relationship_type_ignored():
    p1 = ProcessEntity(entity_id="proc_1", process_id=100, image="cmd.exe")
    p2 = ProcessEntity(entity_id="proc_2", process_id=200, image="powershell.exe")
    act = _activity(
        processes={"proc_1": p1, "proc_2": p2},
        relationships=[("proc_1", "UNKNOWN_REL", "proc_2")],
    )

    data = GraphBuilder().build(act)

    assert data.x.shape == (2, FEATURE_DIM)
    assert data.edge_index.shape == (2, 0)


def test_deterministic_node_ordering():
    # Insert keys out of alphabetical order
    processes = {
        "proc_z": ProcessEntity(entity_id="proc_z", process_id=300, image="z.exe"),
        "proc_a": ProcessEntity(entity_id="proc_a", process_id=100, image="a.exe"),
        "proc_m": ProcessEntity(entity_id="proc_m", process_id=200, image="m.exe"),
    }
    act = _activity(processes=processes)

    data1 = GraphBuilder().build(act)
    data2 = GraphBuilder().build(act)

    assert torch.equal(data1.x, data2.x)


def test_metadata_preservation():
    act = _activity(activity_id="act_unique_123", scenario_id="SDWIN-9999", host="DC01")
    data = GraphBuilder().build(act)

    assert data.activity_id == "act_unique_123"
    assert data.scenario_id == "SDWIN-9999"
    assert data.host == "DC01"
