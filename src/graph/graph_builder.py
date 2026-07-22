from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from src.canon.schema import (
    Activity,
    BaseEntity,
    FileEntity,
    NetworkConnectionEntity,
    ProcessEntity,
    RegistryEntity,
    ServiceEntity,
    UserEntity,
)
from src.graph.types import (
    EDGE_TYPE_INDEX,
    FEATURE_DIM,
    FEATURE_SPEC,
    NODE_TYPE_INDEX,
    NodeType,
    RelationshipType,
)


class GraphBuilder:
    """Converts an Activity into a PyTorch Geometric Data object with rich node & structural features."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        self.feature_dim = feature_dim

    def build(self, activity: Activity) -> Data:
        """Converts an Activity instance into a PyG Data object.

        Returns Data(x, edge_index, edge_attr, activity_id, scenario_id, host, source, start_time, end_time).
        """
        # Step 1: Deterministic Node Extraction & Indexing
        node_id_map, node_entities, node_types = self._extract_and_index_nodes(activity)
        num_nodes = len(node_id_map)

        if num_nodes == 0:
            data = Data(
                x=torch.zeros((0, self.feature_dim), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0,), dtype=torch.long),
                activity_id=activity.activity_id,
                scenario_id=activity.scenario_id or "unknown",
                host=activity.host or "unknown",
                source=activity.source.value if hasattr(activity.source, "value") else str(activity.source),
                start_time=str(activity.start_time),
                end_time=str(activity.end_time),
            )
            self._validate_graph(data, 0, 0)
            return data

        # Step 2: Per-Entity Feature Construction
        x = self._build_node_features(node_entities, node_types, num_nodes)

        # Step 3: Edge Index & Edge Attributes Construction
        edge_index, edge_attr = self._build_edges(activity.relationships, node_id_map)
        num_edges = edge_index.shape[1]

        # Step 4: Add Graph Structural Degree Features & Process Hierarchy Depths
        if num_edges > 0:
            in_degrees = torch.bincount(edge_index[1], minlength=num_nodes).float()
            out_degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()
            x[:, FEATURE_SPEC.index("node_in_degree_log1p")] = torch.log1p(in_degrees)
            x[:, FEATURE_SPEC.index("node_out_degree_log1p")] = torch.log1p(out_degrees)

            # Compute process tree depth
            proc_depths = self._compute_process_depths(activity.processes, node_id_map, num_nodes)
            x[:, FEATURE_SPEC.index("process_depth_log1p")] = torch.log1p(proc_depths)

        # Step 5: Construct PyG Data with Metadata Preservation
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            activity_id=activity.activity_id,
            scenario_id=activity.scenario_id or "unknown",
            host=activity.host or "unknown",
            source=activity.source.value if hasattr(activity.source, "value") else str(activity.source),
            start_time=str(activity.start_time),
            end_time=str(activity.end_time),
        )

        # Step 6: Graph Structure Validation
        self._validate_graph(data, num_nodes, num_edges)

        return data

    def _extract_and_index_nodes(
        self, activity: Activity
    ) -> Tuple[Dict[str, int], List[BaseEntity], List[str]]:
        """Extracts entities from activity dicts in deterministic sorted order."""
        node_id_map: Dict[str, int] = {}
        node_entities: List[BaseEntity] = []
        node_types: List[str] = []

        containers = [
            (activity.processes, NodeType.PROCESS.value),
            (activity.files, NodeType.FILE.value),
            (activity.registry, NodeType.REGISTRY.value),
            (activity.network, NodeType.NETWORK.value),
            (activity.users, NodeType.USER.value),
            (activity.services, NodeType.SERVICE.value),
        ]

        for container, ntype in containers:
            for eid in sorted(container.keys()):
                if eid not in node_id_map:
                    node_id_map[eid] = len(node_id_map)
                    node_entities.append(container[eid])
                    node_types.append(ntype)

        return node_id_map, node_entities, node_types

    def _build_node_features(
        self, entities: List[BaseEntity], ntypes: List[str], num_nodes: int
    ) -> torch.Tensor:
        """Constructs node feature matrix x by delegating to entity-specific builders."""
        x = torch.zeros((num_nodes, self.feature_dim), dtype=torch.float)

        for i, (ent, ntype) in enumerate(zip(entities, ntypes)):
            if isinstance(ent, ProcessEntity):
                x[i] = self._build_process_features(ent)
            elif isinstance(ent, FileEntity):
                x[i] = self._build_file_features(ent)
            elif isinstance(ent, RegistryEntity):
                x[i] = self._build_registry_features(ent)
            elif isinstance(ent, NetworkConnectionEntity):
                x[i] = self._build_network_features(ent)
            elif isinstance(ent, UserEntity):
                x[i] = self._build_user_features(ent)
            elif isinstance(ent, ServiceEntity):
                x[i] = self._build_service_features(ent)
            else:
                x[i, NODE_TYPE_INDEX.get(ntype, 0)] = 1.0

        return x

    def _build_process_features(self, ent: ProcessEntity) -> torch.Tensor:
        vec = torch.zeros(self.feature_dim, dtype=torch.float)
        vec[NODE_TYPE_INDEX[NodeType.PROCESS.value]] = 1.0

        integ = (ent.integrity_level or "").lower()
        if "system" in integ:
            vec[FEATURE_SPEC.index("integ_system")] = 1.0
        elif "high" in integ:
            vec[FEATURE_SPEC.index("integ_high")] = 1.0
        else:
            vec[FEATURE_SPEC.index("integ_medium_low")] = 1.0

        cmd_len = len(ent.command_line or "")
        vec[FEATURE_SPEC.index("cmd_len_log1p")] = float(math.log1p(cmd_len))

        if ent.hashes:
            if "SHA1" in ent.hashes:
                vec[FEATURE_SPEC.index("has_sha1")] = 1.0
            if "MD5" in ent.hashes:
                vec[FEATURE_SPEC.index("has_md5")] = 1.0
            if "SHA256" in ent.hashes:
                vec[FEATURE_SPEC.index("has_sha256")] = 1.0

        return vec

    def _build_file_features(self, ent: FileEntity) -> torch.Tensor:
        vec = torch.zeros(self.feature_dim, dtype=torch.float)
        vec[NODE_TYPE_INDEX[NodeType.FILE.value]] = 1.0

        ext = (ent.extension or "").lower()
        if ext in ("exe", "dll", "sys") or ent.file_type in ("Executable", "Module"):
            vec[FEATURE_SPEC.index("is_executable")] = 1.0

        if ent.hashes:
            if "SHA1" in ent.hashes:
                vec[FEATURE_SPEC.index("has_sha1")] = 1.0
            if "MD5" in ent.hashes:
                vec[FEATURE_SPEC.index("has_md5")] = 1.0
            if "SHA256" in ent.hashes:
                vec[FEATURE_SPEC.index("has_sha256")] = 1.0

        return vec

    def _build_registry_features(self, ent: RegistryEntity) -> torch.Tensor:
        vec = torch.zeros(self.feature_dim, dtype=torch.float)
        vec[NODE_TYPE_INDEX[NodeType.REGISTRY.value]] = 1.0

        hive = (ent.registry_hive or "").upper()
        if "HKLM" in hive or "LOCAL_MACHINE" in hive:
            vec[FEATURE_SPEC.index("reg_hive_hklm")] = 1.0
        elif "HKCU" in hive or "CURRENT_USER" in hive:
            vec[FEATURE_SPEC.index("reg_hive_hkcu")] = 1.0

        kpath = (ent.key_path or "").lower()
        if any(p in kpath for p in ["run", "runonce", "startup", "currentversion\\windows"]):
            vec[FEATURE_SPEC.index("reg_is_run_key")] = 1.0
        if "services" in kpath:
            vec[FEATURE_SPEC.index("reg_is_service_key")] = 1.0
        if "exclusions" in kpath or "windows defender" in kpath:
            vec[FEATURE_SPEC.index("reg_is_defender_exclusion")] = 1.0

        op = (ent.operation or "").lower()
        if any(o in op for o in ["setvalue", "createkey", "write", "modify"]):
            vec[FEATURE_SPEC.index("reg_op_write")] = 1.0

        depth = kpath.count("\\") + kpath.count("/")
        vec[FEATURE_SPEC.index("reg_path_depth_log1p")] = float(math.log1p(depth))

        return vec

    def _build_network_features(self, ent: NetworkConnectionEntity) -> torch.Tensor:
        vec = torch.zeros(self.feature_dim, dtype=torch.float)
        vec[NODE_TYPE_INDEX[NodeType.NETWORK.value]] = 1.0

        proto = (ent.protocol or "").lower()
        if "tcp" in proto:
            vec[FEATURE_SPEC.index("net_tcp")] = 1.0
        elif "udp" in proto:
            vec[FEATURE_SPEC.index("net_udp")] = 1.0

        return vec

    def _build_user_features(self, ent: UserEntity) -> torch.Tensor:
        vec = torch.zeros(self.feature_dim, dtype=torch.float)
        vec[NODE_TYPE_INDEX[NodeType.USER.value]] = 1.0

        if ent.elevated_token is True:
            vec[FEATURE_SPEC.index("user_is_elevated")] = 1.0

        ltype = ent.logon_type
        if ltype in (2, 10):  # Interactive or RemoteInteractive
            vec[FEATURE_SPEC.index("user_logon_interactive")] = 1.0
        elif ltype == 3:  # Network
            vec[FEATURE_SPEC.index("user_logon_network")] = 1.0

        return vec

    def _build_service_features(self, ent: ServiceEntity) -> torch.Tensor:
        vec = torch.zeros(self.feature_dim, dtype=torch.float)
        vec[NODE_TYPE_INDEX[NodeType.SERVICE.value]] = 1.0
        return vec

    def _compute_process_depths(
        self, processes: Dict[str, ProcessEntity], node_id_map: Dict[str, int], num_nodes: int
    ) -> torch.Tensor:
        """Computes hierarchy depth of each process node in the process tree."""
        depths = torch.zeros(num_nodes, dtype=torch.float)
        parent_map = {eid: p.parent_guid for eid, p in processes.items() if p.parent_guid}

        for eid, p in processes.items():
            if eid in node_id_map:
                idx = node_id_map[eid]
                depth = 0
                curr = eid
                visited = set()
                while curr in parent_map and parent_map[curr] and parent_map[curr] not in visited:
                    visited.add(curr)
                    curr = parent_map[curr]
                    depth += 1
                depths[idx] = float(depth)

        return depths

    def _build_edges(
        self, relationships: List[Tuple[str, str, str]], node_id_map: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sources: List[int] = []
        targets: List[int] = []
        edge_types: List[int] = []

        for src, rel, dst in relationships:
            if src in node_id_map and dst in node_id_map:
                if rel in EDGE_TYPE_INDEX:
                    sources.append(node_id_map[src])
                    targets.append(node_id_map[dst])
                    edge_types.append(EDGE_TYPE_INDEX[rel])

        if not sources:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            edge_attr = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_attr

    def _validate_graph(self, data: Data, num_nodes: int, num_edges: int) -> None:
        """Validates graph tensor invariants."""
        assert data.x.shape == (num_nodes, self.feature_dim), (
            f"x shape mismatch: expected ({num_nodes}, {self.feature_dim}), got {data.x.shape}"
        )
        assert not torch.isnan(data.x).any(), "NaN detected in node feature matrix x"
        assert not torch.isinf(data.x).any(), "Inf detected in node feature matrix x"

        assert data.edge_index.shape == (2, num_edges), (
            f"edge_index shape mismatch: expected (2, {num_edges}), got {data.edge_index.shape}"
        )
        assert data.edge_attr.shape == (num_edges,), (
            f"edge_attr shape mismatch: expected ({num_edges},), got {data.edge_attr.shape}"
        )

        if num_edges > 0:
            assert (data.edge_index >= 0).all() and (data.edge_index < num_nodes).all(), (
                "edge_index contains out-of-bounds node indices"
            )
