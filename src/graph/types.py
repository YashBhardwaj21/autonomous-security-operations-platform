from enum import Enum
from typing import Dict, List


class RelationshipType(str, Enum):
    SPAWNED = "spawned"
    LOADED = "loaded"
    CONNECTED_TO = "connected_to"
    CREATED_FILE = "created_file"
    MODIFIED_REGISTRY = "modified_registry"
    ACCESSED_PROCESS = "accessed_process"


class NodeType(str, Enum):
    PROCESS = "Process"
    FILE = "File"
    REGISTRY = "Registry"
    NETWORK = "Network"
    USER = "User"
    SERVICE = "Service"


NODE_TYPE_INDEX: Dict[str, int] = {t.value: i for i, t in enumerate(NodeType)}
EDGE_TYPE_INDEX: Dict[str, int] = {r.value: i for i, r in enumerate(RelationshipType)}

FEATURE_SPEC: List[str] = [
    # Node Type One-Hots (0-5)
    "is_process",
    "is_file",
    "is_registry",
    "is_network",
    "is_user",
    "is_service",
    # Process Attributes (6-11)
    "integ_system",
    "integ_high",
    "integ_medium_low",
    "is_executable",
    "cmd_len_log1p",
    "process_depth_log1p",
    # File/Hash Attributes (12-14)
    "has_sha1",
    "has_md5",
    "has_sha256",
    # Network Attributes (15-16)
    "net_tcp",
    "net_udp",
    # Rich Registry Attributes (17-23)
    "reg_hive_hklm",
    "reg_hive_hkcu",
    "reg_is_run_key",
    "reg_is_service_key",
    "reg_is_defender_exclusion",
    "reg_op_write",
    "reg_path_depth_log1p",
    # Rich User Attributes (24-26)
    "user_is_elevated",
    "user_logon_interactive",
    "user_logon_network",
    # Graph Structural Features (27-28)
    "node_in_degree_log1p",
    "node_out_degree_log1p",
]

FEATURE_DIM: int = len(FEATURE_SPEC)
