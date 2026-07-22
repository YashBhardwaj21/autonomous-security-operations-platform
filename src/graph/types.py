from enum import Enum


class RelationshipType(str, Enum):
    SPAWNED = "spawned"
    LOADED = "loaded"
    CONNECTED_TO = "connected_to"
    CREATED_FILE = "created_file"
    MODIFIED_REGISTRY = "modified_registry"
    ACCESSED_PROCESS = "accessed_process"
