"""Canonical event, entity, and session schema — the single source of truth.

Merged from asop's ``src/domain/schema.py`` (the pydantic model produced by the
live parser) and chosen over asop's flat dataclass ``models.py`` and etbackend's
51-key dict-vectorizer. Every module in ``src/`` writes against these types.

Resolves REPORT.md H12 (no shared schema across the two former repos) and the
asop-internal duplication noted in the merge plan.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    # Attack-labelled telemetry (attribution training population)
    OTRF = "OTRF"
    WINDOWS = "WINDOWS"
    LAB = "LAB"
    ZEEK = "ZEEK"
    SURICATA = "SURICATA"
    # Benign-only populations (UEBA baseline path ONLY — never attribution positive class)
    LANL = "LANL"
    CICIDS = "CICIDS"
    TONIOT = "TONIOT"
    # Held-out external validation (never trained on)
    BOTS = "BOTS"


# Source populations that may ONLY feed the UEBA benign-baseline path.
# Enforced in code (see src/features/labeling.py and src/ueba) — REPORT.md C1/M6.
BENIGN_ONLY_SOURCES = frozenset({SourceType.LANL, SourceType.CICIDS, SourceType.TONIOT})
HELD_OUT_SOURCES = frozenset({SourceType.BOTS})


class CanonicalEvent(BaseModel):
    event_uuid: UUID
    event_source_id: Optional[str] = None  # RecordID / EventRecordID for tracing
    timestamp: datetime
    provider: str
    channel: str
    event_id: int
    host: str
    source: SourceType
    raw_event: Dict[str, Any]
    parser_version: str
    schema_version: str


class BaseEntity(BaseModel):
    entity_id: str  # global unique identifier in the graph


class ProcessEntity(BaseEntity):
    process_guid: Optional[str] = None
    process_id: int
    image: str
    command_line: Optional[str] = None
    parent_guid: Optional[str] = None
    parent_image: Optional[str] = None
    user: Optional[str] = None
    integrity_level: Optional[str] = None
    hashes: Optional[Dict[str, str]] = None
    company: Optional[str] = None
    original_filename: Optional[str] = None


class UserEntity(BaseEntity):
    user_sid: str
    logon_id: Optional[str] = None
    account_name: str
    domain: Optional[str] = None
    logon_type: Optional[int] = None
    authentication_package: Optional[str] = None
    elevated_token: Optional[bool] = None


class HostEntity(BaseEntity):
    hostname: str
    os: Optional[str] = None
    domain: Optional[str] = None
    ip_addresses: Optional[List[str]] = None


class NetworkConnectionEntity(BaseEntity):
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str


class FileEntity(BaseEntity):
    file_path: str
    file_type: Optional[str] = None
    hashes: Optional[Dict[str, str]] = None
    extension: Optional[str] = None


class RegistryEntity(BaseEntity):
    registry_hive: str
    key_path: str
    value_name: Optional[str] = None
    operation: str


class ServiceEntity(BaseEntity):
    service_name: str
    display_name: Optional[str] = None
    binary_path: Optional[str] = None
    start_type: Optional[str] = None


class Activity(BaseModel):
    """A session: related events grouped by entity (host + logon), never time-only.

    Carries the typed entity maps and the relationship tuples the graph builder
    and feature extractors consume.
    """

    activity_id: str
    scenario_id: Optional[str] = None  # provenance (OTRF scenario / dataset id); label source
    host: Optional[str] = None
    logon_id: Optional[str] = None
    source: SourceType = SourceType.OTRF
    start_time: datetime
    end_time: datetime
    events: List[CanonicalEvent] = Field(default_factory=list)
    processes: Dict[str, ProcessEntity] = Field(default_factory=dict)
    users: Dict[str, UserEntity] = Field(default_factory=dict)
    files: Dict[str, FileEntity] = Field(default_factory=dict)
    registry: Dict[str, RegistryEntity] = Field(default_factory=dict)
    network: Dict[str, NetworkConnectionEntity] = Field(default_factory=dict)
    services: Dict[str, ServiceEntity] = Field(default_factory=dict)
    relationships: List[Tuple[str, str, str]] = Field(default_factory=list)


SCHEMA_VERSION = "2.0"
PARSER_VERSION = "asop-merged-1.0"


class FeatureVector(BaseModel):
    """Schema-agnostic feature container (retained from asop models.py, B-line).

    ``space`` records which model consumes it so UEBA and attribution feature
    spaces cannot be silently mixed (REPORT.md M6).
    """

    schema_version: str = SCHEMA_VERSION
    space: str  # "attribution" | "ueba"
    activity_id: Optional[str] = None
    scenario_id: Optional[str] = None
    source: SourceType = SourceType.OTRF
    feature_names: List[str]
    features: List[float]

    def as_array(self) -> "np.ndarray":
        return np.asarray(self.features, dtype=np.float64)

    model_config = {"arbitrary_types_allowed": True}
