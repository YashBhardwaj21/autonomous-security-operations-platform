"""Canonical parsers — vendor telemetry -> CanonicalEvent.

Merged and extended from asop ``src/parser/parser.py``. Changes vs. source:

* Imports the single canonical schema (``src.canon.schema``), not asop's
  private ``src.domain.schema`` (H12).
* Coverage extended per REPORT.md H4: adds Sysmon EventID 10 (ProcessAccess —
  592,932 events, the primary LSASS-access signal for credential dumping),
  Sysmon 7 (ImageLoad), and PowerShell 4103/4104. The 6-EventID / 32.6%-coverage
  gap is closed for the highest-value event types.
* Silent ``datetime.utcnow()`` timestamp substitution (REPORT.md M-1) is removed.
  Events with a missing/unparseable timestamp are SKIPPED and COUNTED in
  ``DropStats`` — temporal order matters for sessions, so we never fabricate it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from dateutil import parser as date_parser

from src.canon.schema import (
    PARSER_VERSION,
    SCHEMA_VERSION,
    BaseEntity,
    CanonicalEvent,
    FileEntity,
    NetworkConnectionEntity,
    ProcessEntity,
    RegistryEntity,
    RelationshipType,
    SourceType,
    UserEntity,
)

_TS_FIELDS = ("@timestamp", "UtcTime", "EventTime", "SystemTime")


def _process_entity_id(event: CanonicalEvent) -> Optional[str]:
    raw, host = event.raw_event, event.host.lower()
    guid = raw.get("ProcessGuid")
    if guid:
        return f"proc_{guid}"
    pid = raw.get("ProcessId")
    if pid:
        return f"proc_{host}_{pid}_{event.timestamp.timestamp()}"
    return None


@dataclass
class DropStats:
    """Counts events dropped (and why) instead of silently corrupting them."""

    no_timestamp: int = 0
    bad_timestamp: int = 0
    no_parser: int = 0
    malformed: int = 0
    by_event_id: Counter = field(default_factory=Counter)

    def total(self) -> int:
        return self.no_timestamp + self.bad_timestamp + self.no_parser + self.malformed


def _parse_timestamp(raw: Dict[str, Any], stats: DropStats) -> Optional[datetime]:
    ts = next((raw[f] for f in _TS_FIELDS if raw.get(f)), None)
    if not ts:
        stats.no_timestamp += 1
        return None
    try:
        return date_parser.parse(ts)
    except (ValueError, TypeError, OverflowError):
        stats.bad_timestamp += 1
        return None


def _base_event(raw: Dict[str, Any], event_id: int, source: SourceType,
                default_provider: str, default_channel: str,
                ts: datetime) -> CanonicalEvent:
    return CanonicalEvent(
        event_uuid=uuid4(),
        event_source_id=str(raw.get("RecordNumber") or raw.get("EventRecordID") or ""),
        timestamp=ts,
        provider=raw.get("Provider_Name", default_provider),
        channel=raw.get("Channel", default_channel),
        event_id=event_id,
        host=str(raw.get("Hostname") or raw.get("Computer") or "Unknown"),
        source=source,
        raw_event=raw,
        parser_version=PARSER_VERSION,
        schema_version=SCHEMA_VERSION,
    )


def _parse_hashes(hashes_str: str) -> Optional[Dict[str, str]]:
    out: Dict[str, str] = {}
    for h in (hashes_str or "").split(","):
        if "=" in h:
            k, v = h.split("=", 1)
            out[k.strip()] = v.strip()
    return out or None


class BaseParser(ABC):
    event_ids: tuple = ()
    provider = "Microsoft-Windows-Sysmon"
    channel = "Microsoft-Windows-Sysmon/Operational"

    def parse_event(self, raw: Dict[str, Any], source: SourceType,
                    stats: DropStats) -> Optional[CanonicalEvent]:
        eid = str(raw.get("EventID"))
        if eid not in {str(e) for e in self.event_ids}:
            return None
        ts = _parse_timestamp(raw, stats)
        if ts is None:
            stats.by_event_id[int(eid)] += 1
            return None
        return _base_event(raw, int(eid), source, self.provider, self.channel, ts)

    @abstractmethod
    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        ...

    def extract_relationships(
        self, event: CanonicalEvent, entities: Dict[str, BaseEntity]
    ) -> List[Tuple[str, str, str]]:
        return []


class SysmonEvent1Parser(BaseParser):
    """Process creation."""

    event_ids = (1,)

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        raw, host = event.raw_event, event.host.lower()
        guid = raw.get("ProcessGuid")
        eid = f"proc_{guid}" if guid else f"proc_{host}_{raw.get('ProcessId')}_{event.timestamp.timestamp()}"
        try:
            pid = int(raw.get("ProcessId", 0))
        except (ValueError, TypeError):
            pid = 0
        ent = ProcessEntity(
            entity_id=eid, process_guid=guid, process_id=pid,
            image=raw.get("Image", ""), command_line=raw.get("CommandLine"),
            parent_guid=raw.get("ParentProcessGuid"), parent_image=raw.get("ParentImage"),
            user=raw.get("User"), integrity_level=raw.get("IntegrityLevel"),
            hashes=_parse_hashes(raw.get("Hashes", "")),
            company=raw.get("Company"), original_filename=raw.get("OriginalFileName"),
        )
        return {eid: ent}

    def extract_relationships(
        self, event: CanonicalEvent, entities: Dict[str, BaseEntity]
    ) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        child_proc_id = next(iter(entities.keys()))
        raw = event.raw_event
        p_guid = str(raw.get("ParentProcessGuid") or "").strip()
        if p_guid and p_guid != "{00000000-0000-0000-0000-000000000000}":
            parent_proc_id = f"proc_{p_guid}"
            return [(parent_proc_id, RelationshipType.SPAWNED.value, child_proc_id)]
        return []


class SysmonEvent3Parser(BaseParser):
    """Network connection."""

    event_ids = (3,)

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        raw = event.raw_event
        s_ip, d_ip = raw.get("SourceIp", ""), raw.get("DestinationIp", "")
        s_p, d_p = raw.get("SourcePort", 0), raw.get("DestinationPort", 0)
        proto = raw.get("Protocol", "")
        tbin = event.timestamp.strftime("%Y%m%d%H%M")
        eid = f"net_{s_ip}_{d_ip}_{s_p}_{d_p}_{proto}_{tbin}"
        ent = NetworkConnectionEntity(
            entity_id=eid, source_ip=s_ip, dest_ip=d_ip,
            source_port=int(s_p) if str(s_p).isdigit() else 0,
            dest_port=int(d_p) if str(d_p).isdigit() else 0, protocol=proto,
        )
        return {eid: ent}

    def extract_relationships(
        self, event: CanonicalEvent, entities: Dict[str, BaseEntity]
    ) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        proc_id = _process_entity_id(event)
        if not proc_id:
            return []
        net_id = next(iter(entities.keys()))
        return [(proc_id, RelationshipType.CONNECTED_TO.value, net_id)]


class SysmonEvent7Parser(BaseParser):
    """Image/DLL load — H4 addition (201,522 events)."""

    event_ids = (7,)

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        raw, host = event.raw_event, event.host.lower()
        path = raw.get("ImageLoaded", "")
        eid = f"file_{host}_{path}"
        ext = path.split(".")[-1] if "." in path else ""
        ent = FileEntity(
            entity_id=eid, file_path=path,
            file_type="Module" if ext.lower() in ("dll", "sys") else "File",
            hashes=_parse_hashes(raw.get("Hashes", "")), extension=ext,
        )
        return {eid: ent}

    def extract_relationships(
        self, event: CanonicalEvent, entities: Dict[str, BaseEntity]
    ) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        proc_id = _process_entity_id(event)
        if not proc_id:
            return []
        file_id = next(iter(entities.keys()))
        return [(proc_id, RelationshipType.LOADED.value, file_id)]


class SysmonEvent10Parser(BaseParser):
    """Process access — H4 addition; the primary LSASS-access / credential-dumping
    signal (592,932 events, the single largest event type, previously dropped)."""

    event_ids = (10,)

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        raw, host = event.raw_event, event.host.lower()
        # The accessed (target) process is the security-relevant entity.
        tguid = raw.get("TargetProcessGUID") or raw.get("TargetProcessGuid")
        timg = raw.get("TargetImage", "")
        eid = f"proc_{tguid}" if tguid else f"proc_{host}_{raw.get('TargetProcessId')}_access"
        try:
            pid = int(raw.get("TargetProcessId", 0))
        except (ValueError, TypeError):
            pid = 0
        ent = ProcessEntity(
            entity_id=eid, process_guid=tguid, process_id=pid, image=timg,
            command_line=None, parent_image=raw.get("SourceImage"),
        )
        return {eid: ent}

    def extract_relationships(
        self, event: CanonicalEvent, entities: Dict[str, BaseEntity]
    ) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        target_proc_id = next(iter(entities.keys()))
        raw, host = event.raw_event, event.host.lower()
        s_guid = raw.get("SourceProcessGUID") or raw.get("SourceProcessGuid")
        s_pid = raw.get("SourceProcessId")
        src_proc_id = f"proc_{s_guid}" if s_guid else (f"proc_{host}_{s_pid}" if s_pid else None)
        if src_proc_id:
            return [(src_proc_id, RelationshipType.ACCESSED_PROCESS.value, target_proc_id)]
        return []


class SysmonEvent11Parser(BaseParser):
    """File create."""

    event_ids = (11,)

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        raw, host = event.raw_event, event.host.lower()
        path = raw.get("TargetFilename", "")
        eid = f"file_{host}_{path}"
        ext = path.split(".")[-1] if "." in path else ""
        ftype = "Executable" if ext.lower() in ("exe", "dll", "sys") else "Document"
        return {eid: FileEntity(entity_id=eid, file_path=path, file_type=ftype,
                                hashes=None, extension=ext)}

    def extract_relationships(
        self, event: CanonicalEvent, entities: Dict[str, BaseEntity]
    ) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        proc_id = _process_entity_id(event)
        if not proc_id:
            return []
        file_id = next(iter(entities.keys()))
        return [(proc_id, RelationshipType.CREATED_FILE.value, file_id)]


class SysmonEvent12Parser(BaseParser):
    """Registry create (12) / value set (13)."""

    event_ids = (12, 13)

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        raw, host = event.raw_event, event.host.lower()
        target = raw.get("TargetObject", "")
        parts = target.split("\\")
        hive = parts[0] if parts else ""
        key_path = "\\".join(parts[1:]) if len(parts) > 1 else ""
        value_name = parts[-1] if event.event_id == 13 and parts else None
        if event.event_id == 13 and len(parts) > 2:
            key_path = "\\".join(parts[1:-1])
        op = "Create" if event.event_id == 12 else "Set"
        if raw.get("EventType") in ("DeleteKey", "DeleteValue"):
            op = "Delete"
        eid = f"reg_{host}_{target}"
        return {eid: RegistryEntity(entity_id=eid, registry_hive=hive, key_path=key_path,
                                    value_name=value_name, operation=op)}

    def extract_relationships(
        self, event: CanonicalEvent, entities: Dict[str, BaseEntity]
    ) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        proc_id = _process_entity_id(event)
        if not proc_id:
            return []
        reg_id = next(iter(entities.keys()))
        return [(proc_id, RelationshipType.MODIFIED_REGISTRY.value, reg_id)]


class SecurityEvent4624Parser(BaseParser):
    """Successful logon."""

    event_ids = (4624,)
    provider = "Microsoft-Windows-Security-Auditing"
    channel = "Security"

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        raw = event.raw_event
        sid = raw.get("TargetUserSid") or raw.get("SubjectUserSid", "")
        acct = raw.get("TargetUserName") or raw.get("SubjectUserName", "")
        dom = raw.get("TargetDomainName") or raw.get("SubjectDomainName", "")
        logon_id = raw.get("TargetLogonId")
        ltype = raw.get("LogonType")
        eid = f"user_{sid}" if sid else f"user_{dom}\\{acct}".lower()
        return {eid: UserEntity(
            entity_id=eid, user_sid=sid, logon_id=logon_id, account_name=acct, domain=dom,
            logon_type=int(ltype) if str(ltype).isdigit() else None,
            authentication_package=raw.get("AuthenticationPackageName"),
            elevated_token=raw.get("ElevatedToken") == "%%1842",
        )}


class PowerShellParser(BaseParser):
    """PowerShell Operational 4103 (module logging) / 4104 (script block) — H4
    addition (~235k events across both PowerShell channels). No dedicated entity;
    the event itself feeds command-line / script feature extraction."""

    event_ids = (4103, 4104)
    provider = "Microsoft-Windows-PowerShell"
    channel = "Microsoft-Windows-PowerShell/Operational"

    def extract_entities(self, event: CanonicalEvent) -> Dict[str, BaseEntity]:
        return {}


class ParserFactory:
    def __init__(self) -> None:
        parsers = [
            SysmonEvent1Parser(), SysmonEvent3Parser(), SysmonEvent7Parser(),
            SysmonEvent10Parser(), SysmonEvent11Parser(), SysmonEvent12Parser(),
            SecurityEvent4624Parser(), PowerShellParser(),
        ]
        self._parsers: Dict[int, BaseParser] = {}
        for p in parsers:
            for eid in p.event_ids:
                self._parsers[eid] = p

    def supported_event_ids(self) -> list:
        return sorted(self._parsers)

    def get_parser(self, event_id: int) -> Optional[BaseParser]:
        return self._parsers.get(event_id)

    def parse(self, raw: Dict[str, Any], source: SourceType,
              stats: DropStats) -> Optional[CanonicalEvent]:
        try:
            eid = int(raw.get("EventID"))
        except (ValueError, TypeError):
            stats.malformed += 1
            return None
        parser = self._parsers.get(eid)
        if parser is None:
            stats.no_parser += 1
            stats.by_event_id[eid] += 1
            return None
        return parser.parse_event(raw, source, stats)
