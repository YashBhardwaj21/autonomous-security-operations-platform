from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from src.canon.schema import (
    Activity,
    CanonicalEvent,
    FileEntity,
    NetworkConnectionEntity,
    ProcessEntity,
    RegistryEntity,
    ServiceEntity,
    SourceType,
    UserEntity,
)
from src.ingestion.parser import ParserFactory

WELL_KNOWN_LOGON_IDS = {"0x0", "0x3e7", "0x3e4", "0x3e5"}
ZERO_GUID = "{00000000-0000-0000-0000-000000000000}"
_LOGON_FIELDS = ("TargetLogonId", "LogonId", "SubjectLogonId")

PROCESS_CREATE_EVENTS = {1}
PROCESS_GUID_EVENTS = {3, 7, 11, 12, 13}
PROCESS_ACCESS_EVENTS = {10}


def is_process_creation(e: CanonicalEvent) -> bool:
    return e.event_id in PROCESS_CREATE_EVENTS


def is_process_activity(e: CanonicalEvent) -> bool:
    return e.event_id in PROCESS_GUID_EVENTS


def is_process_access(e: CanonicalEvent) -> bool:
    return e.event_id in PROCESS_ACCESS_EVENTS


def _logon_id(e: CanonicalEvent) -> Optional[str]:
    for f in _LOGON_FIELDS:
        v = e.raw_event.get(f)
        if v and str(v).lower() not in WELL_KNOWN_LOGON_IDS:
            return str(v)
    return None


@dataclass
class SessionStats:
    process_tree_activities: int = 0
    logon_activities: int = 0
    gap_window_activities: int = 0

    total_events: int = 0
    tree_events: int = 0
    logon_events: int = 0
    gap_events: int = 0

    process_nodes: int = 0
    root_nodes: int = 0
    orphan_nodes: int = 0

    tree_sizes: List[int] = field(default_factory=list)


@dataclass
class ProcessNode:
    guid: str
    parent_guid: Optional[str]
    host: str
    image: str
    events: List[CanonicalEvent] = field(default_factory=list)
    children: List["ProcessNode"] = field(default_factory=list)


class SessionBuilder:
    def __init__(self, factory: Optional[ParserFactory] = None,
                 inactivity_gap_seconds: int = 300,
                 debug: bool = False):
        self.factory = factory or ParserFactory()
        self.gap = timedelta(seconds=inactivity_gap_seconds)
        self.debug = debug
        self.last_stats: Optional[SessionStats] = None

    def _build_process_tree_sessions(
        self,
        events: List[CanonicalEvent],
        scenario_id: Optional[str] = None,
    ) -> Tuple[List[Activity], List[CanonicalEvent], SessionStats]:
        process_nodes: Dict[Tuple[str, str], ProcessNode] = {}
        unattached_events: List[CanonicalEvent] = []

        # Step 1: Create nodes for process creation events (Sysmon EID 1)
        for e in events:
            if is_process_creation(e):
                guid = str(e.raw_event.get("ProcessGuid") or "").strip()
                if guid:
                    p_guid = str(e.raw_event.get("ParentProcessGuid") or "").strip() or None
                    if p_guid == ZERO_GUID:
                        p_guid = None
                    img = str(e.raw_event.get("Image") or e.raw_event.get("NewProcessName") or "").strip()
                    node = ProcessNode(guid=guid, parent_guid=p_guid, host=e.host, image=img)
                    node.events.append(e)
                    process_nodes[(e.host, guid)] = node
                else:
                    unattached_events.append(e)

        # Step 2: Attach child events to process nodes
        for e in events:
            if is_process_creation(e):
                continue

            if is_process_activity(e):
                p_guid = str(e.raw_event.get("ProcessGuid") or "").strip()
                if p_guid and (e.host, p_guid) in process_nodes:
                    process_nodes[(e.host, p_guid)].events.append(e)
                else:
                    unattached_events.append(e)

            elif is_process_access(e):
                src_guid = str(
                    e.raw_event.get("SourceProcessGUID")
                    or e.raw_event.get("SourceProcessGuid")
                    or ""
                ).strip()
                if src_guid and (e.host, src_guid) in process_nodes:
                    process_nodes[(e.host, src_guid)].events.append(e)
                else:
                    unattached_events.append(e)

            else:
                unattached_events.append(e)

        # Step 3: Build parent-child links & track orphan nodes
        orphan_count = 0
        for (host, guid), node in process_nodes.items():
            if node.parent_guid:
                if (host, node.parent_guid) in process_nodes:
                    process_nodes[(host, node.parent_guid)].children.append(node)
                else:
                    orphan_count += 1

        # Step 4: Identify root process nodes
        root_nodes: List[ProcessNode] = [
            node for (host, guid), node in process_nodes.items()
            if not node.parent_guid or (host, node.parent_guid) not in process_nodes
        ]

        # Sort children deterministically by timestamp of first event
        for node in process_nodes.values():
            node.children.sort(key=lambda n: n.events[0].timestamp if n.events else datetime.min)

        # Step 5: DFS to collect tree events with per-tree cycle protection
        tree_activities: List[Activity] = []

        def _collect(n: ProcessNode, acc: List[CanonicalEvent], visited: Set[str]):
            node_key = f"{n.host}::{n.guid}"
            if node_key in visited:
                return
            visited.add(node_key)

            acc.extend(n.events)
            for child in n.children:
                _collect(child, acc, visited)

        for root in root_nodes:
            raw_tree_events: List[CanonicalEvent] = []
            tree_visited: Set[str] = set()
            _collect(root, raw_tree_events, tree_visited)

            if raw_tree_events:
                # Deduplicate by event_uuid and sort by timestamp
                tree_events = sorted(
                    {ev.event_uuid: ev for ev in raw_tree_events}.values(),
                    key=lambda ev: ev.timestamp,
                )
                act = self._build_activity(tree_events, scenario_id, root.host, None)
                tree_activities.append(act)

        tree_sizes = [len(a.events) for a in tree_activities]
        stats = SessionStats(
            process_tree_activities=len(tree_activities),
            tree_events=sum(tree_sizes),
            process_nodes=len(process_nodes),
            root_nodes=len(root_nodes),
            orphan_nodes=orphan_count,
            tree_sizes=tree_sizes,
        )

        # Step 6: Diagnostic logging (when debug=True)
        if self.debug:
            avg_size = (sum(tree_sizes) / len(tree_sizes)) if tree_sizes else 0
            print(f"[PROCESS_TREE_DIAG] Scenario: {scenario_id or 'unknown'}")
            print(f"  Total Nodes  : {len(process_nodes)}")
            print(f"  Root Nodes   : {len(root_nodes)}")
            print(f"  Orphan Nodes : {orphan_count}")
            print(f"  Trees Created: {len(tree_activities)}")
            print(f"  Attached Evs : {sum(tree_sizes)}")
            print(f"  Fallback Evs : {len(unattached_events)}")
            print(f"  Avg Evs/Tree : {avg_size:.1f}")
            print(f"  Largest Tree : {max(tree_sizes, default=0)}")
            print(f"  Smallest Tree: {min(tree_sizes, default=0)}")

        return tree_activities, unattached_events, stats

    def build_sessions(
        self,
        events: List[CanonicalEvent],
        scenario_id: Optional[str] = None,
    ) -> List[Activity]:
        if not events:
            self.last_stats = SessionStats()
            return []

        events = sorted(events, key=lambda e: e.timestamp)

        # Step 7: Process Tree Sessionization
        tree_activities, _, stats = self._build_process_tree_sessions(events, scenario_id)

        # Guarantee no event is assigned twice (consumed event filtering)
        consumed_uuids = {
            ev.event_uuid
            for activity in tree_activities
            for ev in activity.events
        }

        unattached = [e for e in events if e.event_uuid not in consumed_uuids]

        logon_groups: Dict[tuple, List[CanonicalEvent]] = defaultdict(list)
        leftover_by_host: Dict[str, List[CanonicalEvent]] = defaultdict(list)

        for e in unattached:
            lid = _logon_id(e)
            if lid is not None:
                logon_groups[(e.host, lid)].append(e)
            else:
                leftover_by_host[e.host].append(e)

        logon_activities: List[Activity] = []
        logon_events_count = 0
        for (host, lid), evs in logon_groups.items():
            act = self._build_activity(evs, scenario_id, host, lid)
            logon_activities.append(act)
            logon_events_count += len(evs)

        gap_activities: List[Activity] = []
        gap_events_count = 0
        for host, evs in leftover_by_host.items():
            g_acts = self._gap_windows(evs, scenario_id, host)
            gap_activities.extend(g_acts)
            gap_events_count += len(evs)

        stats.total_events = len(events)
        stats.logon_activities = len(logon_activities)
        stats.gap_window_activities = len(gap_activities)
        stats.logon_events = logon_events_count
        stats.gap_events = gap_events_count

        self.last_stats = stats
        sessions: List[Activity] = list(tree_activities) + logon_activities + gap_activities
        return sessions

    def _gap_windows(
        self,
        events: List[CanonicalEvent],
        scenario_id: Optional[str],
        host: str,
    ) -> List[Activity]:
        if not events:
            return []

        print("\n" + "=" * 100)
        print(f"GAP WINDOW AUDIT")
        print(f"Scenario : {scenario_id}")
        print(f"Host     : {host}")
        print(f"Events   : {len(events)}")

        duration = events[-1].timestamp - events[0].timestamp
        print(f"Duration : {duration}")

        gaps = []
        for prev, cur in zip(events, events[1:]):
            gaps.append((cur.timestamp - prev.timestamp).total_seconds())

        if gaps:
            import statistics

            print(f"Largest gap : {max(gaps):.2f} sec")
            print(f"Average gap : {statistics.mean(gaps):.2f} sec")
            print(f"Median gap  : {statistics.median(gaps):.2f} sec")

            over_gap = sum(g > self.gap.total_seconds() for g in gaps)
            print(f"Gaps > {self.gap.total_seconds()} sec : {over_gap}")

        windows = []
        current = [events[0]]

        for prev, cur in zip(events, events[1:]):
            gap = cur.timestamp - prev.timestamp
            if gap > self.gap:
                print(
                    f"NEW WINDOW : "
                    f"{prev.timestamp} -> {cur.timestamp} "
                    f"gap={gap.total_seconds():.2f}s "
                    f"events={len(current)}"
                )
                windows.append(
                    self._build_activity(
                        current,
                        scenario_id,
                        host,
                        None,
                    )
                )
                current = [cur]
            else:
                current.append(cur)

        windows.append(
            self._build_activity(
                current,
                scenario_id,
                host,
                None,
            )
        )

        print(f"Final Windows : {len(windows)}")
        return windows

    def _build_activity(self, events: List[CanonicalEvent], scenario_id: Optional[str],
                        host: str, logon_id: Optional[str]) -> Activity:
        events = sorted(events, key=lambda e: e.timestamp)
        processes: Dict[str, ProcessEntity] = {}
        users: Dict[str, UserEntity] = {}
        files: Dict[str, FileEntity] = {}
        registry: Dict[str, RegistryEntity] = {}
        network: Dict[str, NetworkConnectionEntity] = {}
        services: Dict[str, ServiceEntity] = {}
        raw_relationships: List[Tuple[str, str, str]] = []

        for e in events:
            parser = self.factory.get_parser(e.event_id)
            if parser is None:
                continue
            entities = parser.extract_entities(e)
            for eid, ent in entities.items():
                if isinstance(ent, ProcessEntity):
                    processes[eid] = ent
                elif isinstance(ent, UserEntity):
                    users[eid] = ent
                elif isinstance(ent, FileEntity):
                    files[eid] = ent
                elif isinstance(ent, RegistryEntity):
                    registry[eid] = ent
                elif isinstance(ent, NetworkConnectionEntity):
                    network[eid] = ent
                elif isinstance(ent, ServiceEntity):
                    services[eid] = ent

            raw_relationships.extend(parser.extract_relationships(e, entities))

        valid_nodes = (
            set(processes)
            | set(users)
            | set(files)
            | set(registry)
            | set(network)
            | set(services)
        )
        relationships = list(
            dict.fromkeys(
                (src, rel, dst)
                for src, rel, dst in raw_relationships
                if src in valid_nodes and dst in valid_nodes
            )
        )

        source = events[0].source
        aid = f"{scenario_id or 'sess'}::{host}::{logon_id or 'gap'}::{events[0].timestamp.timestamp():.0f}"
        return Activity(
            activity_id=aid,
            scenario_id=scenario_id,
            host=host,
            logon_id=logon_id,
            source=source,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            events=events,
            processes=processes,
            users=users,
            files=files,
            registry=registry,
            network=network,
            services=services,
            relationships=relationships,
        )