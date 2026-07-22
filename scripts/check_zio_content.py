from collections import Counter
import numpy as np

from src.canon.schema import SourceType
from src.ingestion.otrf import build_dataset, iter_scenarios, read_raw_events, DEFAULT_ROOT
from src.ingestion.parser import ParserFactory, DropStats
from src.sessions.session_builder import SessionBuilder, SessionStats
from src.features.labeling import label_from_metadata

print("=" * 120)
print("ASOP PROCESS-TREE SESSIONIZER AUDIT")
print("=" * 120)

factory = ParserFactory()
builder = SessionBuilder(factory)
stats = DropStats()

all_activities = []
scenario_stats_list = []

# Collect all activities and per-scenario stats directly from the session builder
for scenario_id, meta, zips in iter_scenarios(DEFAULT_ROOT):
    scen_label = label_from_metadata(meta, scenario_id, SourceType.OTRF)
    if scen_label.is_empty():
        continue
    events = []
    for zp in zips:
        for raw in read_raw_events(zp):
            ev = factory.parse(raw, SourceType.OTRF, stats)
            if ev is not None:
                events.append(ev)
    if not events:
        continue

    scen_acts = builder.build_sessions(events, scenario_id=scenario_id)
    all_activities.extend(scen_acts)
    if builder.last_stats:
        scenario_stats_list.append(builder.last_stats)

dataset = build_dataset()

print("\n")
print("=" * 120)
print("DATASET SUMMARY")
print("=" * 120)

print(f"Samples   : {dataset.X.shape[0]}")
print(f"Features  : {dataset.X.shape[1]}")
print(f"Scenarios : {len(set(dataset.scenario_ids))}")
print(f"Labels    : {len(set(dataset.y_primary))}")

print("\nScenario Counts")
print("-" * 120)
for scenario, count in Counter(dataset.scenario_ids).most_common():
    print(f"{scenario:<38} {count}")

print("\nLabel Counts")
print("-" * 120)
for label, count in Counter(dataset.y_primary).most_common():
    print(f"{label:<20} {count}")

# Aggregate Sessionization Stats
tree_acts = sum(s.process_tree_activities for s in scenario_stats_list)
logon_acts = sum(s.logon_activities for s in scenario_stats_list)
gap_acts = sum(s.gap_window_activities for s in scenario_stats_list)

tot_events = sum(s.total_events for s in scenario_stats_list)
tree_events = sum(s.tree_events for s in scenario_stats_list)
logon_events = sum(s.logon_events for s in scenario_stats_list)
gap_events = sum(s.gap_events for s in scenario_stats_list)

proc_nodes = sum(s.process_nodes for s in scenario_stats_list)
root_nodes = sum(s.root_nodes for s in scenario_stats_list)
orphan_nodes = sum(s.orphan_nodes for s in scenario_stats_list)

all_tree_sizes = [size for s in scenario_stats_list for size in s.tree_sizes if size > 0]

mean_tree = float(np.mean(all_tree_sizes)) if all_tree_sizes else 0.0
median_tree = float(np.median(all_tree_sizes)) if all_tree_sizes else 0.0
p95_tree = float(np.percentile(all_tree_sizes, 95)) if all_tree_sizes else 0.0
max_tree = int(np.max(all_tree_sizes)) if all_tree_sizes else 0
min_tree = int(np.min(all_tree_sizes)) if all_tree_sizes else 0

print("\n" + "=" * 120)
print("SESSIONIZATION SUMMARY")
print("=" * 120)

print("\nActivities")
print("----------")
print(f"Process Tree : {tree_acts}")
print(f"Logon Session: {logon_acts}")
print(f"Gap Window   : {gap_acts}")

print("\nEvents")
print("------")
print(f"Total Events          : {tot_events}")
print(f"Tree Events           : {tree_events}")
print(f"Logon Events          : {logon_events}")
print(f"Gap Events            : {gap_events}")

print("\nProcess Graph")
print("-------------")
print(f"Process Nodes         : {proc_nodes}")
print(f"Root Nodes            : {root_nodes}")
print(f"Orphan Nodes          : {orphan_nodes}")

print("\nTree Sizes")
print("----------")
print(f"Average               : {mean_tree:.1f}")
print(f"Median                : {median_tree:.1f}")
print(f"95th Percentile       : {p95_tree:.1f}")
print(f"Largest               : {max_tree}")
print(f"Smallest              : {min_tree}")

# Task 5 — Event Ownership Check (Duplicate-Event Verification)
all_assigned_uuids = [ev.event_uuid for act in all_activities for ev in act.events]
uuid_counts = Counter(all_assigned_uuids)
duplicate_assignments = sum(1 for uuid, count in uuid_counts.items() if count > 1)

print("\n" + "=" * 120)
print("EVENT OWNERSHIP CHECK")
print("=" * 120)
print(f"Total Assigned Event Instances : {len(all_assigned_uuids)}")
print(f"Unique Assigned Event UUIDs    : {len(uuid_counts)}")
print(f"Duplicate Assigned Events      : {duplicate_assignments}")

# Task 4 — Process Tree Inspection (Top 10 Largest Process Trees)
print("\n" + "=" * 120)
print("PROCESS TREE INSPECTION (10 LARGEST TREES)")
print("=" * 120)

tree_activities_all = [act for act in all_activities if act.logon_id is None]
top_10_trees = sorted(tree_activities_all, key=lambda a: len(a.events), reverse=True)[:10]

for idx, act in enumerate(top_10_trees, 1):
    eids = Counter(e.event_id for e in act.events)
    # Find root process image if present
    proc_images = [p.image for p in act.processes.values() if p.image]
    root_img = proc_images[0] if proc_images else "Unknown"
    child_imgs = list(set(proc_images[1:])) if len(proc_images) > 1 else []

    print(f"\n[{idx}] Scenario: {act.scenario_id}")
    print(f"    Root Process  : {root_img}")
    print(f"    Children      : {child_imgs if child_imgs else 'None'}")
    print(f"    Number of Evs : {len(act.events)}")
    print(f"    Event IDs     : {dict(eids)}")
    print(f"    Processes     : {len(act.processes)}")
    print(f"    Files         : {len(act.files)}")
    print(f"    Registry      : {len(act.registry)}")
    print(f"    Network       : {len(act.network)}")
    print(f"    Services      : {len(act.services)}")

print("\nAudit Complete")