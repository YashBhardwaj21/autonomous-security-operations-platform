from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import torch

from src.graph.dataset import ActivityGraphDataset
from src.graph.types import NodeType, RelationshipType


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_level1_hash(data) -> str:
    """Level 1 Topology Hash: node count, edge count, node type breakdown, edge type breakdown."""
    num_nodes = data.num_nodes
    num_edges = data.num_edges

    node_types = list(NodeType)
    rel_types = list(RelationshipType)

    node_counts = [0] * len(node_types)
    for idx in range(num_nodes):
        feat = data.x[idx]
        for t_idx in range(len(node_types)):
            if feat[t_idx] == 1.0:
                node_counts[t_idx] += 1
                break

    edge_counts = [0] * len(rel_types)
    if num_edges > 0:
        for attr in data.edge_attr:
            r_idx = attr.item()
            if 0 <= r_idx < len(rel_types):
                edge_counts[r_idx] += 1

    summary_str = f"{num_nodes}_{num_edges}_{node_counts}_{edge_counts}"
    return hashlib.sha256(summary_str.encode("utf-8")).hexdigest()


def compute_level2_hash(data) -> str:
    """Level 2 Feature Hash: Level 1 + node feature matrix x."""
    hasher = hashlib.sha256()
    hasher.update(compute_level1_hash(data).encode("utf-8"))
    hasher.update(data.x.cpu().numpy().tobytes())
    return hashlib.sha256(hasher.hexdigest().encode("utf-8")).hexdigest()


def compute_level3_hash(data) -> str:
    """Level 3 Exact Tensor Replica Hash: pure graph tensors (x, edge_index, edge_attr), EXCLUDING y and metadata."""
    hasher = hashlib.sha256()
    hasher.update(data.x.cpu().numpy().tobytes())
    hasher.update(data.edge_index.cpu().numpy().tobytes())
    hasher.update(data.edge_attr.cpu().numpy().tobytes())
    return hasher.hexdigest()


def main():
    dataset_path = Path("data/processed/activities")
    if not dataset_path.exists() or not list(dataset_path.glob("activity_*.json")):
        print("No processed activities found in data/processed/activities/")
        return

    print("=" * 90)
    print("3-LEVEL MULTI-TIER GRAPH DATASET LEAKAGE & REPLICA GROUP ANALYSIS")
    print("=" * 90)

    raw_dataset = ActivityGraphDataset(root=str(dataset_path))
    dataset = [raw_dataset[i] for i in range(len(raw_dataset)) if raw_dataset[i].num_nodes > 0]
    num_graphs = len(dataset)

    raw_labels = [dataset[i].y.item() for i in range(num_graphs)]
    unique_labels = sorted(list(set(raw_labels)))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    mapped_labels = [label_to_idx[lbl] for lbl in raw_labels]

    # StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(range(num_graphs), mapped_labels))

    train_graphs = [dataset[i] for i in train_idx]
    val_graphs = [dataset[i] for i in val_idx]

    # Level 1, 2, 3 Hash Collections
    train_l1 = {compute_level1_hash(g) for g in train_graphs}
    train_l2 = {compute_level2_hash(g) for g in train_graphs}

    # Group Level 3 Hashes across dataset
    all_hash_groups = defaultdict(list)
    for g in dataset:
        h3 = compute_level3_hash(g)
        all_hash_groups[h3].append(g)

    # Train vs Val Level 3 Overlap
    train_l3 = {compute_level3_hash(g) for g in train_graphs}

    l1_matches = sum(1 for g in val_graphs if compute_level1_hash(g) in train_l1)
    l2_matches = sum(1 for g in val_graphs if compute_level2_hash(g) in train_l2)
    l3_matches = sum(1 for g in val_graphs if compute_level3_hash(g) in train_l3)

    print(f"Total Non-Empty Graphs : {num_graphs} ({len(train_graphs)} train / {len(val_graphs)} val)")
    print(f"Unique Level 3 Hashes  : {len(all_hash_groups)}")
    print("-" * 90)
    print(f"Level 1 Matches (Structural Topology)  : {l1_matches} / {len(val_graphs)} ({l1_matches/len(val_graphs)*100:.1f}%)")
    print(f"Level 2 Matches (Feature Matrix Same)  : {l2_matches} / {len(val_graphs)} ({l2_matches/len(val_graphs)*100:.1f}%)")
    print(f"Level 3 Matches (Exact Tensor Replica) : {l3_matches} / {len(val_graphs)} ({l3_matches/len(val_graphs)*100:.1f}%)")

    # Detailed Group Replica Analysis
    conflicting_label_hashes = 0
    consistent_label_hashes = 0
    multi_scenario_hashes = 0

    replica_report_list = []

    for h3, group in all_hash_groups.items():
        if len(group) > 1:
            g_labels = set(g.y.item() for g in group)
            g_scenarios = set(g.scenario_id for g in group if hasattr(g, "scenario_id"))
            g_train_count = sum(1 for g in group if g in train_graphs)
            g_val_count = sum(1 for g in group if g in val_graphs)

            is_conflicting = len(g_labels) > 1
            if is_conflicting:
                conflicting_label_hashes += 1
            else:
                consistent_label_hashes += 1

            if len(g_scenarios) > 1:
                multi_scenario_hashes += 1

            replica_report_list.append({
                "hash": h3[:12],
                "total_occurrences": len(group),
                "train_occurrences": g_train_count,
                "val_occurrences": g_val_count,
                "labels": sorted(list(g_labels)),
                "is_label_consistent": not is_conflicting,
                "unique_scenarios_count": len(g_scenarios),
                "scenarios": sorted(list(g_scenarios))[:5],
                "sample_activity_ids": [g.activity_id for g in group[:3]],
            })

    replica_report_list.sort(key=lambda x: x["total_occurrences"], reverse=True)

    print("\n" + "=" * 90)
    print("REPLICA GROUP CHARACTERISTICS (GROUPS WITH >= 2 OCCURRENCES)")
    print("=" * 90)
    print(f"Total Repeated Hash Groups      : {len(replica_report_list)}")
    print(f"Consistent Label Groups         : {consistent_label_hashes} ({consistent_label_hashes/max(len(replica_report_list),1)*100:.1f}%)")
    print(f"Conflicting Label Groups        : {conflicting_label_hashes} ({conflicting_label_hashes/max(len(replica_report_list),1)*100:.1f}%)")
    print(f"Multi-Scenario Spanning Groups  : {multi_scenario_hashes}")

    print("\nTOP 5 REPEATED GRAPH TENSOR GROUPS:")
    print("-" * 90)
    for r in replica_report_list[:5]:
        print(f"Hash: {r['hash']} | Total: {r['total_occurrences']} (Train: {r['train_occurrences']}, Val: {r['val_occurrences']}) | Labels: {r['labels']} | Scenarios: {r['scenarios']}")

    # Save artifact reports
    json_path = REPORTS_DIR / "replica_hash_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(replica_report_list, f, indent=2)

    md_path = REPORTS_DIR / "replica_hash_analysis.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# PyG Level 3 Graph Replica Group Analysis\n\n")
        f.write(f"- **Total Non-Empty Graphs**: {num_graphs}\n")
        f.write(f"- **Unique Level 3 Tensor Hashes**: {len(all_hash_groups)}\n")
        f.write(f"- **Repeated Hash Groups (>=2 graphs)**: {len(replica_report_list)}\n")
        f.write(f"- **Consistent Label Groups**: {consistent_label_hashes}\n")
        f.write(f"- **Conflicting Label Groups**: {conflicting_label_hashes}\n")
        f.write(f"- **Multi-Scenario Spanning Groups**: {multi_scenario_hashes}\n\n")
        f.write("## Top Repeated Graph Tensor Groups\n\n")
        f.write("| Hash (12-char) | Total | Train | Val | Labels | Scenarios Count |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in replica_report_list[:10]:
            f.write(f"| `{r['hash']}` | {r['total_occurrences']} | {r['train_occurrences']} | {r['val_occurrences']} | {r['labels']} | {r['unique_scenarios_count']} |\n")

    print(f"\nSaved replica JSON report to: {json_path.resolve()}")
    print(f"Saved replica MD report to  : {md_path.resolve()}")
    print("=" * 90)


if __name__ == "__main__":
    main()
