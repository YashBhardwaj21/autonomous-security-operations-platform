from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median, stdev
import networkx as nx
import torch

from src.graph.dataset import ActivityGraphDataset
from src.graph.types import FEATURE_DIM, FEATURE_SPEC, RelationshipType, NodeType


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


def pyg_to_networkx_simple(data) -> nx.Graph:
    """Converts PyG Data to an undirected NetworkX Graph for component analysis."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    if data.num_edges > 0:
        edges = [(src.item(), dst.item()) for src, dst in zip(data.edge_index[0], data.edge_index[1])]
        G.add_edges_from(edges)
    return G


def main() -> None:
    dataset_path = Path("data/processed/activities")
    if not dataset_path.exists() or not list(dataset_path.glob("activity_*.json")):
        print("No processed activities found in data/processed/activities/")
        return

    print("=" * 90)
    print("COMPREHENSIVE PYG ACTIVITY GRAPH DATASET AUDIT")
    print("=" * 90)

    dataset = ActivityGraphDataset(root=str(dataset_path))
    num_graphs = len(dataset)

    node_counts: list[int] = []
    edge_counts: list[int] = []
    densities: list[float] = []

    node_type_counter = Counter()
    edge_type_counter = Counter()
    class_counter = Counter()
    graph_entity_presence = Counter()

    empty_graphs = 0
    zero_edge_graphs = 0
    nan_graphs = 0

    # Zero-edge breakdown
    zero_edge_1_node = 0
    zero_edge_2_nodes = 0
    zero_edge_3plus_nodes = 0

    # Connected component metrics
    num_components_list: list[int] = []
    largest_component_sizes: list[int] = []
    largest_component_ratios: list[float] = []

    # Collect all node feature vectors for feature sanity check
    all_node_features: list[torch.Tensor] = []

    node_types = list(NodeType)
    rel_types = list(RelationshipType)

    for i in range(num_graphs):
        data = dataset[i]
        n_nodes = data.num_nodes
        n_edges = data.num_edges

        node_counts.append(n_nodes)
        edge_counts.append(n_edges)

        if torch.isnan(data.x).any():
            nan_graphs += 1

        if n_nodes == 0:
            empty_graphs += 1

        if n_edges == 0:
            zero_edge_graphs += 1
            if n_nodes == 1:
                zero_edge_1_node += 1
            elif n_nodes == 2:
                zero_edge_2_nodes += 1
            elif n_nodes >= 3:
                zero_edge_3plus_nodes += 1

        if n_nodes > 1:
            densities.append((2.0 * n_edges) / (n_nodes * (n_nodes - 1)))

        if n_nodes > 0:
            all_node_features.append(data.x)

        # Node type counts & presence per graph
        present_in_graph = set()
        for idx in range(n_nodes):
            feat = data.x[idx]
            for t_idx, t in enumerate(node_types):
                if feat[t_idx] == 1.0:
                    ntype_val = t.value
                    node_type_counter[ntype_val] += 1
                    present_in_graph.add(ntype_val)
                    break

        for ntype_val in present_in_graph:
            graph_entity_presence[ntype_val] += 1

        # Edge type counts
        if n_edges > 0:
            for attr in data.edge_attr:
                idx = attr.item()
                if 0 <= idx < len(rel_types):
                    edge_type_counter[rel_types[idx].value] += 1

        # Target class counts
        if hasattr(data, "y") and data.y is not None:
            class_counter[data.y.item()] += 1

        # Connected Components Analysis
        if n_nodes > 0:
            G = pyg_to_networkx_simple(data)
            components = list(nx.connected_components(G))
            num_components = len(components)
            largest_comp = max(len(c) for c in components) if components else 0
            ratio = (largest_comp / n_nodes) if n_nodes > 0 else 0.0

            num_components_list.append(num_components)
            largest_component_sizes.append(largest_comp)
            largest_component_ratios.append(ratio)

    avg_nodes = mean(node_counts) if node_counts else 0.0
    std_nodes = stdev(node_counts) if len(node_counts) > 1 else 0.0
    avg_edges = mean(edge_counts) if edge_counts else 0.0
    std_edges = stdev(edge_counts) if len(edge_counts) > 1 else 0.0
    avg_density = mean(densities) if densities else 0.0
    avg_components = mean(num_components_list) if num_components_list else 0.0
    avg_largest_comp_ratio = mean(largest_component_ratios) if largest_component_ratios else 0.0

    # Graph Outliers Breakdown
    outliers_1000 = sum(1 for n in node_counts if n > 1000)
    outliers_5000 = sum(1 for n in node_counts if n > 5000)
    outliers_10000 = sum(1 for n in node_counts if n > 10000)

    # Feature-level statistics
    feature_stats: list[dict] = []
    if all_node_features:
        concat_x = torch.cat(all_node_features, dim=0)  # Shape: (total_nodes, FEATURE_DIM)
        total_nodes = concat_x.shape[0]

        for f_idx in range(FEATURE_DIM):
            col = concat_x[:, f_idx]
            f_name = FEATURE_SPEC[f_idx] if f_idx < len(FEATURE_SPEC) else f"feat_{f_idx}"
            col_min = col.min().item()
            col_max = col.max().item()
            col_mean = col.mean().item()
            col_std = col.std().item() if total_nodes > 1 else 0.0
            nonzero_pct = (torch.count_nonzero(col).item() / total_nodes * 100.0) if total_nodes > 0 else 0.0

            feature_stats.append({
                "idx": f_idx,
                "name": f_name,
                "min": round(col_min, 4),
                "max": round(col_max, 4),
                "mean": round(col_mean, 4),
                "std": round(col_std, 4),
                "nonzero_pct": round(nonzero_pct, 2),
            })

    print(f"\nDATASET SUMMARY METRICS")
    print("-" * 90)
    print(f"Total Graphs               : {num_graphs}")
    print(f"Feature Dimension (F)      : {FEATURE_DIM}")
    print(f"Total Target Classes       : {len(class_counter)}")
    print(f"Nodes / Graph (Mean ± Std) : {avg_nodes:.2f} ± {std_nodes:.2f}")
    print(f"Edges / Graph (Mean ± Std) : {avg_edges:.2f} ± {std_edges:.2f}")
    print(f"Average Density            : {avg_density:.4f}")

    print(f"\nFEATURE-LEVEL SANITY CHECK (ACROSS ALL NODES)")
    print("-" * 90)
    print(f"{'ID':<3} | {'Feature Name':<20} | {'Mean':<8} | {'Std':<8} | {'Min':<6} | {'Max':<6} | {'Non-Zero %':<10}")
    print("-" * 90)
    for fs in feature_stats:
        print(f"{fs['idx']:<3} | {fs['name']:<20} | {fs['mean']:<8.4f} | {fs['std']:<8.4f} | {fs['min']:<6.2f} | {fs['max']:<6.2f} | {fs['nonzero_pct']:<10.2f}%")

    print(f"\nGRAPH SIZE SKEW & OUTLIERS")
    print("-" * 90)
    print(f"Nodes (Min / 25% / Median / 75% / Max) : {min(node_counts)} / {percentile(node_counts, 0.25):.0f} / {median(node_counts):.0f} / {percentile(node_counts, 0.75):.0f} / {max(node_counts)}")
    print(f"Edges (Min / 25% / Median / 75% / Max) : {min(edge_counts)} / {percentile(edge_counts, 0.25):.0f} / {median(edge_counts):.0f} / {percentile(edge_counts, 0.75):.0f} / {max(edge_counts)}")
    print(f"Graphs > 1,000 nodes       : {outliers_1000}")
    print(f"Graphs > 5,000 nodes       : {outliers_5000}")
    print(f"Graphs > 10,000 nodes      : {outliers_10000}")

    print(f"\nZERO-EDGE GRAPH BREAKDOWN (Total Zero-Edge: {zero_edge_graphs} / {num_graphs} = {zero_edge_graphs/num_graphs*100:.1f}%)")
    print("-" * 90)
    print(f"1-Node Zero-Edge Graphs    : {zero_edge_1_node:>5} ({zero_edge_1_node/zero_edge_graphs*100:.1f}%) [Singletons]")
    print(f"2-Node Zero-Edge Graphs    : {zero_edge_2_nodes:>5} ({zero_edge_2_nodes/zero_edge_graphs*100:.1f}%)")
    print(f"3+ Node Zero-Edge Graphs   : {zero_edge_3plus_nodes:>5} ({zero_edge_3plus_nodes/zero_edge_graphs*100:.1f}%)")

    print(f"\nCONNECTED COMPONENTS METRICS")
    print("-" * 90)
    print(f"Average Components / Graph : {avg_components:.2f}")
    print(f"Average Main Component Ratio: {avg_largest_comp_ratio*100:.1f}% of graph nodes")

    print(f"\nCLASS DISTRIBUTION")
    print("-" * 90)
    for k, v in sorted(class_counter.items()):
        print(f"  Class {k:<15} : {v:>5} ({v/num_graphs*100:.1f}%)")

    print(f"\nNODE TYPE DISTRIBUTION")
    print("-" * 90)
    for k, v in node_type_counter.most_common():
        print(f"  {k:<25} : {v:>10}")

    print(f"\nEDGE TYPE DISTRIBUTION")
    print("-" * 90)
    for k, v in edge_type_counter.most_common():
        print(f"  {k:<25} : {v:>10}")

    print("=" * 90)

    # Save summary artifacts
    summary_data = {
        "num_graphs": num_graphs,
        "feature_dim": FEATURE_DIM,
        "num_classes": len(class_counter),
        "mean_nodes": round(avg_nodes, 2),
        "std_nodes": round(std_nodes, 2),
        "mean_edges": round(avg_edges, 2),
        "std_edges": round(std_edges, 2),
        "avg_density": round(avg_density, 4),
        "avg_components": round(avg_components, 2),
        "avg_main_component_ratio": round(avg_largest_comp_ratio, 4),
        "outliers": {
            "gt_1000_nodes": outliers_1000,
            "gt_5000_nodes": outliers_5000,
            "gt_10000_nodes": outliers_10000,
        },
        "zero_edge_breakdown": {
            "1_node": zero_edge_1_node,
            "2_nodes": zero_edge_2_nodes,
            "3plus_nodes": zero_edge_3plus_nodes,
        },
        "feature_sanity_check": feature_stats,
        "class_distribution": dict(class_counter),
        "node_type_counts": dict(node_type_counter),
        "edge_type_counts": dict(edge_type_counter),
    }

    json_path = REPORTS_DIR / "graph_dataset_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    md_path = REPORTS_DIR / "graph_dataset_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# PyG Graph Dataset Comprehensive Statistical Audit\n\n")
        f.write(f"- **Total Graphs**: {num_graphs}\n")
        f.write(f"- **Feature Dimension**: {FEATURE_DIM}\n")
        f.write(f"- **Classes**: {len(class_counter)}\n")
        f.write(f"- **Nodes / Graph (Mean ± Std)**: {avg_nodes:.2f} ± {std_nodes:.2f}\n")
        f.write(f"- **Edges / Graph (Mean ± Std)**: {avg_edges:.2f} ± {std_edges:.2f}\n")
        f.write(f"- **Avg Density**: {avg_density:.4f}\n")
        f.write(f"- **Avg Components / Graph**: {avg_components:.2f}\n")
        f.write(f"- **Main Component Coverage**: {avg_largest_comp_ratio*100:.1f}%\n\n")

        f.write("## Feature-Level Sanity Check\n\n")
        f.write("| ID | Feature Name | Mean | Std | Min | Max | Non-Zero % |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for fs in feature_stats:
            f.write(f"| {fs['idx']} | {fs['name']} | {fs['mean']} | {fs['std']} | {fs['min']} | {fs['max']} | {fs['nonzero_pct']}% |\n")

        f.write("\n## Graph Size Outliers\n\n")
        f.write(f"- **> 1,000 nodes**: {outliers_1000}\n")
        f.write(f"- **> 5,000 nodes**: {outliers_5000}\n")
        f.write(f"- **> 10,000 nodes**: {outliers_10000}\n\n")

        f.write("## Class Distribution\n\n")
        for k, v in sorted(class_counter.items()):
            f.write(f"- **Class {k}**: {v} ({v/num_graphs*100:.1f}%)\n")

    print(f"\nSaved report summary to: {json_path.resolve()}")
    print(f"Saved report markdown to: {md_path.resolve()}")


if __name__ == "__main__":
    main()
