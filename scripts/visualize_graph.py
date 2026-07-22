from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.graph.dataset import ActivityGraphDataset
from src.graph.types import NodeType, RelationshipType, NODE_TYPE_INDEX


OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


NODE_COLORS = {
    NodeType.PROCESS.value: "#1f77b4",   # Blue
    NodeType.FILE.value: "#2ca02c",      # Green
    NodeType.REGISTRY.value: "#ff7f0e",  # Orange
    NodeType.NETWORK.value: "#d62728",   # Red
    NodeType.USER.value: "#9467bd",      # Purple
    NodeType.SERVICE.value: "#8c564b",   # Brown
}


def pyg_to_networkx(data) -> nx.DiGraph:
    G = nx.DiGraph()
    num_nodes = data.num_nodes

    node_types = list(NodeType)
    rel_types = list(RelationshipType)

    # Infer node types from one-hot features in data.x
    for i in range(num_nodes):
        feat = data.x[i]
        ntype = "Unknown"
        for t_idx, t in enumerate(node_types):
            if feat[t_idx] == 1.0:
                ntype = t.value
                break
        G.add_node(i, type=ntype)

    if data.num_edges > 0:
        for src, dst, attr in zip(data.edge_index[0], data.edge_index[1], data.edge_attr):
            rel_idx = attr.item()
            rel_name = rel_types[rel_idx].value if 0 <= rel_idx < len(rel_types) else "rel"
            G.add_edge(src.item(), dst.item(), relation=rel_name)

    return G


def render_graph_plot(G: nx.DiGraph, title: str, save_path: Path) -> None:
    if G.number_of_nodes() == 0:
        return

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Subsample or adjust layout strategy for massive graphs using ego-subgraph expansion
    if num_nodes > 300:
        # Find highest-degree hub node (e.g. primary root process)
        degrees = dict(G.degree())
        hub_node = max(degrees, key=degrees.get)

        # Expand 2-hop ego subgraph around hub to preserve execution flow structure
        ego_nodes = set(nx.single_source_shortest_path_length(G.to_undirected(), hub_node, cutoff=2).keys())
        if len(ego_nodes) > 350:
            ego_nodes = set(sorted(ego_nodes, key=lambda n: degrees.get(n, 0), reverse=True)[:350])

        G_sub = G.subgraph(ego_nodes).copy()
        pos = nx.spring_layout(G_sub, seed=42, iterations=20)
        display_G = G_sub
        is_sampled = True
    else:
        pos = nx.spring_layout(G, seed=42)
        display_G = G
        is_sampled = False

    plt.figure(figsize=(10, 8))
    colors = [NODE_COLORS.get(display_G.nodes[n].get("type", ""), "#7f7f7f") for n in display_G.nodes]

    node_size = 120 if is_sampled else 300
    nx.draw_networkx_nodes(display_G, pos, node_color=colors, node_size=node_size, alpha=0.9)
    nx.draw_networkx_edges(display_G, pos, arrowstyle="->", arrowsize=8 if is_sampled else 12, edge_color="#aaaaaa", width=0.8 if is_sampled else 1.2)

    if not is_sampled:
        labels = {n: f"{n}:{display_G.nodes[n].get('type', '')[:4]}" for n in display_G.nodes}
        nx.draw_networkx_labels(display_G, pos, labels=labels, font_size=8, font_color="white")
        edge_labels = {(u, v): d.get("relation", "") for u, v, d in display_G.edges.data()}
        nx.draw_networkx_edge_labels(display_G, pos, edge_labels=edge_labels, font_size=7)

    plot_title = title
    if is_sampled:
        plot_title += f"\n(2-Hop Ego Subgraph: {display_G.number_of_nodes()} / {num_nodes} nodes & {display_G.number_of_edges()} / {num_edges} edges)"

    plt.title(plot_title, fontsize=11, fontweight="bold")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    dataset_path = Path("data/processed/activities")
    if not dataset_path.exists() or not list(dataset_path.glob("activity_*.json")):
        print("No processed activities found.")
        return

    dataset = ActivityGraphDataset(root=str(dataset_path))
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Sort indices by number of edges
    indexed_graphs = [(i, dataset[i].num_edges, dataset[i].num_nodes) for i in range(len(dataset))]
    indexed_graphs.sort(key=lambda item: item[1])

    smallest_idx = indexed_graphs[0][0]
    median_idx = indexed_graphs[len(indexed_graphs) // 2][0]
    largest_idx = indexed_graphs[-1][0]

    samples = [
        ("Smallest Graph", smallest_idx, OUTPUT_DIR / "graph_sample_smallest.png"),
        ("Median Graph", median_idx, OUTPUT_DIR / "graph_sample_median.png"),
        ("Largest Graph", largest_idx, OUTPUT_DIR / "graph_sample_largest.png"),
    ]

    print("=" * 90)
    print("VISUALIZING SAMPLE PYG ACTIVITY GRAPHS")
    print("=" * 90)

    for label, idx, out_file in samples:
        data = dataset[idx]
        G = pyg_to_networkx(data)
        render_graph_plot(G, f"{label} (ID: {data.activity_id}, Nodes: {data.num_nodes}, Edges: {data.num_edges})", out_file)
        print(f"[{label:<15}] Nodes: {data.num_nodes:>5} | Edges: {data.num_edges:>5} | Saved -> {out_file.name}")

    print("=" * 90)


if __name__ == "__main__":
    main()
