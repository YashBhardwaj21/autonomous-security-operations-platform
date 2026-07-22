from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

DATASET_DIR = Path("data/processed/activities")

VALID_EDGE_TYPES = {
    "spawned",
    "loaded",
    "connected_to",
    "created_file",
    "modified_registry",
    "accessed_process",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def extract_entity_ids(container):

    ids = []

    if container is None:
        return ids

    if isinstance(container, dict):
        return list(container.keys())

    if isinstance(container, list):

        for item in container:

            if isinstance(item, str):
                ids.append(item)

            elif isinstance(item, dict):

                if "entity_id" in item:
                    ids.append(item["entity_id"])

                elif "id" in item:
                    ids.append(item["id"])

    return ids


def get_node_ids(activity):

    nodes = {}

    mapping = {
        "processes": "Process",
        "files": "File",
        "registry": "Registry",
        "network": "Network",
        "users": "User",
        "services": "Service",
    }

    for field, label in mapping.items():

        for eid in extract_entity_ids(activity.get(field)):
            nodes[eid] = label

    return nodes


def connected_components(nodes, edges):

    if not nodes:
        return 0

    adj = {n: set() for n in nodes}

    for src, _, dst in edges:

        if src in adj and dst in adj:
            adj[src].add(dst)
            adj[dst].add(src)

    visited = set()
    cc = 0

    for node in nodes:

        if node in visited:
            continue

        cc += 1

        stack = [node]

        while stack:

            cur = stack.pop()

            if cur in visited:
                continue

            visited.add(cur)

            stack.extend(adj[cur] - visited)

    return cc


def percentile(values, p):

    if not values:
        return 0

    values = sorted(values)

    idx = int((len(values)-1)*p)

    return values[idx]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    files = sorted(DATASET_DIR.glob("activity_*.json"))

    if not files:
        print("No activities found.")
        return

    total_nodes = 0
    total_edges = 0

    node_type_counter = Counter()
    edge_type_counter = Counter()

    isolated_type_counter = Counter()

    component_hist = Counter()

    invalid_edge_types = Counter()

    empty_graphs = []

    duplicate_edges = 0
    dangling_edges = 0
    self_loops = 0

    node_counts = []
    edge_counts = []
    densities = []

    largest_nodes = (0, "")
    largest_edges = (0, "")

    smallest_nodes = (10**9, "")
    smallest_edges = (10**9, "")

    print("=" * 90)
    print("GRAPH DATASET AUDIT")
    print("=" * 90)

    for file in files:

        with open(file, encoding="utf8") as f:
            activity = json.load(f)

        nodes = get_node_ids(activity)
        edges = activity.get("relationships", [])

        node_counts.append(len(nodes))
        edge_counts.append(len(edges))

        total_nodes += len(nodes)
        total_edges += len(edges)

        if len(nodes) == 0:
            empty_graphs.append(file.name)

        for t in nodes.values():
            node_type_counter[t] += 1

        degree = Counter()

        seen = set()

        for edge in edges:

            if len(edge) != 3:
                continue

            src, rel, dst = edge

            edge_type_counter[rel] += 1

            if rel not in VALID_EDGE_TYPES:
                invalid_edge_types[rel] += 1

            if src == dst:
                self_loops += 1

            tup = tuple(edge)

            if tup in seen:
                duplicate_edges += 1

            seen.add(tup)

            if src not in nodes or dst not in nodes:
                dangling_edges += 1

            if src in nodes:
                degree[src] += 1

            if dst in nodes:
                degree[dst] += 1

        for node, typ in nodes.items():

            if degree[node] == 0:
                isolated_type_counter[typ] += 1

        cc = connected_components(nodes, edges)
        component_hist[cc] += 1

        n = len(nodes)

        if n > 1:
            densities.append((2*len(edges))/(n*(n-1)))

        if len(nodes) > largest_nodes[0]:
            largest_nodes = (len(nodes), file.name)

        if len(edges) > largest_edges[0]:
            largest_edges = (len(edges), file.name)

        if len(nodes) < smallest_nodes[0]:
            smallest_nodes = (len(nodes), file.name)

        if len(edges) < smallest_edges[0]:
            smallest_edges = (len(edges), file.name)

    print()
    print("DATASET")
    print("-"*90)

    print(f"Activities                 : {len(files)}")
    print(f"Total Nodes                : {total_nodes}")
    print(f"Total Edges                : {total_edges}")
    print(f"Average Nodes              : {mean(node_counts):.2f}")
    print(f"Average Edges              : {mean(edge_counts):.2f}")
    print(f"Average Density            : {mean(densities):.4f}")

    print()
    print("NODE TYPES")
    print("-"*90)

    for k,v in node_type_counter.most_common():
        print(f"{k:<15}{v:>10}")

    print()
    print("EDGE TYPES")
    print("-"*90)

    for k,v in edge_type_counter.most_common():
        print(f"{k:<25}{v:>10}")

    print()
    print("CONNECTED COMPONENTS")
    print("-"*90)

    for k,v in sorted(component_hist.items()):
        print(f"{k:>2} component(s): {v}")

    print()
    print("ISOLATED NODE TYPES")
    print("-"*90)

    total_iso = sum(isolated_type_counter.values())

    for k,v in isolated_type_counter.most_common():
        print(f"{k:<15}{v:>10}")

    print(f"\nTotal Isolated Nodes       : {total_iso}")

    print()
    print("GRAPH QUALITY")
    print("-"*90)

    print(f"Dangling Edges             : {dangling_edges}")
    print(f"Duplicate Edges            : {duplicate_edges}")
    print(f"Self Loops                 : {self_loops}")
    print(f"Invalid Edge Types         : {sum(invalid_edge_types.values())}")
    print(f"Empty Graphs               : {len(empty_graphs)}")

    print()
    print("GRAPH SIZE DISTRIBUTION")
    print("-"*90)

    print(f"Nodes Min                 : {min(node_counts)}")
    print(f"Nodes 25%                 : {percentile(node_counts,0.25)}")
    print(f"Nodes Median              : {percentile(node_counts,0.50)}")
    print(f"Nodes 75%                 : {percentile(node_counts,0.75)}")
    print(f"Nodes Max                 : {max(node_counts)}")

    print()

    print(f"Edges Min                 : {min(edge_counts)}")
    print(f"Edges 25%                 : {percentile(edge_counts,0.25)}")
    print(f"Edges Median              : {percentile(edge_counts,0.50)}")
    print(f"Edges 75%                 : {percentile(edge_counts,0.75)}")
    print(f"Edges Max                 : {max(edge_counts)}")

    print()
    print("EXTREMES")
    print("-"*90)

    print(f"Largest Nodes             : {largest_nodes[0]} ({largest_nodes[1]})")
    print(f"Largest Edges             : {largest_edges[0]} ({largest_edges[1]})")
    print(f"Smallest Nodes            : {smallest_nodes[0]} ({smallest_nodes[1]})")
    print(f"Smallest Edges            : {smallest_edges[0]} ({smallest_edges[1]})")

    print()

    if empty_graphs:

        print("EMPTY GRAPHS")
        print("-"*90)

        for g in empty_graphs:
            print(g)

    print()
    print("="*90)

    if dangling_edges == 0:
        print("✓ No dangling edges")

    if duplicate_edges == 0:
        print("✓ No duplicate edges")

    if self_loops == 0:
        print("✓ No self loops")

    if len(empty_graphs) == 0:
        print("✓ No empty graphs")

    if sum(invalid_edge_types.values()) == 0:
        print("✓ All edge types valid")

    print("="*90)


if __name__ == "__main__":
    main()