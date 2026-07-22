"""Digital Twin simulator — ported from etbackend, topology fabrication REMOVED.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set

import networkx as nx


class TopologyRequiredError(ValueError):
    pass


class DigitalTwinSimulator:
    TECHNIQUE_COSTS = {
        "T1021": 1.0, "T1021.001": 1.2, "T1021.002": 1.0, "T1078": 0.8,
        "T1550": 0.9, "T1210": 1.5, "T1570": 1.3, "T1080": 2.0,
    }
    DEFAULT_EDGE_COST = 1.5

    def __init__(self):
        self._graph = nx.DiGraph()
        self._asset_data: Dict[str, dict] = {}

    def build(self, assets: List[Dict], edges: List[Dict]) -> None:
        """Build topology from REAL asset inventory + REAL connectivity edges.

        `edges` is required and must be non-empty — no fabricated topology (H10).
        """
        if not edges:
            raise TopologyRequiredError(
                "Digital twin requires real asset-relationship edges. Provide "
                "connectivity from your inventory (data/reference/topology.json). "
                "Fabricated topology was removed (REPORT.md H10)."
            )
        self._graph.clear()
        self._asset_data.clear()
        for a in assets:
            aid = a["asset_id"]
            self._asset_data[aid] = a
            self._graph.add_node(
                aid, name=a.get("name", aid), type=a.get("type", "server"),
                criticality_tier=a.get("criticality_tier", 3),
                network_position=a.get("network_position", "internal"),
            )
        for e in edges:
            fa, ta = e.get("from_asset"), e.get("to_asset")
            if fa in self._graph and ta in self._graph:
                tech = e.get("technique_id", "T1021")
                self._graph.add_edge(
                    fa, ta, edge_type=e.get("edge_type", "connectivity"),
                    technique_id=tech, weight=self.TECHNIQUE_COSTS.get(tech, self.DEFAULT_EDGE_COST),
                )

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def simulate_compromise(self, start_node: str,
                            modifications: Optional[List[Dict]] = None) -> Dict[str, Any]:
        g = self._apply_modifications(self._graph.copy(), modifications or [])
        if start_node not in g:
            return {"reachable_nodes": [], "path_costs": {}, "critical_paths": [],
                    "error": f"Node {start_node} not in topology"}
        lengths, paths = nx.single_source_dijkstra(g, start_node, weight="weight", cutoff=10.0)
        reachable = {n: c for n, c in lengths.items() if n != start_node}
        critical = []
        for tgt, path in paths.items():
            if tgt == start_node or len(path) < 2:
                continue
            if self._asset_data.get(tgt, {}).get("criticality_tier", 3) <= 1:
                techs = [g.edges.get((path[i], path[i + 1]), {}).get("technique_id", "T1021")
                         for i in range(len(path) - 1)]
                critical.append({"path": path, "technique_ids": techs,
                                 "total_cost": round(lengths.get(tgt, 0), 2)})
        critical.sort(key=lambda x: x["total_cost"])
        return {
            "reachable_nodes": list(reachable.keys()),
            "path_costs": {k: round(v, 3) for k, v in reachable.items()},
            "critical_paths": critical[:5],
            "simulation_id": str(uuid.uuid4()),
        }

    def blast_radius(self, start_node: str) -> Dict[str, Any]:
        """Real reachability-based blast radius for the SOAR gate (REPORT.md H2).

        Returns reachable count and how many reachable assets are tier<=1 — the
        gate uses this instead of a fake user_count x static multiplier.
        """
        res = self.simulate_compromise(start_node)
        reachable = res.get("reachable_nodes", [])
        critical_reachable = sum(
            1 for n in reachable if self._asset_data.get(n, {}).get("criticality_tier", 3) <= 1
        )
        return {
            "reachable_count": len(reachable),
            "critical_reachable": critical_reachable,
            "reachable_nodes": reachable,
        }

    def _apply_modifications(self, g: nx.DiGraph, mods: List[Dict]) -> nx.DiGraph:
        for m in mods:
            op = m.get("op")
            if op == "remove_edge" and g.has_edge(m.get("from_asset"), m.get("to_asset")):
                g.remove_edge(m["from_asset"], m["to_asset"])
            elif op == "add_edge" and m.get("from_asset") and m.get("to_asset"):
                g.add_edge(m["from_asset"], m["to_asset"],
                           weight=m.get("params", {}).get("weight", 1.5), technique_id="T1021")
            elif op == "harden_node":
                nid = m.get("node_id")
                if nid in g:
                    for p in list(g.predecessors(nid)):
                        g[p][nid]["weight"] *= 2.5
        return g

    def get_topology(self) -> Dict[str, Any]:
        nodes = [{"asset_id": n, **{k: d.get(k) for k in ("name", "type", "criticality_tier", "network_position")}}
                 for n, d in self._graph.nodes(data=True)]
        edges = [{"from_asset": u, "to_asset": v, "edge_type": d.get("edge_type"),
                  "technique_weight": round(d.get("weight", 1.5), 3)}
                 for u, v, d in self._graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}
