"""The consolidated reasoning graph (a DAG of merged token states)."""
from __future__ import annotations

from typing import Any

from .graph_node import GraphNode

ROOT_ID = -1


class ReasoningGraph:
    """Hold consolidated nodes/edges and expose DAG checks and serialization."""

    def __init__(self, *, heuristic: str, threshold: float) -> None:
        self._heuristic = heuristic
        self._threshold = threshold
        self._nodes: dict[int, GraphNode] = {}
        self._edges: set[tuple[int, int]] = set()
        self._merges: list[dict[str, Any]] = []
        self._raw_node_count = 0
        self._raw_edge_count = 0

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.cluster_id] = node

    def add_edge(self, src: int, dst: int) -> None:
        if src != dst:
            self._edges.add((src, dst))

    def record_merge(self, event: dict[str, Any]) -> None:
        self._merges.append(event)

    def set_raw_counts(self, node_count: int, edge_count: int) -> None:
        self._raw_node_count = node_count
        self._raw_edge_count = edge_count

    @property
    def nodes(self) -> dict[int, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> set[tuple[int, int]]:
        return self._edges

    @property
    def merges(self) -> list[dict[str, Any]]:
        return self._merges

    def in_degree(self) -> dict[int, int]:
        degree = {node_id: 0 for node_id in self._nodes}
        for _src, dst in self._edges:
            if dst in degree:
                degree[dst] += 1
        return degree

    def out_degree(self) -> dict[int, int]:
        degree = {node_id: 0 for node_id in self._nodes}
        for src, _dst in self._edges:
            if src in degree:
                degree[src] += 1
        return degree

    def is_dag(self) -> bool:
        indeg = self.in_degree()
        queue = [node_id for node_id, deg in indeg.items() if deg == 0]
        adjacency: dict[int, list[int]] = {node_id: [] for node_id in self._nodes}
        for src, dst in self._edges:
            if src in adjacency:
                adjacency[src].append(dst)
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for neighbor in adjacency.get(current, []):
                indeg[neighbor] -= 1
                if indeg[neighbor] == 0:
                    queue.append(neighbor)
        return visited == len(self._nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heuristic": self._heuristic,
            "threshold": self._threshold,
            "root_id": ROOT_ID,
            "raw_node_count": self._raw_node_count,
            "raw_edge_count": self._raw_edge_count,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [list(edge) for edge in sorted(self._edges)],
            "merges": self._merges,
            "dag_valid": self.is_dag(),
        }
