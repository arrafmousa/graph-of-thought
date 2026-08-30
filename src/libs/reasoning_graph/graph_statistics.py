"""Compute machine-readable graph statistics and histograms (research plan sections 15, 16)."""
from __future__ import annotations

from typing import Any

from .reasoning_graph import ROOT_ID, ReasoningGraph


class GraphStatistics:
    """Derive diagnostics from a consolidated graph for reports and debugging."""

    def compute(self, graph: ReasoningGraph) -> dict[str, Any]:
        indeg = graph.in_degree()
        outdeg = graph.out_degree()
        node_ids = [nid for nid in graph.nodes if nid != ROOT_ID]
        sizes = [len(graph.nodes[nid].members) for nid in node_ids]
        depths = [graph.nodes[nid].depth for nid in node_ids]
        in_values = [indeg[nid] for nid in node_ids]
        out_values = [outdeg[nid] for nid in node_ids]

        consolidated_nodes = len(node_ids)
        consolidated_edges = len(graph.edges)
        raw_nodes = graph.to_dict()["raw_node_count"]
        raw_edges = graph.to_dict()["raw_edge_count"]

        return {
            "raw_token_nodes": raw_nodes,
            "raw_edges": raw_edges,
            "consolidated_nodes": consolidated_nodes,
            "consolidated_edges": consolidated_edges,
            "node_reduction": (1 - consolidated_nodes / raw_nodes) if raw_nodes else 0.0,
            "merge_events": len(graph.merges),
            "join_nodes": sum(1 for value in in_values if value > 1),
            "branch_nodes": sum(1 for value in out_values if value > 1),
            "terminal_nodes": sum(1 for nid in node_ids if graph.nodes[nid].terminal),
            "max_in_degree": max(in_values) if in_values else 0,
            "max_out_degree": max(out_values) if out_values else 0,
            "mean_in_degree": self._mean(in_values),
            "mean_out_degree": self._mean(out_values),
            "median_in_degree": self._median(in_values),
            "median_out_degree": self._median(out_values),
            "graph_depth": max(depths) if depths else 0,
            "largest_cluster": max(sizes) if sizes else 0,
            "dag_valid": graph.is_dag(),
            "histograms": {
                "in_degree": self._histogram(in_values),
                "out_degree": self._histogram(out_values),
                "cluster_size": self._histogram(sizes),
            },
        }

    @staticmethod
    def _histogram(values: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: int(item[0])))

    @staticmethod
    def _mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _median(values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return float(ordered[mid])
        return (ordered[mid - 1] + ordered[mid]) / 2
