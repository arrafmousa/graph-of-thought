"""Offline latent consolidation of raw traces into a DAG (research plan section 13).

Processes token nodes in increasing generation depth; for each node it finds the
highest-similarity admissible node from another chain (per the merge metric and
candidate filter) and merges when the score meets the threshold — otherwise it
starts a new graph state. Every tentative merge is checked against a reachability
guard so the consolidated graph always stays acyclic.
"""
from __future__ import annotations

from typing import Any

from . import vector_math
from .candidate_filter import CandidateFilter
from .graph_node import GraphNode
from .loaded_chain import LoadedChain
from .merge_metric import MergeMetric
from .reasoning_graph import ROOT_ID, ReasoningGraph
from .representative_selector import RepresentativeSelector


class GraphConsolidator:
    """Build a consolidated :class:`ReasoningGraph` from loaded chains."""

    def __init__(
        self,
        *,
        candidate_filter: CandidateFilter,
        representative_selector: RepresentativeSelector,
    ) -> None:
        self._filter = candidate_filter
        self._selector = representative_selector

    def consolidate(
        self,
        *,
        chains: list[LoadedChain],
        metric: MergeMetric,
        threshold: float,
        pooling_k: int,
        context_window: int,
    ) -> ReasoningGraph:
        graph = ReasoningGraph(heuristic=metric.name, threshold=threshold)
        chain_map = {chain.chain_id: chain for chain in chains}
        token_map = {
            token.key: token for chain in chains for token in chain.tokens
        }
        raw_edges = sum(max(0, len(chain.tokens) - 1) for chain in chains)

        pooled = self._pooled(chains, pooling_k) if metric.requires_hidden() else {}
        context: dict[str, Any] = {"pooled": pooled, "pooling_k": pooling_k}

        graph.add_node(GraphNode(cluster_id=ROOT_ID, depth=-1, members=[], texts=["<root>"]))

        clusters: dict[int, dict[str, Any]] = {}
        node_cluster: dict[tuple[int, int], int] = {}
        adjacency: dict[int, set[int]] = {ROOT_ID: set()}
        next_id = 0

        for key in sorted(token_map, key=lambda k: (k[1], k[0])):
            chain_id, idx = key
            token = token_map[key]
            parent = node_cluster[(chain_id, idx - 1)] if idx > 0 else ROOT_ID

            best_cid = None
            best_score = None
            for cid, cluster in clusters.items():
                rep = token_map[cluster["rep_key"]]
                if not self._filter.admissible(token, rep, cluster["chains"]):
                    continue
                score = metric.score(token, rep, context)
                if score >= threshold and (best_score is None or score > best_score):
                    best_score = score
                    best_cid = cid

            assigned = None
            if best_cid is not None and not self._reaches(adjacency, best_cid, parent):
                assigned = best_cid
                cluster = clusters[best_cid]
                cluster["members"].append(key)
                cluster["chains"].add(chain_id)
                rep = token_map[cluster["rep_key"]]
                decision = self._selector.select(
                    chain_map[rep.chain_id], rep.token_index, chain_map[chain_id], idx
                )
                graph.record_merge(
                    {
                        **decision,
                        "similarity": best_score,
                        "heuristic": metric.name,
                        "merge_threshold": threshold,
                        "cluster_id": best_cid,
                        "node_a": [rep.chain_id, rep.token_index],
                        "node_b": [chain_id, idx],
                        "token_a": rep.text,
                        "token_b": token.text,
                        "depth_a": rep.token_index,
                        "depth_b": idx,
                        "context_a": self._context(chain_map[rep.chain_id], rep.token_index, context_window),
                        "context_b": self._context(chain_map[chain_id], idx, context_window),
                    }
                )
            if assigned is None:
                assigned = next_id
                next_id += 1
                clusters[assigned] = {"members": [key], "chains": {chain_id}, "rep_key": key}
                adjacency[assigned] = set()

            node_cluster[key] = assigned
            adjacency.setdefault(parent, set()).add(assigned)
            graph.add_edge(parent, assigned)

        self._materialize(graph, clusters, token_map, chain_map)
        graph.set_raw_counts(len(token_map), raw_edges)
        return graph

    def _materialize(self, graph, clusters, token_map, chain_map) -> None:
        for cid, cluster in clusters.items():
            members = cluster["members"]
            depth = min(member[1] for member in members)
            texts: list[str] = []
            terminal = False
            for member in members:
                text = token_map[member].text
                if text not in texts:
                    texts.append(text)
                if member[1] == len(chain_map[member[0]].tokens) - 1:
                    terminal = True
            graph.add_node(
                GraphNode(
                    cluster_id=cid,
                    depth=depth,
                    members=members,
                    texts=texts,
                    terminal=terminal,
                )
            )

    @staticmethod
    def _pooled(chains, pooling_k) -> dict[tuple[int, int], list[float]]:
        pooled: dict[tuple[int, int], list[float]] = {}
        for chain in chains:
            for i, token in enumerate(chain.tokens):
                window = chain.tokens[max(0, i - pooling_k + 1) : i + 1]
                vectors = [t.hidden for t in window if t.hidden is not None]
                result = vector_math.mean_pool(vectors)
                if result is not None:
                    pooled[token.key] = result
        return pooled

    @staticmethod
    def _context(chain, upto_index: int, context_window: int) -> str:
        start = max(0, upto_index - context_window + 1)
        return " ".join(token.text for token in chain.tokens[start : upto_index + 1])

    @staticmethod
    def _reaches(adjacency: dict[int, set[int]], start: int, target: int) -> bool:
        """Whether ``target`` is reachable from ``start`` (cycle guard)."""
        stack = [start]
        seen = {start}
        while stack:
            current = stack.pop()
            if current == target:
                return True
            for neighbor in adjacency.get(current, ()):  # type: ignore[arg-type]
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return False
