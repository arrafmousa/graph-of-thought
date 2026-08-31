"""Restrict which node pairs may merge (research plan section 10).

Independently of the similarity heuristic, a merge candidate must come from a
different chain, must not fold a chain onto itself, and must satisfy the
configured depth policy. Ancestor/descendant (cycle) safety is enforced
separately by the consolidator's DAG guard.
"""
from __future__ import annotations

from typing import Union

from .depth_policy import DepthPolicy
from .loaded_token import LoadedToken


class CandidateFilter:
    """Admissibility rules for candidate merges."""

    def __init__(self, *, depth_policy: Union[DepthPolicy, str], max_depth_difference: int) -> None:
        self._depth_policy = (
            depth_policy if isinstance(depth_policy, DepthPolicy) else DepthPolicy(depth_policy)
        )
        self._max_depth_difference = max_depth_difference

    def admissible(
        self, token: LoadedToken, rep: LoadedToken, cluster_chains: set[int]
    ) -> bool:
        if rep.chain_id == token.chain_id:
            return False
        if token.chain_id in cluster_chains:
            return False
        return self._depth_ok(token.token_index, rep.token_index)

    def candidate_depth_range(self, token_depth: int) -> tuple[int, int] | None:
        """Inclusive [low, high] of representative depths worth scanning, or ``None`` for all.

        Representatives are created at their own depth and processed in ascending depth,
        so their depth never exceeds ``token_depth``; this lets the consolidator index
        clusters by depth instead of scanning every cluster.
        """
        if self._depth_policy is DepthPolicy.UNRESTRICTED:
            return None
        if self._depth_policy is DepthPolicy.SAME_DEPTH:
            return (token_depth, token_depth)
        low = max(0, token_depth - self._max_depth_difference)
        return (low, token_depth)

    def _depth_ok(self, token_depth: int, rep_depth: int) -> bool:
        if self._depth_policy is DepthPolicy.UNRESTRICTED:
            return True
        if self._depth_policy is DepthPolicy.SAME_DEPTH:
            return token_depth == rep_depth
        return abs(token_depth - rep_depth) <= self._max_depth_difference
