"""Restrict which node pairs may merge (research plan section 10).

Independently of the similarity heuristic, a merge candidate must come from a
different chain, must not fold a chain onto itself, and must satisfy the
configured depth policy. Ancestor/descendant (cycle) safety is enforced
separately by the consolidator's DAG guard.
"""
from __future__ import annotations

from .loaded_token import LoadedToken


class CandidateFilter:
    """Admissibility rules for candidate merges."""

    def __init__(self, *, depth_policy: str, max_depth_difference: int) -> None:
        allowed = {"same_depth", "absolute_window", "unrestricted"}
        if depth_policy not in allowed:
            raise ValueError(f"Unknown depth_policy '{depth_policy}'. Allowed: {sorted(allowed)}")
        self._depth_policy = depth_policy
        self._max_depth_difference = max_depth_difference

    def admissible(
        self, token: LoadedToken, rep: LoadedToken, cluster_chains: set[int]
    ) -> bool:
        if rep.chain_id == token.chain_id:
            return False
        if token.chain_id in cluster_chains:
            return False
        return self._depth_ok(token.token_index, rep.token_index)

    def _depth_ok(self, token_depth: int, rep_depth: int) -> bool:
        if self._depth_policy == "unrestricted":
            return True
        if self._depth_policy == "same_depth":
            return token_depth == rep_depth
        return abs(token_depth - rep_depth) <= self._max_depth_difference
