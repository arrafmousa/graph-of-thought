"""Discoverable enumeration of node-consolidation heuristics (research plan section 11).

Every merge heuristic is listed here so the full set is knowable from the code and
selectable by configuration. Add a member and register its metric class in
:class:`MergeRegistry` to expose a new heuristic.
"""
from __future__ import annotations

from enum import Enum


class MergeHeuristic(Enum):
    TOKEN_IDENTITY = "token_identity"
    HIDDEN_COSINE = "hidden_cosine"
    POOLED_HIDDEN_COSINE = "pooled_hidden_cosine"
