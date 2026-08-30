"""Abstract contract for a pluggable node-consolidation heuristic (research plan section 11)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .loaded_token import LoadedToken


class MergeMetric(ABC):
    """Score how equivalent two token states are for merging.

    Implementations are selected by ``name`` in configuration. ``context`` carries
    precomputed auxiliary representations (e.g. pooled hidden states) so metrics do
    not each recompute them. A higher score means more similar; the consolidator
    merges when the score meets the configured threshold.
    """

    name: str = "abstract"

    @abstractmethod
    def requires_hidden(self) -> bool:
        """Whether this metric needs hidden-state vectors to be present."""

    @abstractmethod
    def score(self, a: LoadedToken, b: LoadedToken, context: dict[str, Any]) -> float:
        """Return a similarity score in a metric-defined range (cosine-like ~[-1, 1])."""
