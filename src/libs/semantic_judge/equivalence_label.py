"""Semantic-equivalence labels emitted by an LLM judge."""
from __future__ import annotations

from enum import Enum


class EquivalenceLabel(Enum):
    """Ordered labels for how closely two partial reasoning states agree."""

    DIFFERENT = "different"
    PARTIAL = "partial"
    EQUIVALENT = "equivalent"