"""H1: raw last-layer hidden-state cosine similarity (research plan H1)."""
from __future__ import annotations

from typing import Any

from . import vector_math
from .loaded_token import LoadedToken
from .merge_metric import MergeMetric


class HiddenCosineMetric(MergeMetric):
    """Cosine similarity between the two tokens' hidden states."""

    name = "hidden_cosine"

    def requires_hidden(self) -> bool:
        return True

    def score(self, a: LoadedToken, b: LoadedToken, context: dict[str, Any]) -> float:
        return vector_math.cosine(a.hidden, b.hidden)
