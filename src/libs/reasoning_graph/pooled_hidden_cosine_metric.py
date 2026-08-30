"""H3: last-k-token pooled hidden-state cosine similarity (research plan H3)."""
from __future__ import annotations

from typing import Any

from . import vector_math
from .loaded_token import LoadedToken
from .merge_metric import MergeMetric


class PooledHiddenCosineMetric(MergeMetric):
    """Cosine similarity between mean-pooled last-k hidden states.

    The pooled representation for each token is precomputed by the consolidator and
    supplied via ``context['pooled']``; ``k`` is ``context['pooling_k']``.
    """

    name = "pooled_hidden_cosine"

    def requires_hidden(self) -> bool:
        return True

    def score(self, a: LoadedToken, b: LoadedToken, context: dict[str, Any]) -> float:
        pooled = context.get("pooled", {})
        return vector_math.cosine(pooled.get(a.key), pooled.get(b.key))
