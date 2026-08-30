"""H0 baseline: merge only when the generated token id is identical."""
from __future__ import annotations

from typing import Any

from .loaded_token import LoadedToken
from .merge_metric import MergeMetric


class TokenIdentityMetric(MergeMetric):
    """Intentionally weak sanity baseline (research plan H0)."""

    name = "token_identity"

    def requires_hidden(self) -> bool:
        return False

    def score(self, a: LoadedToken, b: LoadedToken, context: dict[str, Any]) -> float:
        return 1.0 if a.token_id == b.token_id else 0.0
