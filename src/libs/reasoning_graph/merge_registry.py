"""Registry mapping merge heuristics to metric classes (research plan section 11)."""
from __future__ import annotations

from typing import Union

from .hidden_cosine_metric import HiddenCosineMetric
from .merge_heuristic import MergeHeuristic
from .merge_metric import MergeMetric
from .pooled_hidden_cosine_metric import PooledHiddenCosineMetric
from .token_identity_metric import TokenIdentityMetric


class MergeRegistry:
    """Create merge metrics by their :class:`MergeHeuristic`."""

    def __init__(self, metrics: dict[MergeHeuristic, type[MergeMetric]]) -> None:
        self._metrics = metrics

    @classmethod
    def with_builtin_metrics(cls) -> "MergeRegistry":
        return cls(
            {
                MergeHeuristic.TOKEN_IDENTITY: TokenIdentityMetric,
                MergeHeuristic.HIDDEN_COSINE: HiddenCosineMetric,
                MergeHeuristic.POOLED_HIDDEN_COSINE: PooledHiddenCosineMetric,
            }
        )

    def available(self) -> list[MergeHeuristic]:
        return list(self._metrics)

    def create(self, heuristic: Union[MergeHeuristic, str]) -> MergeMetric:
        heuristic = self._coerce(heuristic)
        metric_cls = self._metrics.get(heuristic)
        if metric_cls is None:
            available = [h.value for h in self._metrics]
            raise KeyError(f"Unregistered merge heuristic '{heuristic}'. Available: {available}")
        return metric_cls()

    @staticmethod
    def _coerce(heuristic: Union[MergeHeuristic, str]) -> MergeHeuristic:
        if isinstance(heuristic, MergeHeuristic):
            return heuristic
        try:
            return MergeHeuristic(heuristic)
        except ValueError as exc:
            available = [h.value for h in MergeHeuristic]
            raise KeyError(f"Unknown merge heuristic '{heuristic}'. Available: {available}") from exc
