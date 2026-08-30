"""Registry mapping merge-heuristic names to metric classes (research plan section 11)."""
from __future__ import annotations

from .hidden_cosine_metric import HiddenCosineMetric
from .merge_metric import MergeMetric
from .pooled_hidden_cosine_metric import PooledHiddenCosineMetric
from .token_identity_metric import TokenIdentityMetric


class MergeRegistry:
    """Create merge metrics by their registered name."""

    def __init__(self, metrics: dict[str, type[MergeMetric]]) -> None:
        self._metrics = metrics

    @classmethod
    def with_builtin_metrics(cls) -> "MergeRegistry":
        metrics: dict[str, type[MergeMetric]] = {}
        for metric_cls in (
            TokenIdentityMetric,
            HiddenCosineMetric,
            PooledHiddenCosineMetric,
        ):
            metrics[metric_cls.name] = metric_cls
        return cls(metrics)

    def create(self, name: str) -> MergeMetric:
        metric_cls = self._metrics.get(name)
        if metric_cls is None:
            available = sorted(self._metrics)
            raise KeyError(f"Unknown merge heuristic '{name}'. Available: {available}")
        return metric_cls()
