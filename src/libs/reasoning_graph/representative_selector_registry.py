"""Registry mapping representative-selection policies to selector classes.

Keeps the merge winner-selection strategy discoverable and configurable by
:class:`RepresentativePolicy`, even though a single policy exists today (research
plan section 9; user requirement).
"""
from __future__ import annotations

from typing import Union

from .representative_policy import RepresentativePolicy
from .representative_selector import RepresentativeSelector


class RepresentativeSelectorRegistry:
    """Create a representative selector for a given policy."""

    def __init__(self, selectors: dict[RepresentativePolicy, type[RepresentativeSelector]]) -> None:
        self._selectors = selectors

    @classmethod
    def with_builtin_selectors(cls) -> "RepresentativeSelectorRegistry":
        return cls({RepresentativePolicy.RECENT_MEAN_LOGPROB: RepresentativeSelector})

    def available(self) -> list[RepresentativePolicy]:
        return list(self._selectors)

    def create(
        self, policy: Union[RepresentativePolicy, str], *, window: int
    ) -> RepresentativeSelector:
        policy = self._coerce(policy)
        selector_cls = self._selectors.get(policy)
        if selector_cls is None:
            available = [p.value for p in self._selectors]
            raise KeyError(f"Unregistered representative policy '{policy}'. Available: {available}")
        return selector_cls(window=window)

    @staticmethod
    def _coerce(policy: Union[RepresentativePolicy, str]) -> RepresentativePolicy:
        if isinstance(policy, RepresentativePolicy):
            return policy
        try:
            return RepresentativePolicy(policy)
        except ValueError as exc:
            available = [p.value for p in RepresentativePolicy]
            raise KeyError(f"Unknown representative policy '{policy}'. Available: {available}") from exc
