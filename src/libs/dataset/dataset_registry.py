"""Registry mapping dataset provider kinds to their classes.

Selecting a provider by :class:`DatasetProviderKind` in configuration lets a run
switch datasets (or the loading backend) with no code change (research plan section 23).
"""
from __future__ import annotations

from typing import Any, Union

from .dataset_provider import DatasetProvider
from .dataset_provider_kind import DatasetProviderKind
from .gsm8k_dataset import Gsm8kDataset
from .synthetic_dataset import SyntheticDataset


class DatasetRegistry:
    """Create dataset providers by their registered kind."""

    def __init__(self, providers: dict[DatasetProviderKind, type[DatasetProvider]]) -> None:
        self._providers = providers

    @classmethod
    def with_builtin_providers(cls) -> "DatasetRegistry":
        return cls(
            {
                DatasetProviderKind.GSM8K: Gsm8kDataset,
                DatasetProviderKind.SYNTHETIC: SyntheticDataset,
            }
        )

    def available(self) -> list[DatasetProviderKind]:
        return list(self._providers)

    def create(self, kind: Union[DatasetProviderKind, str], **kwargs: Any) -> DatasetProvider:
        kind = self._coerce(kind)
        provider_cls = self._providers.get(kind)
        if provider_cls is None:
            available = [k.value for k in self._providers]
            raise KeyError(f"Unregistered dataset provider '{kind}'. Available: {available}")
        return provider_cls(**kwargs)

    @staticmethod
    def _coerce(kind: Union[DatasetProviderKind, str]) -> DatasetProviderKind:
        if isinstance(kind, DatasetProviderKind):
            return kind
        try:
            return DatasetProviderKind(kind)
        except ValueError as exc:
            available = [k.value for k in DatasetProviderKind]
            raise KeyError(f"Unknown dataset provider '{kind}'. Available: {available}") from exc
