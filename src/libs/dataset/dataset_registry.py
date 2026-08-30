"""Registry mapping dataset provider class names to their classes.

Selecting a provider by name in configuration lets a run switch datasets (or the
loading backend, e.g. Hugging Face vs a local synthetic source) with no code
change (research plan section 23).
"""
from __future__ import annotations

from typing import Any

from .dataset_provider import DatasetProvider
from .gsm8k_dataset import Gsm8kDataset
from .synthetic_dataset import SyntheticDataset


class DatasetRegistry:
    """Create dataset providers by their registered class name."""

    def __init__(self, providers: dict[str, type[DatasetProvider]]) -> None:
        self._providers = providers

    @classmethod
    def with_builtin_providers(cls) -> "DatasetRegistry":
        providers: dict[str, type[DatasetProvider]] = {}
        for provider_cls in (Gsm8kDataset, SyntheticDataset):
            providers[provider_cls.__name__] = provider_cls
        return cls(providers)

    def create(self, name: str, **kwargs: Any) -> DatasetProvider:
        provider_cls = self._providers.get(name)
        if provider_cls is None:
            available = sorted(self._providers)
            raise KeyError(f"Unknown dataset provider '{name}'. Available: {available}")
        return provider_cls(**kwargs)
