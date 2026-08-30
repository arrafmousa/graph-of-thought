"""Registry mapping model provider class names to their classes.

Selecting a provider by name in configuration lets a run switch the model backend
(e.g. a local Hugging Face model vs a synthetic CPU stub, and later a hosted API)
with no code change (research plan section 23; user requirement).
"""
from __future__ import annotations

from typing import Any

from .huggingface_model_provider import HuggingFaceModelProvider
from .model_provider import ModelProvider
from .synthetic_model_provider import SyntheticModelProvider


class ModelProviderRegistry:
    """Create model providers by their registered class name."""

    def __init__(self, providers: dict[str, type[ModelProvider]]) -> None:
        self._providers = providers

    @classmethod
    def with_builtin_providers(cls) -> "ModelProviderRegistry":
        providers: dict[str, type[ModelProvider]] = {}
        for provider_cls in (HuggingFaceModelProvider, SyntheticModelProvider):
            providers[provider_cls.__name__] = provider_cls
        return cls(providers)

    def create(self, name: str, **kwargs: Any) -> ModelProvider:
        provider_cls = self._providers.get(name)
        if provider_cls is None:
            available = sorted(self._providers)
            raise KeyError(f"Unknown model provider '{name}'. Available: {available}")
        return provider_cls(**kwargs)
