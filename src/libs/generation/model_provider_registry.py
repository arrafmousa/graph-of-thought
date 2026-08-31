"""Registry mapping model provider kinds to their classes.

Selecting a provider by :class:`ModelProviderKind` in configuration lets a run switch
the model backend (a local Hugging Face model vs a synthetic CPU stub, and later a
hosted API) with no code change (research plan section 23; user requirement).
"""
from __future__ import annotations

from typing import Any, Union

from .huggingface_model_provider import HuggingFaceModelProvider
from .model_provider import ModelProvider
from .model_provider_kind import ModelProviderKind
from .synthetic_model_provider import SyntheticModelProvider


class ModelProviderRegistry:
    """Create model providers by their registered kind."""

    def __init__(self, providers: dict[ModelProviderKind, type[ModelProvider]]) -> None:
        self._providers = providers

    @classmethod
    def with_builtin_providers(cls) -> "ModelProviderRegistry":
        return cls(
            {
                ModelProviderKind.HUGGINGFACE: HuggingFaceModelProvider,
                ModelProviderKind.SYNTHETIC: SyntheticModelProvider,
            }
        )

    def available(self) -> list[ModelProviderKind]:
        return list(self._providers)

    def create(self, kind: Union[ModelProviderKind, str], **kwargs: Any) -> ModelProvider:
        kind = self._coerce(kind)
        provider_cls = self._providers.get(kind)
        if provider_cls is None:
            available = [k.value for k in self._providers]
            raise KeyError(f"Unregistered model provider '{kind}'. Available: {available}")
        return provider_cls(**kwargs)

    @staticmethod
    def _coerce(kind: Union[ModelProviderKind, str]) -> ModelProviderKind:
        if isinstance(kind, ModelProviderKind):
            return kind
        try:
            return ModelProviderKind(kind)
        except ValueError as exc:
            available = [k.value for k in ModelProviderKind]
            raise KeyError(f"Unknown model provider '{kind}'. Available: {available}") from exc
