"""Registry for semantic-judge inference providers."""
from __future__ import annotations

from typing import Any, Union

from .azure_openai_batch_judge_provider import AzureOpenAIBatchJudgeProvider
from .judge_provider import JudgeProvider
from .judge_provider_kind import JudgeProviderKind
from .synthetic_judge_provider import SyntheticJudgeProvider


class JudgeProviderRegistry:
    """Create judge providers by a discoverable configured kind."""

    def __init__(self, providers: dict[JudgeProviderKind, type[JudgeProvider]]) -> None:
        self._providers = providers

    @classmethod
    def with_builtin_providers(cls) -> "JudgeProviderRegistry":
        return cls(
            {
                JudgeProviderKind.AZURE_OPENAI_BATCH: AzureOpenAIBatchJudgeProvider,
                JudgeProviderKind.SYNTHETIC: SyntheticJudgeProvider,
            }
        )

    def available(self) -> list[JudgeProviderKind]:
        return list(self._providers)

    def create(
        self, kind: Union[JudgeProviderKind, str], **kwargs: Any
    ) -> JudgeProvider:
        resolved = kind if isinstance(kind, JudgeProviderKind) else JudgeProviderKind(kind)
        return self._providers[resolved](**kwargs)