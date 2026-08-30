"""Abstract contract for an interchangeable model provider (AGENTS.md section 5)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .chain_trace import ChainTrace


class ModelProvider(ABC):
    """Sample token-level reasoning traces from a prompt.

    Concrete providers (e.g. a Hugging Face local model, or a hosted API) are
    selected by class name in configuration and constructed with explicit model
    settings. Providers that cannot expose hidden states leave ``TokenNode.hidden``
    as ``None``; downstream latent heuristics then skip such traces.
    """

    @abstractmethod
    def generate(
        self,
        *,
        prompt: str,
        num_chains: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
        seed: int,
    ) -> list[ChainTrace]:
        """Return ``num_chains`` sampled traces for ``prompt``."""

    @abstractmethod
    def provides_hidden_states(self) -> bool:
        """Whether generated tokens carry hidden-state vectors."""
