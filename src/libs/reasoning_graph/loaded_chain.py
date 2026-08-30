"""A chain of loaded token states with its terminal outcome."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .loaded_token import LoadedToken


@dataclass
class LoadedChain:
    """One reconstructed reasoning trace for graph consolidation."""

    chain_id: int
    tokens: list[LoadedToken] = field(default_factory=list)
    completion_text: str = ""
    terminated_reason: str = "max_tokens"
    predicted: Optional[str] = None
    correct: Optional[bool] = None

    def recent_mean_logprob(self, upto_index: int, window: int) -> float:
        start = max(0, upto_index - window + 1)
        recent = self.tokens[start : upto_index + 1]
        if not recent:
            return float("-inf")
        return sum(t.logprob for t in recent) / len(recent)
