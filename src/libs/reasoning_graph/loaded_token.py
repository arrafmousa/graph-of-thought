"""A token state loaded from trace artifacts (graph library's own input type)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoadedToken:
    """One token node reconstructed from ``raw_traces.jsonl`` + ``hidden_states.jsonl``."""

    chain_id: int
    token_index: int
    token_id: int
    text: str
    logprob: float
    hidden: Optional[list[float]] = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.chain_id, self.token_index)
