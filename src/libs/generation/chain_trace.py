"""One sampled chain of thought for a question."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .token_node import TokenNode


@dataclass
class ChainTrace:
    """A complete sampled reasoning trace (research plan sections 3, 5).

    ``predicted`` and ``correct`` are attached after generation by the
    orchestrator using the dataset provider; the generation library itself does
    not judge correctness (keeps libraries isolated).
    """

    chain_id: int
    tokens: list[TokenNode] = field(default_factory=list)
    completion_text: str = ""
    terminated_reason: str = "max_tokens"
    predicted: Optional[str] = None
    correct: Optional[bool] = None

    def recent_mean_logprob(self, window: int) -> float:
        """Mean sampled-token log-probability over the last ``window`` tokens."""
        if not self.tokens:
            return float("-inf")
        recent = self.tokens[-window:]
        return sum(t.logprob for t in recent) / len(recent)

    def metadata(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "completion_text": self.completion_text,
            "terminated_reason": self.terminated_reason,
            "predicted": self.predicted,
            "correct": self.correct,
            "num_tokens": len(self.tokens),
            "tokens": [t.to_metadata() for t in self.tokens],
        }
