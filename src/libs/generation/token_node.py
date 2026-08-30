"""One generated token and its latent/uncertainty features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TokenNode:
    """A single autoregressive step within a chain (research plan section 3.1).

    ``hidden`` is the selected-layer hidden state, already detached to CPU as a
    plain list of floats. ``logprob`` is the log-probability of the sampled
    token under the model at this step.
    """

    token_index: int
    token_id: int
    text: str
    logprob: float
    hidden: Optional[list[float]] = None

    def to_metadata(self) -> dict[str, Any]:
        """Serialize everything except the (separately stored) hidden vector."""
        return {
            "token_index": self.token_index,
            "token_id": self.token_id,
            "text": self.text,
            "logprob": self.logprob,
        }
