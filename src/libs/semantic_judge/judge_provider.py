"""Contract for batched semantic-judge inference."""
from __future__ import annotations

from abc import ABC, abstractmethod


class JudgeProvider(ABC):
    """Generate one structured response for every supplied judge prompt."""

    @abstractmethod
    def generate(self, *, prompts: list[str]) -> list[str]:
        """Return responses in the same order as ``prompts``."""

    @abstractmethod
    def release(self) -> None:
        """Release model resources after all judgments are complete."""