"""Abstract contract for an interchangeable dataset provider (AGENTS.md section 5)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .dataset_entry import DatasetEntry


class DatasetProvider(ABC):
    """Load dataset entries and interpret model completions for one dataset.

    Concrete providers are selected by class name in configuration and receive
    only explicit configuration values (no hidden defaults). A provider owns the
    dataset-specific logic for three things: loading records into
    :class:`DatasetEntry` values, parsing a raw model completion into a predicted
    answer, and judging whether a prediction matches the gold answer.
    """

    @abstractmethod
    def load(
        self, *, split: str, num_questions: int, sample_seed: int
    ) -> list[DatasetEntry]:
        """Return up to ``num_questions`` entries from ``split``."""

    @abstractmethod
    def parse_prediction(self, entry: DatasetEntry, completion: str) -> Optional[str]:
        """Extract the predicted answer from ``completion`` or ``None`` if absent."""

    @abstractmethod
    def is_correct(self, entry: DatasetEntry, predicted: Optional[str]) -> bool:
        """Return whether ``predicted`` matches the entry's gold answer."""

    @abstractmethod
    def answers_equivalent(
        self, first: Optional[str], second: Optional[str]
    ) -> Optional[bool]:
        """Return semantic answer equality, or ``None`` when either answer is invalid."""

    @abstractmethod
    def evaluate_completion(self, entry: DatasetEntry, completion: str) -> dict:
        """Return parsed answer, validity flags, and correctness metadata."""
