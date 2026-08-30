"""Synthetic local dataset: deterministic arithmetic questions, no network.

Used for local CPU development and tests so the full pipeline runs without the
Hugging Face Hub or the ``datasets`` package (AGENTS.md local-cpu-dev skill).
Interchangeable with :class:`Gsm8kDataset` purely through configuration.
"""
from __future__ import annotations

import random
from typing import Optional

from . import number_utils
from .dataset_entry import DatasetEntry
from .dataset_provider import DatasetProvider


class SyntheticDataset(DatasetProvider):
    """Generate reproducible ``a + b`` style questions with numeric gold answers."""

    def __init__(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        dataset_config: str,
        answer_delimiter: str,
    ) -> None:
        self._dataset_id = dataset_id
        self._dataset_revision = dataset_revision
        self._dataset_config = dataset_config
        self._answer_delimiter = answer_delimiter

    def load(self, *, split: str, num_questions: int) -> list[DatasetEntry]:
        rng = random.Random(f"{self._dataset_id}:{self._dataset_revision}:{split}")
        entries: list[DatasetEntry] = []
        for index in range(num_questions):
            a = rng.randint(2, 40)
            b = rng.randint(2, 40)
            gold = a + b
            question = f"A basket holds {a} apples and {b} more are added. How many apples are there in total?"
            entries.append(
                DatasetEntry(
                    dataset=self._dataset_id,
                    question_id=f"{split}-{index:05d}",
                    question=question,
                    gold_answer=str(gold),
                    options=[],
                    metadata={"operands": [a, b]},
                )
            )
        return entries

    def parse_prediction(self, entry: DatasetEntry, completion: str) -> Optional[str]:
        delimiter = self._answer_delimiter
        tail = completion.rsplit(delimiter, 1)[-1] if delimiter in completion else completion
        return number_utils.last_number(tail)

    def is_correct(self, entry: DatasetEntry, predicted: Optional[str]) -> bool:
        if predicted is None:
            return False
        gold = number_utils.last_number(entry.gold_answer)
        return gold is not None and predicted == gold
