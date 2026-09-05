"""GSM8K provider: short-answer grade-school math (openai/gsm8k)."""
from __future__ import annotations

import random
from typing import Optional

from . import number_utils
from .dataset_entry import DatasetEntry
from .dataset_provider import DatasetProvider


class Gsm8kDataset(DatasetProvider):
    """Load GSM8K from the Hugging Face Hub and score numeric answers.

    The gold answer is the number following ``answer_delimiter`` (``####``) in the
    reference solution; predictions are taken as the last number in a completion.
    """

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

    def load(
        self, *, split: str, num_questions: int, sample_seed: int
    ) -> list[DatasetEntry]:
        from datasets import load_dataset  # lazy: heavy dependency (AGENTS.md section 31)

        dataset = load_dataset(
            self._dataset_id,
            self._dataset_config,
            split=split,
            revision=self._dataset_revision,
        )
        entries: list[DatasetEntry] = []
        sample_size = min(num_questions, len(dataset))
        indices = random.Random(sample_seed).sample(range(len(dataset)), sample_size)
        for index in indices:
            record = dataset[index]
            question = str(record["question"])
            reference = str(record["answer"])
            gold = self._extract_gold(reference)
            entries.append(
                DatasetEntry(
                    dataset=self._dataset_id,
                    question_id=f"{split}-{index:05d}",
                    question=question,
                    gold_answer=gold,
                    options=[],
                    metadata={"reference_solution": reference, "source_index": index},
                )
            )
        return entries

    def parse_prediction(self, entry: DatasetEntry, completion: str) -> Optional[str]:
        region = number_utils.extract_delimited(completion, self._answer_delimiter)
        return number_utils.last_number(region if region is not None else completion)

    def is_correct(self, entry: DatasetEntry, predicted: Optional[str]) -> bool:
        if predicted is None:
            return False
        gold = number_utils.last_number(entry.gold_answer)
        return gold is not None and predicted == gold

    def answers_equivalent(
        self, first: Optional[str], second: Optional[str]
    ) -> Optional[bool]:
        if first is None or second is None:
            return None
        return number_utils.last_number(first) == number_utils.last_number(second)

    def evaluate_completion(self, entry: DatasetEntry, completion: str) -> dict:
        predicted = self.parse_prediction(entry, completion)
        return {
            "format_valid": number_utils.has_delimited_pair(
                completion, self._answer_delimiter
            ),
            "parse_valid": predicted is not None,
            "parser": "number",
            "predicted": predicted,
            "correct": self.is_correct(entry, predicted),
        }

    def _extract_gold(self, reference: str) -> str:
        if self._answer_delimiter in reference:
            tail = reference.rsplit(self._answer_delimiter, 1)[-1]
            normalized = number_utils.last_number(tail)
            if normalized is not None:
                return normalized
        normalized = number_utils.last_number(reference)
        return normalized if normalized is not None else reference.strip()
