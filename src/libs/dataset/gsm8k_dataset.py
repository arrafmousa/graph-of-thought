"""GSM8K provider: short-answer grade-school math (openai/gsm8k)."""
from __future__ import annotations

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

    def load(self, *, split: str, num_questions: int) -> list[DatasetEntry]:
        from datasets import load_dataset  # lazy: heavy dependency (AGENTS.md section 31)

        dataset = load_dataset(
            self._dataset_id,
            self._dataset_config,
            split=split,
            revision=self._dataset_revision,
        )
        entries: list[DatasetEntry] = []
        for index in range(min(num_questions, len(dataset))):
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
                    metadata={"reference_solution": reference},
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

    def _extract_gold(self, reference: str) -> str:
        if self._answer_delimiter in reference:
            tail = reference.rsplit(self._answer_delimiter, 1)[-1]
            normalized = number_utils.last_number(tail)
            if normalized is not None:
                return normalized
        normalized = number_utils.last_number(reference)
        return normalized if normalized is not None else reference.strip()
