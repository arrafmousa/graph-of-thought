"""Configurable Hugging Face provider for short-answer math datasets."""
from __future__ import annotations

import random
from typing import Any, Optional, Union

from . import number_utils
from .answer_mode import AnswerMode
from .dataset_entry import DatasetEntry
from .dataset_provider import DatasetProvider


class HuggingFaceMathDataset(DatasetProvider):
    """Load and reproducibly sample one logical math dataset from Hub configs."""

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_id: str,
        dataset_revision: str,
        dataset_configs: list[str],
        question_fields: list[str],
        answer_field: str,
        id_field: str,
        metadata_fields: list[str],
        answer_mode: Union[AnswerMode, str],
        answer_delimiter: str,
        require_answer_delimiter: bool,
        math_parser_timeout_seconds: int,
    ) -> None:
        self._dataset_name = dataset_name
        self._dataset_id = dataset_id
        self._dataset_revision = dataset_revision
        self._dataset_configs = list(dataset_configs)
        self._question_fields = list(question_fields)
        self._answer_field = answer_field
        self._id_field = id_field
        self._metadata_fields = list(metadata_fields)
        self._answer_mode = (
            answer_mode if isinstance(answer_mode, AnswerMode) else AnswerMode(answer_mode)
        )
        self._answer_delimiter = answer_delimiter
        self._require_answer_delimiter = require_answer_delimiter
        self._math_parser_timeout_seconds = math_parser_timeout_seconds
        self._math_parse_cache: dict[str, list[Any]] = {}

    def load(
        self, *, split: str, num_questions: int, sample_seed: int
    ) -> list[DatasetEntry]:
        from datasets import load_dataset

        source_rows: list[tuple[str, int, dict[str, Any]]] = []
        for config_name in self._dataset_configs:
            dataset = load_dataset(
                self._dataset_id,
                config_name,
                split=split,
                revision=self._dataset_revision,
            )
            source_rows.extend(
                (config_name, source_index, dataset[source_index])
                for source_index in range(len(dataset))
            )

        if len(source_rows) < num_questions:
            raise ValueError(
                f"Dataset '{self._dataset_name}' has {len(source_rows)} rows in split "
                f"'{split}', fewer than requested sample size {num_questions}"
            )

        selected = random.Random(sample_seed).sample(range(len(source_rows)), num_questions)
        return [self._entry(source_rows[index]) for index in selected]

    def parse_prediction(self, entry: DatasetEntry, completion: str) -> Optional[str]:
        region = number_utils.extract_delimited(completion, self._answer_delimiter)
        if region is None:
            if self._require_answer_delimiter:
                return None
            region = completion.strip()
        return self._extract(region)

    def is_correct(self, entry: DatasetEntry, predicted: Optional[str]) -> bool:
        return self.answers_equivalent(entry.gold_answer, predicted) is True

    def answers_equivalent(
        self, first: Optional[str], second: Optional[str]
    ) -> Optional[bool]:
        if first is None or second is None:
            return None
        first_text = number_utils.normalize_math_text(first)
        second_text = number_utils.normalize_math_text(second)
        if first_text == second_text:
            return True
        first_parsed = self._parse_math(first)
        second_parsed = self._parse_math(second)
        if first_parsed and second_parsed:
            from math_verify import verify

            return bool(
                verify(
                    first_parsed,
                    second_parsed,
                    timeout_seconds=self._math_parser_timeout_seconds,
                )
            )
        if self._answer_mode is AnswerMode.NUMBER:
            return None
        if not first_text or not second_text:
            return None
        return first_text == second_text

    def evaluate_completion(self, entry: DatasetEntry, completion: str) -> dict:
        region = number_utils.extract_delimited(completion, self._answer_delimiter)
        predicted = self.parse_prediction(entry, completion)
        return {
            "format_valid": number_utils.has_delimited_pair(
                completion, self._answer_delimiter
            ),
            "parse_valid": predicted is not None,
            "parser": self._parser_name(predicted),
            "raw_extracted_answer": region,
            "predicted": predicted,
            "correct": self.is_correct(entry, predicted),
        }

    def _entry(self, source: tuple[str, int, dict[str, Any]]) -> DatasetEntry:
        config_name, source_index, record = source
        question_parts = [str(record[field]).strip() for field in self._question_fields]
        question = "\n".join(part for part in question_parts if part)
        if not question:
            raise ValueError(
                f"Dataset '{self._dataset_name}' row {source_index} has no question text"
            )

        source_id = (
            str(record[self._id_field])
            if self._id_field and self._id_field in record
            else f"{source_index:05d}"
        )
        metadata = {
            "hub_dataset_id": self._dataset_id,
            "source_config": config_name,
            "source_index": source_index,
            "source_id": source_id,
        }
        metadata.update(
            {field: record[field] for field in self._metadata_fields if field in record}
        )
        raw_gold = str(record[self._answer_field])
        gold_region = number_utils.extract_delimited(raw_gold, self._answer_delimiter)
        gold = self._extract(gold_region if gold_region is not None else raw_gold.strip())
        if gold is None:
            raise ValueError(
                f"Dataset '{self._dataset_name}' row {source_index} has an invalid gold answer"
            )
        metadata["raw_gold_answer"] = raw_gold
        return DatasetEntry(
            dataset=self._dataset_name,
            question_id=f"{self._dataset_name}-{config_name}-{source_id}",
            question=question,
            gold_answer=gold,
            options=[],
            metadata=metadata,
        )

    def _extract(self, answer: str) -> Optional[str]:
        stripped = answer.strip()
        if not stripped:
            return None
        parsed = self._parse_math(stripped)
        if parsed:
            return stripped
        if self._answer_mode is AnswerMode.NUMBER:
            return None
        return stripped

    def _parser_name(self, predicted: Optional[str]) -> str:
        if predicted is None:
            return "invalid"
        return "math_verify" if self._parse_math(predicted) else "text"

    def _parse_math(self, answer: str) -> list[Any]:
        if answer not in self._math_parse_cache:
            from math_verify import parse

            stripped = answer.strip()
            sources = [stripped]
            if any(marker in stripped for marker in ("\\", "{", "^", "_")):
                sources.append(f"${stripped}$")
            parsed: list[Any] = []
            for source in sources:
                parsed = parse(source, parsing_timeout=self._math_parser_timeout_seconds)
                if parsed:
                    break
            self._math_parse_cache[answer] = parsed
        return self._math_parse_cache[answer]