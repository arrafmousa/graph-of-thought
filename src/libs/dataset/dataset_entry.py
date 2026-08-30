"""A single dataset example, decoupled from any dataset's raw schema."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetEntry:
    """One reasoning question with its gold answer and prompt formatter.

    ``question_id`` is stable within a dataset/split. ``gold_answer`` is the
    already-extracted expected answer (e.g. the GSM8K number after ``####``).
    ``options`` is the allowed option set for multiple-choice datasets, or an
    empty list for short-answer datasets such as GSM8K.
    """

    dataset: str
    question_id: str
    question: str
    gold_answer: str
    options: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_prompt(self, system_instruction: str) -> str:
        """Render the deterministic prompt shown to the model."""
        parts = [system_instruction.strip(), "", f"Question: {self.question.strip()}"]
        if self.options:
            rendered = "\n".join(self.options)
            parts.extend(["", "Options:", rendered])
        parts.extend(["", "Answer:"])
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "question_id": self.question_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "options": list(self.options),
            "metadata": dict(self.metadata),
        }
