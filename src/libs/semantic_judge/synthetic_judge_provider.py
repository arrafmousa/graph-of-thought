"""Deterministic semantic judge used by CPU tests and smoke runs."""
from __future__ import annotations

import json

from .judge_provider import JudgeProvider


class SyntheticJudgeProvider(JudgeProvider):
    """Return a valid deterministic equivalence judgment without an LLM dependency."""

    def __init__(
        self,
        *,
        model_name: str,
    ) -> None:
        self._model_name = model_name

    def generate(self, *, prompts: list[str]) -> list[str]:
        return [
            json.dumps(
                {
                    "equivalence_label": "equivalent",
                    "equivalence_score": 4,
                    "redundancy_safe": True,
                    "confidence": 1.0,
                    "shared_information": "Synthetic paths encode the same state.",
                    "critical_difference": "",
                }
            )
            for _prompt in prompts
        ]

    def release(self) -> None:
        return None