"""Strict semantic-equivalence classification over partial reasoning paths."""
from __future__ import annotations

import json
from typing import Any

from .equivalence_label import EquivalenceLabel
from .judge_provider import JudgeProvider


class SemanticJudge:
    """Format pair prompts and validate every structured LLM judgment."""

    def __init__(
        self,
        *,
        provider: JudgeProvider,
        pair_prompt_template: str,
        max_score: int,
        parse_attempts: int,
    ) -> None:
        if parse_attempts < 1:
            raise ValueError("parse_attempts must be at least 1")
        self._provider = provider
        self._pair_prompt_template = pair_prompt_template
        self._max_score = max_score
        self._parse_attempts = parse_attempts

    def judge_pairs(self, *, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [
            self._pair_prompt_template.format(
                question=request["question"],
                path_a=request["path_a"],
                path_b=request["path_b"],
            )
            for request in requests
        ]
        results: list[dict[str, Any] | None] = [None] * len(requests)
        pending = list(range(len(requests)))
        failures: dict[int, Exception] = {}
        for attempt in range(self._parse_attempts):
            active_prompts = [prompts[index] for index in pending]
            responses = self._provider.generate(prompts=active_prompts)
            if len(responses) != len(pending):
                raise ValueError(
                    f"Judge returned {len(responses)} responses for {len(pending)} requests"
                )
            retry: list[int] = []
            for index, response in zip(pending, responses):
                try:
                    judgment = self._parse(response)
                except (ValueError, json.JSONDecodeError) as exc:
                    failures[index] = exc
                    retry.append(index)
                    continue
                results[index] = {
                    **requests[index],
                    "judgment": judgment,
                    "raw_response": response,
                    "attempts_used": attempt + 1,
                }
            pending = retry
            if not pending:
                break
        if pending:
            first = pending[0]
            raise ValueError(
                f"Judge response for pair {requests[first].get('pair_id')} remained invalid "
                f"after {self._parse_attempts} attempts: {failures[first]}"
            )
        return [result for result in results if result is not None]

    def release(self) -> None:
        self._provider.release()

    def _parse(self, response: str) -> dict[str, Any]:
        start = response.find("{")
        end = response.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"Judge response does not contain a JSON object: {response!r}")
        result = json.loads(response[start : end + 1])
        required = {
            "equivalence_label",
            "equivalence_score",
            "redundancy_safe",
            "confidence",
            "shared_information",
            "critical_difference",
        }
        missing = sorted(required - result.keys())
        if missing:
            raise ValueError(f"Judge response is missing fields: {missing}")
        labels = {label.value for label in EquivalenceLabel}
        if result["equivalence_label"] not in labels:
            raise ValueError(f"Unknown equivalence label: {result['equivalence_label']!r}")
        score = result["equivalence_score"]
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= self._max_score:
            raise ValueError(
                f"equivalence_score must be an integer from 0 to {self._max_score}"
            )
        if not isinstance(result["redundancy_safe"], bool):
            raise ValueError("redundancy_safe must be boolean")
        confidence = result["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("confidence must be numeric")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return result