"""Azure OpenAI asynchronous Batch API implementation of the semantic judge."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .judge_provider import JudgeProvider

_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled", "canceled"}


class AzureOpenAIBatchJudgeProvider(JudgeProvider):
    """Submit semantic judgments through one configured Azure OpenAI batch deployment."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_path: str,
        deployment: str,
        api_key_environment_variable: str,
        request_url: str,
        batch_endpoint: str,
        completion_window: str,
        system_instruction: str,
        max_completion_tokens: int,
        max_score: int,
        poll_interval_seconds: int,
        max_wait_seconds: int,
        submission_attempts: int,
        submission_retry_seconds: int,
        artifact_dir: Path,
    ) -> None:
        self._endpoint = endpoint
        self._api_path = api_path
        self._deployment = deployment
        self._api_key_environment_variable = api_key_environment_variable
        self._request_url = request_url
        self._batch_endpoint = batch_endpoint
        self._completion_window = completion_window
        self._system_instruction = system_instruction
        self._max_completion_tokens = max_completion_tokens
        self._max_score = max_score
        self._poll_interval_seconds = poll_interval_seconds
        self._max_wait_seconds = max_wait_seconds
        self._submission_attempts = submission_attempts
        self._submission_retry_seconds = submission_retry_seconds
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._batch_number = 0

    def generate(self, *, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        client = self._client()
        batch_number = self._batch_number
        self._batch_number += 1
        input_path = self._artifact_dir / f"batch_{batch_number:03d}_input.jsonl"
        self._write_input(input_path, prompts)
        with input_path.open("rb") as handle:
            input_file = client.files.create(file=handle, purpose="batch")
        batch = self._submit(client, input_file.id)
        status_path = self._artifact_dir / f"batch_{batch_number:03d}_status.json"
        batch = self._wait(client, batch, status_path)
        error_text = self._download_optional(client, batch.error_file_id)
        if error_text:
            (self._artifact_dir / f"batch_{batch_number:03d}_errors.jsonl").write_text(
                error_text, encoding="utf-8"
            )
        if batch.status != "completed":
            raise RuntimeError(f"Azure OpenAI batch {batch.id} ended with status {batch.status}")
        output_text = self._download_required(client, batch.output_file_id)
        (self._artifact_dir / f"batch_{batch_number:03d}_output.jsonl").write_text(
            output_text, encoding="utf-8"
        )
        return self._ordered_responses(output_text, len(prompts))

    def release(self) -> None:
        return None

    def _client(self):
        from openai import OpenAI

        api_key = os.environ.get(self._api_key_environment_variable)
        if not api_key:
            raise RuntimeError(
                f"Required environment variable '{self._api_key_environment_variable}' is not set"
            )
        base_url = f"{self._endpoint.rstrip('/')}/{self._api_path.strip('/')}/"
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _write_input(self, path: Path, prompts: list[str]) -> None:
        schema = self._response_schema()
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, prompt in enumerate(prompts):
                request = {
                    "custom_id": f"request-{index:08d}",
                    "method": "POST",
                    "url": self._request_url,
                    "body": {
                        "model": self._deployment,
                        "messages": [
                            {"role": "system", "content": self._system_instruction},
                            {"role": "user", "content": prompt},
                        ],
                        "max_completion_tokens": self._max_completion_tokens,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "merge_equivalence",
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    },
                }
                handle.write(json.dumps(request) + "\n")

    def _submit(self, client, input_file_id: str):
        last_error = None
        for attempt in range(self._submission_attempts):
            try:
                return client.batches.create(
                    input_file_id=input_file_id,
                    endpoint=self._batch_endpoint,
                    completion_window=self._completion_window,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 == self._submission_attempts:
                    break
                time.sleep(self._submission_retry_seconds * (2**attempt))
        raise RuntimeError(
            f"Azure OpenAI batch submission failed after {self._submission_attempts} attempts"
        ) from last_error

    def _wait(self, client, batch, status_path: Path):
        started = time.monotonic()
        while batch.status not in _TERMINAL_STATUSES:
            status_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
            print(f"Azure batch {batch.id}: {batch.status}", flush=True)
            if time.monotonic() - started >= self._max_wait_seconds:
                client.batches.cancel(batch.id)
                raise TimeoutError(
                    f"Azure OpenAI batch {batch.id} exceeded {self._max_wait_seconds} seconds"
                )
            time.sleep(self._poll_interval_seconds)
            batch = client.batches.retrieve(batch.id)
        status_path.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
        print(f"Azure batch {batch.id}: {batch.status}", flush=True)
        return batch

    @staticmethod
    def _download_required(client, file_id: str | None) -> str:
        if not file_id:
            raise RuntimeError("Completed Azure OpenAI batch has no output file")
        return client.files.content(file_id).text

    @staticmethod
    def _download_optional(client, file_id: str | None) -> str | None:
        return client.files.content(file_id).text if file_id else None

    @staticmethod
    def _ordered_responses(output_text: str, expected: int) -> list[str]:
        responses: dict[int, str] = {}
        failures: list[str] = []
        for line in output_text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id = record["custom_id"]
            index = int(custom_id.rsplit("-", 1)[-1])
            response = record.get("response")
            if not response or response.get("status_code") != 200:
                failures.append(custom_id)
                continue
            choices = response["body"].get("choices", [])
            if not choices:
                failures.append(custom_id)
                continue
            responses[index] = choices[0]["message"]["content"]
        missing = [index for index in range(expected) if index not in responses]
        if failures or missing:
            raise RuntimeError(
                f"Azure OpenAI batch has failed responses {failures} and missing indices {missing}"
            )
        return [responses[index] for index in range(expected)]

    def _response_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "equivalence_label": {
                    "type": "string",
                    "enum": ["different", "partial", "equivalent"],
                },
                "equivalence_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": self._max_score,
                },
                "redundancy_safe": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "shared_information": {"type": "string"},
                "critical_difference": {"type": "string"},
            },
            "required": [
                "equivalence_label",
                "equivalence_score",
                "redundancy_safe",
                "confidence",
                "shared_information",
                "critical_difference",
            ],
            "additionalProperties": False,
        }