"""Tests for strict semantic-judge classification."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from libs.semantic_judge import (
    AzureOpenAIBatchJudgeProvider,
    JudgeProviderRegistry,
    SemanticJudge,
)


def _judge() -> SemanticJudge:
    provider = JudgeProviderRegistry.with_builtin_providers().create(
        "synthetic",
        model_name="synthetic/judge",
    )
    return SemanticJudge(
        provider=provider,
        pair_prompt_template="Question: {question}\nPATH A:\n{path_a}\nPATH B:\n{path_b}",
        max_score=4,
        parse_attempts=2,
    )


def test_judge_returns_one_strict_result_per_pair():
    requests = [
        {
            "pair_id": "q:0:1",
            "question": "What is 2 + 2?",
            "path_a": "Add two and two.",
            "path_b": "Compute 2 + 2.",
        },
        {
            "pair_id": "q:2:3",
            "question": "What is 3 + 3?",
            "path_a": "Add three and three.",
            "path_b": "Compute 3 + 3.",
        },
    ]

    results = _judge().judge_pairs(requests=requests)

    assert [result["pair_id"] for result in results] == ["q:0:1", "q:2:3"]
    assert all(result["judgment"]["equivalence_score"] == 4 for result in results)
    assert all(result["judgment"]["redundancy_safe"] is True for result in results)


def test_judge_provider_kinds_are_discoverable():
    values = {kind.value for kind in JudgeProviderRegistry.with_builtin_providers().available()}
    assert values == {"azure_openai_batch", "synthetic"}


def test_azure_batch_provider_uploads_once_and_restores_request_order(
    monkeypatch, tmp_path
):
    captured = {}

    def create_file(*, file, purpose):
        captured["input"] = file.read().decode("utf-8")
        captured["purpose"] = purpose
        return SimpleNamespace(id="file-input")

    def create_batch(*, input_file_id, endpoint, completion_window):
        captured["batch"] = (input_file_id, endpoint, completion_window)
        return SimpleNamespace(
            id="batch-1",
            status="completed",
            output_file_id="file-output",
            error_file_id=None,
            model_dump_json=lambda indent: json.dumps({"id": "batch-1", "status": "completed"}, indent=indent),
        )

    responses = [
        {
            "custom_id": f"request-{index:08d}",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": f"response {index}"}}]},
            },
        }
        for index in (1, 0)
    ]
    client = SimpleNamespace(
        files=SimpleNamespace(
            create=create_file,
            content=lambda _file_id: SimpleNamespace(
                text="\n".join(json.dumps(response) for response in responses)
            ),
        ),
        batches=SimpleNamespace(create=create_batch),
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs: client))
    provider = AzureOpenAIBatchJudgeProvider(
        endpoint="https://example.cognitiveservices.azure.com/",
        api_path="openai/v1",
        deployment="gpt-5.1",
        api_key_environment_variable="AZURE_OPENAI_API_KEY",
        request_url="/v1/chat/completions",
        batch_endpoint="/chat/completions",
        completion_window="24h",
        system_instruction="Return JSON.",
        max_completion_tokens=256,
        max_score=4,
        poll_interval_seconds=60,
        max_wait_seconds=86400,
        submission_attempts=2,
        submission_retry_seconds=60,
        artifact_dir=tmp_path,
    )

    result = provider.generate(prompts=["first", "second"])

    assert result == ["response 0", "response 1"]
    assert captured["purpose"] == "batch"
    assert captured["batch"] == ("file-input", "/chat/completions", "24h")
    requests = [json.loads(line) for line in captured["input"].splitlines()]
    assert {request["body"]["model"] for request in requests} == {"gpt-5.1"}
    assert requests[0]["url"] == "/v1/chat/completions"
    assert requests[0]["body"]["response_format"]["type"] == "json_schema"
    assert (tmp_path / "batch_000_output.jsonl").is_file()