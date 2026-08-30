"""Tests for the generation library (synthetic provider, codec, trace store)."""
from __future__ import annotations

import json

import pytest

from libs.generation import (
    ModelProviderRegistry,
    SyntheticModelProvider,
    TraceStore,
)
from libs.generation import vector_codec


def _provider() -> SyntheticModelProvider:
    return SyntheticModelProvider(
        model_id="local/synthetic",
        model_revision="v1",
        dtype="float32",
        device="cpu",
        hidden_layer=-1,
    )


def _generate(provider: SyntheticModelProvider):
    return provider.generate(
        prompt="A basket holds 10 apples and 5 more are added. How many in total?",
        num_chains=4,
        max_new_tokens=16,
        temperature=0.8,
        top_p=0.95,
        batch_size=2,
        seed=123,
    )


def test_generation_is_deterministic():
    a = _generate(_provider())
    b = _generate(_provider())
    assert [t.completion_text for t in a] == [t.completion_text for t in b]
    assert [tok.logprob for tok in a[0].tokens] == [tok.logprob for tok in b[0].tokens]


def test_tokens_carry_hidden_states():
    traces = _generate(_provider())
    assert _provider().provides_hidden_states() is True
    for trace in traces:
        assert trace.tokens
        for token in trace.tokens:
            assert token.hidden is not None
            assert len(token.hidden) == 16


def test_recent_mean_logprob_uses_window():
    trace = _generate(_provider())[0]
    value = trace.recent_mean_logprob(window=4)
    assert isinstance(value, float)


def test_vector_codec_round_trip():
    vector = [0.5, -1.25, 3.0, 0.0]
    decoded = vector_codec.decode(vector_codec.encode(vector))
    assert decoded == pytest.approx(vector)


def test_trace_store_writes_expected_artifacts(tmp_path):
    traces = _generate(_provider())
    store = TraceStore(tmp_path / "traces")
    directory = store.write_question(
        question_id="test-00001",
        entry={"question_id": "test-00001"},
        prompt="prompt",
        generation_params={"seed": 123},
        hidden_layer=-1,
        traces=traces,
        stats={"num_chains": len(traces)},
    )
    raw_lines = (directory / "raw_traces.jsonl").read_text(encoding="utf-8").splitlines()
    hidden_lines = (directory / "hidden_states.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == len(traces)
    assert hidden_lines
    record = json.loads(hidden_lines[0])
    assert record["dtype"] == "float32"
    assert vector_codec.decode(record["data"])


def test_registry_rejects_unknown_provider():
    with pytest.raises(KeyError):
        ModelProviderRegistry.with_builtin_providers().create("Missing")
