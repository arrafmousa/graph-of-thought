"""Tests for the dataset factory library."""
from __future__ import annotations

import pytest

from libs.dataset import (
    DatasetEntry,
    DatasetProviderKind,
    DatasetRegistry,
    Gsm8kDataset,
    SyntheticDataset,
)
from libs.dataset import number_utils


def _synthetic() -> SyntheticDataset:
    return SyntheticDataset(
        dataset_id="local/synthetic-arithmetic",
        dataset_revision="v1",
        dataset_config="default",
        answer_delimiter="####",
    )


def test_synthetic_entries_are_deterministic_and_numeric():
    a = _synthetic().load(split="test", num_questions=3)
    b = _synthetic().load(split="test", num_questions=3)
    assert [e.gold_answer for e in a] == [e.gold_answer for e in b]
    assert all(e.gold_answer.lstrip("-").isdigit() for e in a)
    assert len(a) == 3


def test_synthetic_parse_and_correctness():
    provider = _synthetic()
    entry = provider.load(split="test", num_questions=1)[0]
    completion = f"reasoning here\n#### {entry.gold_answer}"
    predicted = provider.parse_prediction(entry, completion)
    assert predicted == entry.gold_answer
    assert provider.is_correct(entry, predicted) is True
    assert provider.is_correct(entry, "999999") is False
    assert provider.is_correct(entry, None) is False


def test_build_prompt_includes_instruction_and_question():
    entry = DatasetEntry(dataset="d", question_id="q1", question="2+2?", gold_answer="4")
    prompt = entry.build_prompt("Solve it.")
    assert "Solve it." in prompt
    assert "2+2?" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_number_utils_last_number():
    assert number_utils.last_number("first 3 then 1,234 apples") == "1234"
    assert number_utils.last_number("no digits") is None
    assert number_utils.last_number("answer is 7.5") == "7.5"


def test_gsm8k_prediction_uses_last_number_after_delimiter():
    provider = Gsm8kDataset(
        dataset_id="openai/gsm8k",
        dataset_revision="main",
        dataset_config="main",
        answer_delimiter="####",
    )
    entry = DatasetEntry(dataset="openai/gsm8k", question_id="q", question="?", gold_answer="18")
    assert provider.parse_prediction(entry, "work ... #### 18") == "18"
    assert provider.is_correct(entry, provider.parse_prediction(entry, "#### 18")) is True


def test_registry_creates_known_and_rejects_unknown():
    registry = DatasetRegistry.with_builtin_providers()
    provider = registry.create(
        "synthetic",
        dataset_id="x",
        dataset_revision="v1",
        dataset_config="c",
        answer_delimiter="####",
    )
    assert isinstance(provider, SyntheticDataset)
    with pytest.raises(KeyError):
        registry.create("Nope")


def test_provider_kinds_are_discoverable():
    registry = DatasetRegistry.with_builtin_providers()
    assert set(registry.available()) == set(DatasetProviderKind)
    assert {k.value for k in DatasetProviderKind} == {"gsm8k", "synthetic"}
