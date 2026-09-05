"""Tests for the dataset factory library."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from libs.dataset import (
    DatasetEntry,
    DatasetProviderKind,
    DatasetRegistry,
    Gsm8kDataset,
    HuggingFaceMathDataset,
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
    a = _synthetic().load(split="test", num_questions=3, sample_seed=17)
    b = _synthetic().load(split="test", num_questions=3, sample_seed=17)
    assert [e.gold_answer for e in a] == [e.gold_answer for e in b]
    assert all(e.gold_answer.lstrip("-").isdigit() for e in a)
    assert len(a) == 3


def test_synthetic_parse_and_correctness():
    provider = _synthetic()
    entry = provider.load(split="test", num_questions=1, sample_seed=17)[0]
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


def test_number_utils_extract_delimited_prefers_last_pair():
    assert number_utils.extract_delimited("work #### 18 ####", "####") == "18"
    assert number_utils.extract_delimited("a #### x #### b #### 42 ####", "####") == "42"
    assert number_utils.extract_delimited("solution #### 18", "####") == "18"
    assert number_utils.extract_delimited("no marker here", "####") is None
    assert number_utils.has_delimited_pair("#### 18 ####", "####") is True
    assert number_utils.has_delimited_pair("#### 18", "####") is False


def test_gsm8k_prediction_uses_last_number_after_delimiter():
    provider = Gsm8kDataset(
        dataset_id="openai/gsm8k",
        dataset_revision="main",
        dataset_config="main",
        answer_delimiter="####",
    )
    entry = DatasetEntry(dataset="openai/gsm8k", question_id="q", question="?", gold_answer="18")
    assert provider.parse_prediction(entry, "work ... #### 18") == "18"
    assert provider.parse_prediction(entry, "work ... #### 18 ####") == "18"
    assert provider.is_correct(entry, provider.parse_prediction(entry, "#### 18 ####")) is True


def test_gsm8k_samples_random_rows_reproducibly(monkeypatch):
    records = [
        {"question": f"question {index}", "answer": f"work #### {index}"}
        for index in range(30)
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: records),
    )
    provider = Gsm8kDataset(
        dataset_id="openai/gsm8k",
        dataset_revision="pinned",
        dataset_config="main",
        answer_delimiter="####",
    )

    first = provider.load(split="test", num_questions=20, sample_seed=41)
    repeated = provider.load(split="test", num_questions=20, sample_seed=41)
    different = provider.load(split="test", num_questions=20, sample_seed=42)

    first_indices = [entry.metadata["source_index"] for entry in first]
    assert first_indices == [entry.metadata["source_index"] for entry in repeated]
    assert first_indices != [entry.metadata["source_index"] for entry in different]
    assert len(first_indices) == len(set(first_indices)) == 20
    assert first_indices != list(range(20))


def test_huggingface_math_combines_configs_and_samples_reproducibly(monkeypatch):
    records = {
        "part-one": [
            {"question": f"first {index}", "answer": str(index)}
            for index in range(15)
        ],
        "part-two": [
            {"question": f"second {index}", "answer": str(index + 15)}
            for index in range(15)
        ],
    }
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda _id, config, **_kwargs: records[config]
        ),
    )
    provider = HuggingFaceMathDataset(
        dataset_name="aime2025",
        dataset_id="opencompass/AIME2025",
        dataset_revision="pinned",
        dataset_configs=["part-one", "part-two"],
        question_fields=["question"],
        answer_field="answer",
        id_field="",
        metadata_fields=[],
        answer_mode="number",
        answer_delimiter="####",
        require_answer_delimiter=True,
        math_parser_timeout_seconds=0,
    )

    first = provider.load(split="test", num_questions=20, sample_seed=73)
    repeated = provider.load(split="test", num_questions=20, sample_seed=73)

    assert [entry.question_id for entry in first] == [entry.question_id for entry in repeated]
    assert len({entry.question_id for entry in first}) == 20
    assert {entry.metadata["source_config"] for entry in first} == {
        "part-one",
        "part-two",
    }


def test_huggingface_math_parses_numeric_and_symbolic_answers():
    numeric = HuggingFaceMathDataset(
        dataset_name="numeric",
        dataset_id="owner/numeric",
        dataset_revision="pinned",
        dataset_configs=["default"],
        question_fields=["question"],
        answer_field="answer",
        id_field="",
        metadata_fields=[],
        answer_mode="number",
        answer_delimiter="####",
        require_answer_delimiter=True,
        math_parser_timeout_seconds=0,
    )
    symbolic = HuggingFaceMathDataset(
        dataset_name="symbolic",
        dataset_id="owner/symbolic",
        dataset_revision="pinned",
        dataset_configs=["default"],
        question_fields=["problem"],
        answer_field="answer",
        id_field="unique_id",
        metadata_fields=["subject"],
        answer_mode="math_text",
        answer_delimiter="####",
        require_answer_delimiter=True,
        math_parser_timeout_seconds=0,
    )

    assert numeric.parse_prediction(
        DatasetEntry("numeric", "q", "?", "1,234"), "work #### 1,234 ####"
    ) == "1,234"
    entry = DatasetEntry("symbolic", "q", "?", r"\left( 3, \frac{\pi}{2} \right)")
    predicted = symbolic.parse_prediction(
        entry, r"work #### \boxed{(3,\frac{\pi}{2})} ####"
    )
    assert symbolic.is_correct(entry, predicted)

    mixed = HuggingFaceMathDataset(
        dataset_name="mixed",
        dataset_id="owner/mixed",
        dataset_revision="pinned",
        dataset_configs=["default"],
        question_fields=["question"],
        answer_field="answer",
        id_field="",
        metadata_fields=[],
        answer_mode="number_or_text",
        answer_delimiter="####",
        require_answer_delimiter=True,
        math_parser_timeout_seconds=0,
    )
    text_entry = DatasetEntry("mixed", "q", "?", "Mrs. Hilt")
    assert mixed.is_correct(text_entry, mixed.parse_prediction(text_entry, "#### Mrs. Hilt ####"))
    assert symbolic.answers_equivalent(r"\frac{14}{3}", "14/3") is True
    assert symbolic.parse_prediction(entry, "answer without delimiter") is None


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
    assert {k.value for k in DatasetProviderKind} == {
        "gsm8k",
        "huggingface_math",
        "synthetic",
    }
