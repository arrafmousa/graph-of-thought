"""Dataset library object: an interchangeable factory of multiple-choice /
short-answer reasoning datasets (see sbc_t4_poc_research_plan.md sections 4, 22).

A run selects a dataset provider by class name in configuration; the provider
loads records from an explicit Hugging Face repo id + revision (or a local
synthetic source) and yields :class:`DatasetEntry` values. Each entry knows how
to format its own prompt and carries the gold answer; the provider knows how to
parse a model completion back into a predicted answer and judge correctness.
"""
from .dataset_entry import DatasetEntry
from .dataset_provider import DatasetProvider
from .dataset_provider_kind import DatasetProviderKind
from .dataset_registry import DatasetRegistry
from .gsm8k_dataset import Gsm8kDataset
from .synthetic_dataset import SyntheticDataset

__all__ = [
    "DatasetEntry",
    "DatasetProvider",
    "DatasetProviderKind",
    "DatasetRegistry",
    "Gsm8kDataset",
    "SyntheticDataset",
]
