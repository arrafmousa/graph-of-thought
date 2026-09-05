"""Discoverable semantic-judge provider implementations."""
from __future__ import annotations

from enum import Enum


class JudgeProviderKind(Enum):
    """Available execution backends for semantic judgments."""

    AZURE_OPENAI_BATCH = "azure_openai_batch"
    SYNTHETIC = "synthetic"