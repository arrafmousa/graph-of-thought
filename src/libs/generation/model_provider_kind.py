"""Discoverable enumeration of available model providers.

Every generation backend is listed here so the full set is knowable from the code
(user requirement). Add a member and register its class in
:class:`ModelProviderRegistry` to expose a new backend.
"""
from __future__ import annotations

from enum import Enum


class ModelProviderKind(Enum):
    HUGGINGFACE = "huggingface"
    SYNTHETIC = "synthetic"
