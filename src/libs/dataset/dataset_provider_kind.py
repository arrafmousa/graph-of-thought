"""Discoverable enumeration of available dataset providers.

Every dataset backend is listed here so the full set is knowable from the code
(user requirement: no isolated classes — everything selectable is an enum, even
when a single option exists today). Add a member here and register its class in
:class:`DatasetRegistry` to expose a new dataset.
"""
from __future__ import annotations

from enum import Enum


class DatasetProviderKind(Enum):
    GSM8K = "gsm8k"
    SYNTHETIC = "synthetic"
