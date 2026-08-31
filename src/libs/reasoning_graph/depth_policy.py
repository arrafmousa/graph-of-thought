"""Discoverable enumeration of merge-candidate depth policies (research plan section 10.3)."""
from __future__ import annotations

from enum import Enum


class DepthPolicy(Enum):
    SAME_DEPTH = "same_depth"
    ABSOLUTE_WINDOW = "absolute_window"
    UNRESTRICTED = "unrestricted"
