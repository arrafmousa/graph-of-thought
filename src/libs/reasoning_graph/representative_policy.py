"""Discoverable enumeration of merge representative-selection policies (research plan section 9).

Currently a single policy exists; it is still an enum so the available set is knowable
from the code and can grow without touching call sites (user requirement).
"""
from __future__ import annotations

from enum import Enum


class RepresentativePolicy(Enum):
    RECENT_MEAN_LOGPROB = "recent_mean_logprob"
