"""Dashboard handle contract as seen by telemetry (AGENTS.md section 5).

A structural protocol exposing the dashboard's named components so telemetry can
bind and update them without importing the dashboard library.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .live_tile import LiveTile


@runtime_checkable
class DashboardHandle(Protocol):
    """Exposes the dashboard's identity and named tile components."""

    name: str

    def components(self) -> dict[str, LiveTile]:
        ...
