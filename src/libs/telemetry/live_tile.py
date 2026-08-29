"""Live tile contract as seen by telemetry (AGENTS.md section 5).

A structural protocol describing what telemetry needs from a dashboard tile,
so the telemetry library does not depend on the dashboard library.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LiveTile(Protocol):
    """A dashboard component telemetry can update."""

    def update(self, *args: Any) -> None:
        ...
