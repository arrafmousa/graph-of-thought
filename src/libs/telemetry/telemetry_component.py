"""A named dashboard component bound to the telemetry writer.

Accessed as ``telemetry.<name>``; calling ``update`` both records a
``tile_update`` telemetry event and pushes the values to the live dashboard
tile, which refreshes the HTML file.
"""
from __future__ import annotations

from typing import Any

from .live_tile import LiveTile


class TelemetryComponent:
    """Binds a dashboard tile to telemetry logging under a component name."""

    def __init__(self, writer: Any, name: str, tile: LiveTile) -> None:
        self._writer = writer
        self._name = name
        self._tile = tile

    def update(self, *args: Any) -> None:
        self._writer.log_tile_update(self._name, list(args))
        self._tile.update(*args)
