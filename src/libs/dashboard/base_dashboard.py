"""Base dashboard: the root of the dashboard class hierarchy (AGENTS.md sections 5, 13).

A dashboard is an object composed of named tile components. The base provides a
single basic tile (``summary``); every other dashboard subclasses it and adds
its own tiles by overriding ``_build``. Callers reach any component with
``dashboard.tile(name)`` (or via the telemetry writer) to update its values.

Live file writing is a separate concern (see ``LiveDashboardWriter``): the base
only notifies a registered listener whenever any component updates.
"""
from __future__ import annotations

from typing import Callable

from . import html_utils
from .table_tile import TableTile
from .tile import Tile


class BaseDashboard:
    """Root dashboard with one basic tile; subclasses add their own layout."""

    name = "base"

    def __init__(self) -> None:
        self._tiles: dict[str, Tile] = {}
        self._run_id = "?"
        self._listeners: list[Callable[[], None]] = []
        self._add("summary", TableTile(["metric", "value"]))
        self._build()

    def _build(self) -> None:
        """Hook for subclasses to add their tiles. The base adds none beyond summary."""

    def _add(self, name: str, tile: Tile) -> None:
        self._tiles[name] = tile
        tile.set_listener(self._notify)

    def add_update_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in self._listeners:
            callback()

    def components(self) -> dict[str, Tile]:
        return self._tiles

    def tile(self, name: str) -> Tile:
        if name not in self._tiles:
            raise KeyError(f"No dashboard component named '{name}'")
        return self._tiles[name]

    def set_run_id(self, run_id: str) -> None:
        self._run_id = run_id

    def render(self, refresh_seconds: int) -> str:
        sections = []
        for name, tile in self._tiles.items():
            sections.append(
                f"<section class=\"panel\"><h2>{html_utils.escape(name)}</h2>"
                f"{tile.render()}</section>"
            )
        return html_utils.document(f"Run {self._run_id}", "".join(sections), refresh_seconds)

    def render_text(self) -> str:
        blocks = [f"=== Run {self._run_id} ==="]
        for name, tile in self._tiles.items():
            blocks.append(f"\n[{name}]\n{tile.render_text()}")
        return "\n".join(blocks)
