"""Graph tile: appends an (x, y) point per update (AGENTS.md section 13)."""
from __future__ import annotations

from typing import Any

from . import html_utils
from .tile import Tile


class GraphTile(Tile):
    """A line chart of (x, y) points. ``update`` appends one point.

    Example:
        graph = GraphTile("step", "score")
        graph.update(0, 1.5)
    """

    def __init__(self, x_label: str, y_label: str) -> None:
        self._x_label = x_label
        self._y_label = y_label
        self._points: list[tuple[float, float]] = []

    def update(self, *args: Any) -> None:
        if len(args) != 2:
            raise ValueError(f"GraphTile.update expects (x, y), got {len(args)} args")
        x, y = args
        self._points.append((float(x), float(y)))
        self._notify()

    def render(self) -> str:
        return html_utils.line_chart(self._points, f"{self._y_label} vs {self._x_label}")
