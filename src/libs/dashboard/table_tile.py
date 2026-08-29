"""Table tile: appends a row per update (AGENTS.md section 13)."""
from __future__ import annotations

from typing import Any

from . import html_utils
from .tile import Tile


class TableTile(Tile):
    """A table with fixed column headers. ``update`` appends one row.

    Example:
        table = TableTile(["title 1", "title 2"])
        table.update("a", "b")
    """

    def __init__(self, columns: list[str]) -> None:
        self._columns = list(columns)
        self._rows: list[list[str]] = []

    def update(self, *cells: Any) -> None:
        if len(cells) != len(self._columns):
            raise ValueError(
                f"TableTile expects {len(self._columns)} cells, got {len(cells)}"
            )
        self._rows.append([str(cell) for cell in cells])
        self._notify()

    def render(self) -> str:
        return html_utils.table(self._columns, self._rows)
