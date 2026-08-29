"""Generic dashboard: base summary plus a values graph and event log."""
from __future__ import annotations

from .base_dashboard import BaseDashboard
from .graph_tile import GraphTile
from .table_tile import TableTile


class GenericDashboard(BaseDashboard):
    """A general-purpose dashboard for arbitrary runs."""

    name = "generic"

    def _build(self) -> None:
        self._add("values", GraphTile("step", "score"))
        self._add("log", TableTile(["time", "event", "message"]))
