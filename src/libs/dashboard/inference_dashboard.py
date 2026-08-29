"""Inference dashboard: base summary plus a latency graph and request log."""
from __future__ import annotations

from .base_dashboard import BaseDashboard
from .graph_tile import GraphTile
from .table_tile import TableTile


class InferenceDashboard(BaseDashboard):
    """A dashboard for model inference runs."""

    name = "inference"

    def _build(self) -> None:
        self._add("values", GraphTile("request", "latency_ms"))
        self._add("log", TableTile(["time", "event", "message"]))
