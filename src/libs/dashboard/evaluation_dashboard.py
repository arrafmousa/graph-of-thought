"""Evaluation dashboard: base summary plus a metric graph and event log."""
from __future__ import annotations

from .base_dashboard import BaseDashboard
from .graph_tile import GraphTile
from .table_tile import TableTile


class EvaluationDashboard(BaseDashboard):
    """A dashboard for evaluation/benchmarking runs."""

    name = "evaluation"

    def _build(self) -> None:
        self._add("values", GraphTile("step", "metric"))
        self._add("log", TableTile(["time", "event", "message"]))
