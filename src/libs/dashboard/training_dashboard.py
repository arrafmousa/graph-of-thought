"""Training dashboard: base summary plus loss and learning-rate graphs and a log."""
from __future__ import annotations

from .base_dashboard import BaseDashboard
from .graph_tile import GraphTile
from .table_tile import TableTile


class TrainingDashboard(BaseDashboard):
    """A dashboard for model training runs."""

    name = "training"

    def _build(self) -> None:
        self._add("values", GraphTile("step", "loss"))
        self._add("learning_rate", GraphTile("step", "learning_rate"))
        self._add("log", TableTile(["time", "event", "message"]))
