"""Dashboard library object: a class hierarchy of tile-composed dashboards.

``BaseDashboard`` provides one basic tile (``summary``). Every other dashboard
subclasses it and defines its own tile layout by overriding ``_build``. Callers
reach any component with ``dashboard.tile(name)`` — or, when a dashboard is
attached to a telemetry writer, via ``telemetry.<name>.update(...)`` — to update
its values. ``LiveDashboardWriter`` persists updates to HTML; ``DashboardRenderer``
regenerates a dashboard from telemetry by reconstructing its subclass.

Add a new dashboard by subclassing ``BaseDashboard`` and registering it in
``DashboardRegistry``.
"""
from .base_dashboard import BaseDashboard
from .dashboard_registry import DashboardRegistry
from .dashboard_renderer import DashboardRenderer
from .evaluation_dashboard import EvaluationDashboard
from .generic_dashboard import GenericDashboard
from .graph_tile import GraphTile
from .inference_dashboard import InferenceDashboard
from .live_dashboard_writer import LiveDashboardWriter
from .table_tile import TableTile
from .tile import Tile
from .training_dashboard import TrainingDashboard

__all__ = [
    "BaseDashboard",
    "DashboardRegistry",
    "DashboardRenderer",
    "EvaluationDashboard",
    "GenericDashboard",
    "GraphTile",
    "InferenceDashboard",
    "LiveDashboardWriter",
    "TableTile",
    "Tile",
    "TrainingDashboard",
]
