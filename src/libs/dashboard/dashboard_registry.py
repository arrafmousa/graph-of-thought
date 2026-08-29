"""Registry mapping dashboard names to dashboard classes (AGENTS.md section 13).

Selecting a dashboard by name creates the corresponding subclass, whose layout
is defined in its own class. Register new dashboards here to make them
selectable by configuration and reconstructable during regeneration.
"""
from __future__ import annotations

from .base_dashboard import BaseDashboard
from .evaluation_dashboard import EvaluationDashboard
from .generic_dashboard import GenericDashboard
from .inference_dashboard import InferenceDashboard
from .training_dashboard import TrainingDashboard


class DashboardRegistry:
    """Create dashboard instances by their registered name."""

    def __init__(self, dashboards: dict[str, type[BaseDashboard]]) -> None:
        self._dashboards = dashboards

    @classmethod
    def with_builtin_dashboards(cls) -> "DashboardRegistry":
        dashboards: dict[str, type[BaseDashboard]] = {}
        for dashboard_cls in (
            GenericDashboard,
            TrainingDashboard,
            InferenceDashboard,
            EvaluationDashboard,
        ):
            dashboards[dashboard_cls.name] = dashboard_cls
        return cls(dashboards)

    def create(self, name: str) -> BaseDashboard:
        dashboard_cls = self._dashboards.get(name)
        if dashboard_cls is None:
            available = sorted(self._dashboards)
            raise KeyError(f"Unknown dashboard '{name}'. Available: {available}")
        return dashboard_cls()
