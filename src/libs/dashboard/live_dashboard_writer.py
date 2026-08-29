"""Live writer: persists a dashboard to HTML whenever a component updates.

Attaches to a dashboard as its update listener; any tile update triggers a
rewrite of the static file, so a browser viewing it (with the refresh meta tag)
updates online as data arrives.
"""
from __future__ import annotations

from pathlib import Path

from .base_dashboard import BaseDashboard


class LiveDashboardWriter:
    """Write a dashboard's HTML to disk after every component update."""

    def __init__(self, dashboard: BaseDashboard, output_path: Path, refresh_seconds: int) -> None:
        self._dashboard = dashboard
        self._output_path = Path(output_path)
        self._refresh_seconds = refresh_seconds
        dashboard.add_update_listener(self.write)

    def write(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(
            self._dashboard.render(self._refresh_seconds), encoding="utf-8"
        )
