"""Regenerate a static dashboard by replaying telemetry (AGENTS.md sections 12, 13).

The dashboard identity is recorded as a ``dashboard_init`` event and each
component update as a ``tile_update`` event. This renderer reconstructs the
dashboard subclass by name and replays the updates, so the dashboard is fully
reproducible from ``telemetry.jsonl`` alone — no running backend required.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base_dashboard import BaseDashboard
from .dashboard_registry import DashboardRegistry


class DashboardRenderer:
    """Rebuild and render a dashboard from recorded telemetry events."""

    def __init__(self, registry: DashboardRegistry) -> None:
        self._registry = registry

    @classmethod
    def default(cls) -> "DashboardRenderer":
        return cls(DashboardRegistry.with_builtin_dashboards())

    def render_events(self, events: list[dict], refresh_seconds: int) -> str:
        dashboard = self._build(events)
        for event in events:
            if event.get("event_type") == "tile_update":
                name = event.get("component")
                args = (event.get("payload") or {}).get("args", [])
                dashboard.tile(name).update(*args)
        return dashboard.render(refresh_seconds)

    def render_run(self, run_dir: Path) -> str:
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        events = self._read_events(run_dir / "telemetry.jsonl")
        html = self.render_events(events, refresh_seconds=0)
        output_path = run_dir / manifest["dashboard"]["path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return html

    def _build(self, events: list[dict]) -> BaseDashboard:
        for event in events:
            if event.get("event_type") == "dashboard_init":
                name = (event.get("payload") or {}).get("dashboard")
                dashboard = self._registry.create(name)
                dashboard.set_run_id(event.get("run_id", "?"))
                return dashboard
        raise KeyError("No 'dashboard_init' event found in telemetry")

    @staticmethod
    def _read_events(path: Path) -> list[dict]:
        events: list[dict] = []
        if not path.is_file():
            return events
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        return events
