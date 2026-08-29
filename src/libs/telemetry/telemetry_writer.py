"""Telemetry writer bound to a dashboard (AGENTS.md sections 11, 13).

Constructed with the dashboard it reports to. Each named dashboard component is
exposed as an attribute (``telemetry.<name>``) whose ``update`` records a
``tile_update`` event and pushes values to the live dashboard. Lifecycle events
are recorded with ``emit``. Every event is appended to ``telemetry.jsonl`` so
the dashboard is reproducible from telemetry alone.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .dashboard_handle import DashboardHandle
from .telemetry_component import TelemetryComponent
from .telemetry_event import TELEMETRY_SCHEMA_VERSION, TelemetryEvent


class TelemetryWriter:
    """Append versioned telemetry and drive a dashboard's named components."""

    def __init__(self, path: Path, run_id: str, dashboard: DashboardHandle) -> None:
        self._path = Path(path)
        self._run_id = run_id
        self._dashboard = dashboard
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._components: dict[str, TelemetryComponent] = {}
        self._bind_components()
        self._log_dashboard_init()

    @property
    def schema_version(self) -> str:
        return TELEMETRY_SCHEMA_VERSION

    @property
    def path(self) -> Path:
        return self._path

    def component(self, name: str) -> TelemetryComponent:
        return self._components[name]

    def emit(
        self,
        event_type: str,
        component: str,
        phase: str,
        *,
        step: Optional[int] = None,
        metrics: Optional[dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[dict[str, Any]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> TelemetryEvent:
        return self._write(
            event_type=event_type,
            component=component,
            phase=phase,
            step=step,
            metrics=metrics if metrics is not None else {},
            latency_ms=latency_ms,
            message=message,
            error=error,
            payload=payload,
        )

    def log_tile_update(self, name: str, args: list[Any]) -> None:
        self._write(
            event_type="tile_update",
            component=name,
            phase="reporting",
            step=None,
            metrics={},
            latency_ms=None,
            message=None,
            error=None,
            payload={"args": args},
        )

    def _bind_components(self) -> None:
        for name, tile in self._dashboard.components().items():
            if hasattr(self, name):
                raise ValueError(
                    f"Dashboard component name '{name}' conflicts with a writer attribute"
                )
            component = TelemetryComponent(self, name, tile)
            self._components[name] = component
            setattr(self, name, component)

    def _log_dashboard_init(self) -> None:
        layout = {
            "dashboard": self._dashboard.name,
            "components": list(self._dashboard.components().keys()),
        }
        self._write(
            event_type="dashboard_init",
            component="dashboard",
            phase="reporting",
            step=None,
            metrics={},
            latency_ms=None,
            message=None,
            error=None,
            payload=layout,
        )

    def _write(
        self,
        *,
        event_type: str,
        component: str,
        phase: str,
        step: Optional[int],
        metrics: dict[str, Any],
        latency_ms: Optional[float],
        message: Optional[str],
        error: Optional[dict[str, Any]],
        payload: Optional[dict[str, Any]],
    ) -> TelemetryEvent:
        event = TelemetryEvent(
            schema_version=TELEMETRY_SCHEMA_VERSION,
            run_id=self._run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            component=component,
            phase=phase,
            step=step,
            metrics=metrics,
            latency_ms=latency_ms,
            message=message,
            error=error,
            payload=payload,
        )
        self._file.write(json.dumps(event.to_dict()) + "\n")
        self._file.flush()
        return event

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
