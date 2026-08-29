import json

import pytest

from libs.dashboard import BaseDashboard, GenericDashboard, GraphTile, LiveDashboardWriter
from libs.telemetry import TELEMETRY_SCHEMA_VERSION, TelemetryWriter


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_writer_logs_dashboard_init_with_name(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    TelemetryWriter(path, run_id="r", dashboard=GenericDashboard()).close()
    events = _read(path)
    assert events[0]["event_type"] == "dashboard_init"
    assert events[0]["payload"]["dashboard"] == "generic"
    assert events[0]["payload"]["components"] == ["summary", "values", "log"]
    assert events[0]["schema_version"] == TELEMETRY_SCHEMA_VERSION


def test_component_update_logs_and_updates_live_dashboard(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    dashboard = GenericDashboard()
    LiveDashboardWriter(dashboard, tmp_path / "dashboard.html", refresh_seconds=1)
    writer = TelemetryWriter(path, run_id="r", dashboard=dashboard)

    writer.component("values").update(0, 1.5)
    writer.component("values").update(1, 2.5)
    writer.component("log").update("t", "step", "msg")
    writer.close()

    events = _read(path)
    updates = [e for e in events if e["event_type"] == "tile_update"]
    assert [e["component"] for e in updates] == ["values", "values", "log"]
    assert updates[0]["payload"]["args"] == [0, 1.5]

    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "polyline" in html  # graph updated the live file


def test_attribute_access_to_components(tmp_path):
    dashboard = GenericDashboard()
    LiveDashboardWriter(dashboard, tmp_path / "dashboard.html", refresh_seconds=0)
    writer = TelemetryWriter(tmp_path / "t.jsonl", run_id="r", dashboard=dashboard)
    writer.values.update(0, 1.0)
    writer.values.update(1, 2.0)
    writer.close()
    assert "polyline" in (tmp_path / "dashboard.html").read_text(encoding="utf-8")


def test_component_name_conflict_raises(tmp_path):
    class _BadDashboard(BaseDashboard):
        name = "bad"

        def _build(self) -> None:
            self._add("emit", GraphTile("x", "y"))

    with pytest.raises(ValueError):
        TelemetryWriter(tmp_path / "t.jsonl", run_id="r", dashboard=_BadDashboard())
