import pytest

from libs.dashboard import DashboardRenderer


def _events() -> list[dict]:
    return [
        {"run_id": "run-1", "event_type": "dashboard_init", "component": "dashboard",
         "phase": "reporting", "payload": {"dashboard": "generic",
                                            "components": ["summary", "values", "log"]}},
        {"run_id": "run-1", "event_type": "tile_update", "component": "values",
         "phase": "reporting", "payload": {"args": [0, 1.0]}},
        {"run_id": "run-1", "event_type": "tile_update", "component": "values",
         "phase": "reporting", "payload": {"args": [1, 2.0]}},
        {"run_id": "run-1", "event_type": "tile_update", "component": "log",
         "phase": "reporting", "payload": {"args": ["t", "step", "hello"]}},
    ]


def test_render_events_reconstructs_dashboard_by_name():
    html = DashboardRenderer.default().render_events(_events(), refresh_seconds=0)
    assert "run-1" in html
    assert "polyline" in html
    assert "hello" in html


def test_render_events_without_init_raises():
    with pytest.raises(KeyError):
        DashboardRenderer.default().render_events([], refresh_seconds=0)
