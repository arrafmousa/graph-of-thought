from libs.dashboard import GenericDashboard, LiveDashboardWriter


def test_writer_persists_on_component_update(tmp_path):
    output = tmp_path / "dashboard.html"
    dashboard = GenericDashboard()
    dashboard.set_run_id("r")
    writer = LiveDashboardWriter(dashboard, output, refresh_seconds=2)

    writer.write()
    assert output.is_file()

    dashboard.tile("values").update(0, 1.5)
    dashboard.tile("values").update(1, 2.5)
    text = output.read_text(encoding="utf-8")
    assert "polyline" in text
    assert 'http-equiv="refresh"' in text
    assert "r" in text
