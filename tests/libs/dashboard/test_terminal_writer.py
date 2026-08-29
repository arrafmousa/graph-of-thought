import io

from libs.dashboard import GenericDashboard, TerminalDashboardWriter


def test_terminal_writer_emits_tables_and_graphs_on_update():
    stream = io.StringIO()
    dashboard = GenericDashboard()
    dashboard.set_run_id("r")
    TerminalDashboardWriter(dashboard, stream)

    dashboard.tile("summary").update("steps", 3)
    dashboard.tile("values").update(0, 1.0)
    dashboard.tile("values").update(1, 2.0)
    dashboard.tile("log").update("t", "step", "msg")

    out = stream.getvalue()
    assert "=== Run r ===" in out
    assert "[summary]" in out and "[values]" in out and "[log]" in out
    assert "score vs step" in out
    assert "metric" in out and "steps" in out


def test_terminal_writer_writes_once_per_update():
    stream = io.StringIO()
    dashboard = GenericDashboard()
    TerminalDashboardWriter(dashboard, stream)
    dashboard.tile("summary").update("a", "b")
    first = stream.getvalue().count("=== Run")
    dashboard.tile("summary").update("c", "d")
    second = stream.getvalue().count("=== Run")
    assert second == first + 1
