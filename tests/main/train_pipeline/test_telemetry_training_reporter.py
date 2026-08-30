import json

from libs.dashboard import LiveDashboardWriter, TrainingDashboard
from libs.telemetry import TelemetryWriter
from main.train_pipeline import TelemetryTrainingReporter


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_reporter_updates_loss_and_lr_components(tmp_path):
    dashboard = TrainingDashboard()
    LiveDashboardWriter(dashboard, tmp_path / "dashboard.html", refresh_seconds=0)
    writer = TelemetryWriter(tmp_path / "telemetry.jsonl", run_id="r", dashboard=dashboard)
    reporter = TelemetryTrainingReporter(writer)

    reporter.report_step(0, {"loss": 1.2, "learning_rate": 5e-5})
    reporter.report_step(1, {"loss": 0.8, "learning_rate": 4e-5})
    reporter.report_message("training complete")
    writer.close()

    events = _read(tmp_path / "telemetry.jsonl")
    updates = [e for e in events if e["event_type"] == "tile_update"]
    components = [e["component"] for e in updates]
    assert components.count("values") == 2
    assert components.count("learning_rate") == 2
    assert any(e["event_type"] == "training_message" for e in events)

    # The loss graph rendered from telemetry shows a line.
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert "polyline" in html
