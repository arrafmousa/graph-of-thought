"""Forward training progress to telemetry and the training dashboard.

Composition point (AGENTS.md section 3.2): implements the training library's
``TrainingReporter`` contract by driving the telemetry writer's dashboard
components (loss graph, learning-rate graph, log table).
"""
from __future__ import annotations

from datetime import datetime, timezone

from libs.telemetry import TelemetryWriter
from libs.training import TrainingReporter

_COMPONENT = "trainer"


class TelemetryTrainingReporter(TrainingReporter):
    """Route per-step metrics and messages into telemetry + dashboard tiles."""

    def __init__(self, telemetry: TelemetryWriter) -> None:
        self._telemetry = telemetry

    def report_step(self, step: int, metrics: dict) -> None:
        if "loss" in metrics:
            self._telemetry.component("values").update(step, metrics["loss"])
        if "learning_rate" in metrics:
            self._telemetry.component("learning_rate").update(step, metrics["learning_rate"])
        self._telemetry.component("log").update(self._now(), f"step {step}", self._format(metrics))
        self._telemetry.emit("training_step", _COMPONENT, "train", step=step, metrics=metrics)

    def report_message(self, message: str) -> None:
        self._telemetry.component("log").update(self._now(), "message", message)
        self._telemetry.emit("training_message", _COMPONENT, "train", message=message)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _format(metrics: dict) -> str:
        return ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
