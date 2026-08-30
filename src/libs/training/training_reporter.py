"""Training progress contract (AGENTS.md section 5).

Isolated training code reports progress through this contract without depending
on the telemetry or dashboard libraries; an orchestrator provides a concrete
implementation that forwards to telemetry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TrainingReporter(ABC):
    """Receives training progress: per-step metrics and human-readable messages."""

    @abstractmethod
    def report_step(self, step: int, metrics: dict) -> None:
        """Report metrics (e.g. loss, learning_rate) at a global training step."""

    @abstractmethod
    def report_message(self, message: str) -> None:
        """Report a lifecycle/status message."""
