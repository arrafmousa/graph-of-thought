"""Structured telemetry event envelope (AGENTS.md section 11.1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TELEMETRY_SCHEMA_VERSION = "1.0.0"


@dataclass
class TelemetryEvent:
    """One machine-readable telemetry event line."""

    schema_version: str
    run_id: str
    timestamp_utc: str
    event_type: str
    component: str
    phase: str
    step: Optional[int] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    error: Optional[dict[str, Any]] = None
    payload: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
