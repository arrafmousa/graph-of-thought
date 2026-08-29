"""Validate run manifests for structural completeness (AGENTS.md section 10)."""
from __future__ import annotations

from typing import Any

from .manifest_error import ManifestError

_REQUIRED_TOP_LEVEL = (
    "manifest_schema_version",
    "run_id",
    "timestamp_start_utc",
    "timestamp_end_utc",
    "status",
    "entrypoint",
    "command",
    "working_directory",
    "git",
    "environment",
    "configuration",
    "configuration_hash",
    "inputs",
    "outputs",
    "model",
    "randomness",
    "telemetry",
    "dashboard",
    "duration_seconds",
    "errors",
)

_VALID_STATUSES = ("initialized", "running", "completed", "failed")


class ManifestValidator:
    """Check required fields, status validity, and internal consistency."""

    def validate(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ManifestError("Manifest root must be an object")

        for field in _REQUIRED_TOP_LEVEL:
            if field not in data:
                raise ManifestError(f"Manifest missing required field '{field}'")

        status = data["status"]
        if status not in _VALID_STATUSES:
            raise ManifestError(f"Manifest status {status!r} is not one of {_VALID_STATUSES}")

        if not isinstance(data["errors"], list):
            raise ManifestError("Manifest 'errors' must be a list")

        for section in ("git", "environment", "configuration", "randomness", "telemetry", "dashboard"):
            if not isinstance(data[section], dict):
                raise ManifestError(f"Manifest '{section}' must be an object")

        randomness = data["randomness"]
        if "seeds" not in randomness:
            raise ManifestError("Manifest 'randomness.seeds' is required")

        telemetry = data["telemetry"]
        for key in ("path", "schema_version"):
            if key not in telemetry:
                raise ManifestError(f"Manifest 'telemetry.{key}' is required")

        dashboard = data["dashboard"]
        for key in ("template", "path"):
            if key not in dashboard:
                raise ManifestError(f"Manifest 'dashboard.{key}' is required")

        if status in ("completed", "failed") and data["timestamp_end_utc"] is None:
            raise ManifestError(f"Status '{status}' requires 'timestamp_end_utc'")
