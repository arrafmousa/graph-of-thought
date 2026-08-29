"""Run reproducibility manifest (AGENTS.md section 10)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0.0"


class RunManifest:
    """Assemble, mutate over the run lifecycle, and persist a run manifest."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        entrypoint: str,
        command: str,
        working_directory: str,
        git: dict[str, Any],
        environment: dict[str, Any],
        configuration: dict[str, Any],
        seeds: dict[str, Any],
        inputs: dict[str, Any],
        model: dict[str, Any],
        telemetry_path: str,
        telemetry_schema_version: str,
        dashboard_template: str,
        dashboard_path: str,
    ) -> "RunManifest":
        data: dict[str, Any] = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp_start_utc": cls._now(),
            "timestamp_end_utc": None,
            "status": "initialized",
            "entrypoint": entrypoint,
            "command": command,
            "working_directory": working_directory,
            "git": git,
            "environment": environment,
            "configuration": configuration,
            "configuration_hash": cls.compute_config_hash(configuration),
            "inputs": inputs,
            "outputs": {},
            "model": model,
            "randomness": {"seeds": seeds},
            "telemetry": {
                "path": telemetry_path,
                "schema_version": telemetry_schema_version,
            },
            "dashboard": {
                "template": dashboard_template,
                "path": dashboard_path,
            },
            "duration_seconds": 0,
            "errors": [],
        }
        return cls(data)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def compute_config_hash(configuration: dict[str, Any]) -> str:
        canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def mark_running(self) -> None:
        self._data["status"] = "running"

    def mark_completed(self, outputs: dict[str, Any], duration_seconds: float) -> None:
        self._data["status"] = "completed"
        self._data["outputs"] = outputs
        self._data["duration_seconds"] = duration_seconds
        self._data["timestamp_end_utc"] = self._now()

    def mark_failed(self, error: dict[str, Any], duration_seconds: float) -> None:
        self._data["status"] = "failed"
        self._data["duration_seconds"] = duration_seconds
        self._data["timestamp_end_utc"] = self._now()
        self.add_error(error)

    def add_error(self, error: dict[str, Any]) -> None:
        self._data["errors"].append(error)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._data, indent=2, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)
