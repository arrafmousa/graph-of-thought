"""Run orchestrator: composes isolated libraries into a reproducible run.

This is the only place library objects are composed (AGENTS.md section 3.2).
It enforces the manifest lifecycle of section 10.2: configuration is validated
and a manifest + telemetry stream are created before any substantive work; the
manifest is finalized, validated, and a dashboard rendered afterwards.
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.config import ConfigLoader
from libs.dashboard import DashboardRegistry, DashboardRenderer, LiveDashboardWriter
from libs.manifest import ManifestValidator, RunManifest
from libs.runcontext import EnvironmentProbe, GitProbe, RunIdFactory
from libs.telemetry import TelemetryWriter

_COMPONENT = "run_orchestrator"


class RunOrchestrator:
    """Execute a configured run and produce manifest, telemetry, and dashboard."""

    def __init__(self, repo_root: Path, schema_path: Path, tracked_packages: list[str]) -> None:
        self._repo_root = Path(repo_root)
        self._config_loader = ConfigLoader(schema_path)
        self._environment_probe = EnvironmentProbe(tracked_packages)
        self._git_probe = GitProbe(repo_root)
        self._run_id_factory = RunIdFactory()
        self._manifest_validator = ManifestValidator()
        self._dashboard_registry = DashboardRegistry.with_builtin_dashboards()
        self._renderer = DashboardRenderer.default()

    def run(self, config_path: Path, entrypoint: str, command: str) -> Path:
        # Steps 1-5 of the manifest lifecycle must succeed before real work.
        config = self._config_loader.load(config_path)
        run_id = self._run_id_factory.create(config["run"]["task_name"])
        run_dir = self._repo_root / "output" / run_id
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        logger = self._configure_logger(run_id, run_dir / "logs" / "run.log")
        template_name = config["run"]["dashboard_template"]
        dashboard = self._dashboard_registry.create(template_name)
        dashboard.set_run_id(run_id)
        live_writer = LiveDashboardWriter(
            dashboard, run_dir / "dashboard.html", config["run"]["dashboard_refresh_seconds"]
        )
        live_writer.write()
        telemetry = TelemetryWriter(run_dir / "telemetry.jsonl", run_id, dashboard=dashboard)
        manifest = RunManifest.create(
            run_id=run_id,
            entrypoint=entrypoint,
            command=command,
            working_directory=str(self._repo_root),
            git=self._git_probe.collect(),
            environment=self._environment_probe.collect(),
            configuration=config,
            seeds=config["randomness"]["seeds"],
            inputs={"config_path": str(config_path)},
            model={},
            telemetry_path="telemetry.jsonl",
            telemetry_schema_version=telemetry.schema_version,
            dashboard_template=template_name,
            dashboard_path="dashboard.html",
        )
        manifest_path = run_dir / "run_manifest.json"
        manifest.save(manifest_path)

        start = time.monotonic()
        manifest.mark_running()
        manifest.save(manifest_path)
        telemetry.emit("run_start", _COMPONENT, "run", message=f"run {run_id} started")
        logger.info("run %s started", run_id)

        try:
            outputs = self._execute_workload(config, run_dir, telemetry, logger)
            duration = time.monotonic() - start
            manifest.mark_completed(outputs, duration)
            telemetry.emit(
                "run_end", _COMPONENT, "run",
                metrics={"duration_seconds": duration}, message="run completed",
            )
            telemetry.component("summary").update("duration_s", round(duration, 3))
        except Exception as exc:  # observed, recorded, then re-raised (section 22)
            duration = time.monotonic() - start
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest.mark_failed(error, duration)
            telemetry.emit("exception", _COMPONENT, "run", error=error, message="run failed")
            logger.exception("run %s failed", run_id)
            self._finalize(manifest, manifest_path, run_dir, telemetry)
            raise
        else:
            self._finalize(manifest, manifest_path, run_dir, telemetry)

        return run_dir

    def _execute_workload(
        self,
        config: dict[str, Any],
        run_dir: Path,
        telemetry: TelemetryWriter,
        logger: logging.Logger,
    ) -> dict[str, Any]:
        seed = config["randomness"]["seeds"]["python"]
        rng = random.Random(seed)
        steps = config["workload"]["steps"]
        label = config["workload"]["step_label"]
        interval = config["workload"]["interval_seconds"]
        value_min = config["workload"]["value_min"]
        value_max = config["workload"]["value_max"]

        values = telemetry.component("values")
        log = telemetry.component("log")

        telemetry.emit(
            "phase_start", _COMPONENT, "workload",
            metrics={"steps": steps, "interval_seconds": interval},
        )
        scores: list[float] = []
        for step in range(steps):
            value = rng.uniform(value_min, value_max)
            scores.append(value)
            values.update(step, value)
            log.update(datetime.now(timezone.utc).isoformat(), f"{label}_step", f"{value:.4f}")
            if interval > 0:
                time.sleep(interval)
        telemetry.emit("phase_end", _COMPONENT, "workload", metrics={"count": len(scores)})
        logger.info("workload produced %d %s values", len(scores), label)

        telemetry.component("summary").update("steps", len(scores))
        telemetry.component("summary").update("value_range", f"[{value_min}, {value_max}]")

        artifact_path = run_dir / "artifacts" / "scores.json"
        artifact_path.write_text(self._to_json(scores), encoding="utf-8")
        telemetry.emit(
            "artifact_created", _COMPONENT, "workload",
            payload={"path": "artifacts/scores.json", "count": len(scores)},
        )

        return {
            "dashboard": "dashboard.html",
            "telemetry": "telemetry.jsonl",
            "artifacts": ["artifacts/scores.json"],
            "score_count": len(scores),
        }

    def _finalize(
        self,
        manifest: RunManifest,
        manifest_path: Path,
        run_dir: Path,
        telemetry: TelemetryWriter,
    ) -> None:
        telemetry.close()
        manifest.save(manifest_path)
        self._manifest_validator.validate(manifest.to_dict())
        self._renderer.render_run(run_dir)

    def _configure_logger(self, run_id: str, log_path: Path) -> logging.Logger:
        logger = logging.getLogger(f"got.run.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    @staticmethod
    def _to_json(value: Any) -> str:
        import json

        return json.dumps(value, indent=2)
