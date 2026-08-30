"""Training orchestrator: fine-tune a classifier with full reproducibility.

Composes isolated libraries (config, runcontext, manifest, telemetry, dashboard,
training) into a reproducible training run following the manifest lifecycle of
AGENTS.md section 10.2. Datasets and models are Hugging Face references named in
configuration (sections 25, 31); weights are written under output/<run_id>/.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from libs.config import ConfigLoader
from libs.dashboard import DashboardRegistry, DashboardRenderer, LiveDashboardWriter, TerminalDashboardWriter
from libs.manifest import ManifestValidator, RunManifest
from libs.runcontext import EnvironmentProbe, GitProbe, RunIdFactory
from libs.telemetry import TelemetryWriter
from libs.training import SentimentFinetuner

from .telemetry_training_reporter import TelemetryTrainingReporter

_COMPONENT = "train_orchestrator"


class TrainOrchestrator:
    """Execute a configured fine-tuning run and produce manifest/telemetry/dashboard."""

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
        config = self._config_loader.load(config_path)
        training = config["training"]
        run_id = self._run_id_factory.create(config["run"]["task_name"])
        run_dir = self._repo_root / "output" / run_id
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)

        logger = self._configure_logger(run_id, run_dir / "logs" / "run.log")
        template_name = config["run"]["dashboard_template"]
        dashboard = self._dashboard_registry.create(template_name)
        dashboard.set_run_id(run_id)
        live_writer = LiveDashboardWriter(
            dashboard, run_dir / "dashboard.html", config["run"]["dashboard_refresh_seconds"]
        )
        live_writer.write()
        if config["run"]["terminal_progress"]:
            TerminalDashboardWriter(dashboard, sys.stdout)
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
            model={
                "model_id": training["model_id"],
                "model_revision": training["model_revision"],
                "dataset_id": training["dataset_id"],
                "dataset_revision": training["dataset_revision"],
                "dataset_config": training["dataset_config"],
                "task": "sentiment-classification",
            },
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
        telemetry.component("summary").update("model", training["model_id"])
        telemetry.component("summary").update(
            "dataset", f"{training['dataset_id']}/{training['dataset_config']}"
        )
        telemetry.component("summary").update("train_samples", training["train_samples"])
        logger.info("training run %s started", run_id)

        try:
            metrics = self._train(training, run_dir, telemetry)
            for key, value in metrics.items():
                telemetry.component("summary").update(key, round(value, 4))
            duration = time.monotonic() - start
            outputs = {
                "dashboard": "dashboard.html",
                "telemetry": "telemetry.jsonl",
                "checkpoints": "checkpoints/final",
                "metrics": metrics,
            }
            manifest.mark_completed(outputs, duration)
            telemetry.emit(
                "run_end", _COMPONENT, "run",
                metrics={"duration_seconds": duration, **metrics}, message="run completed",
            )
        except Exception as exc:  # observed, recorded, re-raised (section 22)
            duration = time.monotonic() - start
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest.mark_failed(error, duration)
            telemetry.emit("exception", _COMPONENT, "run", error=error, message="run failed")
            logger.exception("training run %s failed", run_id)
            self._finalize(manifest, manifest_path, run_dir, telemetry)
            raise
        else:
            self._finalize(manifest, manifest_path, run_dir, telemetry)

        return run_dir

    def _train(
        self, training: dict[str, Any], run_dir: Path, telemetry: TelemetryWriter
    ) -> dict[str, float]:
        finetuner = SentimentFinetuner(
            model_id=training["model_id"],
            model_revision=training["model_revision"],
            dataset_id=training["dataset_id"],
            dataset_revision=training["dataset_revision"],
            dataset_config=training["dataset_config"],
            text_field=training["text_field"],
            label_field=training["label_field"],
            num_labels=training["num_labels"],
            train_samples=training["train_samples"],
            eval_samples=training["eval_samples"],
            num_epochs=training["num_epochs"],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            max_length=training["max_length"],
            precision=training["precision"],
            seed=training["seed"],
        )
        reporter = TelemetryTrainingReporter(telemetry)
        return finetuner.finetune(run_dir, reporter)

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
        logger = logging.getLogger(f"got.train.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger
