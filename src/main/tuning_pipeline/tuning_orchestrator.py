"""Tuning orchestrator: the hyperparameter-setting first phase of the pipeline.

Generates token-level traces once for a representative sample of the dataset (~100
questions), then sweeps merge heuristics x thresholds. For every configuration it
consolidates all questions, aggregates graph features (merges, node reduction, d_in /
d_out and cluster-size histograms, join/branch counts), and collects two sets of merge
samples — the highest-similarity merges and the borderline (near-threshold) merges,
each with both chains' recent token context so a human can judge semantic quality.

It emits **one dashboard per configuration** plus a cross-configuration comparison
dashboard and a machine-readable ``tuning_summary.json``. Once a heuristic + threshold
are chosen here, they are set in the full-run config
(``configs/graph_pipeline/<experiment>/*.json``).

This is the only place the dataset, generation, reasoning-graph, and graph-report
libraries are composed for the tuning phase (AGENTS.md section 3.2); it follows the
manifest lifecycle of section 10.2.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.config import ConfigLoader
from libs.dataset import DatasetProviderKind, DatasetRegistry
from libs.generation import ModelProviderKind, ModelProviderRegistry, TraceStore
from libs.graph_report import TuningComparisonReport, TuningConfigReport
from libs.manifest import ManifestValidator, RunManifest
from libs.reasoning_graph import (
    CandidateFilter,
    DepthPolicy,
    GraphConsolidator,
    GraphStatistics,
    MergeHeuristic,
    MergeRegistry,
    RepresentativePolicy,
    RepresentativeSelectorRegistry,
    TraceLoader,
)
from libs.dashboard import (
    DashboardRegistry,
    DashboardRenderer,
    LiveDashboardWriter,
    TerminalDashboardWriter,
)
from libs.runcontext import EnvironmentProbe, GitProbe, RunIdFactory
from libs.telemetry import TelemetryWriter

_COMPONENT = "tuning_orchestrator"


class TuningOrchestrator:
    """Execute the hyperparameter sweep and emit per-configuration dashboards."""

    def __init__(self, repo_root: Path, schema_path: Path, tracked_packages: list[str]) -> None:
        self._repo_root = Path(repo_root)
        self._config_loader = ConfigLoader(schema_path)
        self._environment_probe = EnvironmentProbe(tracked_packages)
        self._git_probe = GitProbe(repo_root)
        self._run_id_factory = RunIdFactory()
        self._manifest_validator = ManifestValidator()
        self._dashboard_registry = DashboardRegistry.with_builtin_dashboards()
        self._renderer = DashboardRenderer.default()
        self._dataset_registry = DatasetRegistry.with_builtin_providers()
        self._model_registry = ModelProviderRegistry.with_builtin_providers()
        self._merge_registry = MergeRegistry.with_builtin_metrics()
        self._representative_registry = RepresentativeSelectorRegistry.with_builtin_selectors()

    def run(self, config_path: Path, entrypoint: str, command: str) -> Path:
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
                "provider": config["model"]["provider"],
                "model_id": config["model"]["model_id"],
                "model_revision": config["model"]["model_revision"],
                "dataset_provider": config["dataset"]["provider"],
                "dataset_id": config["dataset"]["dataset_id"],
                "dataset_revision": config["dataset"]["dataset_revision"],
                "task": "reasoning-graph-tuning",
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
        logger.info("tuning run %s started", run_id)

        try:
            outputs = self._execute(config, run_dir, telemetry, logger)
            duration = time.monotonic() - start
            manifest.mark_completed(outputs, duration)
            telemetry.emit(
                "run_end", _COMPONENT, "run",
                metrics={"duration_seconds": duration}, message="run completed",
            )
            telemetry.component("summary").update("duration_s", round(duration, 3))
        except Exception as exc:  # observed, recorded, re-raised (AGENTS.md section 22)
            duration = time.monotonic() - start
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest.mark_failed(error, duration)
            telemetry.emit("exception", _COMPONENT, "run", error=error, message="run failed")
            logger.exception("tuning run %s failed", run_id)
            self._finalize(manifest, manifest_path, run_dir, telemetry)
            raise
        else:
            self._finalize(manifest, manifest_path, run_dir, telemetry)

        return run_dir

    def _execute(self, config, run_dir, telemetry, logger) -> dict[str, Any]:
        seed = config["randomness"]["seeds"]["python"]
        traces_root = run_dir / "artifacts" / "traces"
        store = TraceStore(traces_root)

        provider = self._build_dataset(config["dataset"])
        model = self._build_model(config["model"])
        entries = provider.load(
            split=config["dataset"]["split"],
            num_questions=config["dataset"]["num_questions"],
            sample_seed=seed,
        )
        telemetry.component("summary").update("questions", len(entries))
        telemetry.component("summary").update("dataset", config["dataset"]["dataset_id"])
        telemetry.component("summary").update("model", config["model"]["model_id"])

        generation_params = self._generation_params(config, seed)
        question_ids: list[str] = []
        telemetry.emit("phase_start", _COMPONENT, "generation", metrics={"questions": len(entries)})
        for index, entry in enumerate(entries):
            question_ids.append(entry.question_id)
            self._generate_question(entry, index, config, model, provider, generation_params, store, telemetry)
        store.write_index(
            {"dataset_id": config["dataset"]["dataset_id"], "questions": question_ids, "generation": generation_params}
        )
        telemetry.emit("phase_end", _COMPONENT, "generation", metrics={"questions": len(entries)})

        return self._sweep(config, run_dir, traces_root, question_ids, telemetry, logger)

    def _sweep(self, config, run_dir, traces_root, question_ids, telemetry, logger) -> dict[str, Any]:
        tuning = config["tuning"]
        loader = TraceLoader()
        statistics = GraphStatistics()
        consolidator = self._build_consolidator(tuning)
        config_report = TuningConfigReport()
        tuning_root = run_dir / "artifacts" / "tuning"
        tuning_root.mkdir(parents=True, exist_ok=True)

        chains_by_question = {
            qid: loader.load_question(self._question_dir(traces_root, qid))[1] for qid in question_ids
        }

        telemetry.emit("phase_start", _COMPONENT, "tuning", metrics={
            "configs": len(tuning["heuristics"]) * len(tuning["thresholds"])
        })
        rows: list[dict[str, Any]] = []
        dashboards: list[str] = []
        config_index = 0
        for heuristic in tuning["heuristics"]:
            metric = self._merge_registry.create(heuristic)
            for threshold in tuning["thresholds"]:
                per_question_stats: list[dict[str, Any]] = []
                merges: list[dict[str, Any]] = []
                for qid in question_ids:
                    graph = consolidator.consolidate(
                        chains=chains_by_question[qid], metric=metric, threshold=threshold,
                        pooling_k=tuning["pooling_k"], context_window=tuning["context_window"],
                    )
                    stats = statistics.compute(graph)
                    if not stats["dag_valid"]:
                        raise ValueError(f"Non-DAG graph ({qid}, {heuristic}, {threshold})")
                    per_question_stats.append(stats)
                    for merge in graph.merges:
                        merges.append({**merge, "question_id": qid})

                aggregate = self._aggregate(per_question_stats)
                high, edge = self._samples(merges, threshold, tuning)
                label = f"{heuristic}_{threshold}"
                rel_dir = f"artifacts/tuning/{label}"
                (run_dir / rel_dir).mkdir(parents=True, exist_ok=True)
                html = config_report.render(
                    config_label=f"{heuristic} @ {threshold}",
                    stats=aggregate, high_samples=high, edge_samples=edge,
                )
                (run_dir / rel_dir / "dashboard.html").write_text(html, encoding="utf-8")
                dashboards.append(f"{rel_dir}/dashboard.html")

                rows.append(
                    {
                        "config_label": f"{heuristic} @ {threshold}",
                        "heuristic": heuristic,
                        "threshold": threshold,
                        "dashboard_path": f"{label}/dashboard.html",
                        **aggregate,
                    }
                )
                telemetry.component("values").update(config_index, aggregate["mean_node_reduction"])
                telemetry.component("log").update(
                    datetime.now(timezone.utc).isoformat(),
                    f"config {heuristic}@{threshold}",
                    f"merges={aggregate['total_merges']} reduction={aggregate['mean_node_reduction']:.3f}",
                )
                telemetry.emit(
                    "config_evaluated", _COMPONENT, "tuning", step=config_index,
                    metrics={
                        "total_merges": aggregate["total_merges"],
                        "mean_node_reduction": aggregate["mean_node_reduction"],
                        "join_nodes": aggregate["join_nodes"],
                    },
                    payload={"heuristic": heuristic, "threshold": threshold},
                )
                logger.info("config %s @ %s: %d merges", heuristic, threshold, aggregate["total_merges"])
                config_index += 1

        comparison_rel = "artifacts/tuning/comparison.html"
        available = self._available_vocabulary()
        (run_dir / comparison_rel).write_text(
            TuningComparisonReport().render(rows=rows, available=available), encoding="utf-8"
        )
        summary_rel = "artifacts/tuning/tuning_summary.json"
        (run_dir / summary_rel).write_text(
            json.dumps({"configs": rows, "available": available}, indent=2), encoding="utf-8"
        )
        telemetry.emit("phase_end", _COMPONENT, "tuning", metrics={"configs": len(rows)})
        telemetry.component("summary").update("configs_evaluated", len(rows))

        return {
            "dashboard": "dashboard.html",
            "telemetry": "telemetry.jsonl",
            "traces_index": "artifacts/traces/index.json",
            "comparison_dashboard": comparison_rel,
            "config_dashboards": dashboards,
            "tuning_summary": summary_rel,
            "configs_evaluated": len(rows),
        }

    def _generate_question(self, entry, index, config, model, provider, generation_params, store, telemetry) -> None:
        prompt = entry.build_prompt(config["model"]["system_instruction"])
        gen = config["generation"]
        traces = model.generate(
            prompt=prompt, num_chains=gen["num_chains"], max_new_tokens=gen["max_new_tokens"],
            temperature=gen["temperature"], top_p=gen["top_p"], batch_size=gen["batch_size"],
            seed=generation_params["seed"] + index,
        )
        correct = 0
        for trace in traces:
            trace.predicted = provider.parse_prediction(entry, trace.completion_text)
            trace.correct = provider.is_correct(entry, trace.predicted)
            if trace.correct:
                correct += 1
        stats = self._raw_stats(traces, correct)
        store.write_question(
            question_id=entry.question_id, entry=entry.to_dict(), prompt=prompt,
            generation_params=generation_params, hidden_layer=config["model"]["hidden_layer"],
            traces=traces, stats=stats,
        )
        telemetry.emit(
            "question_generated", _COMPONENT, "generation", step=index,
            metrics={"chains": len(traces), "correct": correct},
            payload={"question_id": entry.question_id},
        )

    def _build_dataset(self, dataset_config: dict[str, Any]):
        return self._dataset_registry.create(
            dataset_config["provider"],
            dataset_id=dataset_config["dataset_id"],
            dataset_revision=dataset_config["dataset_revision"],
            dataset_config=dataset_config["dataset_config"],
            answer_delimiter=dataset_config["answer_delimiter"],
        )

    def _build_model(self, model_config: dict[str, Any]):
        return self._model_registry.create(
            model_config["provider"],
            model_id=model_config["model_id"],
            model_revision=model_config["model_revision"],
            dtype=model_config["dtype"],
            device=model_config["device"],
            hidden_layer=model_config["hidden_layer"],
        )

    def _build_consolidator(self, tuning: dict[str, Any]) -> GraphConsolidator:
        candidate_filter = CandidateFilter(
            depth_policy=tuning["depth_policy"], max_depth_difference=tuning["max_depth_difference"]
        )
        selector = self._representative_registry.create(
            tuning["representative_policy"], window=tuning["confidence_window"]
        )
        return GraphConsolidator(candidate_filter=candidate_filter, representative_selector=selector)

    @staticmethod
    def _available_vocabulary() -> dict[str, list[str]]:
        return {
            "merge heuristics": [h.value for h in MergeHeuristic],
            "depth policies": [d.value for d in DepthPolicy],
            "representative policies": [p.value for p in RepresentativePolicy],
            "dataset providers": [d.value for d in DatasetProviderKind],
            "model providers": [m.value for m in ModelProviderKind],
        }

    @staticmethod
    def _aggregate(per_question_stats: list[dict[str, Any]]) -> dict[str, Any]:
        def merge_hist(key: str) -> dict[str, int]:
            combined: dict[str, int] = {}
            for stats in per_question_stats:
                for bucket, count in stats["histograms"][key].items():
                    combined[bucket] = combined.get(bucket, 0) + count
            return dict(sorted(combined.items(), key=lambda item: int(item[0])))

        def hist_mean(hist: dict[str, int]) -> float:
            total = sum(hist.values())
            if total == 0:
                return 0.0
            return sum(int(bucket) * count for bucket, count in hist.items()) / total

        reductions = [s["node_reduction"] for s in per_question_stats]
        in_hist = merge_hist("in_degree")
        out_hist = merge_hist("out_degree")
        cluster_hist = merge_hist("cluster_size")
        return {
            "questions": len(per_question_stats),
            "total_merges": sum(s["merge_events"] for s in per_question_stats),
            "mean_node_reduction": sum(reductions) / len(reductions) if reductions else 0.0,
            "join_nodes": sum(s["join_nodes"] for s in per_question_stats),
            "branch_nodes": sum(s["branch_nodes"] for s in per_question_stats),
            "mean_in_degree": hist_mean(in_hist),
            "mean_out_degree": hist_mean(out_hist),
            "max_in_degree": max((s["max_in_degree"] for s in per_question_stats), default=0),
            "largest_cluster": max((s["largest_cluster"] for s in per_question_stats), default=0),
            "dag_valid": all(s["dag_valid"] for s in per_question_stats),
            "histograms": {"in_degree": in_hist, "out_degree": out_hist, "cluster_size": cluster_hist},
        }

    @staticmethod
    def _samples(merges, threshold, tuning):
        ordered = sorted(merges, key=lambda m: m["similarity"], reverse=True)
        high = ordered[: tuning["high_sample_count"]]
        margin = tuning["edge_margin"]
        edge_pool = [m for m in merges if threshold <= m["similarity"] <= threshold + margin]
        edge = sorted(edge_pool, key=lambda m: m["similarity"])[: tuning["edge_sample_count"]]
        return high, edge

    @staticmethod
    def _generation_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
        gen = config["generation"]
        model = config["model"]
        return {
            "num_chains": gen["num_chains"],
            "max_new_tokens": gen["max_new_tokens"],
            "temperature": gen["temperature"],
            "top_p": gen["top_p"],
            "batch_size": gen["batch_size"],
            "seed": seed,
            "model_id": model["model_id"],
            "model_revision": model["model_revision"],
            "dtype": model["dtype"],
            "device": model["device"],
            "hidden_layer": model["hidden_layer"],
        }

    @staticmethod
    def _raw_stats(traces, correct: int) -> dict[str, Any]:
        lengths = [len(trace.tokens) for trace in traces]
        total_tokens = sum(lengths)
        return {
            "num_chains": len(traces),
            "total_tokens": total_tokens,
            "min_chain_length": min(lengths) if lengths else 0,
            "max_chain_length": max(lengths) if lengths else 0,
            "mean_chain_length": (total_tokens / len(traces)) if traces else 0.0,
            "correct_chains": correct,
            "incorrect_chains": len(traces) - correct,
            "invalid_format_chains": sum(1 for t in traces if t.predicted is None),
        }

    @staticmethod
    def _question_dir(traces_root: Path, question_id: str) -> Path:
        return traces_root / f"question_{question_id.replace('/', '-')}"

    def _finalize(self, manifest, manifest_path, run_dir, telemetry) -> None:
        telemetry.close()
        manifest.save(manifest_path)
        self._manifest_validator.validate(manifest.to_dict())
        self._renderer.render_run(run_dir)

    def _configure_logger(self, run_id: str, log_path: Path) -> logging.Logger:
        logger = logging.getLogger(f"got.tune.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger
