"""Graph orchestrator: compose the isolated libraries into a reproducible run.

This is the only place the dataset, generation, reasoning-graph, and graph-report
libraries are composed (AGENTS.md section 3.2). It follows the manifest lifecycle
of section 10.2: configuration is validated and a manifest + telemetry stream are
created before any substantive work; the manifest is finalized, validated, and a
dashboard rendered afterwards.

Pipeline stages (research plan section 31 milestone):
  1. load an interchangeable dataset provider and model provider (both by class
     name from configuration);
  2. sample token-level traces per question and persist them as intermediate
     artifacts (the documented hand-off for later SBC labeling / detector stages);
  3. consolidate each question's traces into a DAG under a sweep of merge
     heuristics x thresholds (offline, without regenerating traces);
  4. render a standalone HTML graph report for a sampled question.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.config import ConfigLoader
from libs.dashboard import (
    DashboardRegistry,
    DashboardRenderer,
    LiveDashboardWriter,
    TerminalDashboardWriter,
)
from libs.dataset import DatasetRegistry
from libs.generation import ModelProviderRegistry, TraceStore
from libs.graph_report import GraphHtmlReport
from libs.manifest import ManifestValidator, RunManifest
from libs.reasoning_graph import (
    CandidateFilter,
    GraphConsolidator,
    GraphStatistics,
    MergeRegistry,
    RepresentativeSelector,
    TraceLoader,
)
from libs.runcontext import EnvironmentProbe, GitProbe, RunIdFactory
from libs.telemetry import TelemetryWriter

_COMPONENT = "graph_orchestrator"


class GraphOrchestrator:
    """Execute a configured graph-generation run end to end."""

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
                "provider_class": config["model"]["provider_class"],
                "model_id": config["model"]["model_id"],
                "model_revision": config["model"]["model_revision"],
                "dataset_provider_class": config["dataset"]["provider_class"],
                "dataset_id": config["dataset"]["dataset_id"],
                "dataset_revision": config["dataset"]["dataset_revision"],
                "task": "reasoning-graph-generation",
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
        logger.info("run %s started", run_id)

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
            logger.exception("run %s failed", run_id)
            self._finalize(manifest, manifest_path, run_dir, telemetry)
            raise
        else:
            self._finalize(manifest, manifest_path, run_dir, telemetry)

        return run_dir

    def _execute(
        self,
        config: dict[str, Any],
        run_dir: Path,
        telemetry: TelemetryWriter,
        logger: logging.Logger,
    ) -> dict[str, Any]:
        seed = config["randomness"]["seeds"]["python"]
        traces_root = run_dir / "artifacts" / "traces"
        store = TraceStore(traces_root)

        provider = self._build_dataset(config["dataset"])
        model = self._build_model(config["model"])
        telemetry.component("summary").update("dataset", config["dataset"]["dataset_id"])
        telemetry.component("summary").update("model", config["model"]["model_id"])

        entries = provider.load(
            split=config["dataset"]["split"],
            num_questions=config["dataset"]["num_questions"],
        )
        telemetry.component("summary").update("questions", len(entries))
        telemetry.emit("phase_start", _COMPONENT, "generation", metrics={"questions": len(entries)})

        generation_params = self._generation_params(config, seed)
        question_ids: list[str] = []
        for index, entry in enumerate(entries):
            question_ids.append(entry.question_id)
            self._generate_question(
                entry=entry,
                index=index,
                config=config,
                model=model,
                provider=provider,
                generation_params=generation_params,
                store=store,
                telemetry=telemetry,
                logger=logger,
            )
        store.write_index(
            {
                "dataset_id": config["dataset"]["dataset_id"],
                "questions": question_ids,
                "generation": generation_params,
            }
        )
        telemetry.emit("phase_end", _COMPONENT, "generation", metrics={"questions": len(entries)})

        graph_summary = self._build_graphs(config, run_dir, traces_root, question_ids, telemetry)
        report_paths = self._render_reports(config, run_dir, traces_root, question_ids, seed, telemetry)

        return {
            "dashboard": "dashboard.html",
            "telemetry": "telemetry.jsonl",
            "traces_index": "artifacts/traces/index.json",
            "graphs_dir": "artifacts/graphs",
            "reports": report_paths,
            "questions": len(entries),
            "graphs_built": graph_summary["graphs_built"],
        }

    def _generate_question(
        self,
        *,
        entry,
        index: int,
        config: dict[str, Any],
        model,
        provider,
        generation_params: dict[str, Any],
        store: TraceStore,
        telemetry: TelemetryWriter,
        logger: logging.Logger,
    ) -> None:
        prompt = entry.build_prompt(config["model"]["system_instruction"])
        gen = config["generation"]
        traces = model.generate(
            prompt=prompt,
            num_chains=gen["num_chains"],
            max_new_tokens=gen["max_new_tokens"],
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            batch_size=gen["batch_size"],
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
            question_id=entry.question_id,
            entry=entry.to_dict(),
            prompt=prompt,
            generation_params=generation_params,
            hidden_layer=config["model"]["hidden_layer"],
            traces=traces,
            stats=stats,
        )
        accuracy = correct / len(traces) if traces else 0.0
        telemetry.component("values").update(index, accuracy)
        telemetry.component("log").update(
            datetime.now(timezone.utc).isoformat(),
            f"generated {entry.question_id}",
            f"chains={len(traces)} correct={correct} tokens={stats['total_tokens']}",
        )
        telemetry.emit(
            "question_generated", _COMPONENT, "generation",
            step=index, metrics={"chains": len(traces), "correct": correct, "accuracy": accuracy},
            payload={"question_id": entry.question_id},
        )
        logger.info("question %s: %d chains, %d correct", entry.question_id, len(traces), correct)

    def _build_graphs(
        self,
        config: dict[str, Any],
        run_dir: Path,
        traces_root: Path,
        question_ids: list[str],
        telemetry: TelemetryWriter,
    ) -> dict[str, Any]:
        loader = TraceLoader()
        statistics = GraphStatistics()
        consolidator = self._build_consolidator(config["graph"])
        graphs_root = run_dir / "artifacts" / "graphs"
        graphs_root.mkdir(parents=True, exist_ok=True)

        telemetry.emit("phase_start", _COMPONENT, "consolidation")
        built = 0
        for question_id in question_ids:
            _meta, chains = loader.load_question(self._question_dir(traces_root, question_id))
            question_out = graphs_root / f"question_{question_id.replace('/', '-')}"
            question_out.mkdir(parents=True, exist_ok=True)
            for heuristic in config["graph"]["heuristics"]:
                metric = self._merge_registry.create(heuristic)
                if metric.requires_hidden() and not self._has_hidden(chains):
                    telemetry.emit(
                        "heuristic_skipped", _COMPONENT, "consolidation",
                        message=f"{heuristic} needs hidden states; none present",
                        payload={"question_id": question_id, "heuristic": heuristic},
                    )
                    continue
                for threshold in config["graph"]["thresholds"]:
                    graph = consolidator.consolidate(
                        chains=chains,
                        metric=metric,
                        threshold=threshold,
                        pooling_k=config["graph"]["pooling_k"],
                    )
                    stats = statistics.compute(graph)
                    if not stats["dag_valid"]:
                        raise ValueError(
                            f"Consolidated graph is not a DAG ({question_id}, {heuristic}, {threshold})"
                        )
                    name = f"graph_{heuristic}_{threshold}"
                    (question_out / f"{name}.json").write_text(
                        json.dumps({"graph": graph.to_dict(), "stats": stats}, indent=2),
                        encoding="utf-8",
                    )
                    built += 1
                    telemetry.emit(
                        "graph_built", _COMPONENT, "consolidation",
                        metrics={
                            "node_reduction": stats["node_reduction"],
                            "merges": stats["merge_events"],
                            "join_nodes": stats["join_nodes"],
                        },
                        payload={
                            "question_id": question_id,
                            "heuristic": heuristic,
                            "threshold": threshold,
                        },
                    )
        telemetry.emit("phase_end", _COMPONENT, "consolidation", metrics={"graphs_built": built})
        telemetry.component("summary").update("graphs_built", built)
        return {"graphs_built": built}

    def _render_reports(
        self,
        config: dict[str, Any],
        run_dir: Path,
        traces_root: Path,
        question_ids: list[str],
        seed: int,
        telemetry: TelemetryWriter,
    ) -> list[str]:
        if not question_ids:
            return []
        loader = TraceLoader()
        statistics = GraphStatistics()
        consolidator = self._build_consolidator(config["graph"])
        report = GraphHtmlReport()
        reports_root = run_dir / "artifacts" / "reports"
        reports_root.mkdir(parents=True, exist_ok=True)

        heuristic = config["report"]["heuristic"]
        threshold = config["report"]["threshold"]
        metric = self._merge_registry.create(heuristic)
        rng = random.Random(seed)
        sample_count = min(config["report"]["sample_questions"], len(question_ids))
        sampled = rng.sample(question_ids, sample_count)

        telemetry.emit("phase_start", _COMPONENT, "reporting", metrics={"sampled": sample_count})
        paths: list[str] = []
        for question_id in sampled:
            meta, chains = loader.load_question(self._question_dir(traces_root, question_id))
            graph = consolidator.consolidate(
                chains=chains, metric=metric, threshold=threshold,
                pooling_k=config["graph"]["pooling_k"],
            )
            stats = statistics.compute(graph)
            html = report.render(
                question=self._question_view(meta),
                lanes=self._lanes(graph, chains),
                graph=graph.to_dict(),
                stats=stats,
            )
            rel = f"artifacts/reports/graph_report_{question_id.replace('/', '-')}.html"
            (run_dir / rel).write_text(html, encoding="utf-8")
            paths.append(rel)
            telemetry.emit(
                "report_rendered", _COMPONENT, "reporting",
                payload={"question_id": question_id, "path": rel},
            )
        telemetry.emit("phase_end", _COMPONENT, "reporting", metrics={"reports": len(paths)})
        return paths

    def _build_dataset(self, dataset_config: dict[str, Any]):
        return self._dataset_registry.create(
            dataset_config["provider_class"],
            dataset_id=dataset_config["dataset_id"],
            dataset_revision=dataset_config["dataset_revision"],
            dataset_config=dataset_config["dataset_config"],
            answer_delimiter=dataset_config["answer_delimiter"],
        )

    def _build_model(self, model_config: dict[str, Any]):
        return self._model_registry.create(
            model_config["provider_class"],
            model_id=model_config["model_id"],
            model_revision=model_config["model_revision"],
            dtype=model_config["dtype"],
            device=model_config["device"],
            hidden_layer=model_config["hidden_layer"],
        )

    def _build_consolidator(self, graph_config: dict[str, Any]) -> GraphConsolidator:
        candidate_filter = CandidateFilter(
            depth_policy=graph_config["depth_policy"],
            max_depth_difference=graph_config["max_depth_difference"],
        )
        selector = RepresentativeSelector(window=graph_config["confidence_window"])
        return GraphConsolidator(
            candidate_filter=candidate_filter, representative_selector=selector
        )

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
        invalid = sum(1 for trace in traces if trace.predicted is None)
        answers: dict[str, int] = {}
        for trace in traces:
            key = str(trace.predicted)
            answers[key] = answers.get(key, 0) + 1
        total_tokens = sum(lengths)
        return {
            "num_chains": len(traces),
            "total_tokens": total_tokens,
            "min_chain_length": min(lengths) if lengths else 0,
            "max_chain_length": max(lengths) if lengths else 0,
            "mean_chain_length": (total_tokens / len(traces)) if traces else 0.0,
            "correct_chains": correct,
            "incorrect_chains": len(traces) - correct,
            "invalid_format_chains": invalid,
            "answer_distribution": answers,
        }

    @staticmethod
    def _has_hidden(chains) -> bool:
        for chain in chains:
            for token in chain.tokens:
                if token.hidden is not None:
                    return True
        return False

    @staticmethod
    def _question_view(meta: dict[str, Any]) -> dict[str, Any]:
        entry = meta.get("entry", {})
        return {
            "question_id": entry.get("question_id", "?"),
            "question": entry.get("question", ""),
            "gold_answer": entry.get("gold_answer", "?"),
            "dataset": entry.get("dataset", "?"),
        }

    @staticmethod
    def _lanes(graph, chains) -> list[dict[str, Any]]:
        member_cluster: dict[tuple[int, int], int] = {}
        join_clusters: set[int] = set()
        for node in graph.nodes.values():
            chains_in_node = {member[0] for member in node.members}
            if len(chains_in_node) > 1:
                join_clusters.add(node.cluster_id)
            for member in node.members:
                member_cluster[(member[0], member[1])] = node.cluster_id
        lanes: list[dict[str, Any]] = []
        for chain in sorted(chains, key=lambda c: c.chain_id):
            last_index = len(chain.tokens) - 1
            tokens = []
            for token in chain.tokens:
                cluster_id = member_cluster.get((chain.chain_id, token.token_index), -1)
                tokens.append(
                    {
                        "index": token.token_index,
                        "text": token.text,
                        "cluster_id": cluster_id,
                        "join": cluster_id in join_clusters,
                        "terminal": token.token_index == last_index,
                    }
                )
            lanes.append(
                {
                    "chain_id": chain.chain_id,
                    "correct": bool(chain.correct),
                    "predicted": chain.predicted,
                    "tokens": tokens,
                }
            )
        return lanes

    @staticmethod
    def _question_dir(traces_root: Path, question_id: str) -> Path:
        return traces_root / f"question_{question_id.replace('/', '-')}"

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
        logger = logging.getLogger(f"got.graph.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger
