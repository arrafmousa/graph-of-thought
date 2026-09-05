"""Multi-dataset semantic evaluation of reasoning-state graph merges."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from libs.config import ConfigLoader
from libs.dashboard import DashboardRegistry, DashboardRenderer, LiveDashboardWriter
from libs.dashboard import TerminalDashboardWriter
from libs.dataset import DatasetProviderKind, DatasetRegistry
from libs.generation import ModelProviderRegistry, TraceStore
from libs.graph_report import GraphHtmlReport, SemanticEvaluationReport
from libs.manifest import ManifestValidator, RunManifest
from libs.merge_quality import GraphMergeQualityAnalyzer
from libs.reasoning_graph import CandidateFilter, GraphConsolidator, GraphStatistics
from libs.reasoning_graph import MergeRegistry, RepresentativeSelectorRegistry, TraceLoader
from libs.runcontext import EnvironmentProbe, GitProbe, RunIdFactory
from libs.semantic_judge import JudgeProviderKind, JudgeProviderRegistry, SemanticJudge
from libs.telemetry import TelemetryWriter

_COMPONENT = "semantic_evaluation_orchestrator"


class SemanticEvaluationOrchestrator:
    """Generate, consolidate, judge, and report a complete merge experiment."""

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
        self._judge_registry = JudgeProviderRegistry.with_builtin_providers()
        self._quality_analyzer = GraphMergeQualityAnalyzer()

    def run(self, config_path: Path, entrypoint: str, command: str) -> Path:
        config = self._config_loader.load(config_path)
        self._validate_scope(config)
        run_id = self._run_id_factory.create(config["run"]["task_name"])
        run_dir = self._repo_root / "output" / run_id
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        logger = self._configure_logger(run_id, run_dir / "logs" / "run.log")
        template_name = config["run"]["dashboard_template"]
        dashboard = self._dashboard_registry.create(template_name)
        dashboard.set_run_id(run_id)
        LiveDashboardWriter(
            dashboard,
            run_dir / "dashboard.html",
            config["run"]["dashboard_refresh_seconds"],
        ).write()
        if config["run"]["terminal_progress"]:
            TerminalDashboardWriter(dashboard, sys.stdout)
        telemetry = TelemetryWriter(run_dir / "telemetry.jsonl", run_id, dashboard=dashboard)
        manifest = self._create_manifest(
            config, run_id, config_path, entrypoint, command, telemetry.schema_version
        )
        manifest_path = run_dir / "run_manifest.json"
        manifest.save(manifest_path)

        started = time.monotonic()
        manifest.mark_running()
        manifest.save(manifest_path)
        telemetry.emit("run_start", _COMPONENT, "run", message=f"run {run_id} started")
        logger.info("semantic evaluation run %s started", run_id)
        try:
            outputs = self._execute(config, run_dir, telemetry, logger)
            duration = time.monotonic() - started
            manifest.mark_completed(outputs, duration)
            telemetry.emit(
                "run_end",
                _COMPONENT,
                "run",
                metrics={"duration_seconds": duration},
                message="run completed",
            )
            telemetry.component("summary").update("duration_s", round(duration, 3))
        except Exception as exc:
            duration = time.monotonic() - started
            error = {"type": type(exc).__name__, "message": str(exc)}
            manifest.mark_failed(error, duration)
            telemetry.emit("exception", _COMPONENT, "run", error=error, message="run failed")
            logger.exception("semantic evaluation run %s failed", run_id)
            self._finalize(manifest, manifest_path, run_dir, telemetry)
            raise
        else:
            self._finalize(manifest, manifest_path, run_dir, telemetry)
        return run_dir

    def _execute(self, config, run_dir, telemetry, logger) -> dict[str, Any]:
        questions, providers = self._generate_traces(config, run_dir, telemetry, logger)
        graphs, occurrences, requests, skipped_first_token_merges = self._build_graphs(
            config, run_dir, questions, providers, telemetry, logger
        )
        evaluation_root = run_dir / "artifacts" / "evaluation"
        evaluation_root.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(
            evaluation_root / "pair_requests.jsonl", list(requests.values())
        )
        judgment_rows = self._judge_pairs(
            config, run_dir, list(requests.values()), telemetry
        )
        judgments = {row["pair_id"]: row for row in judgment_rows}
        quality_config = config["quality"]
        analysis = self._quality_analyzer.analyze(
            occurrences=occurrences,
            judgments=judgments,
            graphs=graphs,
            max_judge_score=config["judge"]["max_score"],
            semantic_weight=quality_config["semantic_weight"],
            continuation_weight=quality_config["continuation_weight"],
            quality_beta=quality_config["quality_beta"],
            missing_answer_agreement_score=quality_config[
                "missing_answer_agreement_score"
            ],
        )
        pair_occurrences = analysis.pop("pair_occurrences")
        analysis["experiment"] = {
            "datasets": len(config["datasets"]),
            "questions": len(questions),
            "unique_pairs": len(requests),
            "merge_occurrences": len(occurrences),
            "skipped_first_token_merges": skipped_first_token_merges,
            "min_join_token_index": config["tuning"]["min_join_token_index"],
            "graphs": len(graphs),
        }
        graph_reports = self._render_graph_reports(
            config, run_dir, analysis["graphs"], telemetry
        )
        self._write_jsonl(evaluation_root / "pair_judgments.jsonl", judgment_rows)
        self._write_jsonl(
            evaluation_root / "merge_occurrences.jsonl", pair_occurrences
        )
        self._write_jsonl(evaluation_root / "graph_quality.jsonl", analysis["graphs"])
        (evaluation_root / "semantic_summary.json").write_text(
            json.dumps(analysis, indent=2), encoding="utf-8"
        )
        review = sorted(
            pair_occurrences,
            key=lambda row: (
                row["equivalence_score"],
                row["same_final_answer"] is not False,
                row["similarity"],
            ),
        )[: config["report"]["manual_review_pairs"]]
        report_relative = Path("artifacts") / "reports" / "semantic_evaluation.html"
        report_path = run_dir / report_relative
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            SemanticEvaluationReport().render(summary=analysis, review_pairs=review),
            encoding="utf-8",
        )
        telemetry.component("summary").update("datasets", len(config["datasets"]))
        telemetry.component("summary").update("questions", len(questions))
        telemetry.component("summary").update("graphs", len(graphs))
        telemetry.component("summary").update("unique_pairs", len(requests))
        return self._outputs(
            config, questions, graphs, requests, occurrences, graph_reports, report_relative
        )

    def _generate_traces(self, config, run_dir, telemetry, logger):
        traces_root = run_dir / "artifacts" / "traces"
        traces_root.mkdir(parents=True, exist_ok=True)
        model = self._build_model(config["model"])
        questions: list[dict[str, Any]] = []
        providers = {}
        generation_seed = config["randomness"]["seeds"]["generation"]
        global_index = 0
        telemetry.emit("phase_start", _COMPONENT, "generation")
        try:
            for dataset_config in config["datasets"]:
                provider = self._build_dataset(dataset_config)
                providers[dataset_config["name"]] = provider
                entries = provider.load(
                    split=dataset_config["split"],
                    num_questions=dataset_config["num_questions"],
                    sample_seed=dataset_config["sample_seed"],
                )
                dataset_root = traces_root / dataset_config["name"]
                store = TraceStore(dataset_root)
                selected: list[dict[str, Any]] = []
                for entry in entries:
                    entry.dataset = dataset_config["name"]
                    question_seed = generation_seed + global_index
                    self._generate_question(
                        entry,
                        provider,
                        model,
                        config,
                        question_seed,
                        store,
                        telemetry,
                        logger,
                    )
                    questions.append(
                        {
                            "dataset": dataset_config["name"],
                            "question_id": entry.question_id,
                            "trace_dir": store.question_dir(entry.question_id)
                            .relative_to(run_dir)
                            .as_posix(),
                            "entry": entry.to_dict(),
                        }
                    )
                    selected.append(entry.to_dict())
                    global_index += 1
                store.write_index(
                    {
                        "dataset": dataset_config["name"],
                        "dataset_id": dataset_config["dataset_id"],
                        "dataset_revision": dataset_config["dataset_revision"],
                        "sample_seed": dataset_config["sample_seed"],
                        "questions": selected,
                    }
                )
                telemetry.emit(
                    "dataset_generated",
                    _COMPONENT,
                    "generation",
                    metrics={"questions": len(entries)},
                    payload={"dataset": dataset_config["name"]},
                )
        finally:
            model.release()
            gc.collect()
        (traces_root / "index.json").write_text(
            json.dumps(
                {
                    "datasets": [
                        {
                            "name": item["name"],
                            "dataset_id": item["dataset_id"],
                            "dataset_revision": item["dataset_revision"],
                            "sample_seed": item["sample_seed"],
                            "num_questions": item["num_questions"],
                        }
                        for item in config["datasets"]
                    ],
                    "questions": questions,
                    "generation": self._generation_params(config),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        telemetry.emit(
            "phase_end", _COMPONENT, "generation", metrics={"questions": len(questions)}
        )
        return questions, providers

    def _generate_question(
        self, entry, provider, model, config, seed, store, telemetry, logger
    ) -> None:
        prompt = entry.build_prompt(config["model"]["system_instruction"])
        generation = config["generation"]
        traces = model.generate(
            prompt=prompt,
            num_chains=generation["num_chains"],
            max_new_tokens=generation["max_new_tokens"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            batch_size=generation["batch_size"],
            seed=seed,
        )
        correct = 0
        for trace in traces:
            evaluation = provider.evaluate_completion(entry, trace.completion_text)
            trace.predicted = evaluation["predicted"]
            trace.correct = evaluation["correct"]
            trace.answer_evaluation = evaluation
            correct += int(trace.correct)
        store.write_question(
            question_id=entry.question_id,
            entry=entry.to_dict(),
            prompt=prompt,
            generation_params={**self._generation_params(config), "question_seed": seed},
            hidden_layer=config["model"]["hidden_layer"],
            traces=traces,
            stats=self._raw_stats(traces, correct),
        )
        telemetry.emit(
            "question_generated",
            _COMPONENT,
            "generation",
            metrics={"chains": len(traces), "correct": correct},
            payload={"dataset": entry.dataset, "question_id": entry.question_id},
        )
        logger.info("generated %s with %d chains", entry.question_id, len(traces))

    def _build_graphs(self, config, run_dir, questions, providers, telemetry, logger):
        graph_root = run_dir / "artifacts" / "graphs"
        graph_root.mkdir(parents=True, exist_ok=True)
        consolidator = self._build_consolidator(config["tuning"])
        statistics = GraphStatistics()
        metrics = {
            name: self._merge_registry.create(name)
            for name in config["tuning"]["heuristics"]
        }
        occurrences: list[dict[str, Any]] = []
        requests: dict[str, dict[str, Any]] = {}
        graphs: list[dict[str, Any]] = []
        min_join = config["tuning"]["min_join_token_index"]
        skipped_first_token_merges = 0
        telemetry.emit("phase_start", _COMPONENT, "consolidation")
        for question_index, item in enumerate(questions):
            meta, chains = TraceLoader().load_question(run_dir / item["trace_dir"])
            chain_map = {chain.chain_id: chain for chain in chains}
            for heuristic, metric in metrics.items():
                if metric.requires_hidden() and not self._has_hidden(chains):
                    raise ValueError(f"Heuristic '{heuristic}' requires hidden states")
                for threshold in config["tuning"]["thresholds"]:
                    graph = consolidator.consolidate(
                        chains=chains,
                        metric=metric,
                        threshold=threshold,
                        pooling_k=config["tuning"]["pooling_k"],
                        context_window=config["tuning"]["context_window"],
                    )
                    stats = statistics.compute(graph)
                    if not stats["dag_valid"]:
                        raise ValueError(
                            f"Consolidated graph is not a DAG: {item['question_id']}"
                        )
                    graph_id = self._graph_id(item, heuristic, threshold)
                    relative = (
                        Path("artifacts")
                        / "graphs"
                        / item["dataset"]
                        / self._safe(item["question_id"])
                        / f"graph_{heuristic}_{threshold}.json"
                    )
                    output_path = run_dir / relative
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(
                            {
                                "question": meta["entry"],
                                "graph": graph.to_dict(),
                                "stats": stats,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    graphs.append(
                        {
                            "graph_id": graph_id,
                            "dataset": item["dataset"],
                            "question_id": item["question_id"],
                            "heuristic": heuristic,
                            "threshold": threshold,
                            "graph_path": relative.as_posix(),
                            "trace_dir": item["trace_dir"],
                            "stats": stats,
                        }
                    )
                    for merge in graph.merges:
                        if (
                            merge["node_a"][1] < min_join
                            or merge["node_b"][1] < min_join
                        ):
                            skipped_first_token_merges += 1
                            continue
                        occurrence, request = self._merge_records(
                            item,
                            meta["entry"],
                            graph_id,
                            heuristic,
                            threshold,
                            merge,
                            chain_map,
                            providers[item["dataset"]],
                        )
                        occurrences.append(occurrence)
                        requests.setdefault(request["pair_id"], request)
            telemetry.emit(
                "question_graphs_built",
                _COMPONENT,
                "consolidation",
                step=question_index,
                metrics={
                    "graphs": len(metrics) * len(config["tuning"]["thresholds"]),
                },
                payload={
                    "dataset": item["dataset"],
                    "question_id": item["question_id"],
                },
            )
            logger.info("built graph sweep for %s", item["question_id"])
        (graph_root / "index.json").write_text(
            json.dumps({"graphs": graphs}, indent=2), encoding="utf-8"
        )
        telemetry.emit(
            "phase_end",
            _COMPONENT,
            "consolidation",
            metrics={
                "graphs": len(graphs),
                "merge_occurrences": len(occurrences),
                "skipped_first_token_merges": skipped_first_token_merges,
            },
        )
        return graphs, occurrences, requests, skipped_first_token_merges

    def _merge_records(
        self, item, entry, graph_id, heuristic, threshold, merge, chain_map, provider
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        node_a, node_b = sorted((tuple(merge["node_a"]), tuple(merge["node_b"])))
        chain_a = chain_map[node_a[0]]
        chain_b = chain_map[node_b[0]]
        pair_id = hashlib.sha256(
            json.dumps(
                [item["dataset"], item["question_id"], list(node_a), list(node_b)],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        request = {
            "pair_id": pair_id,
            "dataset": item["dataset"],
            "question_id": item["question_id"],
            "question": entry["question"],
            "node_a": list(node_a),
            "node_b": list(node_b),
            "path_a": self._token_text(chain_a.tokens[: node_a[1] + 1]),
            "path_b": self._token_text(chain_b.tokens[: node_b[1] + 1]),
        }
        occurrence = {
            **request,
            "graph_id": graph_id,
                "merge_accepted": True,
            "heuristic": heuristic,
            "threshold": threshold,
            "similarity": merge["similarity"],
            "cluster_id": merge["cluster_id"],
            "winner_chain": merge["winner_chain"],
            "loser_chain": merge["loser_chain"],
            "predicted_a": chain_a.predicted,
            "predicted_b": chain_b.predicted,
            "correct_a": chain_a.correct,
            "correct_b": chain_b.correct,
                "answer_a_format_valid": chain_a.answer_evaluation.get(
                    "format_valid", False
                ),
                "answer_b_format_valid": chain_b.answer_evaluation.get(
                    "format_valid", False
                ),
                "answer_a_parse_valid": chain_a.answer_evaluation.get(
                    "parse_valid", False
                ),
                "answer_b_parse_valid": chain_b.answer_evaluation.get(
                    "parse_valid", False
                ),
                "same_final_answer": provider.answers_equivalent(
                    chain_a.predicted, chain_b.predicted
                ),
            "continuation_a": self._token_text(chain_a.tokens[node_a[1] + 1 :]),
            "continuation_b": self._token_text(chain_b.tokens[node_b[1] + 1 :]),
        }
        return occurrence, request

    def _judge_pairs(self, config, run_dir, requests, telemetry):
        judge = self._build_judge(config["judge"], run_dir)
        telemetry.emit(
            "phase_start",
            _COMPONENT,
            "judging",
            metrics={"unique_pairs": len(requests)},
        )
        try:
            results = judge.judge_pairs(requests=requests)
        finally:
            judge.release()
            gc.collect()
        telemetry.emit(
            "phase_end", _COMPONENT, "judging", metrics={"unique_pairs": len(results)}
        )
        return results

    def _render_graph_reports(self, config, run_dir, graph_rows, telemetry):
        candidates = [row for row in graph_rows if row["accepted_merges"] > 0]
        selected = self._select_graph_reports(candidates, config["report"]["graph_reports"])
        report = GraphHtmlReport()
        paths: list[str] = []
        for row in selected:
            payload = json.loads((run_dir / row["graph_path"]).read_text(encoding="utf-8"))
            _meta, chains = TraceLoader().load_question(run_dir / row["trace_dir"])
            relative = (
                Path("artifacts") / "reports" / "graphs" / f"{row['graph_id']}.html"
            )
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                report.render(
                    question=payload["question"],
                    lanes=self._lanes(payload["graph"], chains),
                    graph=payload["graph"],
                    stats=payload["stats"],
                ),
                encoding="utf-8",
            )
            row["report_path"] = relative.as_posix()
            paths.append(relative.as_posix())
        telemetry.emit(
            "reports_rendered", _COMPONENT, "reporting", metrics={"reports": len(paths)}
        )
        return paths

    def _build_dataset(self, dataset_config):
        if dataset_config["provider"] == DatasetProviderKind.SYNTHETIC.value:
            return self._dataset_registry.create(
                dataset_config["provider"],
                dataset_id=dataset_config["dataset_id"],
                dataset_revision=dataset_config["dataset_revision"],
                dataset_config=dataset_config["dataset_configs"][0],
                answer_delimiter=dataset_config["answer_delimiter"],
            )
        return self._dataset_registry.create(
            dataset_config["provider"],
            dataset_name=dataset_config["name"],
            dataset_id=dataset_config["dataset_id"],
            dataset_revision=dataset_config["dataset_revision"],
            dataset_configs=dataset_config["dataset_configs"],
            question_fields=dataset_config["question_fields"],
            answer_field=dataset_config["answer_field"],
            id_field=dataset_config["id_field"],
            metadata_fields=dataset_config["metadata_fields"],
            answer_mode=dataset_config["answer_mode"],
            answer_delimiter=dataset_config["answer_delimiter"],
            require_answer_delimiter=dataset_config["require_answer_delimiter"],
            math_parser_timeout_seconds=dataset_config[
                "math_parser_timeout_seconds"
            ],
        )

    def _build_model(self, model_config):
        return self._model_registry.create(
            model_config["provider"],
            model_id=model_config["model_id"],
            model_revision=model_config["model_revision"],
            dtype=model_config["dtype"],
            device=model_config["device"],
            hidden_layer=model_config["hidden_layer"],
        )

    def _build_judge(self, judge_config, run_dir):
        if judge_config["provider"] == JudgeProviderKind.SYNTHETIC.value:
            provider = self._judge_registry.create(
                judge_config["provider"], model_name=judge_config["deployment"]
            )
        else:
            provider = self._judge_registry.create(
                judge_config["provider"],
                endpoint=judge_config["endpoint"],
                api_path=judge_config["api_path"],
                deployment=judge_config["deployment"],
                api_key_environment_variable=judge_config[
                    "api_key_environment_variable"
                ],
                request_url=judge_config["request_url"],
                batch_endpoint=judge_config["batch_endpoint"],
                completion_window=judge_config["completion_window"],
                system_instruction=judge_config["system_instruction"],
                max_completion_tokens=judge_config["max_completion_tokens"],
                max_score=judge_config["max_score"],
                poll_interval_seconds=judge_config["poll_interval_seconds"],
                max_wait_seconds=judge_config["max_wait_seconds"],
                submission_attempts=judge_config["submission_attempts"],
                submission_retry_seconds=judge_config["submission_retry_seconds"],
                artifact_dir=run_dir / "artifacts" / "evaluation" / "azure_batch",
            )
        return SemanticJudge(
            provider=provider,
            pair_prompt_template=judge_config["pair_prompt_template"],
            max_score=judge_config["max_score"],
            parse_attempts=judge_config["parse_attempts"],
        )

    def _build_consolidator(self, tuning):
        candidate_filter = CandidateFilter(
            depth_policy=tuning["depth_policy"],
            max_depth_difference=tuning["max_depth_difference"],
        )
        selector = self._representative_registry.create(
            tuning["representative_policy"], window=tuning["confidence_window"]
        )
        return GraphConsolidator(
            candidate_filter=candidate_filter, representative_selector=selector
        )

    def _create_manifest(
        self, config, run_id, config_path, entrypoint, command, telemetry_version
    ):
        return RunManifest.create(
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
                "generator": {
                    "provider": config["model"]["provider"],
                    "model_id": config["model"]["model_id"],
                    "model_revision": config["model"]["model_revision"],
                },
                "judge": {
                    "provider": config["judge"]["provider"],
                    "deployment": config["judge"]["deployment"],
                    "api_path": config["judge"]["api_path"],
                },
                "datasets": [item["dataset_id"] for item in config["datasets"]],
                "task": "reasoning-graph-semantic-evaluation",
            },
            telemetry_path="telemetry.jsonl",
            telemetry_schema_version=telemetry_version,
            dashboard_template=config["run"]["dashboard_template"],
            dashboard_path="dashboard.html",
        )

    @staticmethod
    def _validate_scope(config) -> None:
        experiment = config["experiment"]
        datasets = config["datasets"]
        if len(datasets) < experiment["minimum_datasets"]:
            raise ValueError(
                f"Expected at least {experiment['minimum_datasets']} datasets, got {len(datasets)}"
            )
        names = [dataset["name"] for dataset in datasets]
        if len(names) != len(set(names)):
            raise ValueError("Dataset names must be unique")
        for dataset in datasets:
            if dataset["num_questions"] < experiment["minimum_questions_per_dataset"]:
                raise ValueError(
                    f"Dataset '{dataset['name']}' requests {dataset['num_questions']} questions; "
                    f"minimum is {experiment['minimum_questions_per_dataset']}"
                )
        if not config["tuning"]["heuristics"] or not config["tuning"]["thresholds"]:
            raise ValueError("At least one heuristic and threshold are required")
        if (
            config["quality"]["semantic_weight"]
            + config["quality"]["continuation_weight"]
            <= 0
        ):
            raise ValueError("Quality weights must have a positive sum")

    @staticmethod
    def _generation_params(config):
        return {
            **config["generation"],
            "model_id": config["model"]["model_id"],
            "model_revision": config["model"]["model_revision"],
            "dtype": config["model"]["dtype"],
            "device": config["model"]["device"],
            "hidden_layer": config["model"]["hidden_layer"],
            "generation_seed": config["randomness"]["seeds"]["generation"],
        }

    @staticmethod
    def _raw_stats(traces, correct):
        lengths = [len(trace.tokens) for trace in traces]
        total = sum(lengths)
        return {
            "num_chains": len(traces),
            "total_tokens": total,
            "min_chain_length": min(lengths) if lengths else 0,
            "max_chain_length": max(lengths) if lengths else 0,
            "mean_chain_length": total / len(traces) if traces else 0.0,
            "correct_chains": correct,
            "incorrect_chains": len(traces) - correct,
            "valid_format_chains": sum(
                trace.answer_evaluation.get("format_valid") is True for trace in traces
            ),
            "invalid_format_chains": sum(
                trace.answer_evaluation.get("format_valid") is not True for trace in traces
            ),
            "parse_valid_chains": sum(
                trace.answer_evaluation.get("parse_valid") is True for trace in traces
            ),
            "unparseable_chains": sum(
                trace.answer_evaluation.get("parse_valid") is not True for trace in traces
            ),
        }

    @staticmethod
    def _has_hidden(chains) -> bool:
        return any(token.hidden is not None for chain in chains for token in chain.tokens)

    @staticmethod
    def _token_text(tokens) -> str:
        return "".join(token.text for token in tokens).strip()

    @staticmethod
    def _safe(value: str) -> str:
        return value.replace("/", "-").replace("\\", "-").replace(":", "-")

    @classmethod
    def _graph_id(cls, item, heuristic, threshold) -> str:
        return cls._safe(
            f"{item['dataset']}__{item['question_id']}__{heuristic}__{threshold}"
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    @staticmethod
    def _select_graph_reports(rows, limit):
        selected = []
        represented = set()
        for row in sorted(rows, key=lambda item: item["graph_quality_score"]):
            group = (row["dataset"], row["heuristic"], row["threshold"])
            if group not in represented:
                selected.append(row)
                represented.add(group)
            if len(selected) == limit:
                break
        return selected

    @staticmethod
    def _lanes(graph, chains):
        member_cluster = {}
        join_clusters = set()
        for node in graph["nodes"]:
            chain_ids = {member[0] for member in node["members"]}
            if len(chain_ids) > 1:
                join_clusters.add(node["cluster_id"])
            for member in node["members"]:
                member_cluster[tuple(member)] = node["cluster_id"]
        lanes = []
        for chain in sorted(chains, key=lambda item: item.chain_id):
            last = len(chain.tokens) - 1
            tokens = []
            for token in chain.tokens:
                cluster_id = member_cluster.get((chain.chain_id, token.token_index), -1)
                tokens.append(
                    {
                        "index": token.token_index,
                        "text": token.text,
                        "cluster_id": cluster_id,
                        "join": cluster_id in join_clusters,
                        "terminal": token.token_index == last,
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
    def _outputs(
        config, questions, graphs, requests, occurrences, graph_reports, report_relative
    ):
        artifacts = [
            "artifacts/traces/index.json",
            "artifacts/graphs/index.json",
            "artifacts/evaluation/pair_requests.jsonl",
            "artifacts/evaluation/pair_judgments.jsonl",
            "artifacts/evaluation/merge_occurrences.jsonl",
            "artifacts/evaluation/graph_quality.jsonl",
            "artifacts/evaluation/semantic_summary.json",
            report_relative.as_posix(),
            *graph_reports,
        ]
        return {
            "dashboard": "dashboard.html",
            "telemetry": "telemetry.jsonl",
            "traces_index": "artifacts/traces/index.json",
            "graphs_index": "artifacts/graphs/index.json",
            "pair_requests": "artifacts/evaluation/pair_requests.jsonl",
            "pair_judgments": "artifacts/evaluation/pair_judgments.jsonl",
            "merge_occurrences": "artifacts/evaluation/merge_occurrences.jsonl",
            "graph_quality": "artifacts/evaluation/graph_quality.jsonl",
            "semantic_summary": "artifacts/evaluation/semantic_summary.json",
            "semantic_report": report_relative.as_posix(),
            "graph_reports": graph_reports,
            "datasets_evaluated": len(config["datasets"]),
            "questions_sampled": len(questions),
            "graphs_built": len(graphs),
            "unique_pairs_judged": len(requests),
            "merge_occurrences_count": len(occurrences),
            "artifacts": artifacts,
        }

    def _finalize(self, manifest, manifest_path, run_dir, telemetry) -> None:
        telemetry.close()
        manifest.save(manifest_path)
        self._manifest_validator.validate(manifest.to_dict())
        self._renderer.render_run(run_dir)

    @staticmethod
    def _configure_logger(run_id, log_path):
        logger = logging.getLogger(f"got.semantic.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger