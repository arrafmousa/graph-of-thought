"""End-to-end tests for multi-dataset semantic merge evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from libs.manifest import ManifestValidator
from main.semantic_evaluation_pipeline import SemanticEvaluationOrchestrator

_REPO = Path(__file__).resolve().parents[3]


def _run(tmp_path) -> Path:
    orchestrator = SemanticEvaluationOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "semantic_evaluation_pipeline" / "schema.json",
        tracked_packages=[],
    )
    return orchestrator.run(
        config_path=_REPO
        / "configs"
        / "semantic_evaluation_pipeline"
        / "demo"
        / "synthetic_cpu_semantic_evaluation.json",
        entrypoint="test",
        command="test",
    )


def test_semantic_evaluation_run_produces_complete_offline_artifacts(tmp_path):
    run_dir = _run(tmp_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ManifestValidator().validate(manifest)
    assert manifest["status"] == "completed"
    assert manifest["outputs"]["questions_sampled"] == 2
    assert manifest["outputs"]["graphs_built"] == 8
    for relative in manifest["outputs"]["artifacts"]:
        assert (run_dir / relative).is_file()

    traces = json.loads(
        (run_dir / manifest["outputs"]["traces_index"]).read_text(encoding="utf-8")
    )
    assert traces["datasets"][0]["sample_seed"] == 101
    assert len(traces["questions"]) == 2
    first_trace_dir = run_dir / traces["questions"][0]["trace_dir"]
    raw_stats = json.loads((first_trace_dir / "raw_stats.json").read_text(encoding="utf-8"))
    assert raw_stats["valid_format_chains"] == 3
    assert raw_stats["parse_valid_chains"] == 3

    requests = [
        json.loads(line)
        for line in (run_dir / manifest["outputs"]["pair_requests"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert requests
    assert all("predicted_a" not in request for request in requests)
    assert all("path_a" in request and "path_b" in request for request in requests)

    occurrences = [
        json.loads(line)
        for line in (run_dir / manifest["outputs"]["merge_occurrences"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert occurrences
    assert all("same_final_answer" in row for row in occurrences)
    assert all("answer_a_parse_valid" in row for row in occurrences)
    assert all("answer_b_parse_valid" in row for row in occurrences)
    assert all(row["merge_accepted"] is True for row in occurrences)
    assert all("equivalence_score" in row for row in occurrences)
    graph_index = json.loads(
        (run_dir / manifest["outputs"]["graphs_index"]).read_text(encoding="utf-8")
    )
    accepted_merge_count = 0
    for graph_record in graph_index["graphs"]:
        graph_payload = json.loads(
            (run_dir / graph_record["graph_path"]).read_text(encoding="utf-8")
        )
        accepted_merge_count += len(graph_payload["graph"]["merges"])
    assert len(occurrences) == accepted_merge_count
    assert {request["pair_id"] for request in requests} == {
        occurrence["pair_id"] for occurrence in occurrences
    }

    summary = json.loads(
        (run_dir / manifest["outputs"]["semantic_summary"]).read_text(encoding="utf-8")
    )
    assert len(summary["configurations"]) == 4
    assert len(summary["graphs"]) == 8
    assert all("graph_quality_score" in row for row in summary["graphs"])
    report = (run_dir / manifest["outputs"]["semantic_report"]).read_text(
        encoding="utf-8"
    )
    assert "configuration ranking" in report
    assert "manual merge review" in report


def test_production_config_samples_twenty_questions_from_five_pinned_datasets():
    config = json.loads(
        (
            _REPO
            / "configs"
            / "semantic_evaluation_pipeline"
            / "math"
            / "llama1b_five_dataset_azure_batch.json"
        ).read_text(encoding="utf-8")
    )
    datasets = config["datasets"]

    assert [item["name"] for item in datasets] == [
        "gsm8k",
        "math500",
        "aime2025",
        "svamp",
        "asdiv",
    ]
    assert all(item["num_questions"] == 20 for item in datasets)
    assert len({item["sample_seed"] for item in datasets}) == 5
    assert all(len(item["dataset_revision"]) == 40 for item in datasets)
    assert config["judge"]["provider"] == "azure_openai_batch"
    assert config["judge"]["deployment"] == "gpt-5.1"
    assert config["judge"]["api_path"] == "openai/v1"
    assert config["judge"]["request_url"] == "/v1/chat/completions"