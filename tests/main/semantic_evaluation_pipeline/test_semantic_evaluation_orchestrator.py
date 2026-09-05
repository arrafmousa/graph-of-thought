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


def _demo_config() -> Path:
    return (
        _REPO
        / "configs"
        / "semantic_evaluation_pipeline"
        / "demo"
        / "synthetic_cpu_semantic_evaluation.json"
    )


def test_generate_then_judge_stages_resume_without_regeneration(tmp_path):
    orchestrator = SemanticEvaluationOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "semantic_evaluation_pipeline" / "schema.json",
        tracked_packages=[],
    )
    # Stage 1: GPU-side generation + graphs only, no judging.
    gen_dir = orchestrator.run(
        config_path=_demo_config(), entrypoint="test", command="test",
        stage="generate",
    )
    gen_manifest = json.loads((gen_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ManifestValidator().validate(gen_manifest)
    assert gen_manifest["status"] == "completed"
    assert gen_manifest["outputs"]["stage"] == "generate"
    assert (gen_dir / "artifacts" / "evaluation" / "pair_requests.jsonl").is_file()
    assert (gen_dir / "artifacts" / "evaluation" / "merge_occurrences_raw.jsonl").is_file()
    assert (gen_dir / "artifacts" / "evaluation" / "generation_state.json").is_file()
    # No judging happened yet.
    assert not (gen_dir / "artifacts" / "evaluation" / "pair_judgments.jsonl").is_file()
    assert not (gen_dir / "artifacts" / "reports" / "semantic_evaluation.html").is_file()

    graphs_before = (gen_dir / "artifacts" / "graphs" / "index.json").read_text(
        encoding="utf-8"
    )

    # Stage 2: resume judging on the same run dir, no regeneration.
    judged_dir = orchestrator.run(
        config_path=_demo_config(), entrypoint="test", command="test",
        stage="judge", resume_run_dir=gen_dir,
    )
    assert judged_dir == gen_dir
    graphs_after = (gen_dir / "artifacts" / "graphs" / "index.json").read_text(
        encoding="utf-8"
    )
    assert graphs_before == graphs_after  # generation artifacts untouched

    judged_manifest = json.loads(
        (judged_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    ManifestValidator().validate(judged_manifest)
    assert judged_manifest["status"] == "completed"
    assert judged_manifest["outputs"]["stage"] == "judged"
    assert (judged_dir / "artifacts" / "evaluation" / "pair_judgments.jsonl").is_file()
    assert (judged_dir / "artifacts" / "reports" / "semantic_evaluation.html").is_file()
    summary = json.loads(
        (judged_dir / judged_manifest["outputs"]["semantic_summary"]).read_text(
            encoding="utf-8"
        )
    )
    assert len(summary["graphs"]) == judged_manifest["outputs"]["graphs_built"]


def test_judge_stage_rebuilds_state_from_saved_graphs_when_missing(tmp_path):
    orchestrator = SemanticEvaluationOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "semantic_evaluation_pipeline" / "schema.json",
        tracked_packages=[],
    )
    gen_dir = orchestrator.run(
        config_path=_demo_config(), entrypoint="test", command="test",
        stage="generate",
    )
    # Simulate an older bundle that only has traces + graphs + pair_requests.
    evaluation = gen_dir / "artifacts" / "evaluation"
    (evaluation / "generation_state.json").unlink()
    (evaluation / "merge_occurrences_raw.jsonl").unlink()
    (evaluation / "pair_requests.jsonl").unlink()

    judged_dir = orchestrator.run(
        config_path=_demo_config(), entrypoint="test", command="test",
        stage="judge", resume_run_dir=gen_dir,
    )
    manifest = json.loads((judged_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ManifestValidator().validate(manifest)
    assert manifest["status"] == "completed"
    assert manifest["outputs"]["unique_pairs_judged"] > 0
    assert (judged_dir / "artifacts" / "reports" / "semantic_evaluation.html").is_file()


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
    config = json.loads(
        (
            _REPO
            / "configs"
            / "semantic_evaluation_pipeline"
            / "demo"
            / "synthetic_cpu_semantic_evaluation.json"
        ).read_text(encoding="utf-8")
    )
    min_join = config["tuning"]["min_join_token_index"]
    after_first_token_merges = 0
    first_token_merges = 0
    for graph_record in graph_index["graphs"]:
        graph_payload = json.loads(
            (run_dir / graph_record["graph_path"]).read_text(encoding="utf-8")
        )
        for merge in graph_payload["graph"]["merges"]:
            if merge["node_a"][1] >= min_join and merge["node_b"][1] >= min_join:
                after_first_token_merges += 1
            else:
                first_token_merges += 1
    assert min_join >= 1
    assert first_token_merges > 0
    assert len(occurrences) == after_first_token_merges
    assert all(
        row["node_a"][1] >= min_join and row["node_b"][1] >= min_join
        for row in occurrences
    )
    assert {request["pair_id"] for request in requests} == {
        occurrence["pair_id"] for occurrence in occurrences
    }

    summary_experiment = json.loads(
        (run_dir / manifest["outputs"]["semantic_summary"]).read_text(encoding="utf-8")
    )["experiment"]
    assert summary_experiment["skipped_first_token_merges"] == first_token_merges
    assert summary_experiment["min_join_token_index"] == min_join

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
    assert config["judge"]["api_version"] == "2024-12-01-preview"
    assert config["judge"]["batch_endpoint"] == "/chat/completions"


def test_pilot_config_samples_five_questions_from_five_pinned_datasets():
    config = json.loads(
        (
            _REPO
            / "configs"
            / "semantic_evaluation_pipeline"
            / "math"
            / "llama1b_five_dataset_pilot.json"
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
    assert all(item["num_questions"] == 5 for item in datasets)
    assert config["experiment"]["minimum_questions_per_dataset"] == 5
    assert all(len(item["dataset_revision"]) == 40 for item in datasets)
    assert config["judge"]["provider"] == "azure_openai_batch"
    assert config["generation"]["batch_size"] == 6