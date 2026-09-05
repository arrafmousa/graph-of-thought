"""End-to-end test for the graph pipeline orchestrator (synthetic, CPU-only)."""
from __future__ import annotations

import json
from pathlib import Path

from libs.manifest import ManifestValidator
from main.graph_pipeline import GraphOrchestrator

_REPO = Path(__file__).resolve().parents[3]


def test_graph_run_produces_valid_artifacts(tmp_path):
    orchestrator = GraphOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "graph_pipeline" / "schema.json",
        tracked_packages=[],
    )
    run_dir = orchestrator.run(
        config_path=_REPO / "configs" / "graph_pipeline" / "demo" / "synthetic_cpu_graphs.json",
        entrypoint="test",
        command="test",
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ManifestValidator().validate(manifest)
    assert manifest["status"] == "completed"
    assert (run_dir / "dashboard.html").is_file()
    assert (run_dir / "telemetry.jsonl").is_file()

    # Intermediate trace artifacts (the documented pipeline hand-off).
    index = json.loads((run_dir / "artifacts" / "traces" / "index.json").read_text(encoding="utf-8"))
    assert index["questions"]
    first = index["questions"][0].replace("/", "-")
    question_dir = run_dir / "artifacts" / "traces" / f"question_{first}"
    assert (question_dir / "raw_traces.jsonl").is_file()
    assert (question_dir / "hidden_states.jsonl").is_file()
    assert (question_dir / "raw_stats.json").is_file()

    # Consolidated graphs and their DAG validity.
    graphs = list((run_dir / "artifacts" / "graphs").rglob("graph_*.json"))
    assert graphs
    payload = json.loads(graphs[0].read_text(encoding="utf-8"))
    assert payload["stats"]["dag_valid"] is True

    # Sampled HTML report.
    assert manifest["outputs"]["reports"]
    report_rel = manifest["outputs"]["reports"][0]
    report_path = run_dir / report_rel
    assert report_path.is_file()
    assert "<svg" in report_path.read_text(encoding="utf-8")


def test_graph_run_records_provider_classes(tmp_path):
    orchestrator = GraphOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "graph_pipeline" / "schema.json",
        tracked_packages=[],
    )
    run_dir = orchestrator.run(
        config_path=_REPO / "configs" / "graph_pipeline" / "demo" / "synthetic_cpu_graphs.json",
        entrypoint="test",
        command="test",
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["provider"] == "synthetic"
    assert manifest["model"]["dataset_provider"] == "synthetic"
