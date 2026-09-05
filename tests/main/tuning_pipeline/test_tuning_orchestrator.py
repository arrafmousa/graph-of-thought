"""End-to-end test for the tuning pipeline orchestrator (synthetic, CPU-only)."""
from __future__ import annotations

import json
from pathlib import Path

from libs.manifest import ManifestValidator
from main.tuning_pipeline import TuningOrchestrator

_REPO = Path(__file__).resolve().parents[3]


def _run(tmp_path) -> Path:
    orchestrator = TuningOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "tuning_pipeline" / "schema.json",
        tracked_packages=[],
    )
    return orchestrator.run(
        config_path=_REPO / "configs" / "tuning_pipeline" / "demo" / "synthetic_cpu_merge_sweep.json",
        entrypoint="test",
        command="test",
    )


def test_tuning_produces_per_config_dashboards_and_comparison(tmp_path):
    run_dir = _run(tmp_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    ManifestValidator().validate(manifest)
    assert manifest["status"] == "completed"

    outputs = manifest["outputs"]
    # 3 heuristics x 3 thresholds = 9 configurations, each with its own dashboard.
    assert outputs["configs_evaluated"] == 9
    assert len(outputs["config_dashboards"]) == 9
    for rel in outputs["config_dashboards"]:
        assert (run_dir / rel).is_file()

    comparison = run_dir / outputs["comparison_dashboard"]
    assert comparison.is_file()
    assert "comparison table" in comparison.read_text(encoding="utf-8")


def test_tuning_summary_lists_features_and_vocabulary(tmp_path):
    run_dir = _run(tmp_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / manifest["outputs"]["tuning_summary"]).read_text(encoding="utf-8"))

    assert len(summary["configs"]) == 9
    row = summary["configs"][0]
    for key in ("heuristic", "threshold", "total_merges", "mean_node_reduction", "histograms"):
        assert key in row
    assert "in_degree" in row["histograms"]
    # Everything selectable is discoverable as an enum vocabulary.
    assert "hidden_cosine" in summary["available"]["merge heuristics"]
    assert "recent_mean_logprob" in summary["available"]["representative policies"]


def test_tuning_collects_high_and_edge_merge_samples(tmp_path):
    run_dir = _run(tmp_path)
    # A per-config dashboard should render both sample tables.
    dashboards = list((run_dir / "artifacts" / "tuning").rglob("dashboard.html"))
    assert dashboards
    text = dashboards[0].read_text(encoding="utf-8")
    assert "highest-similarity merges" in text
    assert "borderline (near-threshold) merges" in text
