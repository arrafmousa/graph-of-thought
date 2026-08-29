import sys
from pathlib import Path

from libs.manifest import ManifestValidator
from main.run_pipeline import RunOrchestrator

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

import validate_run  # noqa: E402


def test_end_to_end_run_produces_valid_artifacts(tmp_path):
    orchestrator = RunOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "schema" / "run_config.schema.json",
        tracked_packages=[],
    )
    run_dir = orchestrator.run(
        config_path=_REPO / "configs" / "example_run.json",
        entrypoint="scripts/run.py",
        command="python scripts/run.py --config configs/example_run.json",
    )

    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "telemetry.jsonl").is_file()
    assert (run_dir / "dashboard.html").is_file()
    assert (run_dir / "artifacts" / "scores.json").is_file()

    import json

    data = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    ManifestValidator().validate(data)

    assert validate_run.validate_run(run_dir) == []


def test_workload_values_stay_within_configured_range(tmp_path):
    import json

    config = {
        "schema_version": "1.0.0",
        "run": {
            "task_name": "range-test",
            "dashboard_template": "generic",
            "dashboard_refresh_seconds": 0,
        },
        "randomness": {"seeds": {"python": 42}},
        # interval_seconds 0 keeps the test fast while exercising value bounds.
        "workload": {
            "steps": 25,
            "step_label": "sample",
            "interval_seconds": 0,
            "value_min": 1,
            "value_max": 3,
        },
    }
    config_path = tmp_path / "range.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    orchestrator = RunOrchestrator(
        repo_root=tmp_path,
        schema_path=_REPO / "configs" / "schema" / "run_config.schema.json",
        tracked_packages=[],
    )
    run_dir = orchestrator.run(
        config_path=config_path,
        entrypoint="scripts/dashboard_demo.py",
        command="python scripts/dashboard_demo.py",
    )

    events = [
        json.loads(line)
        for line in (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = [
        e["payload"]["args"][1]
        for e in events
        if e.get("event_type") == "tile_update" and e.get("component") == "values"
    ]
    assert len(scores) == 25
    assert all(1.0 <= s <= 3.0 for s in scores)
    assert "polyline" in (run_dir / "dashboard.html").read_text(encoding="utf-8")
