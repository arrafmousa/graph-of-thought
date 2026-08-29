import pytest

from libs.manifest import ManifestError, ManifestValidator, RunManifest


def _manifest() -> RunManifest:
    return RunManifest.create(
        run_id="run-1",
        entrypoint="scripts/run.py",
        command="python scripts/run.py --config configs/example_run.json",
        working_directory="/repo",
        git={"commit": None, "branch": None, "dirty": False},
        environment={"python_version": "3.12", "platform": "test", "relevant_package_versions": {}},
        configuration={"a": 1, "b": 2},
        seeds={"python": 123},
        inputs={"config_path": "configs/example_run.json"},
        model={},
        telemetry_path="telemetry.jsonl",
        telemetry_schema_version="1.0.0",
        dashboard_template="generic",
        dashboard_path="dashboard.html",
    )


def test_config_hash_is_order_independent():
    h1 = RunManifest.compute_config_hash({"a": 1, "b": 2})
    h2 = RunManifest.compute_config_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_completed_manifest_validates():
    manifest = _manifest()
    manifest.mark_running()
    manifest.mark_completed({"dashboard": "dashboard.html"}, 1.23)
    ManifestValidator().validate(manifest.to_dict())


def test_missing_field_fails_validation():
    data = _manifest().to_dict()
    del data["telemetry"]
    with pytest.raises(ManifestError):
        ManifestValidator().validate(data)


def test_terminal_status_requires_end_timestamp():
    data = _manifest().to_dict()
    data["status"] = "completed"  # end timestamp still None
    with pytest.raises(ManifestError):
        ManifestValidator().validate(data)
