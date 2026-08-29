import json
from pathlib import Path

import pytest

from libs.config import ConfigError, ConfigLoader

_REPO = Path(__file__).resolve().parents[3]
_SCHEMA = _REPO / "configs" / "schema" / "run_config.schema.json"


def _loader() -> ConfigLoader:
    return ConfigLoader(_SCHEMA)


def test_loads_valid_example_config():
    data = _loader().load(_REPO / "configs" / "example_run.json")
    assert data["run"]["task_name"]
    assert data["run"]["dashboard_template"] == "generic"


def test_loads_dashboard_demo_config():
    data = _loader().load(_REPO / "configs" / "dashboard_demo.json")
    assert data["workload"]["interval_seconds"] == 2
    assert data["workload"]["value_min"] == 1
    assert data["workload"]["value_max"] == 3


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        _loader().load(bad)


def test_wrong_type_raises(tmp_path):
    payload = {
        "schema_version": "1.0.0",
        "run": {"task_name": "x", "dashboard_template": "generic", "dashboard_refresh_seconds": 0},
        "randomness": {"seeds": {"python": 1}},
        "workload": {"steps": "not-an-int", "step_label": "t", "interval_seconds": 0,
                     "value_min": 0, "value_max": 1},
    }
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        _loader().load(bad)


def test_enum_violation_raises(tmp_path):
    payload = {
        "schema_version": "1.0.0",
        "run": {"task_name": "x", "dashboard_template": "nope", "dashboard_refresh_seconds": 0},
        "randomness": {"seeds": {"python": 1}},
        "workload": {"steps": 1, "step_label": "t", "interval_seconds": 0,
                     "value_min": 0, "value_max": 1},
    }
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        _loader().load(bad)
