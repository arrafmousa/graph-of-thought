from pathlib import Path

from libs.runcontext import EnvironmentProbe, GitProbe, RunIdFactory


def test_run_id_contains_task_and_is_unique():
    factory = RunIdFactory()
    a = factory.create("my task")
    b = factory.create("my task")
    assert "my-task" in a
    assert a != b


def test_environment_probe_reports_python_version():
    env = EnvironmentProbe(tracked_packages=[]).collect()
    assert env["python_version"]
    assert "platform" in env
    assert env["relevant_package_versions"] == {}


def test_git_probe_returns_expected_keys(tmp_path: Path):
    info = GitProbe(tmp_path).collect()
    assert set(info) == {"commit", "branch", "dirty"}
