"""Architecture/policy tests (AGENTS.md section 19)."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))

import validate_repo  # noqa: E402


def test_repository_passes_policy_validation():
    problems = validate_repo.run_all()
    assert problems == [], "\n".join(problems)


def test_forbidden_import_detection_flags_libs_to_main(tmp_path, monkeypatch):
    obj = "widget"
    path = tmp_path / "src" / "libs" / obj / "thing.py"
    path.parent.mkdir(parents=True)
    path.write_text("from main.run_pipeline import RunOrchestrator\n", encoding="utf-8")
    monkeypatch.setattr(validate_repo, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_repo, "SRC", tmp_path / "src")
    monkeypatch.setattr(validate_repo, "LIBS", tmp_path / "src" / "libs")
    monkeypatch.setattr(validate_repo, "MAIN", tmp_path / "src" / "main")
    problems = validate_repo.check_forbidden_imports()
    assert any("imports from 'main'" in p for p in problems)


def test_sibling_import_detection(tmp_path, monkeypatch):
    path = tmp_path / "src" / "libs" / "alpha" / "thing.py"
    path.parent.mkdir(parents=True)
    path.write_text("from libs.beta import Something\n", encoding="utf-8")
    monkeypatch.setattr(validate_repo, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_repo, "SRC", tmp_path / "src")
    monkeypatch.setattr(validate_repo, "LIBS", tmp_path / "src" / "libs")
    monkeypatch.setattr(validate_repo, "MAIN", tmp_path / "src" / "main")
    problems = validate_repo.check_forbidden_imports()
    assert any("sibling library 'beta'" in p for p in problems)


def test_two_classes_detected(tmp_path, monkeypatch):
    path = tmp_path / "src" / "libs" / "alpha" / "two.py"
    path.parent.mkdir(parents=True)
    path.write_text("class A:\n    pass\n\n\nclass B:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(validate_repo, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_repo, "SRC", tmp_path / "src")
    problems = validate_repo.check_one_class_per_file()
    assert any("multiple classes" in p for p in problems)
