"""Repository architecture validator (AGENTS.md sections 17.1, 19).

Enforces the engineering commandments as executable checks:
- required top-level structure and ignore rules;
- forbidden import directions (libs -> main, libs -> sibling libs);
- one project-owned class per source file;
- tests mirror source structure;
- required configuration schema fields;
- no project-owned default values in configuration modules;
- presence of telemetry/reporting/dashboard infrastructure.

Runs on the Python standard library alone. Exits non-zero on any violation.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
LIBS = SRC / "libs"
MAIN = SRC / "main"


def check_structure() -> list[str]:
    required = [
        "AGENTS.md",
        ".env.example",
        ".gitignore",
        "configs",
        "scripts",
        "src/libs",
        "src/main",
        "tests",
        "scripts/validate_repo.py",
        "scripts/validate_run.py",
        "scripts/render_dashboard.py",
        "src/libs/telemetry",
        "src/libs/dashboard",
    ]
    return [f"Missing required path: {rel}" for rel in required if not (REPO_ROOT / rel).exists()]


def check_ignore_rules() -> list[str]:
    problems: list[str] = []
    gitignore = REPO_ROOT / ".gitignore"
    if not gitignore.is_file():
        return ["Missing .gitignore"]
    lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
    if ".env" not in lines:
        problems.append(".gitignore must ignore '.env'")
    if "output/" not in lines and "output" not in lines:
        problems.append(".gitignore must ignore 'output/'")
    return problems


def _iter_py(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*.py")):
        yield path


def check_one_class_per_file() -> list[str]:
    problems: list[str] = []
    for path in _iter_py(SRC):
        tree = _parse(path, problems)
        if tree is None:
            continue
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        if len(classes) > 1:
            names = ", ".join(c.name for c in classes)
            problems.append(f"{_rel(path)}: defines multiple classes ({names})")
    return problems


def check_forbidden_imports() -> list[str]:
    problems: list[str] = []
    for path in _iter_py(LIBS):
        obj = _library_object(path)
        if obj is None:
            continue
        tree = _parse(path, problems)
        if tree is None:
            continue
        package_parts = _package_parts(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    problems.extend(_check_absolute(alias.name, obj, path))
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    problems.extend(
                        _check_relative(node.level, package_parts, obj, path)
                    )
                elif node.module:
                    problems.extend(_check_absolute(node.module, obj, path))
    return problems


def _check_absolute(module: str, obj: str, path: Path) -> list[str]:
    head = module.split(".")
    if head[0] == "main":
        return [f"{_rel(path)}: library imports from 'main' ({module})"]
    if head[0] == "libs" and len(head) >= 2 and head[1] != obj:
        return [f"{_rel(path)}: library '{obj}' imports sibling library '{head[1]}'"]
    return []


def _check_relative(level: int, package_parts: list[str], obj: str, path: Path) -> list[str]:
    keep = len(package_parts) - (level - 1)
    base = package_parts[:keep] if keep > 0 else []
    if len(base) < 2 or base[0] != "libs" or base[1] != obj:
        return [
            f"{_rel(path)}: relative import (level {level}) escapes library object '{obj}'"
        ]
    return []


def check_test_alignment() -> list[str]:
    problems: list[str] = []
    for base, kind in ((LIBS, "libs"), (MAIN, "main")):
        if not base.exists():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir() or child.name == "__pycache__":
                continue
            if not any(child.rglob("*.py")):
                continue
            expected = REPO_ROOT / "tests" / kind / child.name
            if not expected.is_dir():
                problems.append(f"Missing mirrored tests directory: tests/{kind}/{child.name}")
    return problems


def check_config_schema() -> list[str]:
    problems: list[str] = []
    for schema_path in sorted((REPO_ROOT / "configs").glob("*/schema.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if not schema.get("required"):
            problems.append(f"{_rel(schema_path)}: must declare non-empty 'required' fields")
    return problems


def check_config_layout() -> list[str]:
    """AGENTS.md section 7.5: configs/<orchestrator>/<experiment>/<config>.json."""
    configs_dir = REPO_ROOT / "configs"
    if not configs_dir.is_dir():
        return ["Missing configs/ directory"]
    problems: list[str] = []
    for entry in sorted(configs_dir.iterdir()):
        if entry.name == "__pycache__":
            continue
        if entry.is_file():
            if entry.name != "README.md":
                problems.append(
                    f"{_rel(entry)}: config files must live in configs/<orchestrator>/<experiment>/"
                )
            continue
        if not (MAIN / entry.name).is_dir():
            problems.append(
                f"configs/{entry.name}: no matching orchestrator src/main/{entry.name}"
            )
        if not (entry / "schema.json").is_file():
            problems.append(f"configs/{entry.name}: missing schema.json")
        if not (entry / "README.md").is_file():
            problems.append(f"configs/{entry.name}: missing README.md describing its configs")
        for config in sorted(entry.rglob("*.json")):
            if config.name == "schema.json" and config.parent == entry:
                continue
            depth = len(config.relative_to(entry).parts)
            if depth != 2:
                problems.append(
                    f"{_rel(config)}: expected configs/<orchestrator>/<experiment>/<config>.json"
                )
    for orchestrator in sorted(MAIN.iterdir()):
        if not orchestrator.is_dir() or orchestrator.name == "__pycache__":
            continue
        if not (configs_dir / orchestrator.name).is_dir():
            problems.append(
                f"src/main/{orchestrator.name}: no matching configs/{orchestrator.name} folder"
            )
    return problems


def check_config_defaults() -> list[str]:
    """Configuration modules must not encode project-owned default values."""
    problems: list[str] = []
    config_dir = LIBS / "config"
    for path in _iter_py(config_dir):
        tree = _parse(path, problems)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = list(node.args.defaults) + list(node.args.kw_defaults)
                for default in defaults:
                    if default is None:
                        continue
                    if isinstance(default, ast.Constant) and default.value is None:
                        continue
                    problems.append(
                        f"{_rel(path)}: function '{node.name}' has a project-owned default value"
                    )
    return problems


def _library_object(path: Path) -> str | None:
    parts = path.relative_to(LIBS).parts
    if len(parts) < 2:
        return None
    return parts[0]


def _package_parts(path: Path) -> list[str]:
    return list(path.relative_to(SRC).with_suffix("").parts[:-1])


def _parse(path: Path, problems: list[str]):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        problems.append(f"{_rel(path)}: syntax error: {exc}")
        return None


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def run_all() -> list[str]:
    problems: list[str] = []
    problems.extend(check_structure())
    problems.extend(check_ignore_rules())
    problems.extend(check_one_class_per_file())
    problems.extend(check_forbidden_imports())
    problems.extend(check_test_alignment())
    problems.extend(check_config_schema())
    problems.extend(check_config_layout())
    problems.extend(check_config_defaults())
    return problems


def main() -> int:
    problems = run_all()
    if problems:
        print("REPOSITORY VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Repository validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
