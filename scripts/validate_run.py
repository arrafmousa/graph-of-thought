"""Validate a completed run directory (AGENTS.md section 17.2).

Usage:
    python scripts/validate_run.py output/<run_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from libs.dashboard import DashboardRenderer  # noqa: E402
from libs.manifest import ManifestValidator, RunManifest  # noqa: E402
from libs.telemetry import TELEMETRY_SCHEMA_VERSION  # noqa: E402


def validate_run(run_dir: Path) -> list[str]:
    problems: list[str] = []
    if not run_dir.is_dir():
        return [f"Run directory does not exist: {run_dir}"]

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return [f"Missing manifest: {manifest_path}"]

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        ManifestValidator().validate(data)
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation problem
        problems.append(f"Manifest invalid: {exc}")

    expected_hash = RunManifest.compute_config_hash(data.get("configuration", {}))
    if data.get("configuration_hash") != expected_hash:
        problems.append("configuration_hash does not match configuration contents")

    telemetry_path = run_dir / data.get("telemetry", {}).get("path", "telemetry.jsonl")
    if not telemetry_path.is_file():
        problems.append(f"Missing telemetry: {telemetry_path}")
    else:
        problems.extend(_validate_telemetry(telemetry_path))

    dashboard_rel = data.get("dashboard", {}).get("path", "dashboard.html")
    if not (run_dir / dashboard_rel).is_file():
        problems.append(f"Missing dashboard: {run_dir / dashboard_rel}")

    for rel in data.get("outputs", {}).get("artifacts", []):
        if not (run_dir / rel).is_file():
            problems.append(f"Referenced artifact missing: {rel}")

    try:
        DashboardRenderer.default().render_run(run_dir)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"Dashboard could not be regenerated: {exc}")

    status = data.get("status")
    if status not in ("completed", "failed"):
        problems.append(f"Run status is not terminal: {status}")

    return problems


def _validate_telemetry(path: Path) -> list[str]:
    problems: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"telemetry line {i} is not valid JSON: {exc}")
            continue
        if event.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
            problems.append(
                f"telemetry line {i} schema_version mismatch: {event.get('schema_version')}"
            )
        for key in ("run_id", "timestamp_utc", "event_type", "component", "phase"):
            if key not in event:
                problems.append(f"telemetry line {i} missing '{key}'")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: python scripts/validate_run.py output/<run_id>", file=sys.stderr)
        return 2
    problems = validate_run(Path(argv[0]))
    if problems:
        print("RUN VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Run validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
