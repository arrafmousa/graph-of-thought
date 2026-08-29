"""Dashboard preview run (AGENTS.md sections 10, 12).

Runs the demo workload — a random value in [value_min, value_max] emitted every
``interval_seconds`` — and opens the dashboard immediately so you can watch it
update live (the page auto-refreshes every ``dashboard_refresh_seconds``). When
the run finishes, the orchestrator writes a final static dashboard.

Usage:
    python scripts/dashboard_demo.py
"""
from __future__ import annotations

import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from main.run_pipeline import RunOrchestrator  # noqa: E402

CONFIG_PATH = _REPO_ROOT / "configs" / "dashboard_demo.json"


def _find_new_dashboard(output_root: Path, known: set[Path]) -> Optional[Path]:
    if not output_root.exists():
        return None
    for run_dir in output_root.iterdir():
        if run_dir in known:
            continue
        dashboard = run_dir / "dashboard.html"
        if dashboard.is_file():
            return dashboard
    return None


def main() -> int:
    orchestrator = RunOrchestrator(
        repo_root=_REPO_ROOT,
        schema_path=_REPO_ROOT / "configs" / "schema" / "run_config.schema.json",
        tracked_packages=[],
    )
    output_root = _REPO_ROOT / "output"
    known = set(output_root.iterdir()) if output_root.exists() else set()

    result: dict[str, Path] = {}
    error: dict[str, BaseException] = {}

    def _run() -> None:
        try:
            result["run_dir"] = orchestrator.run(
                config_path=CONFIG_PATH,
                entrypoint="scripts/dashboard_demo.py",
                command="python scripts/dashboard_demo.py",
            )
        except BaseException as exc:  # surfaced after join (AGENTS.md section 22)
            error["exc"] = exc

    worker = threading.Thread(target=_run)
    worker.start()

    # Open the dashboard as soon as it appears so the browser refreshes live.
    dashboard: Optional[Path] = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and worker.is_alive():
        dashboard = _find_new_dashboard(output_root, known)
        if dashboard is not None:
            webbrowser.open(dashboard.as_uri())
            print(f"Watching live dashboard: {dashboard}")
            break
        time.sleep(0.2)

    worker.join()
    if "exc" in error:
        raise error["exc"]

    run_dir = result.get("run_dir")
    print(f"Run complete: {run_dir}")
    if dashboard is None and run_dir is not None:
        print(f"Open the dashboard: {run_dir / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
