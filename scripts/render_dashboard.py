"""Render (or re-render) a run's static HTML dashboard (AGENTS.md section 12).

Usage:
    python scripts/render_dashboard.py output/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from libs.dashboard import DashboardRenderer  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: python scripts/render_dashboard.py output/<run_id>", file=sys.stderr)
        return 2
    run_dir = Path(argv[0])
    if not (run_dir / "run_manifest.json").is_file():
        print(f"No run_manifest.json in {run_dir}", file=sys.stderr)
        return 2
    DashboardRenderer.default().render_run(run_dir)
    print(f"Dashboard rendered: {run_dir / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
