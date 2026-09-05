"""Entrypoint for a reproducible run (AGENTS.md sections 10, 26).

Usage:
    python scripts/run.py --config configs/run_pipeline/demo/synthetic_workload_smoke.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from main.run_pipeline import RunOrchestrator  # noqa: E402

# Third-party packages whose versions are recorded for reproducibility.
# Explicit and empty until real dependencies are introduced (AGENTS.md section 25).
TRACKED_PACKAGES: list[str] = []


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the Graph-of-Thought pipeline.")
    parser.add_argument("--config", required=True, help="Path to a run configuration JSON file.")
    args = parser.parse_args(argv)

    orchestrator = RunOrchestrator(
        repo_root=_REPO_ROOT,
        schema_path=_REPO_ROOT / "configs" / "run_pipeline" / "schema.json",
        tracked_packages=TRACKED_PACKAGES,
    )
    command = "python " + " ".join(["scripts/run.py", *argv])
    run_dir = orchestrator.run(
        config_path=Path(args.config),
        entrypoint="scripts/run.py",
        command=command,
    )
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
