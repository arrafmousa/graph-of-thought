"""Entrypoint for the tuning (hyperparameter-setting) phase (AGENTS.md sections 10, 26, 31).

Sweeps merge heuristics x thresholds over a representative dataset sample and emits one
dashboard per configuration plus a comparison dashboard and ``tuning_summary.json``.
Use the chosen heuristic + threshold in the full-run config for ``generate_graphs.py``.

Usage:
    python scripts/tune_graph.py --config configs/tuning_pipeline/demo/synthetic_cpu_merge_sweep.json
    python scripts/tune_graph.py --config configs/tuning_pipeline/gsm8k/llama1b_gsm8k_merge_sweep.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from main.tuning_pipeline import TuningOrchestrator  # noqa: E402

TRACKED_PACKAGES: list[str] = ["torch", "transformers", "datasets"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Tune merge heuristics/thresholds for reasoning graphs.")
    parser.add_argument("--config", required=True, help="Path to a tuning configuration JSON file.")
    args = parser.parse_args(argv)

    orchestrator = TuningOrchestrator(
        repo_root=_REPO_ROOT,
        schema_path=_REPO_ROOT / "configs" / "tuning_pipeline" / "schema.json",
        tracked_packages=TRACKED_PACKAGES,
    )
    command = "python " + " ".join(["scripts/tune_graph.py", *argv])
    run_dir = orchestrator.run(
        config_path=Path(args.config),
        entrypoint="scripts/tune_graph.py",
        command=command,
    )
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
