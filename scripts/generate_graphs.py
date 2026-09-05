"""Entrypoint for the reasoning-graph generation POC (AGENTS.md sections 10, 26, 31).

Generates token-level traces for a dataset, consolidates them into DAGs under a
sweep of latent-merge heuristics/thresholds, and renders a standalone HTML graph
report for a sampled question. Dataset and model are selected entirely by class
name in the configuration file (no code changes to switch either).

Usage:
    python scripts/generate_graphs.py --config configs/graph_pipeline/demo/synthetic_cpu_graphs.json
    python scripts/generate_graphs.py --config configs/graph_pipeline/gsm8k/llama1b_gsm8k_graphs.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from main.graph_pipeline import GraphOrchestrator  # noqa: E402

# Third-party packages whose versions are recorded for reproducibility (section 25).
TRACKED_PACKAGES: list[str] = ["torch", "transformers", "datasets"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate reasoning graphs from token traces.")
    parser.add_argument("--config", required=True, help="Path to a graph configuration JSON file.")
    args = parser.parse_args(argv)

    orchestrator = GraphOrchestrator(
        repo_root=_REPO_ROOT,
        schema_path=_REPO_ROOT / "configs" / "graph_pipeline" / "schema.json",
        tracked_packages=TRACKED_PACKAGES,
    )
    command = "python " + " ".join(["scripts/generate_graphs.py", *argv])
    run_dir = orchestrator.run(
        config_path=Path(args.config),
        entrypoint="scripts/generate_graphs.py",
        command=command,
    )
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
