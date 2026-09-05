"""Run multi-dataset semantic evaluation of reasoning-graph merges."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
load_dotenv(_REPO_ROOT / ".env")

from main.semantic_evaluation_pipeline import SemanticEvaluationOrchestrator  # noqa: E402

TRACKED_PACKAGES: list[str] = ["torch", "transformers", "datasets", "openai"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate reasoning-state merges across configured math datasets."
    )
    parser.add_argument("--config", required=True, help="Path to an evaluation JSON config.")
    args = parser.parse_args(argv)
    orchestrator = SemanticEvaluationOrchestrator(
        repo_root=_REPO_ROOT,
        schema_path=_REPO_ROOT / "configs" / "semantic_evaluation_pipeline" / "schema.json",
        tracked_packages=TRACKED_PACKAGES,
    )
    command = "python " + " ".join(["scripts/evaluate_merges.py", *argv])
    run_dir = orchestrator.run(
        config_path=Path(args.config),
        entrypoint="scripts/evaluate_merges.py",
        command=command,
    )
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))