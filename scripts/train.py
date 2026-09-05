"""Entrypoint for a fine-tuning run (AGENTS.md sections 10, 14, 31).

Usage:
    python scripts/train.py --config configs/train_pipeline/sst2/distilbert_sst2_finetune.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from main.train_pipeline import TrainOrchestrator  # noqa: E402

# Package versions recorded for reproducibility (AGENTS.md section 25).
TRACKED_PACKAGES: list[str] = ["torch", "transformers", "datasets", "accelerate"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a sentiment classifier.")
    parser.add_argument("--config", required=True, help="Path to a training configuration JSON file.")
    args = parser.parse_args(argv)

    orchestrator = TrainOrchestrator(
        repo_root=_REPO_ROOT,
        schema_path=_REPO_ROOT / "configs" / "train_pipeline" / "schema.json",
        tracked_packages=TRACKED_PACKAGES,
    )
    command = "python " + " ".join(["scripts/train.py", *argv])
    run_dir = orchestrator.run(
        config_path=Path(args.config),
        entrypoint="scripts/train.py",
        command=command,
    )
    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
