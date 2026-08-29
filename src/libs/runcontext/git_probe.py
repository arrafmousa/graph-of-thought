"""Capture git provenance for a run (AGENTS.md section 10.1)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class GitProbe:
    """Collect commit, branch, and dirty state for a working directory."""

    def __init__(self, working_directory: Path) -> None:
        self._cwd = Path(working_directory)

    def collect(self) -> dict[str, Any]:
        commit = self._run(["rev-parse", "HEAD"])
        branch = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        status = self._run(["status", "--porcelain"])
        return {
            "commit": commit,
            "branch": branch,
            "dirty": status is not None and status != "",
        }

    def _run(self, args: list[str]) -> Any:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(self._cwd),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()
