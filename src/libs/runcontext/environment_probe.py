"""Capture reproducibility-relevant environment metadata (AGENTS.md section 10.1)."""
from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any


class EnvironmentProbe:
    """Collect Python, platform, and selected package version information."""

    def __init__(self, tracked_packages: list[str]) -> None:
        self._tracked_packages = tracked_packages

    def collect(self) -> dict[str, Any]:
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "relevant_package_versions": self._package_versions(),
        }

    def _package_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for name in self._tracked_packages:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "not-installed"
        return versions
