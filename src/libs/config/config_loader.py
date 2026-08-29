"""Load and validate run configuration from disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_error import ConfigError
from .config_validator import ConfigValidator


class ConfigLoader:
    """Load a JSON run configuration and validate it against a JSON schema file."""

    def __init__(self, schema_path: Path) -> None:
        self._schema_path = Path(schema_path)
        if not self._schema_path.is_file():
            raise ConfigError(f"Schema file not found: {self._schema_path}")
        schema = json.loads(self._schema_path.read_text(encoding="utf-8"))
        self._validator = ConfigValidator(schema)

    def load(self, config_path: Path) -> dict[str, Any]:
        path = Path(config_path)
        if not path.is_file():
            raise ConfigError(f"Configuration file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Configuration file is not valid JSON: {path} ({exc})") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Configuration root must be an object: {path}")
        self._validator.validate(data)
        return data
