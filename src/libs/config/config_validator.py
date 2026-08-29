"""Explicit configuration validation against a JSON-Schema subset.

The validator intentionally supports only the subset of JSON Schema this
repository uses so it can run on the Python standard library alone. Every
required project-owned field must be declared in the schema; missing fields
fail validation loudly (AGENTS.md sections 7 and 16 — no hidden fallbacks).
"""
from __future__ import annotations

from typing import Any

from .config_error import ConfigError

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


class ConfigValidator:
    """Validate a configuration mapping against a declared schema."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def validate(self, data: Any) -> None:
        self._validate_node(self._schema, data, "$")

    def _validate_node(self, schema: dict[str, Any], value: Any, path: str) -> None:
        expected_type = schema.get("type")
        if expected_type is not None:
            self._check_type(expected_type, value, path)

        if expected_type == "object":
            self._validate_object(schema, value, path)
        elif expected_type == "array":
            self._validate_array(schema, value, path)

        enum = schema.get("enum")
        if enum is not None and value not in enum:
            raise ConfigError(f"{path}: value {value!r} is not one of {enum}")

        minimum = schema.get("minimum")
        if minimum is not None and isinstance(value, (int, float)) and value < minimum:
            raise ConfigError(f"{path}: value {value} is below minimum {minimum}")

    def _check_type(self, expected_type: str, value: Any, path: str) -> None:
        allowed = _TYPE_MAP.get(expected_type)
        if allowed is None:
            raise ConfigError(f"{path}: unknown schema type {expected_type!r}")
        # bool is a subclass of int; reject it where a numeric type is expected.
        if expected_type in ("integer", "number") and isinstance(value, bool):
            raise ConfigError(f"{path}: expected {expected_type}, got boolean")
        if not isinstance(value, allowed):
            raise ConfigError(
                f"{path}: expected {expected_type}, got {type(value).__name__}"
            )

    def _validate_object(self, schema: dict[str, Any], value: dict[str, Any], path: str) -> None:
        for field in schema.get("required", []):
            if field not in value:
                raise ConfigError(f"{path}: missing required field '{field}'")
        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                self._validate_node(child_schema, value[key], f"{path}.{key}")

    def _validate_array(self, schema: dict[str, Any], value: list[Any], path: str) -> None:
        item_schema = schema.get("items")
        if item_schema is None:
            return
        for index, item in enumerate(value):
            self._validate_node(item_schema, item, f"{path}[{index}]")
