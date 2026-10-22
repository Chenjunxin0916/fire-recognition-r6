"""Strict YAML configuration loading shared by command-line entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is missing or malformed."""


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one UTF-8 YAML mapping and reject empty or malformed documents."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read YAML configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"configuration must be a mapping: {path}")
    return dict(value)


def require_mapping(value: Mapping[str, Any], key: str, *, context: str) -> dict[str, Any]:
    """Return a required nested mapping with a contextual error."""

    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ConfigError(f"{context}.{key} must be a mapping")
    return dict(nested)


def require_value(value: Mapping[str, Any], key: str, *, context: str) -> Any:
    """Return a required scalar or nested value."""

    if key not in value or value[key] is None:
        raise ConfigError(f"missing required configuration field {context}.{key}")
    return value[key]


def require_string(value: Mapping[str, Any], key: str, *, context: str) -> str:
    """Return a required non-empty string configuration value."""

    raw = require_value(value, key, context=context)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string")
    return raw.strip()


def require_bool(value: Mapping[str, Any], key: str, *, context: str) -> bool:
    """Return a required boolean without accepting integer coercion."""

    raw = require_value(value, key, context=context)
    if type(raw) is not bool:
        raise ConfigError(f"{context}.{key} must be a boolean")
    return raw


def require_int(
    value: Mapping[str, Any], key: str, *, context: str, minimum: int | None = None
) -> int:
    """Return a required integer and optionally enforce a lower bound."""

    raw = require_value(value, key, context=context)
    if type(raw) is not int:
        raise ConfigError(f"{context}.{key} must be an integer")
    if minimum is not None and raw < minimum:
        raise ConfigError(f"{context}.{key} must be >= {minimum}")
    return raw


def require_float(
    value: Mapping[str, Any], key: str, *, context: str, minimum: float | None = None
) -> float:
    """Return a required finite numeric value."""

    raw = require_value(value, key, context=context)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{context}.{key} must be numeric")
    result = float(raw)
    if result != result or result in {float("inf"), float("-inf")}:  # NaN/Infinity
        raise ConfigError(f"{context}.{key} must be finite")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{context}.{key} must be >= {minimum}")
    return result


def resolve_project_path(project_root: Path, raw: str) -> Path:
    """Resolve a config path relative to the project root."""

    path = Path(raw)
    return path if path.is_absolute() else project_root / path
