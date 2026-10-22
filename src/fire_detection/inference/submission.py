"""Strict competition submission generation and validation."""

from __future__ import annotations

import codecs
import json
import math
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fire_detection.inference.oof import canonical_filename
from fire_detection.utils import atomic_write_json

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def list_test_filenames(test_dir: Path, *, expected_count: int | None = None) -> tuple[str, ...]:
    """List supported test images from exactly one directory level."""

    if not test_dir.is_dir():
        raise FileNotFoundError(f"test image directory does not exist: {test_dir}")
    names = tuple(
        sorted(
            path.name
            for path in test_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )
    if not names:
        raise ValueError(f"no supported test images found in {test_dir}")
    if expected_count is not None and len(names) != expected_count:
        raise ValueError(f"expected {expected_count} test images, found {len(names)}")
    return names


def predictions_to_submission(
    predictions: pd.DataFrame,
    *,
    expected_filenames: Iterable[str],
    threshold: float,
    filename_column: str = "filename",
    score_column: str = "probability",
) -> dict[str, int]:
    """Convert probabilities to a strictly matched integer submission mapping."""

    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a non-boolean real number")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    missing_columns = [
        column for column in (filename_column, score_column) if column not in predictions.columns
    ]
    if missing_columns:
        raise ValueError(f"prediction CSV missing columns: {missing_columns}")
    if predictions.empty:
        raise ValueError("predictions must not be empty")

    normalized = predictions[[filename_column, score_column]].copy()
    normalized[filename_column] = [
        canonical_filename(value) for value in normalized[filename_column]
    ]
    if normalized[filename_column].duplicated().any():
        raise ValueError("predictions contain duplicate filenames after path normalization")
    scores: list[float] = []
    for index, value in normalized[score_column].items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{score_column}[{index}] must be a non-boolean real number")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{score_column}[{index}] must be finite and in [0, 1]")
        scores.append(score)
    normalized[score_column] = scores

    expected = tuple(sorted(canonical_filename(name) for name in expected_filenames))
    if len(set(expected)) != len(expected):
        raise ValueError("expected test filenames are not unique")
    actual = set(normalized[filename_column])
    if actual != set(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(f"prediction filename mismatch; missing={missing[:5]}, extra={extra[:5]}")
    score_by_name = dict(zip(normalized[filename_column], normalized[score_column], strict=True))
    return {name: int(score_by_name[name] >= threshold_value) for name in expected}


def write_submission(path: Path, payload: dict[str, int]) -> None:
    """Validate and write a submission as sorted UTF-8 JSON without a BOM."""

    validate_submission_payload(payload, tuple(payload))
    atomic_write_json(path, payload)


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_submission_strict(path: Path) -> dict[str, Any]:
    """Parse UTF-8 JSON while rejecting BOM, duplicate keys, and NaN/Infinity."""

    raw = path.read_bytes()
    if raw.startswith(codecs.BOM_UTF8):
        raise ValueError("submission JSON must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("submission JSON must be valid UTF-8") from exc
    payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("submission JSON must contain one top-level object")
    return payload


def validate_submission_payload(
    payload: dict[str, Any],
    expected_filenames: Iterable[str],
    *,
    expected_count: int | None = None,
) -> None:
    """Validate exact filenames and exact integer 0/1 values."""

    expected = tuple(sorted(canonical_filename(name) for name in expected_filenames))
    if expected_count is not None and len(expected) != expected_count:
        raise ValueError(
            f"expected filename source has {len(expected)} entries, not {expected_count}"
        )
    if len(payload) != len(expected):
        raise ValueError(f"submission has {len(payload)} keys, expected {len(expected)}")
    if set(payload) != set(expected):
        missing = sorted(set(expected).difference(payload))
        extra = sorted(set(payload).difference(expected))
        raise ValueError(f"submission filename mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for filename, value in payload.items():
        if canonical_filename(filename) != filename:
            raise ValueError(f"submission key must be a basename: {filename!r}")
        if type(value) is not int or value not in {0, 1}:
            raise ValueError(f"submission value for {filename!r} must be integer 0 or 1")
