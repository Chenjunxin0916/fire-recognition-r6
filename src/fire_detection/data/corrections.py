"""Explicit, versioned corrections applied only to derived annotations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fire_detection.data.schema import AnnotationRecord, DataValidationError, ImageRecord


@dataclass(frozen=True)
class BboxCorrection:
    """One acknowledged annotation correction."""

    annotation_id: int
    action: str
    reason: str
    expected_bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OrientationCorrection:
    """One acknowledged physical EXIF orientation correction."""

    filename: str
    expected_orientation: int
    action: str
    reason: str


@dataclass(frozen=True)
class CorrectionManifest:
    """Validated correction manifest."""

    version: str
    bbox: dict[int, BboxCorrection]
    orientation: dict[str, OrientationCorrection]


@dataclass(frozen=True)
class AppliedBboxCorrection:
    """Audit record for a correction made in derived labels."""

    annotation_id: int
    filename: str
    action: str
    reason: str
    original_bbox: tuple[float, float, float, float]
    corrected_bbox: tuple[float, float, float, float]


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataValidationError(f"{field} must be a mapping")
    return value


def _as_bbox(value: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise DataValidationError(f"{field} must contain four numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise DataValidationError(f"{field} must contain only real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise DataValidationError(f"{field} must contain only finite numbers")
        result.append(number)
    return (result[0], result[1], result[2], result[3])


def load_correction_manifest(path: Path) -> CorrectionManifest:
    """Load the explicit YAML correction manifest without applying it."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DataValidationError(f"cannot read correction manifest {path}: {exc}") from exc
    root = _as_mapping(raw, "correction manifest")
    version = root.get("version")
    if not isinstance(version, str) or not version:
        raise DataValidationError("correction manifest version must be a non-empty string")

    bbox: dict[int, BboxCorrection] = {}
    bbox_items = root.get("bbox_corrections")
    if not isinstance(bbox_items, list):
        raise DataValidationError("bbox_corrections must be a list")
    for offset, item in enumerate(bbox_items):
        row = _as_mapping(item, f"bbox_corrections[{offset}]")
        annotation_id = row.get("annotation_id")
        if type(annotation_id) is not int or annotation_id < 0:
            raise DataValidationError("correction annotation_id must be a non-negative integer")
        action = row.get("action")
        reason = row.get("reason")
        if action not in {"crop_to_image", "clamp_float_boundary"}:
            raise DataValidationError(f"unsupported bbox correction action: {action!r}")
        if not isinstance(reason, str) or not reason:
            raise DataValidationError("bbox correction reason must be non-empty")
        if annotation_id in bbox:
            raise DataValidationError(f"duplicate correction for annotation {annotation_id}")
        bbox[annotation_id] = BboxCorrection(
            annotation_id,
            action,
            reason,
            _as_bbox(row.get("expected_bbox"), f"bbox_corrections[{offset}].expected_bbox"),
        )

    orientation: dict[str, OrientationCorrection] = {}
    orientation_items = root.get("orientation_corrections")
    if not isinstance(orientation_items, list):
        raise DataValidationError("orientation_corrections must be a list")
    for offset, item in enumerate(orientation_items):
        row = _as_mapping(item, f"orientation_corrections[{offset}]")
        filename = row.get("filename")
        expected = row.get("expected_orientation")
        action = row.get("action")
        reason = row.get("reason")
        if not isinstance(filename, str) or not filename:
            raise DataValidationError("orientation correction filename must be non-empty")
        if type(expected) is not int or expected not in range(2, 9):
            raise DataValidationError("expected_orientation must be an integer from 2 through 8")
        if action != "exif_transpose":
            raise DataValidationError(f"unsupported orientation action: {action!r}")
        if not isinstance(reason, str) or not reason:
            raise DataValidationError("orientation correction reason must be non-empty")
        if filename in orientation:
            raise DataValidationError(f"duplicate orientation correction for {filename}")
        orientation[filename] = OrientationCorrection(filename, expected, action, reason)
    return CorrectionManifest(version, bbox, orientation)


def bbox_overflow(bbox: tuple[float, float, float, float], width: int, height: int) -> float:
    """Return the maximum number of pixels a box extends beyond an image."""

    x, y, box_width, box_height = bbox
    return max(0.0, -x, -y, x + box_width - width, y + box_height - height)


def apply_bbox_correction(
    annotation: AnnotationRecord,
    image: ImageRecord,
    correction: BboxCorrection,
) -> AppliedBboxCorrection:
    """Validate and clip one acknowledged box for derived labels."""

    if annotation.id != correction.annotation_id:
        raise DataValidationError("annotation/correction id mismatch")
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(annotation.bbox, correction.expected_bbox, strict=True)
    ):
        raise DataValidationError(
            f"annotation {annotation.id} no longer matches the reviewed correction manifest"
        )
    x, y, width, height = annotation.bbox
    x1 = min(max(x, 0.0), float(image.width))
    y1 = min(max(y, 0.0), float(image.height))
    x2 = min(max(x + width, 0.0), float(image.width))
    y2 = min(max(y + height, 0.0), float(image.height))
    corrected = (x1, y1, x2 - x1, y2 - y1)
    if corrected[2] <= 0.0 or corrected[3] <= 0.0:
        raise DataValidationError(
            f"correction produces an empty box for annotation {annotation.id}"
        )
    return AppliedBboxCorrection(
        annotation.id,
        image.file_name,
        correction.action,
        correction.reason,
        annotation.bbox,
        corrected,
    )
