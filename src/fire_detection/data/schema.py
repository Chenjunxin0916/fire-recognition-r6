"""Strict, path-safe readers for the competition annotations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class DataValidationError(ValueError):
    """Raised when source metadata cannot be used without guessing."""


@dataclass(frozen=True)
class ImageRecord:
    """One COCO image entry."""

    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class AnnotationRecord:
    """One COCO detection annotation."""

    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class CategoryRecord:
    """One COCO category."""

    id: int
    name: str


@dataclass(frozen=True)
class DatasetIndex:
    """Validated immutable view of all source annotations."""

    images: tuple[ImageRecord, ...]
    annotations: tuple[AnnotationRecord, ...]
    categories: tuple[CategoryRecord, ...]
    labels: dict[str, int]

    @property
    def images_by_id(self) -> dict[int, ImageRecord]:
        """Return images keyed by COCO id."""

        return {image.id: image for image in self.images}

    @property
    def images_by_name(self) -> dict[str, ImageRecord]:
        """Return images keyed by normalized filename."""

        return {image.file_name: image for image in self.images}

    @property
    def annotations_by_image(self) -> dict[int, tuple[AnnotationRecord, ...]]:
        """Return annotations grouped by image id."""

        result: dict[int, list[AnnotationRecord]] = {image.id: [] for image in self.images}
        for annotation in self.annotations:
            result[annotation.image_id].append(annotation)
        return {key: tuple(value) for key, value in result.items()}


def normalize_filename(value: str) -> str:
    """Normalize Windows or POSIX separators to a safe basename."""

    if not isinstance(value, str) or not value.strip():
        raise DataValidationError("filename must be a non-empty string")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.name in {"", ".", ".."}:
        raise DataValidationError(f"invalid filename: {value!r}")
    return path.name


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DataValidationError(f"expected a JSON object in {path}")
    return raw


def _strict_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DataValidationError(f"{field} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise DataValidationError(f"{field} must be >= {minimum}, got {value}")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{field} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise DataValidationError(f"{field} must be finite, got {value!r}")
    return result


def _required_list(root: dict[str, Any], field: str) -> list[Any]:
    value = root.get(field)
    if not isinstance(value, list):
        raise DataValidationError(f"COCO field {field!r} must be a list")
    return value


def _load_images(root: dict[str, Any]) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    ids: set[int] = set()
    names: set[str] = set()
    for offset, raw in enumerate(_required_list(root, "images")):
        if not isinstance(raw, dict):
            raise DataValidationError(f"images[{offset}] must be an object")
        image_id = _strict_int(raw.get("id"), f"images[{offset}].id", minimum=0)
        raw_name = raw.get("file_name")
        if not isinstance(raw_name, str):
            raise DataValidationError(f"images[{offset}].file_name must be a string")
        name = normalize_filename(raw_name)
        width = _strict_int(raw.get("width"), f"images[{offset}].width", minimum=1)
        height = _strict_int(raw.get("height"), f"images[{offset}].height", minimum=1)
        if image_id in ids:
            raise DataValidationError(f"duplicate COCO image id: {image_id}")
        if name in names:
            raise DataValidationError(f"duplicate COCO filename after normalization: {name}")
        ids.add(image_id)
        names.add(name)
        records.append(ImageRecord(image_id, name, width, height))
    if not records:
        raise DataValidationError("COCO images must not be empty")
    return tuple(sorted(records, key=lambda item: item.file_name))


def _load_annotations(root: dict[str, Any], image_ids: set[int]) -> tuple[AnnotationRecord, ...]:
    records: list[AnnotationRecord] = []
    ids: set[int] = set()
    for offset, raw in enumerate(_required_list(root, "annotations")):
        if not isinstance(raw, dict):
            raise DataValidationError(f"annotations[{offset}] must be an object")
        annotation_id = _strict_int(raw.get("id"), f"annotations[{offset}].id", minimum=0)
        image_id = _strict_int(raw.get("image_id"), f"annotations[{offset}].image_id", minimum=0)
        category_id = _strict_int(
            raw.get("category_id"), f"annotations[{offset}].category_id", minimum=0
        )
        iscrowd = _strict_int(raw.get("iscrowd", 0), f"annotations[{offset}].iscrowd")
        bbox_raw = raw.get("bbox")
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            raise DataValidationError(f"annotations[{offset}].bbox must contain four numbers")
        bbox = (
            _finite_float(bbox_raw[0], f"annotations[{offset}].bbox[0]"),
            _finite_float(bbox_raw[1], f"annotations[{offset}].bbox[1]"),
            _finite_float(bbox_raw[2], f"annotations[{offset}].bbox[2]"),
            _finite_float(bbox_raw[3], f"annotations[{offset}].bbox[3]"),
        )
        area = _finite_float(raw.get("area"), f"annotations[{offset}].area")
        if annotation_id in ids:
            raise DataValidationError(f"duplicate COCO annotation id: {annotation_id}")
        if image_id not in image_ids:
            raise DataValidationError(
                f"annotation {annotation_id} references unknown image id {image_id}"
            )
        ids.add(annotation_id)
        records.append(AnnotationRecord(annotation_id, image_id, category_id, bbox, area, iscrowd))
    return tuple(sorted(records, key=lambda item: item.id))


def _load_categories(root: dict[str, Any]) -> tuple[CategoryRecord, ...]:
    records: list[CategoryRecord] = []
    ids: set[int] = set()
    for offset, raw in enumerate(_required_list(root, "categories")):
        if not isinstance(raw, dict):
            raise DataValidationError(f"categories[{offset}] must be an object")
        category_id = _strict_int(raw.get("id"), f"categories[{offset}].id", minimum=0)
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise DataValidationError(f"categories[{offset}].name must be non-empty")
        if category_id in ids:
            raise DataValidationError(f"duplicate category id: {category_id}")
        ids.add(category_id)
        records.append(CategoryRecord(category_id, name))
    if not records:
        raise DataValidationError("COCO categories must not be empty")
    return tuple(sorted(records, key=lambda item: item.id))


def _load_labels(path: Path) -> dict[str, int]:
    root = _read_json_object(path)
    labels: dict[str, int] = {}
    for raw_name, raw_label in root.items():
        name = normalize_filename(raw_name)
        label = _strict_int(raw_label, f"image label for {raw_name!r}")
        if label not in {0, 1}:
            raise DataValidationError(f"image label for {raw_name!r} must be 0 or 1")
        if name in labels:
            raise DataValidationError(f"duplicate image label after normalization: {name}")
        labels[name] = label
    if not labels:
        raise DataValidationError("image-level labels must not be empty")
    return labels


def load_dataset_index(coco_json: Path, image_labels_json: Path) -> DatasetIndex:
    """Load and strictly validate COCO and image-level label metadata."""

    root = _read_json_object(coco_json)
    images = _load_images(root)
    image_ids = {image.id for image in images}
    annotations = _load_annotations(root, image_ids)
    categories = _load_categories(root)
    labels = _load_labels(image_labels_json)
    image_names = {image.file_name for image in images}
    if set(labels) != image_names:
        missing = sorted(image_names - set(labels))
        extra = sorted(set(labels) - image_names)
        raise DataValidationError(
            f"COCO/image-label filename mismatch; missing={missing[:10]}, extra={extra[:10]}"
        )
    return DatasetIndex(images, annotations, categories, labels)
