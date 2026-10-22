"""Strict explicit horizontal-flip TTA helpers for Round 3 OOF inference."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from fire_detection.inference.slicing import (
    LoadedTile,
    MappedTileBox,
    TileWindow,
    generate_tile_windows,
)
from fire_detection.models.features import RawBox


class HorizontalFlipTTAError(ValueError):
    """Raised when the fixed horizontal-flip TTA contract cannot be satisfied."""


@dataclass(frozen=True)
class HorizontalFlipTTASpec:
    """Validated two-view TTA settings used by detector and classifier OOF export."""

    config_path: Path
    config_sha256: str
    name: str
    transform: str
    detector_view_reduction: str
    classifier_view_score: str
    aggregation: str
    views: tuple[str, str]

    def provenance(self) -> dict[str, str]:
        """Return stable row-level provenance for a TTA OOF artifact."""

        return {
            "tta_config_sha256": self.config_sha256,
            "tta_transform": self.transform,
            "tta_aggregation": self.aggregation,
        }

    def detector_provenance(self) -> dict[str, str]:
        """Return detector-specific TTA provenance, including within-view reduction."""

        return {
            **self.provenance(),
            "tta_detector_view_reduction": self.detector_view_reduction,
        }

    def classifier_provenance(self) -> dict[str, str]:
        """Return classifier-specific TTA provenance, including the per-view score."""

        return {
            **self.provenance(),
            "tta_classifier_view_score": self.classifier_view_score,
        }

    def manifest_record(self) -> dict[str, object]:
        """Return the immutable algorithm contract recorded in an OOF manifest."""

        return {
            "name": self.name,
            "transform": self.transform,
            "detector_per_view_reduction": self.detector_view_reduction,
            "classifier_per_view_score": self.classifier_view_score,
            "aggregation": self.aggregation,
            "views": list(self.views),
        }


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HorizontalFlipTTAError(f"{field} must be a mapping")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise HorizontalFlipTTAError(
            f"{field} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HorizontalFlipTTAError(f"{field} must be a non-empty string")
    return value.strip()


def load_horizontal_flip_tta_spec(path: Path) -> HorizontalFlipTTASpec:
    """Load the only supported TTA: original/horizontal-flip arithmetic mean."""

    config_path = path.resolve()
    try:
        raw_bytes = config_path.read_bytes()
        raw = yaml.safe_load(raw_bytes)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise HorizontalFlipTTAError(f"cannot read TTA config {config_path}: {exc}") from exc
    root = _mapping(raw, "TTA config")
    _exact_keys(
        root,
        {"schema_version", "experiment", "transform", "detector", "classifier", "aggregation"},
        "TTA config",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise HorizontalFlipTTAError("TTA schema_version must be 1")

    experiment = _mapping(root["experiment"], "experiment")
    transform = _mapping(root["transform"], "transform")
    detector = _mapping(root["detector"], "detector")
    classifier = _mapping(root["classifier"], "classifier")
    aggregation = _mapping(root["aggregation"], "aggregation")
    _exact_keys(experiment, {"name"}, "experiment")
    _exact_keys(transform, {"name"}, "transform")
    _exact_keys(detector, {"per_view_reduction"}, "detector")
    _exact_keys(classifier, {"per_view_score"}, "classifier")
    _exact_keys(aggregation, {"method", "views"}, "aggregation")

    transform_name = _required_string(transform["name"], "transform.name")
    detector_reduction = _required_string(
        detector["per_view_reduction"], "detector.per_view_reduction"
    )
    classifier_score = _required_string(classifier["per_view_score"], "classifier.per_view_score")
    aggregation_name = _required_string(aggregation["method"], "aggregation.method")
    raw_views = aggregation["views"]
    if not isinstance(raw_views, list) or any(not isinstance(view, str) for view in raw_views):
        raise HorizontalFlipTTAError("aggregation.views must be a string list")
    views = tuple(raw_views)

    if transform_name != "horizontal_flip":
        raise HorizontalFlipTTAError("transform.name must be horizontal_flip")
    if detector_reduction != "max_full_slice":
        raise HorizontalFlipTTAError("detector.per_view_reduction must be max_full_slice")
    if classifier_score != "positive_class_probability":
        raise HorizontalFlipTTAError("classifier.per_view_score must be positive_class_probability")
    if aggregation_name != "arithmetic_mean":
        raise HorizontalFlipTTAError(
            "aggregation.method must be arithmetic_mean; TTA max is forbidden"
        )
    if views != ("original", "horizontal_flip"):
        raise HorizontalFlipTTAError(
            "aggregation.views must be exactly [original, horizontal_flip]"
        )

    return HorizontalFlipTTASpec(
        config_path=config_path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        name=_required_string(experiment["name"], "experiment.name"),
        transform=transform_name,
        detector_view_reduction=detector_reduction,
        classifier_view_score=classifier_score,
        aggregation=aggregation_name,
        views=(views[0], views[1]),
    )


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HorizontalFlipTTAError(f"{field} must be a non-boolean real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise HorizontalFlipTTAError(f"{field} must be finite and in [0, 1]")
    return result


def arithmetic_mean_score(original: object, horizontal_flip: object) -> float:
    """Average exactly two view scores; no max or voting is applied across views."""

    values = (
        _probability(original, "original score"),
        _probability(horizontal_flip, "horizontal-flip score"),
    )
    return math.fsum(values) / 2.0


def reduce_detector_view(full_score: object, slice_score: object) -> float:
    """Apply the frozen within-view ``max(full, slice)`` detector reduction."""

    return max(
        _probability(full_score, "full-image score"),
        _probability(slice_score, "slice score"),
    )


def detector_tta_features(
    *,
    original_full_score: object,
    original_slice_score: object,
    horizontal_flip_full_score: object,
    horizontal_flip_slice_score: object,
) -> dict[str, float]:
    """Build explicit detector view scores and their arithmetic-mean TTA score."""

    original = reduce_detector_view(original_full_score, original_slice_score)
    flipped = reduce_detector_view(horizontal_flip_full_score, horizontal_flip_slice_score)
    return {
        "det_original_full_slice_conf": original,
        "det_hflip_max_conf": _probability(
            horizontal_flip_full_score, "horizontal-flip full-image score"
        ),
        "slice_hflip_max_conf": _probability(
            horizontal_flip_slice_score, "horizontal-flip slice score"
        ),
        "det_hflip_full_slice_conf": flipped,
        "det_tta_mean_conf": arithmetic_mean_score(original, flipped),
    }


def classifier_tta_features(
    *, original_probability: object, horizontal_flip_probability: object
) -> dict[str, float]:
    """Build explicit classifier view probabilities and their arithmetic mean."""

    original = _probability(original_probability, "original classifier probability")
    flipped = _probability(horizontal_flip_probability, "horizontal-flip classifier probability")
    return {
        "cls_original_probability": original,
        "cls_hflip_probability": flipped,
        "cls_tta_mean_probability": arithmetic_mean_score(original, flipped),
    }


def _image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = image_shape
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise HorizontalFlipTTAError("image_shape must contain positive integer height and width")
    return height, width


def restore_horizontal_flip_xyxy(
    xyxy: Sequence[object], *, image_shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Map one box from a horizontally flipped view back to original pixels."""

    height, width = _image_shape(image_shape)
    if len(xyxy) != 4:
        raise HorizontalFlipTTAError("xyxy must contain exactly four coordinates")
    values: list[float] = []
    for index, value in enumerate(xyxy):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HorizontalFlipTTAError(f"xyxy[{index}] must be a non-boolean real number")
        number = float(value)
        if not math.isfinite(number):
            raise HorizontalFlipTTAError(f"xyxy[{index}] must be finite")
        values.append(number)
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise HorizontalFlipTTAError("xyxy has reversed coordinates")
    clipped_x1 = max(0.0, min(float(width), x1))
    clipped_x2 = max(0.0, min(float(width), x2))
    clipped_y1 = max(0.0, min(float(height), y1))
    clipped_y2 = max(0.0, min(float(height), y2))
    return (
        float(width) - clipped_x2,
        clipped_y1,
        float(width) - clipped_x1,
        clipped_y2,
    )


def restore_horizontal_flip_box(box: RawBox, *, image_shape: tuple[int, int]) -> RawBox:
    """Restore one flipped-view ``RawBox`` while retaining confidence and class."""

    return RawBox(
        restore_horizontal_flip_xyxy(box.xyxy, image_shape=image_shape),
        box.conf,
        box.class_id,
    )


def restore_horizontal_flip_window(
    window: TileWindow, *, image_shape: tuple[int, int]
) -> dict[str, int]:
    """Describe a transformed tile window in original-image coordinates."""

    height, width = _image_shape(image_shape)
    if not (0 <= window.x1 < window.x2 <= width and 0 <= window.y1 < window.y2 <= height):
        raise HorizontalFlipTTAError("horizontal-flip tile window lies outside the image")
    return {
        "x1": width - window.x2,
        "y1": window.y1,
        "x2": width - window.x1,
        "y2": window.y2,
    }


def raw_horizontal_flip_tile_record(
    window: TileWindow,
    boxes: Sequence[MappedTileBox],
    *,
    image_shape: tuple[int, int],
) -> dict[str, object]:
    """Record flipped-local, flipped-global, and restored-global tile coordinates."""

    restored_boxes = []
    for box in boxes:
        restored_boxes.append(
            {
                "tile_xyxy": list(box.tile_xyxy),
                "horizontal_flip_xyxy": list(box.xyxy),
                "xyxy": list(restore_horizontal_flip_xyxy(box.xyxy, image_shape=image_shape)),
                "conf": box.conf,
                "class_id": box.class_id,
            }
        )
    return {
        "tile_index": window.index,
        "horizontal_flip_window": window.to_dict(),
        "window": restore_horizontal_flip_window(window, image_shape=image_shape),
        "boxes": restored_boxes,
    }


def load_horizontal_flipped_image(path: Path) -> tuple[tuple[int, int], Image.Image]:
    """Decode, EXIF-normalize, and explicitly mirror one image for TTA inference."""

    try:
        with Image.open(path) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = original.size
            flipped = ImageOps.mirror(original)
            original.close()
    except (OSError, UnidentifiedImageError) as exc:
        raise HorizontalFlipTTAError(
            f"cannot read image for horizontal-flip TTA {path}: {exc}"
        ) from exc
    return (height, width), flipped


def image_tiles(image: Image.Image, *, size: int, stride: int) -> tuple[LoadedTile, ...]:
    """Create deterministic detached crops from an already transformed image."""

    width, height = image.size
    windows = generate_tile_windows(width=width, height=height, size=size, stride=stride)
    return tuple(
        LoadedTile(
            window=window,
            image=image.crop((window.x1, window.y1, window.x2, window.y2)),
        )
        for window in windows
    )
