"""Pure feature extraction helpers for detector and classifier predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


class PredictionValidationError(ValueError):
    """Raised when a model result cannot satisfy the OOF prediction contract."""


@dataclass(frozen=True)
class RawBox:
    """One postprocessed YOLO26 end-to-end detection in original-image pixels."""

    xyxy: tuple[float, float, float, float]
    conf: float
    class_id: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["xyxy"] = list(self.xyxy)
        return payload


@dataclass(frozen=True)
class DetectionFeatures:
    """Fixed image-level detector features used by the first OOF baseline."""

    det_max_conf: float
    det_top3_mean: float
    det_box_count: int
    det_largest_box_ratio: float
    det_total_box_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        """Return stable CSV column names and values."""

        return asdict(self)


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionValidationError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise PredictionValidationError(f"{field} must be finite and in [0, 1]")
    return result


def _finite_coordinate(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionValidationError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise PredictionValidationError(f"{field} must be finite")
    return result


def extract_detection_features(
    *,
    xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    image_shape: tuple[int, int],
    count_threshold: float = 0.1,
) -> tuple[DetectionFeatures, tuple[RawBox, ...]]:
    """Convert official YOLO26 end-to-end boxes into fixed image-level features.

    Top-3 means the arithmetic mean of up to three available boxes; missing boxes are not
    zero-padded. Area ratios use the sum of individual box areas and are not clipped to one.
    """

    if len(xyxy) != len(confidences) or len(xyxy) != len(class_ids):
        raise PredictionValidationError(
            "box coordinates, confidences, and class IDs differ in length"
        )
    height, width = image_shape
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise PredictionValidationError(
            "image_shape must contain positive integer height and width"
        )
    threshold = _finite_probability(count_threshold, "count_threshold")
    boxes: list[RawBox] = []
    for index, (coords, raw_conf, raw_class) in enumerate(
        zip(xyxy, confidences, class_ids, strict=True)
    ):
        if len(coords) != 4:
            raise PredictionValidationError(f"xyxy[{index}] must contain four coordinates")
        values = cast(
            tuple[float, float, float, float],
            tuple(
                _finite_coordinate(value, f"xyxy[{index}][{offset}]")
                for offset, value in enumerate(coords)
            ),
        )
        x1, y1, x2, y2 = values
        if x2 < x1 or y2 < y1:
            raise PredictionValidationError(f"xyxy[{index}] has reversed coordinates")
        conf = _finite_probability(raw_conf, f"confidences[{index}]")
        if type(raw_class) is not int or raw_class < 0:
            raise PredictionValidationError(f"class_ids[{index}] must be a non-negative integer")
        boxes.append(RawBox(values, conf, raw_class))

    boxes.sort(key=lambda item: item.conf, reverse=True)
    if not boxes:
        return DetectionFeatures(0.0, 0.0, 0, 0.0, 0.0), ()

    scores = [box.conf for box in boxes]
    image_area = float(height * width)
    areas = [
        max(0.0, min(float(width), box.xyxy[2]) - max(0.0, box.xyxy[0]))
        * max(0.0, min(float(height), box.xyxy[3]) - max(0.0, box.xyxy[1]))
        for box in boxes
    ]
    top3 = scores[:3]
    features = DetectionFeatures(
        det_max_conf=scores[0],
        det_top3_mean=sum(top3) / len(top3),
        det_box_count=sum(score >= threshold for score in scores),
        det_largest_box_ratio=max(areas) / image_area,
        det_total_box_ratio=sum(areas) / image_area,
    )
    return features, tuple(boxes)


def resolve_positive_class_index(
    names: Mapping[int, str] | Sequence[str], positive_name: str = "1"
) -> int:
    """Resolve the classifier positive index by class name, never array position."""

    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    matches = [int(index) for index, name in items if str(name) == positive_name]
    if len(matches) != 1:
        raise PredictionValidationError(
            f"expected exactly one classifier class named {positive_name!r}, found {matches}"
        )
    return matches[0]


def classifier_positive_probability(
    *,
    names: Mapping[int, str] | Sequence[str],
    probabilities: Sequence[float],
    positive_name: str = "1",
) -> float:
    """Return the validated positive-class probability from a classification result."""

    index = resolve_positive_class_index(names, positive_name)
    if index >= len(probabilities):
        raise PredictionValidationError(
            f"positive class index {index} exceeds probability vector length {len(probabilities)}"
        )
    values = [
        _finite_probability(value, f"probabilities[{offset}]")
        for offset, value in enumerate(probabilities)
    ]
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise PredictionValidationError(
            f"classification probabilities must sum to 1, found {total}"
        )
    return values[index]


def tensor_to_list(value: Any) -> list[Any]:
    """Convert an Ultralytics tensor/array-like object to a plain list."""

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "tolist"):
        return cast(list[Any], current.tolist())
    return list(current)


def extract_ultralytics_detection_result(
    result: Any,
) -> tuple[str, DetectionFeatures, tuple[RawBox, ...]]:
    """Extract original-pixel features from one Ultralytics detection ``Results`` object."""

    filename = Path(str(result.path)).name
    boxes = result.boxes
    if boxes is None:
        features, raw = extract_detection_features(
            xyxy=(), confidences=(), class_ids=(), image_shape=tuple(result.orig_shape)
        )
        return filename, features, raw
    xyxy = tensor_to_list(boxes.xyxy)
    confidences = tensor_to_list(boxes.conf)
    class_values = tensor_to_list(boxes.cls)
    class_ids: list[int] = []
    for offset, value in enumerate(class_values):
        numeric = float(value)
        if not numeric.is_integer():
            raise PredictionValidationError(
                f"predicted class ID at {offset} is not integral: {value}"
            )
        class_ids.append(int(numeric))
    features, raw = extract_detection_features(
        xyxy=xyxy,
        confidences=confidences,
        class_ids=class_ids,
        image_shape=tuple(result.orig_shape),
    )
    return filename, features, raw


def extract_ultralytics_classification_result(result: Any) -> tuple[str, float]:
    """Extract the filename and class ``"1"`` probability from Ultralytics results."""

    if result.probs is None:
        raise PredictionValidationError("classification result has no probability vector")
    filename = Path(str(result.path)).name
    probability = classifier_positive_probability(
        names=result.names,
        probabilities=tensor_to_list(result.probs.data),
    )
    return filename, probability
