"""Strict, training-independent detector inference configuration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from fire_detection.models.training import TrainingSpec


class DetectorInferenceConfigurationError(ValueError):
    """Raised when a detector inference config is missing or contradictory."""


@dataclass(frozen=True)
class SliceInferenceSpec:
    """Validated overlapping-window settings for detector inference."""

    enabled: bool
    size: int
    overlap: float
    stride: int


@dataclass(frozen=True)
class DetectorInferenceSpec:
    """Resolved full-image and optional sliced detector inference settings."""

    config_path: Path
    config_sha256: str
    name: str
    inference_imgsz: int
    conf: float
    max_det: int
    augment: bool
    slicing: SliceInferenceSpec

    def provenance(self) -> dict[str, str | int | float | bool]:
        """Return stable OOF columns that distinguish training and inference settings."""

        return {
            "inference_config_sha256": self.config_sha256,
            "inference_imgsz": self.inference_imgsz,
            "slice_enabled": self.slicing.enabled,
            "slice_size": self.slicing.size,
            "slice_overlap": self.slicing.overlap,
            "slice_stride": self.slicing.stride,
        }

    def raw_inference(self, *, training_imgsz: int) -> dict[str, Any]:
        """Return JSON-serializable raw-box inference provenance."""

        return {
            "conf": self.conf,
            "max_det": self.max_det,
            "augment": self.augment,
            "end2end": True,
            "postprocess": "official_yolo26_topk_conf_filter",
            "training_imgsz": training_imgsz,
            "full_image_imgsz": self.inference_imgsz,
            "config_sha256": self.config_sha256,
            "slice": {
                "enabled": self.slicing.enabled,
                "size": self.slicing.size,
                "overlap": self.slicing.overlap,
                "stride": self.slicing.stride,
                "overlap_box_fusion": "none",
            },
        }


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectorInferenceConfigurationError(f"{field} must be a mapping")
    return dict(value)


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise DetectorInferenceConfigurationError(
            f"{field} keys mismatch: missing={missing}, unknown={unknown}"
        )


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DetectorInferenceConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise DetectorInferenceConfigurationError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise DetectorInferenceConfigurationError(f"{field} must be a boolean")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectorInferenceConfigurationError(f"{field} must be a real number")
    result = float(value)
    above_maximum = result > maximum if maximum_inclusive else result >= maximum
    if not math.isfinite(result) or result < minimum or above_maximum:
        bracket = "]" if maximum_inclusive else ")"
        raise DetectorInferenceConfigurationError(
            f"{field} must be finite and in [{minimum}, {maximum}{bracket}"
        )
    return result


def _slice_stride(size: int, overlap: float) -> int:
    raw_stride = size * (1.0 - overlap)
    stride = round(raw_stride)
    if not math.isclose(raw_stride, stride, rel_tol=0.0, abs_tol=1e-9):
        raise DetectorInferenceConfigurationError(
            "slice.size * (1 - slice.overlap) must produce an integer stride"
        )
    if stride <= 0:
        raise DetectorInferenceConfigurationError("slice stride must be positive")
    return stride


def load_detector_inference_spec(path: Path) -> DetectorInferenceSpec:
    """Load and strictly validate a detector inference YAML file."""

    config_path = path.resolve()
    try:
        raw_bytes = config_path.read_bytes()
        raw = yaml.safe_load(raw_bytes)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DetectorInferenceConfigurationError(
            f"cannot read detector inference config {config_path}: {exc}"
        ) from exc
    root = _mapping(raw, "detector inference config")
    _exact_keys(
        root,
        {"schema_version", "experiment", "full_image", "slice", "prediction"},
        "config",
    )
    if _integer(root["schema_version"], "schema_version", minimum=1) != 1:
        raise DetectorInferenceConfigurationError("schema_version must be 1")

    experiment = _mapping(root["experiment"], "experiment")
    full_image = _mapping(root["full_image"], "full_image")
    slicing = _mapping(root["slice"], "slice")
    prediction = _mapping(root["prediction"], "prediction")
    _exact_keys(experiment, {"name"}, "experiment")
    _exact_keys(full_image, {"imgsz"}, "full_image")
    _exact_keys(slicing, {"enabled", "size", "overlap"}, "slice")
    _exact_keys(prediction, {"conf", "max_det", "augment"}, "prediction")

    size = _integer(slicing["size"], "slice.size", minimum=32)
    overlap = _number(
        slicing["overlap"],
        "slice.overlap",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    slice_spec = SliceInferenceSpec(
        enabled=_boolean(slicing["enabled"], "slice.enabled"),
        size=size,
        overlap=overlap,
        stride=_slice_stride(size, overlap),
    )
    return DetectorInferenceSpec(
        config_path=config_path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        name=_string(experiment["name"], "experiment.name"),
        inference_imgsz=_integer(full_image["imgsz"], "full_image.imgsz", minimum=32),
        conf=_number(prediction["conf"], "prediction.conf", minimum=0.0, maximum=1.0),
        max_det=_integer(prediction["max_det"], "prediction.max_det", minimum=1),
        augment=_boolean(prediction["augment"], "prediction.augment"),
        slicing=slice_spec,
    )


def inference_spec_from_training(spec: TrainingSpec) -> DetectorInferenceSpec:
    """Represent the legacy training-config inference settings in the new contract."""

    size = spec.imgsz
    return DetectorInferenceSpec(
        config_path=spec.config_path,
        config_sha256=hashlib.sha256(spec.config_path.read_bytes()).hexdigest(),
        name=f"{spec.experiment_name}_legacy_full_image",
        inference_imgsz=spec.imgsz,
        conf=spec.inference_conf,
        max_det=spec.inference_max_det,
        augment=spec.inference_augment,
        slicing=SliceInferenceSpec(enabled=False, size=size, overlap=0.0, stride=size),
    )
