"""Strict five-Fold test prediction for the frozen detector/classifier winner."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol

import numpy as np
import pandas as pd

from fire_detection.config import (
    ConfigError,
    load_yaml_mapping,
    require_bool,
    require_float,
    require_int,
    require_mapping,
    require_string,
    resolve_project_path,
)
from fire_detection.environment import collect_environment_report
from fire_detection.inference.detector_config import (
    DetectorInferenceSpec,
    load_detector_inference_spec,
)
from fire_detection.inference.oof import canonical_filename
from fire_detection.inference.slicing import (
    LoadedTile,
    load_image_tiles,
    map_tile_boxes,
    raw_tile_record,
)
from fire_detection.inference.submission import IMAGE_SUFFIXES, list_test_filenames
from fire_detection.inference.test_input import OfficialTestInputError, inspect_image_file
from fire_detection.inference.tta import (
    HorizontalFlipTTAError,
    HorizontalFlipTTASpec,
    arithmetic_mean_score,
    classifier_tta_features,
    detector_tta_features,
    image_tiles,
    load_horizontal_flip_tta_spec,
    load_horizontal_flipped_image,
    raw_horizontal_flip_tile_record,
    restore_horizontal_flip_box,
)
from fire_detection.models.features import (
    PredictionValidationError,
    RawBox,
    classifier_positive_probability,
    extract_ultralytics_detection_result,
    tensor_to_list,
)
from fire_detection.models.offline import (
    configure_imported_ultralytics,
    sha256_file,
    validate_offline_runtime,
)
from fire_detection.models.training import normalize_ultralytics_device
from fire_detection.utils import atomic_write_csv, atomic_write_json

EXPECTED_FOLDS = frozenset(range(5))
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TestPredictionError(ValueError):
    """Raised when final test inference cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class TestPredictionSpec:
    """Decision-complete final test-inference configuration."""

    project_root: Path
    config_path: Path
    experiment_name: str
    seed: int
    selection_manifest: Path
    candidate: str
    offline_manifest: Path
    yolo_config_dir: Path
    ultralytics_version: str
    test_images: Path
    expected_count: int
    detector_inference_config: Path
    detector_training_imgsz: int
    classifier_imgsz: int
    positive_class_name: str
    classifier_augment: bool
    classifier_weight: float
    requested_device: str
    ultralytics_device: str
    output_dir: Path
    predictions_filename: str
    boxes_filename: str
    manifest_filename: str
    schema_version: int = 1
    tta: HorizontalFlipTTASpec | None = None


@dataclass(frozen=True)
class ModelArtifact:
    """One verified final-selection checkpoint and its clean training manifest."""

    task: str
    fold: int
    model_name: str
    imgsz: int
    seed: int
    weights_path: Path
    weights_sha256: str
    run_manifest_path: Path
    run_manifest_sha256: str

    def manifest_record(self, root: Path) -> dict[str, object]:
        """Return portable model provenance for the inference manifest."""

        return {
            "fold": self.fold,
            "model_name": self.model_name,
            "imgsz": self.imgsz,
            "seed": self.seed,
            "weights": {
                "path": _portable_path(self.weights_path, root),
                "sha256": self.weights_sha256,
            },
            "run_manifest": {
                "path": _portable_path(self.run_manifest_path, root),
                "sha256": self.run_manifest_sha256,
            },
        }


@dataclass(frozen=True)
class FinalSelectionContract:
    """The subset of the validated final-selection manifest used for inference."""

    candidate: str
    classifier_weight: float
    manifest_sha256: str
    detector_models: tuple[ModelArtifact, ...]
    classifier_models: tuple[ModelArtifact, ...]
    stable_threshold: float | None = None


@dataclass(frozen=True)
class TestImage:
    """One fully decoded test image with content provenance."""

    filename: str
    path: Path
    sha256: str
    width: int
    height: int

    def manifest_record(self) -> dict[str, object]:
        """Return non-sensitive test-set provenance without an absolute path."""

        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class DetectorFoldPrediction:
    """Full-image and sliced detector result for one model and image."""

    full_score: float
    slice_score: float
    boxes: tuple[RawBox, ...] = ()
    tiles: tuple[Mapping[str, object], ...] = ()
    hflip_full_score: float | None = None
    hflip_slice_score: float | None = None
    hflip_boxes: tuple[RawBox, ...] = ()
    hflip_tiles: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class ClassifierFoldPrediction:
    """Original and explicit horizontal-flip classifier probabilities."""

    original_score: float
    hflip_score: float


class PredictionRunner(Protocol):
    """Injectable boundary around model loading and Ultralytics prediction."""

    def predict_detector(
        self,
        model: ModelArtifact,
        images: Sequence[TestImage],
        inference: DetectorInferenceSpec,
        *,
        device: str,
    ) -> Mapping[str, DetectorFoldPrediction]: ...

    def predict_classifier(
        self,
        model: ModelArtifact,
        images: Sequence[TestImage],
        *,
        device: str,
        imgsz: int,
        positive_class_name: str,
        augment: bool,
    ) -> Mapping[str, float | ClassifierFoldPrediction]: ...

    def environment(self) -> Mapping[str, object]: ...


SelectionValidator = Callable[[Path, Path], object]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TestPredictionError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ConfigError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _plain_filename(value: str, field: str) -> str:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.name != value or windows.name != value or value in {"", ".", ".."}:
        raise ConfigError(f"{field} must be a plain filename")
    return value


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TestPredictionError(f"{name} must be a non-boolean real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TestPredictionError(f"{name} must be finite and in [0, 1]")
    return result


def load_test_prediction_spec(
    path: Path,
    *,
    project_root: Path,
    test_dir_override: Path | None = None,
    output_dir_override: Path | None = None,
    device_override: str | None = None,
    expected_count_override: int | None = None,
) -> TestPredictionSpec:
    """Load and strictly validate the final five-Fold test-inference YAML."""

    root = project_root.resolve()
    config_path = path.resolve() if path.is_absolute() else (root / path).resolve()
    raw = load_yaml_mapping(config_path)
    schema_version = require_int(raw, "schema_version", context="predict_test", minimum=1)
    if schema_version not in {1, 2}:
        raise ConfigError("predict_test.schema_version must be 1 or 2")
    expected_root_keys = {
        "schema_version",
        "experiment",
        "selection",
        "offline",
        "data",
        "detector",
        "classifier",
        "fusion",
        "runtime",
        "output",
    }
    if schema_version == 2:
        expected_root_keys.add("tta")
    _exact_keys(
        raw,
        expected_root_keys,
        "predict_test",
    )
    experiment = require_mapping(raw, "experiment", context="predict_test")
    selection = require_mapping(raw, "selection", context="predict_test")
    offline = require_mapping(raw, "offline", context="predict_test")
    data = require_mapping(raw, "data", context="predict_test")
    detector = require_mapping(raw, "detector", context="predict_test")
    classifier = require_mapping(raw, "classifier", context="predict_test")
    fusion = require_mapping(raw, "fusion", context="predict_test")
    runtime = require_mapping(raw, "runtime", context="predict_test")
    output = require_mapping(raw, "output", context="predict_test")
    tta_config: Mapping[str, Any] | None = None
    if schema_version == 2:
        tta_config = require_mapping(raw, "tta", context="predict_test")
        _exact_keys(tta_config, {"config"}, "predict_test.tta")
    _exact_keys(experiment, {"name", "seed"}, "predict_test.experiment")
    _exact_keys(selection, {"manifest", "candidate"}, "predict_test.selection")
    _exact_keys(
        offline,
        {"manifest", "yolo_config_dir", "ultralytics_version"},
        "predict_test.offline",
    )
    _exact_keys(data, {"test_images", "expected_count"}, "predict_test.data")
    _exact_keys(detector, {"inference_config", "training_imgsz"}, "predict_test.detector")
    _exact_keys(
        classifier,
        {"imgsz", "positive_class_name", "augment"},
        "predict_test.classifier",
    )
    _exact_keys(fusion, {"method", "classifier_weight"}, "predict_test.fusion")
    _exact_keys(runtime, {"device"}, "predict_test.runtime")
    _exact_keys(
        output,
        {"directory", "predictions_filename", "boxes_filename", "manifest_filename"},
        "predict_test.output",
    )

    if require_string(fusion, "method", context="predict_test.fusion") != "weighted_average":
        raise ConfigError("predict_test.fusion.method must be weighted_average")
    classifier_weight = require_float(
        fusion, "classifier_weight", context="predict_test.fusion", minimum=0.0
    )
    if classifier_weight > 1.0:
        raise ConfigError("predict_test.fusion.classifier_weight must be <= 1.0")
    if schema_version == 1 and classifier_weight != 0.5:
        raise ConfigError("predict_test.fusion.classifier_weight must be exactly 0.5")
    if require_bool(classifier, "augment", context="predict_test.classifier") is not False:
        raise ConfigError(
            "predict_test.classifier.augment must be false; built-in Ultralytics TTA is not allowed"
        )
    positive_name = require_string(
        classifier, "positive_class_name", context="predict_test.classifier"
    )
    if positive_name != "1":
        raise ConfigError("predict_test.classifier.positive_class_name must be '1'")

    configured_count = require_int(data, "expected_count", context="predict_test.data", minimum=1)
    if expected_count_override is not None:
        if type(expected_count_override) is not int or expected_count_override < 1:
            raise ConfigError("expected_count override must be a positive integer")
        expected_count = expected_count_override
    else:
        expected_count = configured_count
    requested_device = device_override or require_string(
        runtime, "device", context="predict_test.runtime"
    )
    test_images = test_dir_override or resolve_project_path(
        root, require_string(data, "test_images", context="predict_test.data")
    )
    output_dir = output_dir_override or resolve_project_path(
        root, require_string(output, "directory", context="predict_test.output")
    )
    filenames = (
        _plain_filename(
            require_string(output, key, context="predict_test.output"),
            f"predict_test.output.{key}",
        )
        for key in ("predictions_filename", "boxes_filename", "manifest_filename")
    )
    predictions_filename, boxes_filename, manifest_filename = tuple(filenames)
    if len({predictions_filename, boxes_filename, manifest_filename}) != 3:
        raise ConfigError("predict_test output filenames must be distinct")

    tta_spec: HorizontalFlipTTASpec | None = None
    if tta_config is not None:
        tta_path = resolve_project_path(
            root,
            require_string(tta_config, "config", context="predict_test.tta"),
        ).resolve()
        try:
            tta_spec = load_horizontal_flip_tta_spec(tta_path)
        except HorizontalFlipTTAError as exc:
            raise ConfigError(f"invalid predict_test TTA config: {exc}") from exc

    return TestPredictionSpec(
        project_root=root,
        config_path=config_path,
        schema_version=schema_version,
        experiment_name=require_string(experiment, "name", context="predict_test.experiment"),
        seed=require_int(experiment, "seed", context="predict_test.experiment", minimum=0),
        selection_manifest=resolve_project_path(
            root, require_string(selection, "manifest", context="predict_test.selection")
        ).resolve(),
        candidate=require_string(selection, "candidate", context="predict_test.selection"),
        offline_manifest=resolve_project_path(
            root, require_string(offline, "manifest", context="predict_test.offline")
        ).resolve(),
        yolo_config_dir=resolve_project_path(
            root, require_string(offline, "yolo_config_dir", context="predict_test.offline")
        ).resolve(),
        ultralytics_version=require_string(
            offline, "ultralytics_version", context="predict_test.offline"
        ),
        test_images=test_images.resolve(),
        expected_count=expected_count,
        detector_inference_config=resolve_project_path(
            root,
            require_string(detector, "inference_config", context="predict_test.detector"),
        ).resolve(),
        detector_training_imgsz=require_int(
            detector, "training_imgsz", context="predict_test.detector", minimum=32
        ),
        classifier_imgsz=require_int(
            classifier, "imgsz", context="predict_test.classifier", minimum=32
        ),
        positive_class_name=positive_name,
        classifier_augment=False,
        classifier_weight=classifier_weight,
        tta=tta_spec,
        requested_device=requested_device,
        ultralytics_device=normalize_ultralytics_device(requested_device),
        output_dir=output_dir.resolve(),
        predictions_filename=predictions_filename,
        boxes_filename=boxes_filename,
        manifest_filename=manifest_filename,
    )


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise TestPredictionError(f"artifact must be inside the project root: {path}") from exc


def _resolve_portable_path(value: object, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TestPredictionError(f"{name}.path must be a non-empty string")
    raw = value.strip()
    portable = PurePosixPath(raw.replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in portable.parts:
        raise TestPredictionError(f"{name}.path must be repository-relative")
    resolved = (root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise TestPredictionError(f"{name}.path escapes the project root") from exc
    return resolved


def _sha256_value(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise TestPredictionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _verified_record(
    value: object,
    *,
    root: Path,
    name: str,
    expected_path: Path | None = None,
) -> tuple[Path, str]:
    record = _mapping(value, name)
    path = _resolve_portable_path(record.get("path"), root=root, name=name)
    if expected_path is not None and path != expected_path.resolve():
        raise TestPredictionError(f"{name}.path does not match the required artifact")
    digest = _sha256_value(record.get("sha256"), f"{name}.sha256")
    if not path.is_file():
        raise TestPredictionError(f"{name} does not exist: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise TestPredictionError(f"{name}.sha256 drifted: expected {digest}, found {actual}")
    return path, digest


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise TestPredictionError(f"{name} contains non-standard JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TestPredictionError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TestPredictionError(f"cannot read {name} {path}: {exc}") from exc
    return dict(_mapping(value, name))


def _strict_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TestPredictionError(f"{name} must be an integer >= {minimum}")
    return value


def _model_artifacts(
    value: object,
    *,
    task: str,
    root: Path,
    expected_seed: int,
    expected_imgsz: int,
) -> tuple[ModelArtifact, ...]:
    if not isinstance(value, list):
        raise TestPredictionError(f"components.{task}.models must be a list")
    artifacts: list[ModelArtifact] = []
    seen_folds: set[int] = set()
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    manifest_task = "detect" if task == "detector" else "classify"
    for offset, raw_model in enumerate(value):
        model = _mapping(raw_model, f"components.{task}.models[{offset}]")
        fold = _strict_integer(model.get("fold"), f"components.{task}.models[{offset}].fold")
        if fold not in EXPECTED_FOLDS or fold in seen_folds:
            raise TestPredictionError(
                f"{task} models contain a missing/duplicate/invalid Fold {fold}"
            )
        weights_path, weights_hash = _verified_record(
            model.get("weights"), root=root, name=f"{task} Fold {fold} weights"
        )
        run_path, run_hash = _verified_record(
            model.get("run_manifest"), root=root, name=f"{task} Fold {fold} run manifest"
        )
        if weights_path.name != "last.pt":
            raise TestPredictionError(f"{task} Fold {fold} must use last.pt")
        run = _load_json_object(run_path, f"{task} Fold {fold} run manifest")
        expected_fields: dict[str, object] = {
            "schema_version": 1,
            "status": "complete",
            "task": manifest_task,
            "validation_fold": fold,
            "last_weights": _portable_path(weights_path, root),
            "last_weights_sha256": weights_hash,
            "imgsz": expected_imgsz,
            "seed": expected_seed,
        }
        for field, expected in expected_fields.items():
            actual = run.get(field)
            if type(actual) is not type(expected) or actual != expected:
                raise TestPredictionError(
                    f"{task} Fold {fold} run manifest {field} mismatch: "
                    f"expected {expected!r}, found {actual!r}"
                )
        git = _mapping(run.get("git"), f"{task} Fold {fold} run manifest git")
        if git.get("dirty") is not False:
            raise TestPredictionError(f"{task} Fold {fold} run manifest is dirty")
        model_name = run.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise TestPredictionError(f"{task} Fold {fold} model_name is invalid")
        if weights_path in seen_paths or weights_hash in seen_hashes:
            raise TestPredictionError(f"{task} Fold checkpoints must be distinct")
        seen_folds.add(fold)
        seen_paths.add(weights_path)
        seen_hashes.add(weights_hash)
        artifacts.append(
            ModelArtifact(
                task=task,
                fold=fold,
                model_name=model_name,
                imgsz=expected_imgsz,
                seed=expected_seed,
                weights_path=weights_path,
                weights_sha256=weights_hash,
                run_manifest_path=run_path,
                run_manifest_sha256=run_hash,
            )
        )
    if seen_folds != EXPECTED_FOLDS:
        raise TestPredictionError(
            f"{task} models must contain exactly Fold 0..4, found {sorted(seen_folds)}"
        )
    return tuple(sorted(artifacts, key=lambda item: item.fold))


def _default_selection_validator(path: Path, project_root: Path) -> object:
    from fire_detection.ensemble import selection

    validator = getattr(selection, "validate_final_selection_manifest", None)
    if validator is None or not callable(validator):
        raise TestPredictionError(
            "final-selection public validator is unavailable: validate_final_selection_manifest"
        )
    return validator(path, project_root=project_root)


def _validate_selection_tta(
    manifest: Mapping[str, Any],
    *,
    spec: TestPredictionSpec,
) -> float | None:
    """Match the prediction TTA config and stable threshold to the frozen selection."""

    raw_tta = manifest.get("tta")
    if spec.tta is None:
        if raw_tta is not None:
            raise TestPredictionError("schema-v1 prediction cannot use a TTA final selection")
        return None
    selection_tta = _mapping(raw_tta, "final-selection TTA")
    expected_contract = spec.tta.manifest_record()
    if set(selection_tta) != {*expected_contract, "config"}:
        raise TestPredictionError("final-selection TTA fields differ from the frozen contract")
    actual_contract = {key: value for key, value in selection_tta.items() if key != "config"}
    if actual_contract != expected_contract:
        raise TestPredictionError("final-selection TTA algorithm contract drifted")
    _tta_path, tta_digest = _verified_record(
        selection_tta.get("config"),
        root=spec.project_root,
        name="final-selection TTA config",
        expected_path=spec.tta.config_path,
    )
    if tta_digest != spec.tta.config_sha256:
        raise TestPredictionError("final-selection TTA config differs from prediction config")

    profiles = _mapping(manifest.get("threshold_profiles"), "final-selection threshold profiles")
    stable = _mapping(profiles.get("stable"), "final-selection stable threshold profile")
    return _probability(stable.get("threshold"), "final-selection stable threshold")


def load_final_selection_contract(
    spec: TestPredictionSpec,
    *,
    validator: SelectionValidator | None = None,
) -> FinalSelectionContract:
    """Validate final selection, then independently recheck ten clean checkpoints."""

    if not spec.selection_manifest.is_file():
        raise TestPredictionError(
            f"final-selection manifest does not exist: {spec.selection_manifest}"
        )
    validated = (validator or _default_selection_validator)(
        spec.selection_manifest, spec.project_root
    )
    if isinstance(validated, tuple):
        if not validated:
            raise TestPredictionError("final-selection validator returned an empty tuple")
        validated = validated[0]
    manifest = _mapping(validated, "validated final-selection manifest")
    selection_value = manifest.get("selection")
    if isinstance(selection_value, Mapping):
        candidate = selection_value.get("candidate")
    else:
        candidate = manifest.get("candidate")
    if candidate != spec.candidate:
        raise TestPredictionError(
            f"final-selection candidate mismatch: expected {spec.candidate!r}, found {candidate!r}"
        )
    fusion = _mapping(manifest.get("fusion"), "final-selection fusion")
    if fusion.get("method") != "weighted_average":
        raise TestPredictionError("final-selection fusion method must be weighted_average")
    classifier_weight = _probability(
        fusion.get("classifier_weight"), "final-selection classifier_weight"
    )
    if classifier_weight != spec.classifier_weight:
        raise TestPredictionError(
            "final-selection classifier weight differs from prediction config"
        )
    detector_weight = _probability(fusion.get("detector_weight"), "final-selection detector_weight")
    if not math.isclose(
        detector_weight,
        1.0 - classifier_weight,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise TestPredictionError("final-selection detector weight is inconsistent")
    if spec.tta is None:
        if classifier_weight != 0.5:
            raise TestPredictionError("schema-v1 final-selection classifier weight must be 0.5")
    else:
        expected_score_columns = {
            "detector_score_column": "det_tta_mean_conf",
            "classifier_score_column": "cls_tta_mean_probability",
        }
        for field, expected in expected_score_columns.items():
            if fusion.get(field) != expected:
                raise TestPredictionError(
                    f"final-selection fusion {field} must be {expected!r} for TTA"
                )
    stable_threshold = _validate_selection_tta(manifest, spec=spec)

    components = _mapping(manifest.get("components"), "final-selection components")
    detector = _mapping(components.get("detector"), "final-selection detector component")
    classifier = _mapping(components.get("classifier"), "final-selection classifier component")
    if "inference_config" in detector:
        _verified_record(
            detector.get("inference_config"),
            root=spec.project_root,
            name="final-selection detector inference config",
            expected_path=spec.detector_inference_config,
        )
    detector_models = _model_artifacts(
        detector.get("models"),
        task="detector",
        root=spec.project_root,
        expected_seed=spec.seed,
        expected_imgsz=spec.detector_training_imgsz,
    )
    classifier_models = _model_artifacts(
        classifier.get("models"),
        task="classifier",
        root=spec.project_root,
        expected_seed=spec.seed,
        expected_imgsz=spec.classifier_imgsz,
    )
    all_hashes = [item.weights_sha256 for item in (*detector_models, *classifier_models)]
    if len(set(all_hashes)) != 10:
        raise TestPredictionError("final selection must bind ten distinct checkpoint hashes")
    return FinalSelectionContract(
        candidate=str(candidate),
        classifier_weight=classifier_weight,
        manifest_sha256=sha256_file(spec.selection_manifest),
        detector_models=detector_models,
        classifier_models=classifier_models,
        stable_threshold=stable_threshold,
    )


def validate_test_image_paths(
    paths: Sequence[Path], *, expected_count: int
) -> tuple[TestImage, ...]:
    """Fully decode, hash, and uniquely name an explicit flat test-image set."""

    if type(expected_count) is not int or expected_count < 1:
        raise TestPredictionError("expected_count must be a positive integer")
    if len(paths) != expected_count:
        raise TestPredictionError(f"expected {expected_count} test images, found {len(paths)}")
    images: list[TestImage] = []
    names: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise TestPredictionError(f"test image does not exist: {path}")
        filename = canonical_filename(path.name)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise TestPredictionError(f"unsupported test image extension: {path}")
        if filename in names:
            raise TestPredictionError(
                f"test images contain duplicate filename after path normalization: {filename}"
            )
        try:
            record = inspect_image_file(path)
        except OfficialTestInputError as exc:
            raise TestPredictionError(str(exc)) from exc
        images.append(
            TestImage(
                filename,
                path.resolve(),
                str(record["sha256"]),
                int(record["width"]),
                int(record["height"]),
            )
        )
        names.add(filename)
    return tuple(sorted(images, key=lambda item: item.filename))


def inspect_test_images(test_dir: Path, *, expected_count: int) -> tuple[TestImage, ...]:
    """Enumerate exactly one directory level and fully validate every test image."""

    filenames = list_test_filenames(test_dir, expected_count=expected_count)
    return validate_test_image_paths(
        tuple(test_dir / filename for filename in filenames), expected_count=expected_count
    )


def _normalize_result_mapping(
    values: Mapping[str, Any], *, expected_names: Sequence[str], name: str
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_name, value in values.items():
        filename = canonical_filename(raw_name)
        if filename in normalized:
            raise TestPredictionError(f"{name} contains duplicate filename {filename!r}")
        normalized[filename] = value
    expected = set(expected_names)
    if set(normalized) != expected:
        missing = sorted(expected.difference(normalized))
        extra = sorted(set(normalized).difference(expected))
        raise TestPredictionError(
            f"{name} filename mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    return normalized


def aggregate_fold_predictions(
    filenames: Sequence[str],
    detector: Mapping[int, Mapping[str, DetectorFoldPrediction]],
    classifier: Mapping[int, Mapping[str, float | ClassifierFoldPrediction]],
    *,
    classifier_weight: float,
    tta: HorizontalFlipTTASpec | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Build deterministic five-Fold probabilities under schema-v1 or frozen flip TTA."""

    if set(detector) != EXPECTED_FOLDS or set(classifier) != EXPECTED_FOLDS:
        raise TestPredictionError(
            "prediction inputs must contain exactly detector/classifier Fold 0..4"
        )
    weight = _probability(classifier_weight, "classifier_weight")
    if tta is None and weight != 0.5:
        raise TestPredictionError("classifier_weight must be exactly 0.5")
    if tta is None and threshold is not None:
        raise TestPredictionError("schema-v1 aggregation cannot apply a threshold")
    if tta is not None and threshold is None:
        raise TestPredictionError("TTA aggregation requires the frozen stable threshold")
    stable_threshold = None if threshold is None else _probability(threshold, "stable threshold")
    names = tuple(sorted(canonical_filename(name) for name in filenames))
    if len(set(names)) != len(names):
        raise TestPredictionError("prediction filenames are not unique")
    detector_rows = {
        fold: _normalize_result_mapping(
            detector[fold], expected_names=names, name=f"detector Fold {fold}"
        )
        for fold in sorted(EXPECTED_FOLDS)
    }
    classifier_rows = {
        fold: _normalize_result_mapping(
            classifier[fold], expected_names=names, name=f"classifier Fold {fold}"
        )
        for fold in sorted(EXPECTED_FOLDS)
    }
    rows: list[dict[str, object]] = []
    for filename in names:
        row: dict[str, object] = {"filename": filename}
        detector_scores: list[float] = []
        classifier_scores: list[float] = []
        for fold in sorted(EXPECTED_FOLDS):
            prediction = detector_rows[fold][filename]
            if not isinstance(prediction, DetectorFoldPrediction):
                raise TestPredictionError(
                    f"detector Fold {fold} result for {filename} has the wrong type"
                )
            full = _probability(prediction.full_score, f"detector Fold {fold} full score")
            sliced = _probability(prediction.slice_score, f"detector Fold {fold} slice score")
            classifier_prediction = classifier_rows[fold][filename]
            if tta is None:
                score = max(full, sliced)
                cls_score = _probability(classifier_prediction, f"classifier Fold {fold} score")
                row[f"detector_fold{fold}_full"] = full
                row[f"detector_fold{fold}_slice"] = sliced
                row[f"detector_fold{fold}"] = score
                row[f"classifier_fold{fold}"] = cls_score
            else:
                if prediction.hflip_full_score is None or prediction.hflip_slice_score is None:
                    raise TestPredictionError(
                        f"detector Fold {fold} result for {filename} omits horizontal-flip scores"
                    )
                if not isinstance(classifier_prediction, ClassifierFoldPrediction):
                    raise TestPredictionError(
                        f"classifier Fold {fold} result for {filename} omits horizontal-flip score"
                    )
                detector_features = detector_tta_features(
                    original_full_score=full,
                    original_slice_score=sliced,
                    horizontal_flip_full_score=prediction.hflip_full_score,
                    horizontal_flip_slice_score=prediction.hflip_slice_score,
                )
                classifier_features = classifier_tta_features(
                    original_probability=classifier_prediction.original_score,
                    horizontal_flip_probability=classifier_prediction.hflip_score,
                )
                score = detector_features["det_tta_mean_conf"]
                cls_score = classifier_features["cls_tta_mean_probability"]
                prefix = f"detector_fold{fold}"
                row[f"{prefix}_original_full"] = full
                row[f"{prefix}_original_slice"] = sliced
                row[f"{prefix}_original"] = detector_features["det_original_full_slice_conf"]
                row[f"{prefix}_hflip_full"] = detector_features["det_hflip_max_conf"]
                row[f"{prefix}_hflip_slice"] = detector_features["slice_hflip_max_conf"]
                row[f"{prefix}_hflip"] = detector_features["det_hflip_full_slice_conf"]
                row[f"{prefix}_tta_mean"] = score
                cls_prefix = f"classifier_fold{fold}"
                row[f"{cls_prefix}_original"] = classifier_features["cls_original_probability"]
                row[f"{cls_prefix}_hflip"] = classifier_features["cls_hflip_probability"]
                row[f"{cls_prefix}_tta_mean"] = cls_score
            detector_scores.append(score)
            classifier_scores.append(cls_score)
        detector_mean = math.fsum(detector_scores) / len(EXPECTED_FOLDS)
        classifier_mean = math.fsum(classifier_scores) / len(EXPECTED_FOLDS)
        probability = (1.0 - weight) * detector_mean + weight * classifier_mean
        row["detector_mean"] = _probability(detector_mean, "detector_mean")
        row["classifier_mean"] = _probability(classifier_mean, "classifier_mean")
        row["probability"] = _probability(probability, "probability")
        if stable_threshold is not None:
            row["threshold"] = stable_threshold
            row["prediction"] = int(probability >= stable_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


@contextmanager
def _source_manifest(images: Sequence[TestImage]) -> Any:
    with tempfile.TemporaryDirectory(prefix="fire-test-sources-") as temporary:
        path = Path(temporary) / "sources.txt"
        path.write_text(
            "".join(f"{image.path}\n" for image in images), encoding="utf-8", newline="\n"
        )
        yield path


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        return


def _verify_detector_model(model: Any) -> None:
    raw_names = model.names
    items = raw_names.items() if isinstance(raw_names, Mapping) else enumerate(raw_names)
    names = {int(index): str(name) for index, name in items}
    if names != {0: "fire"}:
        raise PredictionValidationError(
            f"detector class mapping must be {{0: 'fire'}}, found {names}"
        )
    head = model.model.model[-1]
    if not bool(getattr(head, "end2end", False)):
        raise PredictionValidationError("selected detector checkpoint is not YOLO26 end-to-end")


def _sliced_prediction(
    model: Any,
    image: TestImage,
    inference: DetectorInferenceSpec,
    *,
    device: str,
) -> tuple[float, tuple[Mapping[str, object], ...]]:
    image_shape, tiles = load_image_tiles(
        image.path, size=inference.slicing.size, stride=inference.slicing.stride
    )
    try:
        maximum = 0.0
        raw_tiles: list[Mapping[str, object]] = []
        for tile in tiles:
            results = list(
                model.predict(
                    source=tile.image,
                    stream=True,
                    imgsz=inference.slicing.size,
                    conf=inference.conf,
                    max_det=inference.max_det,
                    augment=False,
                    device=device,
                    verbose=False,
                    save=False,
                )
            )
            if len(results) != 1:
                raise TestPredictionError(
                    f"detector Fold tile inference returned {len(results)} results for "
                    f"{image.filename} tile {tile.window.index}"
                )
            result = results[0]
            expected_shape = (tile.window.height, tile.window.width)
            if tuple(int(value) for value in result.orig_shape) != expected_shape:
                raise TestPredictionError(
                    f"tile shape mismatch for {image.filename} tile {tile.window.index}"
                )
            _name, _features, boxes = extract_ultralytics_detection_result(result)
            if any(box.class_id != 0 for box in boxes):
                raise TestPredictionError(
                    f"detector produced a non-fire class for {image.filename}"
                )
            if boxes:
                maximum = max(maximum, boxes[0].conf)
            mapped = map_tile_boxes(tile.window, boxes, image_shape=image_shape)
            raw_tiles.append(raw_tile_record(tile.window, mapped))
        return maximum, tuple(raw_tiles)
    finally:
        for tile in tiles:
            tile.image.close()


def _horizontal_flip_detector_prediction(
    model: Any,
    image: TestImage,
    inference: DetectorInferenceSpec,
    *,
    device: str,
) -> tuple[
    float,
    float,
    tuple[RawBox, ...],
    tuple[Mapping[str, object], ...],
]:
    """Infer one explicit flipped view and restore every coordinate to the original image."""

    image_shape, flipped = load_horizontal_flipped_image(image.path)
    tiles: tuple[LoadedTile, ...] = ()
    try:
        results = list(
            model.predict(
                source=flipped,
                stream=True,
                imgsz=inference.inference_imgsz,
                conf=inference.conf,
                max_det=inference.max_det,
                augment=False,
                device=device,
                verbose=False,
                save=False,
            )
        )
        if len(results) != 1:
            raise TestPredictionError(
                f"detector horizontal-flip inference returned {len(results)} results for "
                f"{image.filename}"
            )
        result = results[0]
        if tuple(int(value) for value in result.orig_shape) != image_shape:
            raise TestPredictionError(
                f"horizontal-flip full-image shape mismatch for {image.filename}"
            )
        _name, full_features, full_boxes = extract_ultralytics_detection_result(result)
        if any(box.class_id != 0 for box in full_boxes):
            raise TestPredictionError(
                f"detector produced a non-fire class for flipped {image.filename}"
            )
        restored_boxes = tuple(
            restore_horizontal_flip_box(box, image_shape=image_shape) for box in full_boxes
        )

        maximum = 0.0
        raw_tiles: list[Mapping[str, object]] = []
        tiles = image_tiles(flipped, size=inference.slicing.size, stride=inference.slicing.stride)
        for tile in tiles:
            tile_results = list(
                model.predict(
                    source=tile.image,
                    stream=True,
                    imgsz=inference.slicing.size,
                    conf=inference.conf,
                    max_det=inference.max_det,
                    augment=False,
                    device=device,
                    verbose=False,
                    save=False,
                )
            )
            if len(tile_results) != 1:
                raise TestPredictionError(
                    "detector horizontal-flip tile inference returned "
                    f"{len(tile_results)} results for {image.filename} tile {tile.window.index}"
                )
            tile_result = tile_results[0]
            expected_shape = (tile.window.height, tile.window.width)
            if tuple(int(value) for value in tile_result.orig_shape) != expected_shape:
                raise TestPredictionError(
                    f"horizontal-flip tile shape mismatch for {image.filename} "
                    f"tile {tile.window.index}"
                )
            _tile_name, _features, tile_boxes = extract_ultralytics_detection_result(tile_result)
            if any(box.class_id != 0 for box in tile_boxes):
                raise TestPredictionError(
                    f"detector produced a non-fire class for flipped {image.filename}"
                )
            if tile_boxes:
                maximum = max(maximum, tile_boxes[0].conf)
            mapped = map_tile_boxes(tile.window, tile_boxes, image_shape=image_shape)
            raw_tiles.append(
                raw_horizontal_flip_tile_record(
                    tile.window,
                    mapped,
                    image_shape=image_shape,
                )
            )
        return full_features.det_max_conf, maximum, restored_boxes, tuple(raw_tiles)
    finally:
        for tile in tiles:
            tile.image.close()
        flipped.close()


def _horizontal_flip_classifier_probability(
    model: Any,
    image: TestImage,
    *,
    device: str,
    imgsz: int,
    positive_class_name: str,
) -> float:
    """Infer exactly one mirrored classifier view without built-in Ultralytics TTA."""

    _image_shape, flipped = load_horizontal_flipped_image(image.path)
    try:
        results = list(
            model.predict(
                source=flipped,
                stream=True,
                imgsz=imgsz,
                augment=False,
                device=device,
                verbose=False,
                save=False,
            )
        )
        if len(results) != 1 or results[0].probs is None:
            raise TestPredictionError(
                f"classifier horizontal-flip inference is invalid for {image.filename}"
            )
        result = results[0]
        return classifier_positive_probability(
            names=result.names,
            probabilities=tensor_to_list(result.probs.data),
            positive_name=positive_class_name,
        )
    finally:
        flipped.close()


class UltralyticsPredictionRunner:
    """Offline Ultralytics implementation of the injectable prediction boundary."""

    def __init__(self, spec: TestPredictionSpec) -> None:
        runtime = validate_offline_runtime(
            project_root=spec.project_root,
            manifest_path=spec.offline_manifest,
            yolo_config_dir=spec.yolo_config_dir,
            required_assets=("yolo26s.pt", "yolo26s-cls.pt", "yolo26n.pt", "Arial.ttf"),
            expected_ultralytics_version=spec.ultralytics_version,
        )
        configure_imported_ultralytics(runtime)
        import torch

        from ultralytics import YOLO

        if spec.ultralytics_device != "cpu" and not torch.cuda.is_available():
            raise TestPredictionError(
                f"CUDA device {spec.requested_device!r} was requested but CUDA is unavailable"
            )
        random.seed(spec.seed)
        np.random.seed(spec.seed)
        torch.manual_seed(spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.seed)
        self._spec = spec
        self._yolo = YOLO
        self._environment = collect_environment_report().to_dict()
        self._active_model_key: tuple[str, str, str] | None = None
        self._active_model: Any | None = None

    def _activate_model(self, model: ModelArtifact, *, detector: bool) -> Any:
        """Keep one checkpoint resident while the official runner persists its shards."""

        key = (model.task, str(model.weights_path), model.weights_sha256)
        if self._active_model_key == key and self._active_model is not None:
            return self._active_model
        self.close()
        loaded = self._yolo(str(model.weights_path), verbose=False)
        try:
            if detector:
                _verify_detector_model(loaded)
        except BaseException:
            _release_model(loaded)
            raise
        self._active_model_key = key
        self._active_model = loaded
        return loaded

    def close(self) -> None:
        """Release the currently resident checkpoint, if any."""

        loaded = self._active_model
        self._active_model = None
        self._active_model_key = None
        if loaded is not None:
            _release_model(loaded)

    def predict_detector(
        self,
        model: ModelArtifact,
        images: Sequence[TestImage],
        inference: DetectorInferenceSpec,
        *,
        device: str,
    ) -> Mapping[str, DetectorFoldPrediction]:
        yolo = self._activate_model(model, detector=True)
        expected = {image.filename: image for image in images}
        full: dict[str, tuple[float, tuple[RawBox, ...]]] = {}
        try:
            with _source_manifest(images) as sources:
                results = yolo.predict(
                    source=str(sources),
                    stream=True,
                    imgsz=inference.inference_imgsz,
                    conf=inference.conf,
                    max_det=inference.max_det,
                    augment=False,
                    device=device,
                    verbose=False,
                    save=False,
                )
                for result in results:
                    filename, features, boxes = extract_ultralytics_detection_result(result)
                    if filename not in expected or filename in full:
                        raise TestPredictionError(
                            f"detector Fold {model.fold} returned unexpected/duplicate {filename!r}"
                        )
                    if any(box.class_id != 0 for box in boxes):
                        raise TestPredictionError(
                            f"detector Fold {model.fold} produced a non-fire class for {filename}"
                        )
                    full[filename] = (features.det_max_conf, boxes)
            if set(full) != set(expected):
                raise TestPredictionError(f"detector Fold {model.fold} omitted test images")
            predictions: dict[str, DetectorFoldPrediction] = {}
            for image in images:
                slice_score, tiles = _sliced_prediction(yolo, image, inference, device=device)
                full_score, boxes = full[image.filename]
                if self._spec.tta is None:
                    predictions[image.filename] = DetectorFoldPrediction(
                        full_score=full_score,
                        slice_score=slice_score,
                        boxes=boxes,
                        tiles=tiles,
                    )
                    continue
                hflip_full, hflip_slice, hflip_boxes, hflip_tiles = (
                    _horizontal_flip_detector_prediction(
                        yolo,
                        image,
                        inference,
                        device=device,
                    )
                )
                predictions[image.filename] = DetectorFoldPrediction(
                    full_score=full_score,
                    slice_score=slice_score,
                    boxes=boxes,
                    tiles=tiles,
                    hflip_full_score=hflip_full,
                    hflip_slice_score=hflip_slice,
                    hflip_boxes=hflip_boxes,
                    hflip_tiles=hflip_tiles,
                )
            return predictions
        except BaseException:
            self.close()
            raise

    def predict_classifier(
        self,
        model: ModelArtifact,
        images: Sequence[TestImage],
        *,
        device: str,
        imgsz: int,
        positive_class_name: str,
        augment: bool,
    ) -> Mapping[str, float | ClassifierFoldPrediction]:
        if augment:
            raise TestPredictionError("classifier TTA is forbidden")
        yolo = self._activate_model(model, detector=False)
        expected = {image.filename for image in images}
        probabilities: dict[str, float] = {}
        try:
            with _source_manifest(images) as sources:
                results = yolo.predict(
                    source=str(sources),
                    stream=True,
                    imgsz=imgsz,
                    augment=False,
                    device=device,
                    verbose=False,
                    save=False,
                )
                for result in results:
                    filename = canonical_filename(str(result.path))
                    if filename not in expected or filename in probabilities:
                        raise TestPredictionError(
                            "classifier Fold "
                            f"{model.fold} returned unexpected/duplicate {filename!r}"
                        )
                    if result.probs is None:
                        raise TestPredictionError(
                            f"classifier Fold {model.fold} returned no probabilities for {filename}"
                        )
                    probabilities[filename] = classifier_positive_probability(
                        names=result.names,
                        probabilities=tensor_to_list(result.probs.data),
                        positive_name=positive_class_name,
                    )
            if set(probabilities) != expected:
                raise TestPredictionError(f"classifier Fold {model.fold} omitted test images")
            if self._spec.tta is None:
                return probabilities
            return {
                image.filename: ClassifierFoldPrediction(
                    original_score=probabilities[image.filename],
                    hflip_score=_horizontal_flip_classifier_probability(
                        yolo,
                        image,
                        device=device,
                        imgsz=imgsz,
                        positive_class_name=positive_class_name,
                    ),
                )
                for image in images
            }
        except BaseException:
            self.close()
            raise

    def environment(self) -> Mapping[str, object]:
        return self._environment


def _close_prediction_runner(runner: PredictionRunner) -> None:
    """Close a runner that exposes an optional model-cache cleanup hook."""

    close = getattr(runner, "close", None)
    if callable(close):
        close()


def _validate_inference_spec(path: Path) -> DetectorInferenceSpec:
    inference = load_detector_inference_spec(path)
    expected = (
        inference.inference_imgsz == 640
        and inference.slicing.enabled
        and inference.slicing.size == 640
        and inference.slicing.overlap == 0.2
        and inference.slicing.stride == 512
        and inference.augment is False
    )
    if not expected:
        raise TestPredictionError(
            "final detector inference must be full 640 plus slice 640 at 20% overlap, "
            "without built-in TTA"
        )
    return inference


def _raw_box_payload(
    filename: str,
    model: ModelArtifact,
    prediction: DetectorFoldPrediction,
    inference: DetectorInferenceSpec,
    *,
    project_root: Path,
    tta: HorizontalFlipTTASpec | None = None,
) -> dict[str, object]:
    full_score = _probability(prediction.full_score, "full detector score")
    slice_score = _probability(prediction.slice_score, "slice detector score")
    boxes: list[dict[str, object]] = []
    for index, box in enumerate(prediction.boxes):
        if box.class_id != 0:
            raise TestPredictionError(f"raw detector box {index} has a non-fire class")
        _probability(box.conf, f"raw detector box {index} confidence")
        if any(not math.isfinite(float(value)) for value in box.xyxy):
            raise TestPredictionError(f"raw detector box {index} coordinates are non-finite")
        boxes.append(box.to_dict())
    payload: dict[str, object] = {
        "filename": filename,
        "model_fold": model.fold,
        "model_name": model.model_name,
        "weights_path": _portable_path(model.weights_path, project_root),
        "weights_sha256": model.weights_sha256,
        "full_max_conf": full_score,
        "slice_max_conf": slice_score,
        "full_slice_max_conf": max(full_score, slice_score),
        "inference": inference.raw_inference(training_imgsz=model.imgsz),
        "boxes": boxes,
        "tiles": [dict(tile) for tile in prediction.tiles],
    }
    if tta is None:
        return payload
    if prediction.hflip_full_score is None or prediction.hflip_slice_score is None:
        raise TestPredictionError("TTA detector result omits horizontal-flip scores")
    features = detector_tta_features(
        original_full_score=full_score,
        original_slice_score=slice_score,
        horizontal_flip_full_score=prediction.hflip_full_score,
        horizontal_flip_slice_score=prediction.hflip_slice_score,
    )
    hflip_boxes: list[dict[str, object]] = []
    for index, box in enumerate(prediction.hflip_boxes):
        if box.class_id != 0:
            raise TestPredictionError(f"horizontal-flip detector box {index} has a non-fire class")
        _probability(box.conf, f"horizontal-flip detector box {index} confidence")
        if any(not math.isfinite(float(value)) for value in box.xyxy):
            raise TestPredictionError(
                f"horizontal-flip detector box {index} coordinates are non-finite"
            )
        hflip_boxes.append(box.to_dict())
    payload["horizontal_flip"] = {
        "coordinates_restored_to_original": True,
        "full_max_conf": features["det_hflip_max_conf"],
        "slice_max_conf": features["slice_hflip_max_conf"],
        "full_slice_max_conf": features["det_hflip_full_slice_conf"],
        "boxes": hflip_boxes,
        "tiles": [dict(tile) for tile in prediction.hflip_tiles],
    }
    payload["tta_mean_conf"] = features["det_tta_mean_conf"]
    return payload


def _test_bundle_sha256(images: Sequence[TestImage]) -> str:
    payload = [image.manifest_record() for image in images]
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_jsonl_record(handle: Any, payload: Mapping[str, object]) -> None:
    handle.write(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _validate_restored_boxes(
    value: object,
    *,
    width: int,
    height: int,
    name: str,
) -> None:
    """Validate that declared restored xyxy coordinates lie in the original image."""

    if not isinstance(value, list):
        raise TestPredictionError(f"{name} must be a list")
    for index, raw_box in enumerate(value):
        box = _mapping(raw_box, f"{name}[{index}]")
        xyxy = box.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            raise TestPredictionError(f"{name}[{index}].xyxy must have four coordinates")
        coordinates: list[float] = []
        for coordinate in xyxy:
            if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
                raise TestPredictionError(f"{name}[{index}].xyxy is not numeric")
            number = float(coordinate)
            if not math.isfinite(number):
                raise TestPredictionError(f"{name}[{index}].xyxy is non-finite")
            coordinates.append(number)
        x1, y1, x2, y2 = coordinates
        if not (0.0 <= x1 <= x2 <= float(width) and 0.0 <= y1 <= y2 <= float(height)):
            raise TestPredictionError(
                f"{name}[{index}].xyxy is not restored to original-image bounds"
            )
        _probability(box.get("conf"), f"{name}[{index}].conf")
        if _strict_integer(box.get("class_id"), f"{name}[{index}].class_id") != 0:
            raise TestPredictionError(f"{name}[{index}] has a non-fire class")


def _runner_environment(runner: PredictionRunner) -> dict[str, object]:
    return dict(_mapping(runner.environment(), "prediction runner environment"))


def _tta_manifest_record(tta: HorizontalFlipTTASpec, *, root: Path) -> dict[str, object]:
    """Serialize the exact explicit-view TTA algorithm and immutable config hash."""

    return {
        **tta.manifest_record(),
        "config": {
            "path": _portable_path(tta.config_path, root),
            "sha256": tta.config_sha256,
        },
    }


def _fusion_manifest(
    spec: TestPredictionSpec,
    selection: FinalSelectionContract,
) -> dict[str, object]:
    """Return the schema-specific deterministic fusion and threshold contract."""

    if spec.tta is None:
        return {
            "method": "weighted_average",
            "detector_weight": 0.5,
            "classifier_weight": 0.5,
            "detector_fold_aggregation": "arithmetic_mean",
            "classifier_fold_aggregation": "arithmetic_mean",
            "detector_per_fold": "max(full_image,slice)",
            "threshold_applied": False,
            "voting": False,
            "tta": False,
        }
    if selection.stable_threshold is None:
        raise TestPredictionError("TTA final selection omits the stable threshold")
    return {
        "method": "weighted_average",
        "detector_weight": 1.0 - selection.classifier_weight,
        "classifier_weight": selection.classifier_weight,
        "detector_fold_aggregation": "arithmetic_mean",
        "classifier_fold_aggregation": "arithmetic_mean",
        "detector_per_view": "max(full_image,slice)",
        "detector_tta_aggregation": "arithmetic_mean(original,horizontal_flip)",
        "classifier_per_view": "positive_class_probability",
        "classifier_tta_aggregation": "arithmetic_mean(original,horizontal_flip)",
        "threshold_applied": True,
        "threshold": selection.stable_threshold,
        "prediction_rule": "probability >= threshold",
        "voting": False,
        "tta": True,
    }


def run_test_prediction(
    spec: TestPredictionSpec,
    *,
    runner: PredictionRunner | None = None,
    selection_validator: SelectionValidator | None = None,
) -> dict[str, Any]:
    """Run ten frozen checkpoints and atomically publish raw test probabilities."""

    if spec.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite test prediction output: {spec.output_dir}")
    selection = load_final_selection_contract(spec, validator=selection_validator)
    images = inspect_test_images(spec.test_images, expected_count=spec.expected_count)
    inference = _validate_inference_spec(spec.detector_inference_config)
    active_runner = runner or UltralyticsPredictionRunner(spec)
    spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = spec.output_dir.parent / f".{spec.output_dir.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    detector_predictions: dict[int, Mapping[str, DetectorFoldPrediction]] = {}
    classifier_predictions: dict[int, Mapping[str, float | ClassifierFoldPrediction]] = {}
    raw_count = 0
    started = time.perf_counter()
    raw_path = staging / spec.boxes_filename
    try:
        with raw_path.open("x", encoding="utf-8", newline="\n") as raw_handle:
            for model in selection.detector_models:
                predictions = active_runner.predict_detector(
                    model, images, inference, device=spec.ultralytics_device
                )
                normalized = _normalize_result_mapping(
                    predictions,
                    expected_names=[image.filename for image in images],
                    name=f"detector Fold {model.fold}",
                )
                typed: dict[str, DetectorFoldPrediction] = {}
                for image in images:
                    prediction = normalized[image.filename]
                    if not isinstance(prediction, DetectorFoldPrediction):
                        raise TestPredictionError(
                            f"detector Fold {model.fold} result has the wrong type"
                        )
                    typed[image.filename] = prediction
                    payload = _raw_box_payload(
                        image.filename,
                        model,
                        prediction,
                        inference,
                        project_root=spec.project_root,
                        tta=spec.tta,
                    )
                    _write_jsonl_record(raw_handle, payload)
                    raw_count += 1
                detector_predictions[model.fold] = typed
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        for model in selection.classifier_models:
            classifier_predictions[model.fold] = active_runner.predict_classifier(
                model,
                images,
                device=spec.ultralytics_device,
                imgsz=spec.classifier_imgsz,
                positive_class_name=spec.positive_class_name,
                augment=False,
            )
        predictions = aggregate_fold_predictions(
            [image.filename for image in images],
            detector_predictions,
            classifier_predictions,
            classifier_weight=selection.classifier_weight,
            tta=spec.tta,
            threshold=selection.stable_threshold,
        )
        predictions_path = staging / spec.predictions_filename
        atomic_write_csv(predictions_path, predictions)
        elapsed = time.perf_counter() - started
        manifest: dict[str, Any] = {
            "schema_version": spec.schema_version,
            "status": "complete",
            "task": "five_fold_test_prediction",
            "experiment_name": spec.experiment_name,
            "seed": spec.seed,
            "candidate": selection.candidate,
            "config": {
                "path": _portable_path(spec.config_path, spec.project_root),
                "sha256": sha256_file(spec.config_path),
            },
            "selection_manifest": {
                "path": _portable_path(spec.selection_manifest, spec.project_root),
                "sha256": selection.manifest_sha256,
            },
            "detector_inference_config": {
                "path": _portable_path(spec.detector_inference_config, spec.project_root),
                "sha256": inference.config_sha256,
                **inference.provenance(),
            },
            "models": {
                "detector": [
                    model.manifest_record(spec.project_root) for model in selection.detector_models
                ],
                "classifier": [
                    model.manifest_record(spec.project_root)
                    for model in selection.classifier_models
                ],
            },
            "test_images": {
                "directory": _portable_path(spec.test_images, spec.project_root),
                "count": len(images),
                "bundle_sha256": _test_bundle_sha256(images),
                "files": [image.manifest_record() for image in images],
            },
            "fusion": _fusion_manifest(spec, selection),
            "runtime": {
                "requested_device": spec.requested_device,
                "ultralytics_device": spec.ultralytics_device,
                "elapsed_seconds": elapsed,
                "environment": _runner_environment(active_runner),
            },
            "outputs": {
                "test_predictions_csv": {
                    "path": _portable_path(
                        spec.output_dir / spec.predictions_filename, spec.project_root
                    ),
                    "sha256": sha256_file(predictions_path),
                    "rows": len(predictions),
                },
                "detector_boxes_jsonl": {
                    "path": _portable_path(
                        spec.output_dir / spec.boxes_filename, spec.project_root
                    ),
                    "sha256": sha256_file(raw_path),
                    "rows": raw_count,
                },
            },
        }
        if spec.tta is not None:
            manifest["tta"] = _tta_manifest_record(spec.tta, root=spec.project_root)
        atomic_write_json(staging / spec.manifest_filename, manifest)
        staging.replace(spec.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        _close_prediction_runner(active_runner)
    return manifest


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int,
    classifier_weight: float,
    tta: HorizontalFlipTTASpec | None,
    threshold: float | None,
) -> tuple[str, ...]:
    """Recompute every saved view, Fold mean, fusion score, threshold, and decision."""

    expected_columns = ["filename"]
    for fold in sorted(EXPECTED_FOLDS):
        if tta is None:
            expected_columns.extend(
                [
                    f"detector_fold{fold}_full",
                    f"detector_fold{fold}_slice",
                    f"detector_fold{fold}",
                    f"classifier_fold{fold}",
                ]
            )
        else:
            expected_columns.extend(
                [
                    f"detector_fold{fold}_original_full",
                    f"detector_fold{fold}_original_slice",
                    f"detector_fold{fold}_original",
                    f"detector_fold{fold}_hflip_full",
                    f"detector_fold{fold}_hflip_slice",
                    f"detector_fold{fold}_hflip",
                    f"detector_fold{fold}_tta_mean",
                    f"classifier_fold{fold}_original",
                    f"classifier_fold{fold}_hflip",
                    f"classifier_fold{fold}_tta_mean",
                ]
            )
    expected_columns.extend(["detector_mean", "classifier_mean", "probability"])
    if tta is not None:
        expected_columns.extend(["threshold", "prediction"])
    if list(frame.columns) != expected_columns:
        raise TestPredictionError(
            "test prediction CSV columns differ from the frozen raw-probability schema"
        )
    if len(frame) != expected_rows:
        raise TestPredictionError(
            f"test prediction CSV has {len(frame)} rows, expected {expected_rows}"
        )
    raw_names = tuple(frame["filename"].tolist())
    if any(not isinstance(value, str) for value in raw_names):
        raise TestPredictionError("test prediction filenames must be strings")
    names = tuple(canonical_filename(value) for value in raw_names)
    if raw_names != names:
        raise TestPredictionError("test prediction filenames must be plain basenames")
    if len(set(names)) != len(names) or list(names) != sorted(names):
        raise TestPredictionError("test prediction filenames must be unique and sorted")
    for index, row in frame.iterrows():
        detector_scores: list[float] = []
        classifier_scores: list[float] = []
        for fold in sorted(EXPECTED_FOLDS):
            if tta is None:
                full = _probability(row[f"detector_fold{fold}_full"], "detector full score")
                sliced = _probability(row[f"detector_fold{fold}_slice"], "detector slice score")
                combined = _probability(row[f"detector_fold{fold}"], "detector Fold score")
                classifier = _probability(row[f"classifier_fold{fold}"], "classifier Fold score")
                if combined != max(full, sliced):
                    raise TestPredictionError(
                        f"test prediction row {index} detector Fold {fold} is not max(full,slice)"
                    )
            else:
                prefix = f"detector_fold{fold}"
                full = _probability(row[f"{prefix}_original_full"], "original full score")
                sliced = _probability(row[f"{prefix}_original_slice"], "original slice score")
                hflip_full = _probability(row[f"{prefix}_hflip_full"], "horizontal-flip full score")
                hflip_slice = _probability(
                    row[f"{prefix}_hflip_slice"], "horizontal-flip slice score"
                )
                features = detector_tta_features(
                    original_full_score=full,
                    original_slice_score=sliced,
                    horizontal_flip_full_score=hflip_full,
                    horizontal_flip_slice_score=hflip_slice,
                )
                expected_detector_fields = {
                    f"{prefix}_original": features["det_original_full_slice_conf"],
                    f"{prefix}_hflip": features["det_hflip_full_slice_conf"],
                    f"{prefix}_tta_mean": features["det_tta_mean_conf"],
                }
                for field, expected in expected_detector_fields.items():
                    actual = _probability(row[field], field)
                    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
                        raise TestPredictionError(
                            f"test prediction row {index} {field} is inconsistent"
                        )
                combined = features["det_tta_mean_conf"]
                cls_prefix = f"classifier_fold{fold}"
                cls_features = classifier_tta_features(
                    original_probability=row[f"{cls_prefix}_original"],
                    horizontal_flip_probability=row[f"{cls_prefix}_hflip"],
                )
                classifier = cls_features["cls_tta_mean_probability"]
                saved_classifier = _probability(
                    row[f"{cls_prefix}_tta_mean"], "classifier TTA mean"
                )
                if not math.isclose(
                    saved_classifier,
                    classifier,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise TestPredictionError(
                        f"test prediction row {index} classifier Fold {fold} mean is inconsistent"
                    )
            detector_scores.append(combined)
            classifier_scores.append(classifier)
        detector_mean = math.fsum(detector_scores) / len(EXPECTED_FOLDS)
        classifier_mean = math.fsum(classifier_scores) / len(EXPECTED_FOLDS)
        probability = (
            1.0 - classifier_weight
        ) * detector_mean + classifier_weight * classifier_mean
        if not math.isclose(
            _probability(row["detector_mean"], "detector_mean"),
            detector_mean,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise TestPredictionError(f"test prediction row {index} detector_mean is inconsistent")
        if not math.isclose(
            _probability(row["classifier_mean"], "classifier_mean"),
            classifier_mean,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise TestPredictionError(
                f"test prediction row {index} classifier_mean is inconsistent"
            )
        if not math.isclose(
            _probability(row["probability"], "probability"),
            probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise TestPredictionError(f"test prediction row {index} probability is inconsistent")
        if tta is not None:
            if threshold is None:
                raise TestPredictionError("TTA prediction validation requires a threshold")
            saved_threshold = _probability(row["threshold"], "stable threshold")
            if not math.isclose(
                saved_threshold,
                threshold,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise TestPredictionError(f"test prediction row {index} threshold is inconsistent")
            saved_prediction = row["prediction"]
            if (
                isinstance(saved_prediction, (bool, np.bool_))
                or not isinstance(saved_prediction, (int, np.integer))
                or int(saved_prediction) not in {0, 1}
                or int(saved_prediction) != int(probability >= threshold)
            ):
                raise TestPredictionError(f"test prediction row {index} prediction is inconsistent")
    return names


def validate_test_inference_manifest(
    path: Path,
    project_root: Path,
    *,
    selection_validator: SelectionValidator | None = None,
) -> tuple[dict[str, Any], str]:
    """Revalidate a published inference manifest, its CSV formula, and raw-box JSONL."""

    root = project_root.resolve()
    manifest_path = path.resolve() if path.is_absolute() else (root / path).resolve()
    manifest = _load_json_object(manifest_path, "test inference manifest")
    schema_version = _strict_integer(
        manifest.get("schema_version"), "test inference manifest schema_version", minimum=1
    )
    if schema_version not in {1, 2}:
        raise TestPredictionError("test inference manifest schema_version is invalid")
    expected_scalars: dict[str, object] = {
        "status": "complete",
        "task": "five_fold_test_prediction",
    }
    for field, expected in expected_scalars.items():
        if type(manifest.get(field)) is not type(expected) or manifest.get(field) != expected:
            raise TestPredictionError(f"test inference manifest {field} is invalid")
    config_path, _config_digest = _verified_record(
        manifest.get("config"), root=root, name="test inference config"
    )
    detector_inference_path, _detector_inference_digest = _verified_record(
        manifest.get("detector_inference_config"),
        root=root,
        name="detector inference config",
    )
    test_images = _mapping(manifest.get("test_images"), "test inference test_images")
    expected_rows = _strict_integer(test_images.get("count"), "test_images.count", minimum=1)
    test_directory = _resolve_portable_path(
        test_images.get("directory"), root=root, name="test_images.directory"
    )
    if not test_directory.is_dir():
        raise TestPredictionError(f"test image directory does not exist: {test_directory}")
    bundle = _sha256_value(test_images.get("bundle_sha256"), "test_images.bundle_sha256")
    files = test_images.get("files")
    if not isinstance(files, list) or len(files) != expected_rows:
        raise TestPredictionError("test_images.files does not match test_images.count")
    serialized_manifest_files = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if hashlib.sha256(serialized_manifest_files.encode("utf-8")).hexdigest() != bundle:
        raise TestPredictionError("test_images.bundle_sha256 is inconsistent")
    current_images = inspect_test_images(test_directory, expected_count=expected_rows)
    current_file_records = [image.manifest_record() for image in current_images]
    serialized_current_files = json.dumps(
        current_file_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if serialized_manifest_files != serialized_current_files:
        raise TestPredictionError("test image content, dimensions, or SHA-256 drifted")
    if _test_bundle_sha256(current_images) != bundle:
        raise TestPredictionError("test image bundle SHA-256 drifted")

    spec = load_test_prediction_spec(
        config_path,
        project_root=root,
        test_dir_override=test_directory,
        output_dir_override=manifest_path.parent,
        expected_count_override=expected_rows,
    )
    if schema_version != spec.schema_version:
        raise TestPredictionError("test inference schema_version drifted from config")
    manifest_scalars: dict[str, object] = {
        "experiment_name": spec.experiment_name,
        "seed": spec.seed,
        "candidate": spec.candidate,
    }
    for field, expected in manifest_scalars.items():
        if type(manifest.get(field)) is not type(expected) or manifest.get(field) != expected:
            raise TestPredictionError(f"test inference manifest {field} drifted from config")
    if manifest_path.name != spec.manifest_filename:
        raise TestPredictionError("test inference manifest filename drifted from config")
    if detector_inference_path != spec.detector_inference_config:
        raise TestPredictionError("detector inference config path drifted from prediction config")

    selection_path, selection_digest = _verified_record(
        manifest.get("selection_manifest"),
        root=root,
        name="final-selection manifest",
        expected_path=spec.selection_manifest,
    )
    selection = load_final_selection_contract(spec, validator=selection_validator)
    if selection_path != spec.selection_manifest or selection.manifest_sha256 != selection_digest:
        raise TestPredictionError("final-selection manifest provenance drifted")
    if spec.tta is None:
        if "tta" in manifest:
            raise TestPredictionError("schema-v1 test inference manifest must not contain TTA")
    else:
        actual_tta = _mapping(manifest.get("tta"), "test inference TTA")
        expected_tta = _tta_manifest_record(spec.tta, root=root)
        if actual_tta != expected_tta:
            raise TestPredictionError("test inference TTA contract or config hash drifted")
    expected_models = {
        "detector": [model.manifest_record(root) for model in selection.detector_models],
        "classifier": [model.manifest_record(root) for model in selection.classifier_models],
    }
    models = _mapping(manifest.get("models"), "test inference models")
    if set(models) != {"detector", "classifier"}:
        raise TestPredictionError("test inference models must contain detector and classifier")
    serialized_models = json.dumps(
        dict(models), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    serialized_expected_models = json.dumps(
        expected_models, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if serialized_models != serialized_expected_models:
        raise TestPredictionError("test inference models drifted from final selection")

    expected_fusion = _fusion_manifest(spec, selection)
    fusion = _mapping(manifest.get("fusion"), "test inference fusion")
    if set(fusion) != set(expected_fusion) or any(
        type(fusion.get(field)) is not type(expected) or fusion.get(field) != expected
        for field, expected in expected_fusion.items()
    ):
        raise TestPredictionError("test inference fusion contract drifted")

    outputs = _mapping(manifest.get("outputs"), "test inference outputs")
    predictions_record = _mapping(
        outputs.get("test_predictions_csv"), "outputs.test_predictions_csv"
    )
    predictions_path, _digest = _verified_record(
        predictions_record,
        root=root,
        name="test predictions CSV",
        expected_path=manifest_path.parent / spec.predictions_filename,
    )
    if (
        _strict_integer(
            predictions_record.get("rows"), "outputs.test_predictions_csv.rows", minimum=1
        )
        != expected_rows
    ):
        raise TestPredictionError("test prediction row count does not match test image count")
    try:
        frame = pd.read_csv(predictions_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise TestPredictionError(f"cannot read test prediction CSV: {exc}") from exc
    names = _validate_prediction_frame(
        frame,
        expected_rows=expected_rows,
        classifier_weight=selection.classifier_weight,
        tta=spec.tta,
        threshold=selection.stable_threshold,
    )
    current_names = tuple(image.filename for image in current_images)
    if names != current_names:
        raise TestPredictionError("test prediction filenames differ from current test images")
    frame_by_name = frame.set_index("filename")
    image_dimensions = {image.filename: (image.width, image.height) for image in current_images}
    name_set = set(names)

    boxes_record = _mapping(outputs.get("detector_boxes_jsonl"), "outputs.detector_boxes_jsonl")
    boxes_path, _boxes_digest = _verified_record(
        boxes_record,
        root=root,
        name="detector boxes JSONL",
        expected_path=manifest_path.parent / spec.boxes_filename,
    )
    expected_box_rows = expected_rows * len(EXPECTED_FOLDS)
    if (
        _strict_integer(boxes_record.get("rows"), "outputs.detector_boxes_jsonl.rows", minimum=1)
        != expected_box_rows
    ):
        raise TestPredictionError("detector box row count must equal five times the image count")
    seen: set[tuple[str, int]] = set()
    try:
        with boxes_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise TestPredictionError(f"detector boxes JSONL line {line_number} is empty")
                record = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        TestPredictionError(
                            f"detector boxes JSONL contains non-standard constant {value}"
                        )
                    ),
                )
                item = _mapping(record, f"detector boxes JSONL line {line_number}")
                filename = canonical_filename(item.get("filename"))
                fold = _strict_integer(item.get("model_fold"), "detector box model_fold")
                if filename not in name_set or fold not in EXPECTED_FOLDS:
                    raise TestPredictionError("detector boxes JSONL has an unexpected image/Fold")
                key = (filename, fold)
                if key in seen:
                    raise TestPredictionError("detector boxes JSONL has duplicate image/Fold rows")
                full_score = _probability(item.get("full_max_conf"), "detector boxes full_max_conf")
                slice_score = _probability(
                    item.get("slice_max_conf"), "detector boxes slice_max_conf"
                )
                combined_score = _probability(
                    item.get("full_slice_max_conf"), "detector boxes combined score"
                )
                if combined_score != max(full_score, slice_score):
                    raise TestPredictionError(
                        "detector boxes combined score is not max(full,slice)"
                    )
                if spec.tta is None:
                    csv_score = _probability(
                        frame_by_name.at[filename, f"detector_fold{fold}"],
                        "test prediction detector Fold score",
                    )
                    if not math.isclose(
                        combined_score,
                        csv_score,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise TestPredictionError(
                            "detector boxes combined score differs from test prediction CSV"
                        )
                else:
                    original_csv = _probability(
                        frame_by_name.at[filename, f"detector_fold{fold}_original"],
                        "test prediction detector original-view score",
                    )
                    if not math.isclose(
                        combined_score,
                        original_csv,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise TestPredictionError(
                            "detector original-view score differs from test prediction CSV"
                        )
                    horizontal_flip = _mapping(
                        item.get("horizontal_flip"),
                        "detector boxes horizontal_flip",
                    )
                    if horizontal_flip.get("coordinates_restored_to_original") is not True:
                        raise TestPredictionError(
                            "horizontal-flip detector coordinates are not declared restored"
                        )
                    hflip_full = _probability(
                        horizontal_flip.get("full_max_conf"),
                        "horizontal-flip detector full score",
                    )
                    hflip_slice = _probability(
                        horizontal_flip.get("slice_max_conf"),
                        "horizontal-flip detector slice score",
                    )
                    hflip_combined = _probability(
                        horizontal_flip.get("full_slice_max_conf"),
                        "horizontal-flip detector combined score",
                    )
                    if hflip_combined != max(hflip_full, hflip_slice):
                        raise TestPredictionError(
                            "horizontal-flip detector score is not max(full,slice)"
                        )
                    saved_hflip_scores = {
                        f"detector_fold{fold}_hflip_full": hflip_full,
                        f"detector_fold{fold}_hflip_slice": hflip_slice,
                        f"detector_fold{fold}_hflip": hflip_combined,
                    }
                    for field, expected in saved_hflip_scores.items():
                        actual = _probability(
                            frame_by_name.at[filename, field],
                            f"test prediction {field}",
                        )
                        if not math.isclose(
                            actual,
                            expected,
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        ):
                            raise TestPredictionError(
                                "horizontal-flip detector raw score differs from prediction CSV"
                            )
                    expected_tta_mean = arithmetic_mean_score(
                        combined_score,
                        hflip_combined,
                    )
                    raw_tta_mean = _probability(
                        item.get("tta_mean_conf"), "detector boxes TTA mean"
                    )
                    csv_tta_mean = _probability(
                        frame_by_name.at[filename, f"detector_fold{fold}_tta_mean"],
                        "test prediction detector TTA mean",
                    )
                    for actual in (raw_tta_mean, csv_tta_mean):
                        if not math.isclose(
                            actual,
                            expected_tta_mean,
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        ):
                            raise TestPredictionError(
                                "detector TTA mean differs from raw views or prediction CSV"
                            )
                    width, height = image_dimensions[filename]
                    _validate_restored_boxes(
                        horizontal_flip.get("boxes"),
                        width=width,
                        height=height,
                        name="horizontal-flip detector boxes",
                    )
                    raw_tiles = horizontal_flip.get("tiles")
                    if not isinstance(raw_tiles, list):
                        raise TestPredictionError("horizontal-flip detector tiles must be a list")
                    for tile_index, raw_tile in enumerate(raw_tiles):
                        tile = _mapping(
                            raw_tile,
                            f"horizontal-flip detector tiles[{tile_index}]",
                        )
                        _validate_restored_boxes(
                            tile.get("boxes"),
                            width=width,
                            height=height,
                            name=f"horizontal-flip detector tiles[{tile_index}].boxes",
                        )
                seen.add(key)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TestPredictionError(f"cannot read detector boxes JSONL: {exc}") from exc
    expected_pairs = {(filename, fold) for filename in names for fold in EXPECTED_FOLDS}
    if seen != expected_pairs:
        raise TestPredictionError("detector boxes JSONL image/Fold coverage is incomplete")
    return manifest, sha256_file(manifest_path)


__all__ = [
    "ClassifierFoldPrediction",
    "DetectorFoldPrediction",
    "FinalSelectionContract",
    "ModelArtifact",
    "PredictionRunner",
    "TestImage",
    "TestPredictionError",
    "TestPredictionSpec",
    "aggregate_fold_predictions",
    "inspect_test_images",
    "load_final_selection_contract",
    "load_test_prediction_spec",
    "run_test_prediction",
    "validate_test_inference_manifest",
    "validate_test_image_paths",
]
