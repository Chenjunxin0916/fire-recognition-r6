"""Unified PyTorch deployment interface for the frozen ten-model YOLO26 ensemble."""

from __future__ import annotations

import gc
import json
import math
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

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
    DetectorInferenceConfigurationError,
    DetectorInferenceSpec,
    load_detector_inference_spec,
)
from fire_detection.inference.oof import canonical_filename
from fire_detection.inference.test_prediction import (
    ClassifierFoldPrediction,
    DetectorFoldPrediction,
    FinalSelectionContract,
    ModelArtifact,
    PredictionRunner,
    SelectionValidator,
    TestImage,
    TestPredictionError,
    TestPredictionSpec,
    _horizontal_flip_classifier_probability,
    _horizontal_flip_detector_prediction,
    _sliced_prediction,
    _source_manifest,
    _verify_detector_model,
    aggregate_fold_predictions,
    load_final_selection_contract,
    load_test_prediction_spec,
    validate_test_image_paths,
)
from fire_detection.models.features import (
    PredictionValidationError,
    RawBox,
    classifier_positive_probability,
    extract_ultralytics_detection_result,
    resolve_positive_class_index,
    tensor_to_list,
)
from fire_detection.models.offline import (
    configure_imported_ultralytics,
    validate_offline_runtime,
)


class DeploymentConfigError(ValueError):
    """Raised when deployment configuration or frozen metadata is invalid."""


class DeploymentPredictionError(ValueError):
    """Raised when an input image or model result cannot be predicted safely."""


class PredictorNotReadyError(RuntimeError):
    """Raised when an inference endpoint has no initialized predictor."""


@dataclass(frozen=True)
class DeploymentSpec:
    """Strict deployment settings layered over the frozen test-prediction contract."""

    project_root: Path
    config_path: Path
    prediction_config: Path
    selection_manifest: Path
    threshold_profile: str
    device: str
    max_batch_size: int
    max_upload_bytes: int
    box_confidence: float
    max_boxes: int
    gradio_enabled: bool
    fastapi_enabled: bool
    host: str
    port: int
    onnx_enabled: bool
    tensorrt_enabled: bool


@dataclass(frozen=True)
class FireBox:
    """One explanatory detection from a detector Fold and inference view."""

    fold: int
    source: Literal["full", "slice"]
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe box record."""

        return {
            "fold": self.fold,
            "source": self.source,
            "xyxy": list(self.xyxy),
            "confidence": self.confidence,
            "class_id": self.class_id,
        }


@dataclass(frozen=True)
class FirePrediction:
    """Image-level fire decision plus explanatory detector boxes."""

    filename: str
    detector_score: float
    classifier_score: float
    fused_score: float
    label: int
    threshold: float
    boxes: tuple[FireBox, ...]
    elapsed_ms: float
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe prediction payload."""

        return {
            "filename": self.filename,
            "detector_score": self.detector_score,
            "classifier_score": self.classifier_score,
            "fused_score": self.fused_score,
            "label": self.label,
            "threshold": self.threshold,
            "boxes": [box.to_dict() for box in self.boxes],
            "elapsed_ms": self.elapsed_ms,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class FirePredictionBatch:
    """One atomic batch prediction and its shared wall-clock duration."""

    predictions: tuple[FirePrediction, ...]
    elapsed_ms: float
    threshold: float
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe batch payload."""

        return {
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "elapsed_ms": self.elapsed_ms,
            "threshold": self.threshold,
            "manifest_sha256": self.manifest_sha256,
        }


class PreloadablePredictionRunner(PredictionRunner, Protocol):
    """Prediction runner that can preload and later release selected checkpoints."""

    def preload(self, contract: FinalSelectionContract) -> None: ...

    def close(self) -> None: ...


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise DeploymentConfigError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _probability(
    value: object,
    name: str,
    *,
    error_type: type[ValueError] = DeploymentConfigError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{name} must be a non-boolean real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise error_type(f"{name} must be finite and in [0, 1]")
    return result


def load_deployment_spec(path: Path, *, project_root: Path) -> DeploymentSpec:
    """Load the deployment YAML without silently inventing model defaults."""

    root = project_root.resolve()
    config_path = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        raw = load_yaml_mapping(config_path)
        _exact_keys(
            raw,
            {
                "schema_version",
                "application",
                "model",
                "limits",
                "visualization",
                "onnx_runtime",
            },
            "deployment",
        )
        if require_int(raw, "schema_version", context="deployment", minimum=1) != 1:
            raise DeploymentConfigError("deployment.schema_version must be 1")
        application = require_mapping(raw, "application", context="deployment")
        model = require_mapping(raw, "model", context="deployment")
        limits = require_mapping(raw, "limits", context="deployment")
        visualization = require_mapping(raw, "visualization", context="deployment")
        onnx = require_mapping(raw, "onnx_runtime", context="deployment")
        _exact_keys(
            application,
            {"gradio_enabled", "fastapi_enabled", "host", "port"},
            "deployment.application",
        )
        _exact_keys(
            model,
            {"prediction_config", "selection_manifest", "threshold_profile", "device"},
            "deployment.model",
        )
        _exact_keys(
            limits,
            {"max_batch_size", "max_upload_bytes"},
            "deployment.limits",
        )
        _exact_keys(
            visualization,
            {"box_confidence", "max_boxes"},
            "deployment.visualization",
        )
        _exact_keys(
            onnx,
            {"enabled", "providers", "tensorrt_enabled"},
            "deployment.onnx_runtime",
        )
        providers = onnx.get("providers")
        if (
            not isinstance(providers, list)
            or not providers
            or not all(isinstance(provider, str) and provider.strip() for provider in providers)
        ):
            raise DeploymentConfigError(
                "deployment.onnx_runtime.providers must be a non-empty string list"
            )
        threshold_profile = require_string(model, "threshold_profile", context="deployment.model")
        if threshold_profile not in {"stable", "recall", "precision"}:
            raise DeploymentConfigError(
                "deployment.model.threshold_profile must be stable, recall, or precision"
            )
        host = require_string(application, "host", context="deployment.application")
        port = require_int(application, "port", context="deployment.application", minimum=1)
        if port > 65535:
            raise DeploymentConfigError("deployment.application.port must be <= 65535")
        box_confidence = require_float(
            visualization,
            "box_confidence",
            context="deployment.visualization",
            minimum=0.0,
        )
        if box_confidence > 1.0:
            raise DeploymentConfigError("deployment.visualization.box_confidence must be <= 1.0")
        return DeploymentSpec(
            project_root=root,
            config_path=config_path,
            prediction_config=resolve_project_path(
                root,
                require_string(model, "prediction_config", context="deployment.model"),
            ).resolve(),
            selection_manifest=resolve_project_path(
                root,
                require_string(model, "selection_manifest", context="deployment.model"),
            ).resolve(),
            threshold_profile=threshold_profile,
            device=require_string(model, "device", context="deployment.model"),
            max_batch_size=require_int(
                limits, "max_batch_size", context="deployment.limits", minimum=1
            ),
            max_upload_bytes=require_int(
                limits, "max_upload_bytes", context="deployment.limits", minimum=1
            ),
            box_confidence=box_confidence,
            max_boxes=require_int(
                visualization, "max_boxes", context="deployment.visualization", minimum=1
            ),
            gradio_enabled=require_bool(
                application, "gradio_enabled", context="deployment.application"
            ),
            fastapi_enabled=require_bool(
                application, "fastapi_enabled", context="deployment.application"
            ),
            host=host,
            port=port,
            onnx_enabled=require_bool(onnx, "enabled", context="deployment.onnx_runtime"),
            tensorrt_enabled=require_bool(
                onnx, "tensorrt_enabled", context="deployment.onnx_runtime"
            ),
        )
    except (ConfigError, OSError) as exc:
        raise DeploymentConfigError(f"cannot load deployment config {config_path}: {exc}") from exc


def _read_manifest(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise DeploymentConfigError(f"selection manifest contains non-standard value {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DeploymentConfigError(
                    f"selection manifest contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentConfigError(f"cannot read selection manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeploymentConfigError("selection manifest root must be an object")
    return value


def _manifest_threshold(path: Path, profile: str) -> float:
    manifest = _read_manifest(path)
    profiles = manifest.get("threshold_profiles")
    if not isinstance(profiles, Mapping):
        raise DeploymentConfigError("selection manifest has no threshold_profiles mapping")
    selected = profiles.get(profile)
    if not isinstance(selected, Mapping):
        raise DeploymentConfigError(f"selection manifest has no {profile!r} threshold profile")
    return _probability(selected.get("threshold"), f"threshold_profiles.{profile}.threshold")


def deployment_threshold(path: Path, *, project_root: Path) -> float:
    """Read the configured manifest threshold without initializing model weights."""

    spec = load_deployment_spec(path, project_root=project_root)
    return _manifest_threshold(spec.selection_manifest, spec.threshold_profile)


class CachedUltralyticsPredictionRunner:
    """Offline Ultralytics runner that keeps the ten selected PyTorch models in memory."""

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
            raise PredictorNotReadyError(
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
        self._models: dict[tuple[str, int], Any] = {}

    def preload(self, contract: FinalSelectionContract) -> None:
        """Load and validate every detector/classifier checkpoint exactly once."""

        if self._models:
            raise PredictorNotReadyError("deployment models are already loaded")
        try:
            for artifact in contract.detector_models:
                model = self._yolo(str(artifact.weights_path), verbose=False)
                _verify_detector_model(model)
                self._models[(artifact.task, artifact.fold)] = model
            for artifact in contract.classifier_models:
                model = self._yolo(str(artifact.weights_path), verbose=False)
                resolve_positive_class_index(model.names, "1")
                self._models[(artifact.task, artifact.fold)] = model
        except (OSError, RuntimeError, PredictionValidationError) as exc:
            self.close()
            raise PredictorNotReadyError(f"cannot preload frozen YOLO26 ensemble: {exc}") from exc
        if len(self._models) != 10:
            self.close()
            raise PredictorNotReadyError(
                f"frozen deployment requires ten models, loaded {len(self._models)}"
            )

    def _model(self, artifact: ModelArtifact) -> Any:
        try:
            return self._models[(artifact.task, artifact.fold)]
        except KeyError as exc:
            raise PredictorNotReadyError(
                f"{artifact.task} Fold {artifact.fold} is not preloaded"
            ) from exc

    def predict_detector(
        self,
        model: ModelArtifact,
        images: Sequence[TestImage],
        inference: DetectorInferenceSpec,
        *,
        device: str,
    ) -> Mapping[str, DetectorFoldPrediction]:
        yolo = self._model(model)
        expected = {image.filename: image for image in images}
        full: dict[str, tuple[float, tuple[RawBox, ...]]] = {}
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
                    raise DeploymentPredictionError(
                        f"detector Fold {model.fold} returned unexpected/duplicate {filename!r}"
                    )
                if any(box.class_id != 0 for box in boxes):
                    raise DeploymentPredictionError(
                        f"detector Fold {model.fold} produced a non-fire class for {filename}"
                    )
                full[filename] = (features.det_max_conf, boxes)
        if set(full) != set(expected):
            raise DeploymentPredictionError(f"detector Fold {model.fold} omitted input images")
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
            raise DeploymentPredictionError("classifier TTA is forbidden by the frozen contract")
        yolo = self._model(model)
        expected = {image.filename for image in images}
        probabilities: dict[str, float] = {}
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
                    raise DeploymentPredictionError(
                        f"classifier Fold {model.fold} returned unexpected/duplicate {filename!r}"
                    )
                if result.probs is None:
                    raise DeploymentPredictionError(
                        f"classifier Fold {model.fold} returned no probabilities for {filename}"
                    )
                probabilities[filename] = classifier_positive_probability(
                    names=result.names,
                    probabilities=tensor_to_list(result.probs.data),
                    positive_name=positive_class_name,
                )
        if set(probabilities) != expected:
            raise DeploymentPredictionError(f"classifier Fold {model.fold} omitted input images")
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

    def environment(self) -> Mapping[str, object]:
        return self._environment

    def close(self) -> None:
        """Release cached model references and CUDA allocator cache."""

        self._models.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return


def _coordinates(value: object, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise DeploymentPredictionError(f"{name} must contain four coordinates")
    coordinates: list[float] = []
    for offset, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, Real) or not math.isfinite(float(raw)):
            raise DeploymentPredictionError(f"{name}[{offset}] must be finite")
        coordinates.append(float(raw))
    if coordinates[2] < coordinates[0] or coordinates[3] < coordinates[1]:
        raise DeploymentPredictionError(f"{name} contains reversed coordinates")
    return cast(tuple[float, float, float, float], tuple(coordinates))


def _explanatory_boxes(
    detector: Mapping[int, Mapping[str, DetectorFoldPrediction]],
    filename: str,
    *,
    confidence_threshold: float,
    max_boxes: int,
) -> tuple[FireBox, ...]:
    boxes: list[FireBox] = []
    for fold in sorted(detector):
        prediction = detector[fold][filename]
        for raw in prediction.boxes:
            if raw.conf >= confidence_threshold:
                boxes.append(FireBox(fold, "full", raw.xyxy, raw.conf, raw.class_id))
        for tile_offset, tile in enumerate(prediction.tiles):
            raw_boxes = tile.get("boxes")
            if not isinstance(raw_boxes, list):
                raise DeploymentPredictionError(
                    f"detector Fold {fold} tile {tile_offset} boxes must be a list"
                )
            for box_offset, raw_box in enumerate(raw_boxes):
                if not isinstance(raw_box, Mapping):
                    raise DeploymentPredictionError(
                        f"detector Fold {fold} tile {tile_offset} box {box_offset} is invalid"
                    )
                confidence = _probability(
                    raw_box.get("conf"),
                    f"detector Fold {fold} tile {tile_offset} box {box_offset}.conf",
                    error_type=DeploymentPredictionError,
                )
                class_id = raw_box.get("class_id")
                if type(class_id) is not int or class_id < 0:
                    raise DeploymentPredictionError(
                        f"detector Fold {fold} tile {tile_offset} box {box_offset}.class_id "
                        "must be a non-negative integer"
                    )
                if confidence >= confidence_threshold:
                    boxes.append(
                        FireBox(
                            fold,
                            "slice",
                            _coordinates(
                                raw_box.get("xyxy"),
                                f"detector Fold {fold} tile {tile_offset} box {box_offset}.xyxy",
                            ),
                            confidence,
                            class_id,
                        )
                    )
    boxes.sort(key=lambda item: (-item.confidence, item.fold, item.source, item.xyxy))
    return tuple(boxes[:max_boxes])


class FirePredictor:
    """Thread-safe image-level predictor bound to one frozen final-selection manifest."""

    def __init__(
        self,
        *,
        deployment_spec: DeploymentSpec,
        prediction_spec: TestPredictionSpec,
        contract: FinalSelectionContract,
        inference_spec: DetectorInferenceSpec,
        runner: PredictionRunner,
        threshold: float,
    ) -> None:
        self.deployment_spec = deployment_spec
        self.prediction_spec = prediction_spec
        self.contract = contract
        self.inference_spec = inference_spec
        self._runner = runner
        self._threshold = _probability(threshold, "deployment threshold")
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        path: Path = Path("configs/deploy.yaml"),
        *,
        project_root: Path | None = None,
        runner: PredictionRunner | None = None,
        selection_validator: SelectionValidator | None = None,
        runner_factory: Callable[[TestPredictionSpec], PreloadablePredictionRunner] | None = None,
    ) -> FirePredictor:
        """Validate the frozen selection, verify ten hashes, and preload PyTorch weights."""

        try:
            deployment = load_deployment_spec(path, project_root=project_root or Path.cwd())
            if deployment.onnx_enabled or deployment.tensorrt_enabled:
                raise DeploymentConfigError(
                    "FirePredictor is the PyTorch runtime; ONNX and TensorRT must remain disabled"
                )
            prediction = load_test_prediction_spec(
                deployment.prediction_config,
                project_root=deployment.project_root,
                device_override=deployment.device,
            )
            if prediction.selection_manifest != deployment.selection_manifest:
                raise DeploymentConfigError(
                    "deployment selection_manifest differs from the frozen prediction config"
                )
            contract = load_final_selection_contract(
                prediction,
                validator=selection_validator,
            )
            inference = load_detector_inference_spec(prediction.detector_inference_config)
            if inference.augment:
                raise DeploymentConfigError("deployment detector TTA must remain disabled")
            if not inference.slicing.enabled:
                raise DeploymentConfigError("frozen deployment detector must use slicing")
            threshold = _manifest_threshold(
                deployment.selection_manifest, deployment.threshold_profile
            )
            selected_runner: PredictionRunner
            if runner is not None:
                selected_runner = runner
            else:
                preloadable = (runner_factory or CachedUltralyticsPredictionRunner)(prediction)
                preloadable.preload(contract)
                selected_runner = preloadable
            return cls(
                deployment_spec=deployment,
                prediction_spec=prediction,
                contract=contract,
                inference_spec=inference,
                runner=selected_runner,
                threshold=threshold,
            )
        except (
            ConfigError,
            DetectorInferenceConfigurationError,
            TestPredictionError,
            OSError,
        ) as exc:
            raise DeploymentConfigError(f"cannot initialize FirePredictor: {exc}") from exc

    @property
    def threshold(self) -> float:
        """Return the immutable manifest-selected deployment threshold."""

        return self._threshold

    @property
    def manifest_sha256(self) -> str:
        """Return the verified final-selection manifest digest."""

        return self.contract.manifest_sha256

    @property
    def device(self) -> str:
        """Return the configured user-facing device string."""

        return self.prediction_spec.requested_device

    @property
    def model_count(self) -> int:
        """Return the number of verified detector and classifier Fold models."""

        return len(self.contract.detector_models) + len(self.contract.classifier_models)

    def environment(self) -> dict[str, object]:
        """Return the runner environment captured during initialization."""

        return dict(self._runner.environment())

    def predict(self, path: Path, *, threshold: float | None = None) -> FirePrediction:
        """Predict one image; a threshold override affects only this returned label."""

        return self.predict_batch((path,), threshold=threshold).predictions[0]

    def predict_batch(
        self,
        paths: Sequence[Path],
        *,
        threshold: float | None = None,
    ) -> FirePredictionBatch:
        """Predict a flat image batch with the frozen full+slice, five-Fold ensemble."""

        if not paths:
            raise DeploymentPredictionError("at least one image is required")
        if len(paths) > self.deployment_spec.max_batch_size:
            raise DeploymentPredictionError(
                f"batch contains {len(paths)} images; maximum is "
                f"{self.deployment_spec.max_batch_size}"
            )
        normalized_names: set[str] = set()
        for path in paths:
            filename = canonical_filename(path.name)
            normalized = filename.casefold()
            if normalized in normalized_names:
                raise DeploymentPredictionError(
                    f"input images contain a case-insensitive duplicate filename: {filename}"
                )
            normalized_names.add(normalized)
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise DeploymentPredictionError(
                    f"cannot inspect input image {path}: {exc}"
                ) from exc
            if size > self.deployment_spec.max_upload_bytes:
                raise DeploymentPredictionError(
                    f"input image {filename!r} exceeds "
                    f"{self.deployment_spec.max_upload_bytes} bytes"
                )
        selected_threshold = (
            self._threshold
            if threshold is None
            else _probability(
                threshold,
                "request threshold",
                error_type=DeploymentPredictionError,
            )
        )
        try:
            images = validate_test_image_paths(tuple(paths), expected_count=len(paths))
        except TestPredictionError as exc:
            raise DeploymentPredictionError(str(exc)) from exc

        started = time.perf_counter()
        detector: dict[int, Mapping[str, DetectorFoldPrediction]] = {}
        classifier: dict[int, Mapping[str, float | ClassifierFoldPrediction]] = {}
        try:
            with self._lock:
                for artifact in self.contract.detector_models:
                    detector[artifact.fold] = self._runner.predict_detector(
                        artifact,
                        images,
                        self.inference_spec,
                        device=self.prediction_spec.ultralytics_device,
                    )
                for artifact in self.contract.classifier_models:
                    classifier[artifact.fold] = self._runner.predict_classifier(
                        artifact,
                        images,
                        device=self.prediction_spec.ultralytics_device,
                        imgsz=self.prediction_spec.classifier_imgsz,
                        positive_class_name=self.prediction_spec.positive_class_name,
                        augment=False,
                    )
                frame = aggregate_fold_predictions(
                    tuple(image.filename for image in images),
                    detector,
                    classifier,
                    classifier_weight=self.contract.classifier_weight,
                    tta=self.prediction_spec.tta,
                    threshold=self.contract.stable_threshold,
                )
        except (
            TestPredictionError,
            PredictionValidationError,
            DeploymentPredictionError,
            PredictorNotReadyError,
            OSError,
            RuntimeError,
        ) as exc:
            raise DeploymentPredictionError(f"YOLO26 ensemble inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        by_filename = {str(row["filename"]): row for row in frame.to_dict(orient="records")}
        predictions: list[FirePrediction] = []
        for image in images:
            row = by_filename[image.filename]
            fused = float(row["probability"])
            predictions.append(
                FirePrediction(
                    filename=image.filename,
                    detector_score=float(row["detector_mean"]),
                    classifier_score=float(row["classifier_mean"]),
                    fused_score=fused,
                    label=int(fused >= selected_threshold),
                    threshold=selected_threshold,
                    boxes=_explanatory_boxes(
                        detector,
                        image.filename,
                        confidence_threshold=self.deployment_spec.box_confidence,
                        max_boxes=self.deployment_spec.max_boxes,
                    ),
                    elapsed_ms=elapsed_ms,
                    manifest_sha256=self.manifest_sha256,
                )
            )
        return FirePredictionBatch(
            predictions=tuple(predictions),
            elapsed_ms=elapsed_ms,
            threshold=selected_threshold,
            manifest_sha256=self.manifest_sha256,
        )

    def close(self) -> None:
        """Release model resources when supported by the configured runner."""

        close = getattr(self._runner, "close", None)
        if callable(close):
            close()
